import asyncio
import json
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from Backend.audio.audio_utils import (
    TARGET_SAMPLE_RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    StreamResampler,
)
from Backend.audio.buffer import AudioBuffer
from Backend.audio.vad import VoiceActivityDetector
from Backend.audio.background_mixer import BackgroundMixer
from Backend.stt_plugin import CohereArabicSTT

from Backend.agent.graph import agent_graph
from Backend.agent.interview_manager import InterviewManager
from Backend.tts_plugin import VoiceTutTTS

from Backend.utils.logger import (
    logger,
    get_conversation_logger,
)


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"

RESULTS_DIR = Path(
    "Backend/data/interviews"
)

# Persistent TTS cache.
#
# Static items from questions.json are cached by:
#
#     question_id + speaker
#
# This includes:
#
#     welcome
#     q1_name
#     q2_domain
#     q3_experience
#     q4_education
#     q5_skills_tools
#     q6_english
#     no_speech_retry
#     wrong_answer_retry
#     waiting_for_summary
#
# The goodbye message is also cached separately.
TTS_CACHE_DIR = Path(
    "Backend/data/tts_cache"
)

GOODBYE_ID = "__goodbye__"

GOODBYE_MESSAGE = (
    "شكراً جزيلاً لوقت حضرتك، بيانات حضرتك اتسجلت وهنتواصل معاك قريب. "
    "مع السلامة."
)

# ============================================================
# CONTINUOUS AUDIO STREAM
# ============================================================

# VoiceTut is expected to produce 24 kHz mono int16 PCM.
TTS_SAMPLE_RATE = 24000
TTS_SAMPLE_WIDTH = 2  # int16

# We send audio to the client in small real-time frames.
#
# 20 ms @ 24 kHz:
#
#     24000 * 0.020 = 480 samples
#
#     480 * 2 bytes = 960 bytes
#
AUDIO_FRAME_DURATION = 0.020

AUDIO_FRAME_SAMPLES = int(
    TTS_SAMPLE_RATE
    * AUDIO_FRAME_DURATION
)

AUDIO_FRAME_BYTES = (
    AUDIO_FRAME_SAMPLES
    * TTS_SAMPLE_WIDTH
)


# ============================================================
# BACKGROUND AUDIO
# ============================================================

BACKGROUND_AMBIENCE_PATH = (
    "Backend/data/sounds/processed/call_center.wav"
)

BACKGROUND_RING_PATH = (
    "Backend/data/sounds/processed/phone_ring.wav"
)


# ============================================================
# INTERVIEW RESULT STORAGE
# ============================================================

def _write_candidate_json(
    candidate: dict,
    session_id: str,
) -> str:
    """
    Blocking file write.

    Called through asyncio.to_thread() so the event loop is never
    blocked by file I/O.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{session_id}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            candidate,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


# ============================================================
# PERSISTENT TTS CACHE
# ============================================================

def _cache_path(
    question_id: str,
    speaker: str,
) -> Path:
    """
    Build the persistent cache path.

    Speaker is part of the filename so changing the VoiceTut
    speaker automatically creates a separate cache.
    """

    safe_speaker = speaker.replace(
        "/",
        "_",
    )

    return (
        TTS_CACHE_DIR
        / f"{question_id}__{safe_speaker}.wav"
    )


def _read_cached_wav(
    path: Path,
) -> tuple[bytes, int]:
    """
    Blocking WAV read.
    """

    with wave.open(
        str(path),
        "rb",
    ) as wav_file:

        sample_rate = (
            wav_file.getframerate()
        )

        pcm = wav_file.readframes(
            wav_file.getnframes()
        )

    return pcm, sample_rate


def _write_cached_wav(
    path: Path,
    audio: bytes,
    sample_rate: int,
) -> None:
    """
    Blocking WAV write.
    """

    TTS_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            sample_rate
        )
        wav_file.writeframes(audio)


async def _get_or_synthesize(
    tts: VoiceTutTTS,
    item_id: str,
    text: str,
) -> tuple[bytes, int]:
    """
    Get an item's audio from the persistent cache.

    If it doesn't exist, synthesize it once and save it.

    Used for:
        - normal questions
        - retry messages
        - goodbye
    """

    path = _cache_path(
        item_id,
        tts.speaker,
    )

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    if await asyncio.to_thread(
        path.exists,
    ):

        logger.info(
            "[TTS] Cache hit: %s",
            item_id,
        )

        return await asyncio.to_thread(
            _read_cached_wav,
            path,
        )

    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

    logger.info(
        "[TTS] Cache miss, synthesizing: %s",
        item_id,
    )

    audio, sample_rate = (
        await tts.synthesize(text)
    )

    await asyncio.to_thread(
        _write_cached_wav,
        path,
        audio,
        sample_rate,
    )

    return audio, sample_rate


def _start_tts_pregeneration(
    interview_manager: InterviewManager,
    tts: VoiceTutTTS,
) -> dict:
    """
    Start loading/synthesizing every static TTS item concurrently.

    Dynamic summary text is NOT included because its text changes
    depending on the candidate.
    """

    tasks = {}

    for item in interview_manager.questions:

        tasks[item["id"]] = (
            asyncio.create_task(
                _get_or_synthesize(
                    tts,
                    item["id"],
                    item["question_ar"],
                )
            )
        )

    # Goodbye is static but isn't in questions.json.
    tasks[GOODBYE_ID] = (
        asyncio.create_task(
            _get_or_synthesize(
                tts,
                GOODBYE_ID,
                GOODBYE_MESSAGE,
            )
        )
    )

    logger.info(
        "[TTS] Started pre-generation for %d static items",
        len(tasks),
    )

    return tasks


# ============================================================
# TTS AUDIO VALIDATION
# ============================================================

def _validate_tts_audio(
    audio: bytes,
    sample_rate: int,
    item_id: str,
) -> None:
    """
    Validate that TTS audio matches the continuous output stream.

    The background stream and TTS must both be:

        24 kHz
        mono
        signed 16-bit PCM
        raw PCM
    """

    if sample_rate != TTS_SAMPLE_RATE:

        raise ValueError(
            f"TTS sample rate mismatch for "
            f"{item_id}: "
            f"expected {TTS_SAMPLE_RATE}, "
            f"got {sample_rate}"
        )

    if len(audio) % TTS_SAMPLE_WIDTH != 0:

        raise ValueError(
            f"TTS PCM byte count is not divisible "
            f"by {TTS_SAMPLE_WIDTH}: "
            f"{len(audio)}"
        )


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

async def websocket_endpoint(
    websocket: WebSocket,
):

    await websocket.accept()

    call_id = str(
        uuid.uuid4()
    )

    conversation_logger = (
        get_conversation_logger(call_id)
    )

    logger.info(
        "[WS] Call started: %s",
        call_id,
    )

    conversation_logger.info(
        "========== CALL STARTED =========="
    )

    conversation_logger.info(
        "Call ID: %s",
        call_id,
    )

    logger.info(
        "[WS] Client connected"
    )

    session_id = (
        f"{int(time.time())}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    # ========================================================
    # SESSION COMPONENTS
    # ========================================================

    audio_buffer = AudioBuffer()

    vad = VoiceActivityDetector()

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    # Stateful resampler.
    #
    # Do NOT reset this between utterances.
    resampler = StreamResampler(
        source_rate=INPUT_SAMPLE_RATE,
        target_rate=TARGET_SAMPLE_RATE,
    )

    interview_manager = InterviewManager(
        QUESTIONS_PATH
    )

    # ========================================================
    # BACKGROUND MIXER
    # ========================================================

    background_mixer = None

    try:

        ambience_path = Path(
            BACKGROUND_AMBIENCE_PATH
        )

        ring_path = Path(
            BACKGROUND_RING_PATH
        )

        if (
            ambience_path.exists()
            and ring_path.exists()
        ):

            background_mixer = (
                BackgroundMixer(
                    ambience_path=str(
                        ambience_path
                    ),
                    ring_path=str(
                        ring_path
                    ),

                    # These are deliberately subtle.
                    #
                    # The standalone mixer test can use much higher
                    # values, but for a real voice call the background
                    # should sit underneath the agent's voice.
                    ambience_volume=0.035,
                    ring_volume=0.06,

                    ring_min_interval=18.0,
                    ring_max_interval=40.0,
                )
            )

            logger.info(
                "[AUDIO] Background mixer initialized"
            )

            logger.info(
                "[AUDIO] Ambience: %s",
                ambience_path,
            )

            logger.info(
                "[AUDIO] Ring: %s",
                ring_path,
            )

        else:

            logger.warning(
                "[AUDIO] Background audio disabled. "
                "Missing file(s): ambience=%s | ring=%s",
                ambience_path.exists(),
                ring_path.exists(),
            )

    except Exception:

        logger.exception(
            "[AUDIO] Failed to initialize "
            "background mixer"
        )

        background_mixer = None

    # ========================================================
    # RETRY MESSAGES
    # ========================================================
    #
    # These are system_message entries from questions.json.
    #
    # They are cached independently:
    #
    #     no_speech_retry.wav
    #     wrong_answer_retry.wav
    #
    # They are NEVER combined with the original question.

    retry_messages = {
        item["id"]: item
        for item in interview_manager.questions
        if item.get("type")
        == "system_message"
    }

    # ========================================================
    # TTS PRE-GENERATION
    # ========================================================

    tts_tasks = (
        _start_tts_pregeneration(
            interview_manager,
            tts,
        )
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    state = {
        "transcript": "",
        "candidate": {},
        "questions": interview_manager.questions,
        "current_question_index": -1,
        "current_question": None,
        "response": "",
        "interview_finished": False,
        "extraction_success": False,
        "failure_reason": None,
        "mode": "interview",
    }

    speech_active = False

    # While monotonic time is below this value, incoming microphone
    # audio is ignored.
    mic_muted_until = 0.0

    # ========================================================
    # CONTINUOUS TTS QUEUE
    # ========================================================
    #
    # Each queue item is:
    #
    #     (audio, playback_finished_event)
    #
    # The audio output loop consumes the audio in real time.
    #
    # This is important because `speak()` must NOT directly write
    # audio to the WebSocket anymore.
    #
    # Otherwise we would have:
    #
    #     background task -> websocket
    #     speak()         -> websocket
    #
    # at the same time.

    tts_audio_queue = asyncio.Queue()

    # This flag controls the lifetime of the continuous audio loop.
    audio_output_running = True

    # ========================================================
    # CONTINUOUS AUDIO OUTPUT LOOP
    # ========================================================

    async def audio_output_loop():
        """
        Continuously send the caller's outgoing audio stream.

        Every frame contains:

            background ambience
                +
            phone ring when scheduled
                +
            TTS when currently playing

        This loop continues while:

            - user is speaking
            - VAD is running
            - STT is processing
            - LangGraph is running
            - LLM is extracting
            - TTS is synthesizing
            - no TTS is playing

        Only the outgoing audio stream is affected.

        The user's microphone audio is NEVER mixed with
        background audio.
        """

        current_tts = b""
        current_tts_position = 0
        current_tts_finished_event = None

        while audio_output_running:

            try:

                # =================================================
                # GET BACKGROUND FRAME
                # =================================================

                if background_mixer is not None:

                    background = (
                        background_mixer.next_frame(
                            AUDIO_FRAME_SAMPLES
                        )
                    )

                else:

                    background = np.zeros(
                        AUDIO_FRAME_SAMPLES,
                        dtype=np.float32,
                    )

                # =================================================
                # GET NEXT TTS WHEN CURRENT TTS ENDS
                # =================================================

                if (
                    current_tts_position
                    >= len(current_tts)
                ):

                    # Tell the caller that the previous TTS
                    # has actually finished playing.
                    if (
                        current_tts
                        and current_tts_finished_event
                    ):

                        current_tts_finished_event.set()

                    current_tts = b""
                    current_tts_position = 0
                    current_tts_finished_event = None

                    # Do not block here.
                    #
                    # Background audio must continue even when
                    # there is no TTS.
                    try:

                        (
                            next_tts,
                            finished_event,
                        ) = (
                            tts_audio_queue.get_nowait()
                        )

                        current_tts = next_tts

                        current_tts_finished_event = (
                            finished_event
                        )

                    except asyncio.QueueEmpty:

                        pass

                # =================================================
                # MIX TTS INTO BACKGROUND
                # =================================================

                output = background.copy()

                if current_tts:

                    remaining_bytes = (
                        len(current_tts)
                        - current_tts_position
                    )

                    frame_bytes = min(
                        AUDIO_FRAME_BYTES,
                        remaining_bytes,
                    )

                    tts_chunk = current_tts[
                        current_tts_position:
                        current_tts_position
                        + frame_bytes
                    ]

                    tts_samples = np.frombuffer(
                        tts_chunk,
                        dtype="<i2",
                    ).astype(
                        np.float32
                    )

                    output[
                        :len(tts_samples)
                    ] += tts_samples

                    current_tts_position += (
                        frame_bytes
                    )

                # =================================================
                # CLIP
                # =================================================

                output = np.clip(
                    output,
                    -32768.0,
                    32767.0,
                )

                # =================================================
                # CONVERT TO RAW INT16 PCM
                # =================================================

                output_pcm = (
                    output
                    .astype(
                        "<i2",
                        copy=False,
                    )
                    .tobytes()
                )

                # =================================================
                # SEND FRAME
                # =================================================

                await websocket.send_bytes(
                    output_pcm
                )

                # =================================================
                # REAL-TIME PACING
                # =================================================

                await asyncio.sleep(
                    AUDIO_FRAME_DURATION
                )

            except asyncio.CancelledError:

                break

            except WebSocketDisconnect:

                break

            except Exception:

                logger.exception(
                    "[AUDIO] Continuous output loop failed"
                )

                await asyncio.sleep(
                    AUDIO_FRAME_DURATION
                )

    # ========================================================
    # SPEAK ONE TTS ITEM
    # ========================================================

    async def speak(
        item_id: str,
        text: str,
    ):
        """
        Generate/load one TTS item and put it into the
        continuous audio stream.

        Background audio is NOT handled here.

        `audio_output_loop()` handles the actual WebSocket
        output and mixes TTS with the background.
        """

        nonlocal speech_active
        nonlocal mic_muted_until

        if not text:
            return

        conversation_logger.info(
            "AGENT | %s",
            text,
        )

        # ----------------------------------------------------
        # Immediately stop accepting candidate speech.
        # ----------------------------------------------------

        mic_muted_until = float("inf")

        speech_active = False

        audio_buffer.clear()

        vad.reset()

        # ----------------------------------------------------
        # Get cached/pre-generated TTS or synthesize dynamically.
        # ----------------------------------------------------

        task = tts_tasks.get(
            item_id
        )

        try:

            if task is not None:

                logger.info(
                    "[TTS] Awaiting cached/pre-generated item: %s",
                    item_id,
                )

                audio, sample_rate = (
                    await task
                )

            else:

                logger.info(
                    "[TTS] Synthesizing dynamic item: %s",
                    item_id,
                )

                audio, sample_rate = (
                    await tts.synthesize(
                        text
                    )
                )

            # ------------------------------------------------
            # Validate audio format.
            # ------------------------------------------------

            _validate_tts_audio(
                audio,
                sample_rate,
                item_id,
            )

            duration_s = (
                len(audio)
                / (
                    sample_rate
                    * TTS_SAMPLE_WIDTH
                )
            )

            logger.info(
                "[TTS] Queueing %s | %.2fs | %d bytes",
                item_id,
                duration_s,
                len(audio),
            )

            # ------------------------------------------------
            # Tell client TTS is beginning.
            #
            # This does NOT mean the server starts sending
            # audio directly here.
            # The continuous audio stream is already running.
            # ------------------------------------------------

            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            # ------------------------------------------------
            # Create completion event.
            #
            # This event is set by audio_output_loop() only
            # AFTER the audio has actually been played through
            # the outgoing stream.
            # ------------------------------------------------

            playback_finished = (
                asyncio.Event()
            )

            await tts_audio_queue.put(
                (
                    audio,
                    playback_finished,
                )
            )

            # ------------------------------------------------
            # Wait for actual playback completion.
            # ------------------------------------------------

            await playback_finished.wait()

            # ------------------------------------------------
            # TTS has actually finished.
            # ------------------------------------------------

            mic_muted_until = (
                time.monotonic()
            )

            await websocket.send_json({
                "type": "tts_end",
            })

            logger.info(
                "[TTS] Finished playback: %s",
                item_id,
            )

        except WebSocketDisconnect:

            raise

        except Exception as e:

            mic_muted_until = 0.0

            logger.exception(
                "[TTS] Failed to speak %s",
                item_id,
            )

            try:

                await websocket.send_json({
                    "type": "error",
                    "stage": "tts",
                    "message": str(e),
                })

            except WebSocketDisconnect:

                raise

    # ========================================================
    # SPEAK RETRY
    # ========================================================

    async def speak_retry(
        retry_id: str,
        question: dict,
    ):
        """
        Speak:

            retry message
                ↓
            original question

        Both remain separate TTS items.
        """

        retry_message = (
            retry_messages.get(
                retry_id
            )
        )

        if retry_message is None:

            logger.error(
                "[TTS] Retry message '%s' "
                "does not exist in questions.json",
                retry_id,
            )

            await speak(
                question["id"],
                question["question_ar"],
            )

            return

        # ----------------------------------------------------
        # 1. Retry clarification
        # ----------------------------------------------------

        await speak(
            retry_message["id"],
            retry_message["question_ar"],
        )

        # ----------------------------------------------------
        # 2. Original question
        # ----------------------------------------------------

        await speak(
            question["id"],
            question["question_ar"],
        )

    # ========================================================
    # HANDLE INVALID ANSWER
    # ========================================================

    async def handle_invalid_answer():
        """
        Handle a failed interview answer.

        no_speech:
            no_speech_retry -> original question

        wrong_answer:
            wrong_answer_retry -> original question

        Summary failures:
            repeat summary
        """

        current_question = (
            state.get(
                "current_question"
            )
        )

        if current_question is None:
            return

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        if state.get("mode") == "summary":

            logger.info(
                "[INTERVIEW] Repeating summary"
            )

            await speak(
                current_question["id"],
                current_question[
                    "question_ar"
                ],
            )

            return

        # ----------------------------------------------------
        # NORMAL QUESTION
        # ----------------------------------------------------

        failure_reason = state.get(
            "failure_reason"
        )

        if (
            failure_reason
            == "wrong_answer"
        ):

            retry_id = (
                "wrong_answer_retry"
            )

        else:

            retry_id = (
                "no_speech_retry"
            )

        logger.info(
            "[INTERVIEW] Retry | "
            "question=%s | reason=%s | retry_id=%s",
            current_question["id"],
            failure_reason,
            retry_id,
        )

        await websocket.send_json({
            "type": "answer_not_understood",
            "question_id": (
                current_question["id"]
            ),
            "failure_reason": (
                failure_reason
            ),
        })

        await websocket.send_json({
            "type": "question",
            "id": current_question["id"],
            "text": current_question[
                "question_ar"
            ],
            "is_welcome": False,
            "repeat": True,
            "failure_reason": (
                failure_reason
            ),
        })

        await speak_retry(
            retry_id,
            current_question,
        )

    # ========================================================
    # START CONTINUOUS AUDIO OUTPUT
    # ========================================================
    #
    # IMPORTANT:
    #
    # This starts BEFORE the welcome TTS.
    #
    # Therefore the background exists for the whole call,
    # not only while TTS is playing.

    audio_output_task = asyncio.create_task(
        audio_output_loop()
    )

    # ========================================================
    # START SESSION
    # ========================================================

    try:

        logger.info(
            "[INTERVIEW] Starting interview"
        )

        # ====================================================
        # WELCOME
        # ====================================================

        welcome_question = (
            interview_manager.get_question(
                0
            )
        )

        if welcome_question is None:

            logger.error(
                "[INTERVIEW] "
                "Welcome question not found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state[
            "current_question_index"
        ] = 0

        state[
            "current_question"
        ] = welcome_question

        state["response"] = (
            welcome_question[
                "question_ar"
            ]
        )

        await websocket.send_json({
            "type": "question",
            "id": welcome_question[
                "id"
            ],
            "text": welcome_question[
                "question_ar"
            ],
            "is_welcome": True,
        })

        await speak(
            welcome_question["id"],
            welcome_question[
                "question_ar"
            ],
        )

        # ====================================================
        # FIRST QUESTION
        # ====================================================

        first_question_index = 1

        first_question = (
            interview_manager.get_question(
                first_question_index
            )
        )

        if first_question is None:

            logger.error(
                "[INTERVIEW] "
                "No first interview question found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state[
            "current_question_index"
        ] = first_question_index

        state[
            "current_question"
        ] = first_question

        state["response"] = (
            first_question[
                "question_ar"
            ]
        )

        state["transcript"] = ""

        state[
            "extraction_success"
        ] = False

        await websocket.send_json({
            "type": "question",
            "id": first_question[
                "id"
            ],
            "text": first_question[
                "question_ar"
            ],
            "is_welcome": False,
            "repeat": False,
        })

        await speak(
            first_question["id"],
            first_question[
                "question_ar"
            ],
        )

        # ====================================================
        # AUDIO LOOP
        # ====================================================

        while True:

            message = (
                await websocket.receive()
            )

            # =================================================
            # TEXT MESSAGE
            # =================================================

            if (
                message.get("text")
                is not None
            ):

                text = message[
                    "text"
                ]

                logger.info(
                    "[WS] Received text: %s",
                    text,
                )

                await websocket.send_json({
                    "type": (
                        "message_received"
                    ),
                    "message": text,
                })

                continue

            # =================================================
            # AUDIO MESSAGE
            # =================================================

            audio_data = message.get(
                "bytes"
            )

            if not audio_data:
                continue

            # =================================================
            # IGNORE AUDIO WHILE BOT SPEAKS
            # =================================================

            if (
                time.monotonic()
                < mic_muted_until
            ):

                continue

            # =================================================
            # RESAMPLE 24kHz -> 16kHz
            # =================================================

            audio = (
                pcm16_to_float32(
                    audio_data
                )
            )

            audio = resampler.process(
                audio
            )

            normalized_audio = (
                float32_to_pcm16(
                    audio
                )
            )

            # =================================================
            # VAD
            # =================================================

            vad_events = vad.process(
                normalized_audio
            )

            # -------------------------------------------------
            # SPEECH START
            # -------------------------------------------------

            for event in vad_events:

                if event == "speech_start":

                    logger.info(
                        "[VAD] Speech started"
                    )

                    speech_active = True

                    audio_buffer.clear()

                    await websocket.send_json({
                        "type": (
                            "speech_started"
                        ),
                    })

            # -------------------------------------------------
            # BUFFER CURRENT CHUNK
            # -------------------------------------------------

            if speech_active:

                audio_buffer.add(
                    normalized_audio
                )

            # -------------------------------------------------
            # SPEECH END
            # -------------------------------------------------

            for event in vad_events:

                if event != "speech_end":
                    continue

                logger.info(
                    "[VAD] Speech ended | "
                    "Audio: %d bytes",
                    audio_buffer.size(),
                )

                speech_active = False

                utterance = (
                    audio_buffer.get_audio()
                )

                await websocket.send_json({
                    "type": (
                        "speech_ended"
                    ),
                    "bytes": len(
                        utterance
                    ),
                    "sample_rate": (
                        TARGET_SAMPLE_RATE
                    ),
                })

                # =============================================
                # STT
                # =============================================

                transcript = ""

                if utterance:

                    try:

                        transcript = (
                            await stt.transcribe(
                                audio_bytes=(
                                    utterance
                                ),
                                sample_rate=(
                                    TARGET_SAMPLE_RATE
                                ),
                            )
                        )

                        transcript = (
                            transcript.strip()
                        )

                    except WebSocketDisconnect:

                        raise

                    except Exception as e:

                        logger.exception(
                            "[STT] "
                            "Transcription failed"
                        )

                        await websocket.send_json({
                            "type": "error",
                            "stage": "stt",
                            "message": str(e),
                        })

                        audio_buffer.clear()
                        vad.reset()

                        continue

                logger.info(
                    "[STT] Transcript: %s",
                    transcript
                    or "<EMPTY>",
                )

                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                })

                conversation_logger.info(
                    "CANDIDATE | %s",
                    transcript,
                )

                # =============================================
                # EMPTY TRANSCRIPT
                # =============================================
                #
                # Empty STT result means:
                #
                #     no usable speech detected
                #
                # It must still trigger the retry flow.

                if not transcript:

                    state[
                        "failure_reason"
                    ] = "no_speech"

                    state[
                        "extraction_success"
                    ] = False

                    await handle_invalid_answer()

                    audio_buffer.clear()
                    vad.reset()

                    continue

                # =============================================
                # LANGGRAPH
                # =============================================

                try:

                    state[
                        "transcript"
                    ] = transcript

                    state[
                        "extraction_success"
                    ] = False

                    logger.info(
                        "[AGENT] Processing answer | "
                        "question=%s",
                        state[
                            "current_question"
                        ]["id"],
                    )

                    result = (
                        await agent_graph.ainvoke(
                            state
                        )
                    )

                    state = result

                    logger.info(
                        "[AGENT] Candidate: %s",
                        state[
                            "candidate"
                        ],
                    )

                    logger.info(
                        "[AGENT] "
                        "Extraction success: %s",
                        state[
                            "extraction_success"
                        ],
                    )

                    # =========================================
                    # WAITING FOR SUMMARY
                    # =========================================

                    if (
                        state.get("mode")
                        == "waiting_for_summary"
                    ):

                        waiting_message = (
                            state[
                                "current_question"
                            ]
                        )

                        logger.info(
                            "[INTERVIEW] "
                            "Waiting before summary"
                        )

                        await websocket.send_json({
                            "type": (
                                "system_message"
                            ),
                            "id": (
                                waiting_message[
                                    "id"
                                ]
                            ),
                            "text": (
                                waiting_message[
                                    "question_ar"
                                ]
                            ),
                        })

                        await speak(
                            waiting_message[
                                "id"
                            ],
                            waiting_message[
                                "question_ar"
                            ],
                        )

                        # Tell the graph what we want
                        # to do next.

                        state[
                            "mode"
                        ] = (
                            "building_summary"
                        )

                        state[
                            "current_question"
                        ] = None

                        state[
                            "transcript"
                        ] = ""

                        state[
                            "extraction_success"
                        ] = False

                        state[
                            "failure_reason"
                        ] = None

                        result = (
                            await agent_graph.ainvoke(
                                state
                            )
                        )

                        state = result

                    # =========================================
                    # INVALID ANSWER
                    # =========================================

                    if not state[
                        "extraction_success"
                    ]:

                        await handle_invalid_answer()

                        audio_buffer.clear()
                        vad.reset()

                        continue

                    # =========================================
                    # INTERVIEW FINISHED
                    # =========================================

                    if state[
                        "interview_finished"
                    ]:

                        logger.info(
                            "[INTERVIEW] "
                            "Interview completed"
                        )

                        logger.info(
                            "[INTERVIEW] "
                            "Final candidate data: %s",
                            state[
                                "candidate"
                            ],
                        )

                        # -------------------------------------
                        # Save candidate data
                        # -------------------------------------

                        try:

                            saved_path = (
                                await asyncio.to_thread(
                                    _write_candidate_json,
                                    state[
                                        "candidate"
                                    ],
                                    session_id,
                                )
                            )

                            logger.info(
                                "[INTERVIEW] "
                                "Saved candidate data: %s",
                                saved_path,
                            )

                        except Exception:

                            logger.exception(
                                "[INTERVIEW] "
                                "Failed to save "
                                "candidate data"
                            )

                        # -------------------------------------
                        # Tell client interview is complete
                        # -------------------------------------

                        await websocket.send_json({
                            "type": (
                                "interview_complete"
                            ),
                            "candidate": (
                                state[
                                    "candidate"
                                ]
                            ),
                        })

                        # -------------------------------------
                        # Goodbye
                        # -------------------------------------

                        await speak(
                            GOODBYE_ID,
                            GOODBYE_MESSAGE,
                        )

                        return

                    # =========================================
                    # NEXT QUESTION
                    # =========================================

                    next_question = (
                        state.get(
                            "current_question"
                        )
                    )

                    if next_question is not None:

                        logger.info(
                            "[INTERVIEW] "
                            "Next question: %s",
                            next_question[
                                "id"
                            ],
                        )

                        await websocket.send_json({
                            "type": (
                                "question"
                            ),
                            "id": (
                                next_question[
                                    "id"
                                ]
                            ),
                            "text": (
                                next_question[
                                    "question_ar"
                                ]
                            ),
                            "is_welcome": False,
                            "repeat": False,
                        })

                        await speak(
                            next_question[
                                "id"
                            ],
                            next_question[
                                "question_ar"
                            ],
                        )

                except WebSocketDisconnect:

                    raise

                except Exception as e:

                    logger.exception(
                        "[AGENT] Processing failed"
                    )

                    await websocket.send_json({
                        "type": "error",
                        "stage": "agent",
                        "message": str(e),
                    })

                # ---------------------------------------------
                # CLEAN UP UTTERANCE
                # ---------------------------------------------

                audio_buffer.clear()

                vad.reset()

                # Do NOT reset the resampler here.
                #
                # It must remain continuous across the entire
                # microphone stream.

    except WebSocketDisconnect:

        logger.info(
            "[WS] Client disconnected"
        )

    except RuntimeError as e:

        if "disconnect message" in str(e):

            logger.info(
                "[WS] Client disconnected"
            )

        else:

            raise

    finally:

        # ====================================================
        # STOP CONTINUOUS AUDIO
        # ====================================================

        audio_output_running = False

        if audio_output_task:

            audio_output_task.cancel()

            try:

                await audio_output_task

            except asyncio.CancelledError:

                pass

        # ====================================================
        # CANCEL UNUSED TTS PRE-GENERATION
        # ====================================================

        for task in tts_tasks.values():

            if not task.done():

                task.cancel()

        # ====================================================
        # CLEANUP
        # ====================================================

        logger.info(
            "[WS] Connection closed"
        )

        conversation_logger.info(
            "========== CALL ENDED =========="
        )