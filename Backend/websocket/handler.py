import asyncio
import json
import logging
import time
import uuid
import wave
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from Backend.audio.audio_utils import (
    TARGET_SAMPLE_RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    StreamResampler,
)
from Backend.audio.buffer import AudioBuffer
from Backend.audio.recorder import AnswerRecorder
from Backend.audio.vad import VoiceActivityDetector
from Backend.stt_plugin import CohereArabicSTT

from Backend.agent.graph import agent_graph
from Backend.agent.interview_manager import InterviewManager
from Backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"

RESULTS_DIR = Path("Backend/data/interviews")

# Persistent TTS cache.
#
# Static items from questions.json are cached by:
#
#     question_id + speaker
#
# This includes every entry in questions.json — every real
# question, plus the system_message entries:
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
# The goodbye message is also cached separately (it isn't in
# questions.json). The dynamic summary itself is the one thing that
# is NEVER cached — its text depends on the candidate's answers.
TTS_CACHE_DIR = Path("Backend/data/tts_cache")

GOODBYE_ID = "__goodbye__"

# Matches graph.py's SUMMARY_QUESTION_ID — the synthetic question id
# the graph uses once it builds the end-of-interview summary.
SUMMARY_QUESTION_ID = "summary"

# id of the "please wait a moment" line stored in questions.json,
# spoken once right before the (always-dynamic) summary itself.
WAITING_FOR_SUMMARY_ID = "waiting_for_summary"

GOODBYE_MESSAGE = (
    "شكراً جزيلاً لوقت حضرتك، بيانات حضرتك اتسجلت وهنتواصل معاك قريب. "
    "مع السلامة."
)

# Keep WebSocket audio messages below common message-size limits.
TTS_SEND_CHUNK_SIZE = 32 * 1024


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

    output_path = RESULTS_DIR / f"{session_id}.json"

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

    Speaker is part of the filename so changing the VoiceTut speaker
    automatically creates a separate cache.
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

        sample_rate = wav_file.getframerate()

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
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _get_or_synthesize(
    tts: VoiceTutTTS,
    item_id: str,
    text: str,
) -> tuple[bytes, int]:
    """
    Get an item's audio from the persistent cache.

    If it doesn't exist, synthesize it once and save it.

    This function is used for every static item: normal questions,
    retry system messages, the waiting-for-summary line, and
    goodbye.
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

    audio, sample_rate = await tts.synthesize(
        text
    )

    await asyncio.to_thread(
        _write_cached_wav,
        path,
        audio,
        sample_rate,
    )

    return audio, sample_rate


def _find_question_by_id(
    interview_manager: InterviewManager,
    question_id: str,
) -> dict | None:

    for question in interview_manager.questions:

        if question.get("id") == question_id:
            return question

    return None


def _start_tts_pregeneration(
    interview_manager: InterviewManager,
    tts: VoiceTutTTS,
) -> dict:
    """
    Start loading/synthesizing every static TTS item concurrently.

    This includes every system_message entry in questions.json
    (the retry clarification lines and the waiting-for-summary
    line) — nothing special-cased, they're just items like any
    other question.

    Dynamic summary text is NOT included because its text changes
    depending on the candidate.
    """

    tasks = {}

    for item in interview_manager.questions:

        tasks[item["id"]] = asyncio.create_task(
            _get_or_synthesize(
                tts,
                item["id"],
                item["question_ar"],
            )
        )

    # Goodbye is static but isn't in questions.json.
    tasks[GOODBYE_ID] = asyncio.create_task(
        _get_or_synthesize(
            tts,
            GOODBYE_ID,
            GOODBYE_MESSAGE,
        )
    )

    logger.info(
        "[TTS] Started pre-generation for %d static items",
        len(tasks),
    )

    return tasks


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

async def websocket_endpoint(
    websocket: WebSocket,
):

    await websocket.accept()

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

    # One WAV per candidate answer, written to
    # Backend/data/recordings/<session_id>/ alongside a manifest
    # linking each file to its question and transcript.
    recorder = AnswerRecorder(session_id)

    vad = VoiceActivityDetector()

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    # Stateful resampler.
    #
    # Do not reset this between utterances.
    resampler = StreamResampler(
        source_rate=INPUT_SAMPLE_RATE,
        target_rate=TARGET_SAMPLE_RATE,
    )

    interview_manager = InterviewManager(
        QUESTIONS_PATH
    )

    # ========================================================
    # RETRY MESSAGES
    # ========================================================
    #
    # These are the system_message entries from questions.json.
    #
    # They are cached independently:
    #
    #     no_speech_retry.wav
    #     wrong_answer_retry.wav
    #     waiting_for_summary.wav
    #
    # They are NEVER combined with the original question or the
    # summary text — always spoken as their own separate clip.

    retry_messages = {
        item["id"]: item
        for item in interview_manager.questions
        if item.get("type") == "system_message"
    }

    # ========================================================
    # TTS PRE-GENERATION
    # ========================================================

    tts_tasks = _start_tts_pregeneration(
        interview_manager,
        tts,
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
    # SPEAK ONE TTS ITEM
    # ========================================================

    async def speak(
        item_id: str,
        text: str,
    ):
        """
        Speak exactly ONE TTS item.

        Static items normally come from tts_tasks.

        Dynamic items such as the summary are synthesized on demand.

        This function never combines two pieces of text.
        """

        nonlocal speech_active
        nonlocal mic_muted_until

        if not text:
            return

        # Immediately stop accepting candidate speech.
        mic_muted_until = float("inf")

        speech_active = False

        audio_buffer.clear()

        vad.reset()

        task = tts_tasks.get(
            item_id
        )

        try:

            # ------------------------------------------------
            # Cached / pre-generated item
            # ------------------------------------------------

            if task is not None:

                logger.info(
                    "[TTS] Awaiting cached/pre-generated item: %s",
                    item_id,
                )

                audio, sample_rate = await task

            # ------------------------------------------------
            # Dynamic item
            # ------------------------------------------------

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
            # Calculate playback duration
            # ------------------------------------------------

            duration_s = (
                len(audio)
                / (sample_rate * 2)
            )

            mic_muted_until = (
                time.monotonic()
                + duration_s
            )

            logger.info(
                "[TTS] Sending %s | %.2fs | %d bytes",
                item_id,
                duration_s,
                len(audio),
            )

            # ------------------------------------------------
            # Start TTS
            # ------------------------------------------------

            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            # ------------------------------------------------
            # Send audio in chunks — keeps individual WebSocket
            # frames below common message-size limits, which may
            # be what caused earlier long/silent audio (like the
            # dynamic summary) to never finish sending.
            # ------------------------------------------------

            for offset in range(
                0,
                len(audio),
                TTS_SEND_CHUNK_SIZE,
            ):

                chunk = audio[
                    offset:
                    offset + TTS_SEND_CHUNK_SIZE
                ]

                await websocket.send_bytes(
                    chunk
                )

            # ------------------------------------------------
            # End TTS
            # ------------------------------------------------

            await websocket.send_json({
                "type": "tts_end",
            })

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
        Speak the retry message and then the original question.

        Example:

            no_speech_retry
                    ↓
            q4_education

        Both are separate cached TTS items — never concatenated,
        so neither ever needs on-demand synthesis after the first
        pregeneration pass.
        """

        retry_message = retry_messages.get(
            retry_id
        )

        if retry_message is None:

            logger.error(
                "[TTS] Retry message '%s' "
                "does not exist in questions.json",
                retry_id,
            )

            # Don't invent another TTS phrase.
            # Just repeat the original question.
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
    # SPEAK SUMMARY (with waiting line first)
    # ========================================================

    async def speak_summary(
        summary_question: dict,
    ):
        """
        The summary's own audio is always synthesized live — its
        text depends on the candidate's actual answers, so unlike
        every other item it can never be pre-cached. Play the short,
        already-cached "please wait a moment" line first so there's
        no dead air while the real summary is still being generated,
        then speak the summary itself.

        Safe to call these as two back-to-back speak()s: nothing in
        between awaits websocket.receive(), so mic muting stays
        continuous across both — the second speak() call overwrites
        mic_muted_until before the loop ever gets a chance to read
        incoming audio again.
        """

        waiting_question = retry_messages.get(
            WAITING_FOR_SUMMARY_ID
        )

        if waiting_question:

            logger.info(
                "[INTERVIEW] Speaking wait-for-summary line"
            )

            await speak(
                waiting_question["id"],
                waiting_question["question_ar"],
            )

        await speak(
            summary_question["id"],
            summary_question["question_ar"],
        )

    # ========================================================
    # HANDLE INVALID ANSWER
    # ========================================================

    async def handle_invalid_answer():
        """
        Handle a failed interview answer.

        Uses the failure_reason produced by LangGraph.

        no_speech:
            no_speech_retry -> original question

        wrong_answer:
            wrong_answer_retry -> original question

        Summary failures are handled separately because a summary
        is dynamic and should simply be replayed.
        """

        current_question = state.get(
            "current_question"
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
                current_question["question_ar"],
            )

            return

        # ----------------------------------------------------
        # NORMAL QUESTION
        # ----------------------------------------------------

        failure_reason = state.get(
            "failure_reason"
        )

        if failure_reason == "wrong_answer":

            retry_id = (
                "wrong_answer_retry"
            )

        else:

            retry_id = (
                "no_speech_retry"
            )

        logger.info(
            "[INTERVIEW] Retry | question=%s "
            "| reason=%s | retry_id=%s",
            current_question["id"],
            failure_reason,
            retry_id,
        )

        await websocket.send_json({
            "type": "answer_not_understood",
            "question_id": current_question["id"],
            "failure_reason": failure_reason,
        })

        await websocket.send_json({
            "type": "question",
            "id": current_question["id"],
            "text": current_question[
                "question_ar"
            ],
            "is_welcome": False,
            "repeat": True,
            "failure_reason": failure_reason,
        })

        await speak_retry(
            retry_id,
            current_question,
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
            interview_manager.get_question(0)
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

        state["current_question_index"] = 0

        state["current_question"] = (
            welcome_question
        )

        state["response"] = (
            welcome_question["question_ar"]
        )

        await websocket.send_json({
            "type": "question",
            "id": welcome_question["id"],
            "text": welcome_question[
                "question_ar"
            ],
            "is_welcome": True,
        })

        await speak(
            welcome_question["id"],
            welcome_question["question_ar"],
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

        state["current_question_index"] = (
            first_question_index
        )

        state["current_question"] = (
            first_question
        )

        state["response"] = (
            first_question["question_ar"]
        )

        state["transcript"] = ""

        state["extraction_success"] = False

        await websocket.send_json({
            "type": "question",
            "id": first_question["id"],
            "text": first_question[
                "question_ar"
            ],
            "is_welcome": False,
            "repeat": False,
        })

        await speak(
            first_question["id"],
            first_question["question_ar"],
        )

        # ====================================================
        # AUDIO LOOP
        # ====================================================

        while True:

            message = await websocket.receive()

            # =================================================
            # TEXT MESSAGE
            # =================================================

            if message.get("text") is not None:

                text = message["text"]

                logger.info(
                    "[WS] Received text: %s",
                    text,
                )

                await websocket.send_json({
                    "type": "message_received",
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

            audio = pcm16_to_float32(
                audio_data
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
                        "type": "speech_started"
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
                    "type": "speech_ended",
                    "bytes": len(utterance),
                    "sample_rate": TARGET_SAMPLE_RATE,
                })

                # =============================================
                # STT
                # =============================================

                transcript = ""

                if utterance:

                    try:

                        transcript = (
                            await stt.transcribe(
                                audio_bytes=utterance,
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
                    transcript or "<EMPTY>",
                )

                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                })

                # =============================================
                # SAVE THE ANSWER RECORDING
                # =============================================
                #
                # Placed here, after STT but BEFORE any of the
                # retry/graph branching below, so every utterance
                # the candidate produces is captured — including
                # ones that get rejected and repeated, and ones
                # STT couldn't transcribe at all (those are
                # exactly the recordings worth listening back to
                # when debugging a bad transcript).
                #
                # The write itself runs off the event loop and
                # never raises, so a disk problem can't interrupt
                # a live call.

                current_question_id = (
                    state["current_question"]["id"]
                    if state.get("current_question")
                    else "unknown"
                )

                recording_path = await recorder.save(
                    audio=utterance,
                    sample_rate=TARGET_SAMPLE_RATE,
                    question_id=current_question_id,
                    transcript=transcript,
                )

                if recording_path:

                    logger.info(
                        "[RECORD] Saved answer audio: %s",
                        recording_path,
                    )

                else:

                    logger.warning(
                        "[RECORD] Failed to save answer audio "
                        "for question %s",
                        current_question_id,
                    )

                # =============================================
                # EMPTY TRANSCRIPT
                # =============================================
                #
                # An empty transcript means the candidate's speech
                # could not be understood at all — this is always a
                # no_speech retry. Handled directly here, without
                # ever invoking the graph: there's nothing for the
                # LLM to reason about (no real content exists), and
                # sending an empty transcript into route_from_start
                # would fall through to its "initialize" fallback,
                # which would incorrectly reset the whole interview
                # back to the welcome message instead of repeating
                # the current question.

                if not transcript:

                    state["failure_reason"] = (
                        "no_speech"
                    )

                    state["extraction_success"] = (
                        False
                    )

                    await handle_invalid_answer()

                    audio_buffer.clear()
                    vad.reset()

                    continue

                # =============================================
                # LANGGRAPH
                # =============================================

                try:

                    state["transcript"] = (
                        transcript
                    )

                    state["extraction_success"] = (
                        False
                    )

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
                        state["candidate"],
                    )

                    logger.info(
                        "[AGENT] Extraction success: %s",
                        state[
                            "extraction_success"
                        ],
                    )

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
                            state["candidate"],
                        )

                        # -------------------------------------
                        # Save candidate data
                        # -------------------------------------

                        try:

                            saved_path = (
                                await asyncio.to_thread(
                                    _write_candidate_json,
                                    state["candidate"],
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
                                state["candidate"]
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

                    next_question = state.get(
                        "current_question"
                    )

                    if next_question is not None:

                        logger.info(
                            "[INTERVIEW] "
                            "Next question: %s",
                            next_question["id"],
                        )

                        await websocket.send_json({
                            "type": "question",
                            "id": next_question["id"],
                            "text": next_question[
                                "question_ar"
                            ],
                            "is_welcome": False,
                            "repeat": False,
                        })

                        # The summary is the one item that's always
                        # synthesized live, so speak_summary() plays
                        # the cached "please wait" line first to
                        # cover the gap. Every other question is
                        # already fully cached and plays instantly.
                        if (
                            next_question["id"]
                            == SUMMARY_QUESTION_ID
                        ):

                            await speak_summary(
                                next_question
                            )

                        else:

                            await speak(
                                next_question["id"],
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
                # It must remain continuous across the whole
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

        # Stop unfinished TTS cache-generation tasks when the
        # session ends.
        for task in tts_tasks.values():

            if not task.done():
                task.cancel()

        logger.info(
            "[WS] Connection closed"
        )