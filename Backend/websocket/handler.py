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
from Backend.audio.vad import VoiceActivityDetector
from Backend.stt_plugin import CohereArabicSTT

from Backend.agent.graph import agent_graph
from Backend.agent.interview_manager import InterviewManager
from Backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"

RESULTS_DIR = Path("Backend/data/interviews")

# Persistent cache of pre-synthesized question audio. Survives
# server restarts — since questions.json is static and identical
# for every candidate, audio only ever needs to be generated once,
# not once per session.
TTS_CACHE_DIR = Path("Backend/data/tts_cache")

GOODBYE_ID = "__goodbye__"

GOODBYE_MESSAGE = (
    "شكراً جزيلاً لوقت حضرتك، بيانات حضرتك اتسجلت وهنتواصل معاك قريب. "
    "مع السلامة."
)

# Sent as multiple smaller WebSocket frames instead of one big
# send_bytes() call. Every other piece of audio (real questions) is
# short enough that one frame never came close to a size limit — but
# the end-of-interview summary reads back every collected field in
# one paragraph, and at 24kHz 16-bit PCM that can comfortably exceed
# a default per-message WebSocket size cap, which closes the
# connection at the protocol level (below where our own except
# Exception in speak() can catch and report it). Chunking sidesteps
# the limit regardless of exactly where it sits.
TTS_SEND_CHUNK_SIZE = 32 * 1024


def _write_candidate_json(candidate: dict, session_id: str) -> str:
    """
    Blocking file write — always call this via asyncio.to_thread().
    Runs exactly once, at the very end of an interview (after the
    audio pipeline for this session is already done), but keeping it
    off the event loop costs nothing and means it can never delay
    other concurrent sessions sharing this server.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / f"{session_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidate, f, ensure_ascii=False, indent=2)

    return str(output_path)


# ============================================================
# PERSISTENT TTS CACHE
# ============================================================
# All blocking file I/O below is always called via asyncio.to_thread()
# from the async helpers, so it never blocks the event loop.

def _cache_path(question_id: str, speaker: str) -> Path:
    # Speaker is part of the key: if VOICETUT_SPEAKER ever changes,
    # old cached audio for the previous speaker is simply never
    # matched again, rather than being served incorrectly.
    safe_speaker = speaker.replace("/", "_")
    return TTS_CACHE_DIR / f"{question_id}__{safe_speaker}.wav"


def _read_cached_wav(path: Path):
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
    return pcm, sample_rate


def _write_cached_wav(path: Path, audio: bytes, sample_rate: int) -> None:
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _get_or_synthesize(
    tts: VoiceTutTTS,
    question_id: str,
    text: str,
) -> tuple[bytes, int]:
    """
    Returns cached audio for (question_id, speaker) if it already
    exists on disk from a previous run; otherwise synthesizes it via
    VoiceTut once and writes it to disk for every future connection
    — including after the server itself restarts — to reuse.
    """

    path = _cache_path(question_id, tts.speaker)

    if await asyncio.to_thread(path.exists):

        logger.info(
            "[TTS] Cache hit: %s",
            question_id,
        )

        return await asyncio.to_thread(_read_cached_wav, path)

    logger.info(
        "[TTS] Cache miss, synthesizing: %s",
        question_id,
    )

    audio, sample_rate = await tts.synthesize(text)

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
    Fire off cache-or-synthesize for every question in this session,
    all at once, as background asyncio tasks — not awaited here.

    On the very first run against a given questions.json + speaker,
    this behaves exactly like before: N concurrent VoiceTut calls.
    On every run after that, most/all of these resolve instantly
    from disk, so start-of-session latency drops close to zero.
    """

    tasks = {}

    for question in interview_manager.questions:

        tasks[question["id"]] = asyncio.create_task(
            _get_or_synthesize(
                tts,
                question["id"],
                question["question_ar"],
            )
        )

    tasks[GOODBYE_ID] = asyncio.create_task(
        _get_or_synthesize(
            tts,
            GOODBYE_ID,
            GOODBYE_MESSAGE,
        )
    )

    logger.info(
        "[TTS] Pre-generation started for %d items",
        len(tasks),
    )

    return tasks


async def websocket_endpoint(
    websocket: WebSocket,
):

    await websocket.accept()

    logger.info("[WS] Client connected")

    session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # =====================================================
    # SESSION COMPONENTS
    # =====================================================

    audio_buffer = AudioBuffer()

    vad = VoiceActivityDetector()

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    # Stateful — carries a small sample tail across chunks so
    # resampling doesn't introduce a discontinuity at every chunk
    # boundary. One per connection; do NOT reset this on
    # speech_start/speech_end, only if the whole session is rebuilt.
    resampler = StreamResampler(
        source_rate=INPUT_SAMPLE_RATE,
        target_rate=TARGET_SAMPLE_RATE,
    )

    interview_manager = InterviewManager(
        QUESTIONS_PATH
    )

    # =====================================================
    # KICK OFF PARALLEL TTS PRE-GENERATION (CACHE-BACKED)
    # =====================================================
    # Every question's audio starts loading (from cache) or
    # synthesizing (on a true cache miss) right now, in the
    # background, concurrently — before the welcome message has even
    # been sent. speak() below will await the matching task instead
    # of calling tts.synthesize() itself.

    tts_tasks = _start_tts_pregeneration(
        interview_manager,
        tts,
    )

    # =====================================================
    # INITIAL SESSION STATE
    # =====================================================

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

    # =====================================================
    # MIC MUTING WHILE THE AGENT IS SPEAKING
    # =====================================================
    # A monotonic "unmute at" timestamp rather than a boolean flag +
    # timer task: the main audio loop already runs on every incoming
    # chunk regardless of what the agent is doing, so a plain
    # time.monotonic() comparison on each chunk is enough — no
    # separate task to schedule, track, or cancel.
    #
    # Starts at +inf below, i.e. "muted until told otherwise" —
    # speak() is what pushes this forward to a real deadline once
    # audio is actually ready, and pulls it back down on completion
    # or failure. This means the mic is muted the instant we decide
    # to speak (even during the brief pre-generated-audio lookup),
    # not only once playback starts.
    mic_muted_until = 0.0

    # =====================================================
    # TTS HELPER
    # =====================================================

    async def speak(question_id: str, text: str):
        """
        question_id must match an id from questions.json, or
        GOODBYE_ID. Falls back to synthesizing on the spot only if,
        for some reason, no pre-generated task exists for this id
        (e.g. a question id that isn't in tts_tasks) — this keeps
        speak() working even if pre-generation and the actual
        question set ever drift apart.

        Mutes the mic for the duration of this call plus the audio's
        actual playback length, so anything the candidate says before
        or during the question is dropped rather than transcribed.
        """

        nonlocal mic_muted_until, speech_active

        if not text:
            return

        # Mute immediately, before anything else — covers the (short)
        # cache lookup or on-demand synthesis time too, not just
        # actual playback. Also drop/reset any capture that might
        # already be in progress so nothing bleeds across the
        # boundary between "candidate was talking" and "agent starts
        # talking".
        mic_muted_until = float("inf")
        speech_active = False
        audio_buffer.clear()
        vad.reset()

        task = tts_tasks.get(question_id)

        try:

            if task is not None:

                logger.info(
                    "[TTS] Awaiting pre-generated audio: %s",
                    question_id,
                )

                audio, sample_rate = await task

            else:

                logger.warning(
                    "[TTS] No pre-generated audio for %s, "
                    "synthesizing on demand",
                    question_id,
                )

                audio, sample_rate = await tts.synthesize(text)

            logger.info(
                "[TTS] Audio ready: %d bytes at %d Hz",
                len(audio),
                sample_rate,
            )

            # 16-bit mono PCM: 2 bytes per sample.
            duration_s = len(audio) / (sample_rate * 2)

            mic_muted_until = time.monotonic() + duration_s

            logger.info(
                "[TTS] Mic muted for %.2fs while playing: %s",
                duration_s,
                question_id,
            )

            # Tell the client that TTS audio is coming
            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            # Send audio as multiple smaller frames instead of one
            # send_bytes() call — see TTS_SEND_CHUNK_SIZE comment at
            # the top of this file for why. The client buffers these
            # and plays them as one clip on tts_end, so this is
            # invisible to actual playback behavior.
            for offset in range(0, len(audio), TTS_SEND_CHUNK_SIZE):
                chunk = audio[offset:offset + TTS_SEND_CHUNK_SIZE]
                await websocket.send_bytes(chunk)

            # Tell the client that TTS audio is finished
            await websocket.send_json({
                "type": "tts_end",
            })

            logger.info(
                "[TTS] Audio sent to client (%d bytes, %d frame(s))",
                len(audio),
                -(-len(audio) // TTS_SEND_CHUNK_SIZE) if audio else 0,
            )

        except WebSocketDisconnect:
            raise

        except Exception as e:

            # Don't leave the mic muted forever if synthesis failed.
            mic_muted_until = 0.0

            logger.exception(
                "[TTS] Synthesis failed"
            )

            try:
                await websocket.send_json({
                    "type": "error",
                    "stage": "tts",
                    "message": str(e),
                })
            except WebSocketDisconnect:
                raise

    try:

        # =================================================
        # START INTERVIEW
        # =================================================

        logger.info(
            "[INTERVIEW] Starting interview..."
        )

        # -------------------------------------------------
        # WELCOME
        # -------------------------------------------------

        welcome_question = interview_manager.get_question(0)

        if welcome_question is None:

            logger.warning(
                "[INTERVIEW] No welcome question found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state["current_question_index"] = 0
        state["current_question"] = welcome_question
        state["response"] = welcome_question["question_ar"]

        logger.info(
            "[INTERVIEW] Welcome: %s",
            welcome_question["id"],
        )

        await websocket.send_json({
            "type": "question",
            "id": welcome_question["id"],
            "text": welcome_question["question_ar"],
            "is_welcome": True,
        })

        # Speak welcome — awaits the welcome task specifically, and
        # mutes the mic for its actual playback duration.
        await speak(
            welcome_question["id"],
            welcome_question["question_ar"],
        )

        # -------------------------------------------------
        # FIRST REAL QUESTION
        # -------------------------------------------------

        first_question_index = 1

        first_question = interview_manager.get_question(
            first_question_index
        )

        if first_question is None:

            logger.warning(
                "[INTERVIEW] No interview questions found"
            )

            await websocket.send_json({
                "type": "interview_complete",
            })

            return

        state["current_question_index"] = first_question_index
        state["current_question"] = first_question
        state["response"] = first_question["question_ar"]
        state["transcript"] = ""
        state["extraction_success"] = False

        logger.info(
            "[INTERVIEW] First question: %s",
            first_question["id"],
        )

        await websocket.send_json({
            "type": "question",
            "id": first_question["id"],
            "text": first_question["question_ar"],
            "is_welcome": False,
            "repeat": False,
        })

        # Speak first question
        await speak(
            first_question["id"],
            first_question["question_ar"],
        )

        # =================================================
        # AUDIO LOOP
        # =================================================

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

            audio_data = message.get("bytes")

            if audio_data is None:
                continue

            if not audio_data:
                continue

            # =================================================
            # DROP AUDIO WHILE THE AGENT IS SPEAKING
            # =================================================
            # Checked first, before any processing at all, so audio
            # captured while a question is being (or about to be)
            # spoken never reaches the resampler, VAD, or buffer —
            # not "ignored after the fact", genuinely never handled.

            if time.monotonic() < mic_muted_until:
                continue

            # =================================================
            # NORMALIZE AUDIO
            # =================================================
            # Use the stateful StreamResampler here, not the one-shot
            # resample_audio() — this runs per incoming chunk, and a
            # stateless resample per chunk introduces a discontinuity
            # at every chunk boundary. resample_audio() is still the
            # right call for one-shot buffers (e.g. inside stt_plugin,
            # which resamples a whole finished utterance at once).

            audio = pcm16_to_float32(
                audio_data
            )

            audio = resampler.process(
                audio
            )

            normalized_audio = float32_to_pcm16(
                audio
            )

            # =================================================
            # VAD
            # =================================================

            vad_events = vad.process(
                normalized_audio
            )

            # -------------------------------------------------
            # Handle speech_start first so a fresh buffer is ready
            # to receive this chunk below.
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
            # Append this chunk while we're "in speech" — this is
            # what guarantees the chunk containing speech_end is
            # included in the buffer before get_audio() is called
            # below. (Previously this only ran AFTER the event loop,
            # so the chunk where speech_end fired was silently
            # dropped from the utterance.)
            # -------------------------------------------------

            if speech_active:

                audio_buffer.add(
                    normalized_audio
                )

            # -------------------------------------------------
            # Now handle speech_end, with this chunk already in
            # the buffer.
            # -------------------------------------------------

            for event in vad_events:

                if event == "speech_end":

                    logger.info(
                        "[VAD] Speech ended | Audio: %d bytes",
                        audio_buffer.size(),
                    )

                    speech_active = False

                    utterance = audio_buffer.get_audio()

                    logger.info(
                        "[AUDIO] Complete utterance | %d bytes",
                        len(utterance),
                    )

                    await websocket.send_json({
                        "type": "speech_ended",
                        "bytes": len(utterance),
                        "sample_rate": TARGET_SAMPLE_RATE,
                    })

                    # =========================================
                    # STT
                    # =========================================

                    if utterance:

                        logger.info(
                            "[STT] Transcribing utterance..."
                        )

                        try:

                            transcript = await stt.transcribe(
                                audio_bytes=utterance,
                                sample_rate=TARGET_SAMPLE_RATE,
                            )

                            transcript = transcript.strip()

                            logger.info(
                                "[STT] Transcript: %s",
                                transcript or "<EMPTY>",
                            )

                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript,
                            })

                            # =================================
                            # LANGGRAPH
                            # =================================

                            if transcript:

                                logger.info(
                                    "[AGENT] Processing candidate answer..."
                                )

                                try:

                                    state["transcript"] = transcript
                                    state["extraction_success"] = False

                                    logger.info(
                                        "[AGENT] Current question: %s",
                                        state["current_question"]["id"],
                                    )

                                    result = await agent_graph.ainvoke(
                                        state
                                    )

                                    state = result

                                    logger.info(
                                        "[AGENT] Candidate: %s",
                                        state["candidate"],
                                    )

                                    logger.info(
                                        "[AGENT] Extraction success: %s",
                                        state["extraction_success"],
                                    )

                                    # =================================
                                    # INVALID ANSWER
                                    # =================================

                                    if not state["extraction_success"]:

                                        current_question = state[
                                            "current_question"
                                        ]

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Answer not understood. "
                                            "Repeating: %s",
                                            current_question["id"],
                                        )

                                        await websocket.send_json({
                                            "type": "answer_not_understood",
                                            "question_id": (
                                                current_question["id"]
                                            ),
                                        })

                                        await websocket.send_json({
                                            "type": "question",
                                            "id": current_question["id"],
                                            "text": (
                                                current_question[
                                                    "question_ar"
                                                ]
                                            ),
                                            "is_welcome": False,
                                            "repeat": True,
                                        })

                                        await speak(
                                            current_question["id"],
                                            current_question[
                                                "question_ar"
                                            ],
                                        )

                                        continue

                                    # =================================
                                    # INTERVIEW FINISHED
                                    # =================================

                                    if state["interview_finished"]:

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Interview completed"
                                        )

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Final candidate data: %s",
                                            state["candidate"],
                                        )

                                        # Save the extracted data (not
                                        # audio) off the event loop —
                                        # runs once, doesn't touch the
                                        # realtime audio path at all.
                                        try:
                                            saved_path = await asyncio.to_thread(
                                                _write_candidate_json,
                                                state["candidate"],
                                                session_id,
                                            )
                                            logger.info(
                                                "[INTERVIEW] Saved "
                                                "candidate data to %s",
                                                saved_path,
                                            )
                                        except Exception:
                                            logger.exception(
                                                "[INTERVIEW] Failed "
                                                "to save candidate "
                                                "data"
                                            )

                                        # Send the summary first so the
                                        # client can display it right
                                        # away, before the goodbye
                                        # audio finishes generating.
                                        await websocket.send_json({
                                            "type": "interview_complete",
                                            "candidate": (
                                                state["candidate"]
                                            ),
                                        })

                                        await speak(
                                            GOODBYE_ID,
                                            GOODBYE_MESSAGE,
                                        )

                                        return

                                    # =================================
                                    # NEXT QUESTION
                                    # =================================

                                    next_question = state[
                                        "current_question"
                                    ]

                                    if next_question:

                                        logger.info(
                                            "[INTERVIEW] "
                                            "Next question: %s",
                                            next_question["id"],
                                        )

                                        await websocket.send_json({
                                            "type": "question",
                                            "id": next_question["id"],
                                            "text": (
                                                next_question[
                                                    "question_ar"
                                                ]
                                            ),
                                            "is_welcome": False,
                                            "repeat": False,
                                        })

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

                        except WebSocketDisconnect:
                            raise

                        except Exception as e:

                            logger.exception(
                                "[STT] Transcription failed"
                            )

                            await websocket.send_json({
                                "type": "error",
                                "stage": "stt",
                                "message": str(e),
                            })

                    # -----------------------------------------
                    # CLEAN UP UTTERANCE
                    # -----------------------------------------

                    audio_buffer.clear()

                    vad.reset()

                    # Note: resampler is NOT reset here — it must
                    # keep tracking the raw input stream continuously
                    # regardless of speech/silence state.

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

        # Cancel any TTS pre-generation tasks that are still running
        # when the session ends (e.g. candidate hangs up early) so
        # they don't keep the event loop doing pointless work for a
        # connection that's already gone.
        for task in tts_tasks.values():
            if not task.done():
                task.cancel()

        logger.info(
            "[WS] Connection closed"
        )