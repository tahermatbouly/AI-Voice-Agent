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
from Backend.agent.interview_manager import InterviewManager, value_cache_id
from Backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"

RESULTS_DIR = Path("Backend/data/interviews")

TTS_CACHE_DIR = Path("Backend/data/tts_cache")

TTS_SEND_CHUNK_SIZE = 32 * 1024

# Extra mute after estimated playback so silence is measured only once
# the client has finished playing (clients buffer until tts_end).
TTS_TAIL_SECONDS = 0.4

DEFAULT_SILENCE_MS = 700


# ============================================================
# RESULT STORAGE
# ============================================================

def _write_candidate_json(
    candidate: dict,
    session_id: str,
) -> str:

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
# TTS CACHE
# ============================================================

def _cache_path(
    item_id: str,
    speaker: str,
) -> Path:

    safe_speaker = speaker.replace(
        "/",
        "_",
    )

    return (
        TTS_CACHE_DIR
        / f"{item_id}__{safe_speaker}.wav"
    )


def _read_cached_wav(
    path: Path,
) -> tuple[bytes, int]:

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

    path = _cache_path(
        item_id,
        tts.speaker,
    )

    logger.info(
        "[TTS] Checking cache | id=%s | path=%s",
        item_id,
        path,
    )

    if await asyncio.to_thread(
        path.exists
    ):

        logger.info(
            "[TTS] Cache hit | %s",
            item_id,
        )

        return await asyncio.to_thread(
            _read_cached_wav,
            path,
        )

    logger.info(
        "[TTS] Cache miss | %s",
        item_id,
    )

    logger.info(
        "[TTS] Generating voice | id=%s | text=%s",
        item_id,
        text,
    )

    audio, sample_rate = await tts.synthesize(
        text
    )

    logger.info(
        "[TTS] Generated | id=%s | bytes=%d | rate=%d",
        item_id,
        len(audio),
        sample_rate,
    )

    await asyncio.to_thread(
        _write_cached_wav,
        path,
        audio,
        sample_rate,
    )

    logger.info(
        "[TTS] Cache written | %s",
        path,
    )

    return audio, sample_rate


async def _pregenerate_static_tts(
    interview_manager: InterviewManager,
    tts: VoiceTutTTS,
) -> dict:

    items = interview_manager.static_tts_items()

    logger.info(
        "[TTS] Static TTS items found: %d",
        len(items),
    )

    if not items:

        logger.warning(
            "[TTS] No static TTS items were returned "
            "by InterviewManager"
        )

        return {}

    tasks = {}

    for item_id, text in items:

        logger.info(
            "[TTS] Preparing static clip | id=%s | text=%s",
            item_id,
            text,
        )

        tasks[item_id] = asyncio.create_task(
            _get_or_synthesize(
                tts,
                item_id,
                text,
            )
        )

    logger.info(
        "[TTS] Waiting for %d static clips...",
        len(tasks),
    )

    results = {}

    for item_id, task in tasks.items():

        try:

            results[item_id] = await task

            logger.info(
                "[TTS] Ready | %s",
                item_id,
            )

        except Exception:

            logger.exception(
                "[TTS] Failed to generate | %s",
                item_id,
            )

            raise

    logger.info(
        "[TTS] Static TTS pre-generation complete"
    )

    return {
        item_id: asyncio.create_task(
            _return_cached_audio(audio)
        )
        for item_id, (audio, sample_rate)
        in results.items()
    }


async def _return_cached_audio(
    audio: bytes,
    sample_rate: int,
) -> tuple[bytes, int]:

    return audio, sample_rate


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

    recorder = AnswerRecorder(
        session_id
    )

    vad = VoiceActivityDetector(
        min_silence_duration_ms=DEFAULT_SILENCE_MS
    )

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    resampler = StreamResampler(
        source_rate=INPUT_SAMPLE_RATE,
        target_rate=TARGET_SAMPLE_RATE,
    )

    interview_manager = InterviewManager(
        QUESTIONS_PATH
    )

    logger.info(
        "[INTERVIEW] InterviewManager loaded"
    )

    logger.info(
        "[TTS] Cache directory: %s",
        TTS_CACHE_DIR.resolve(),
    )

    # ========================================================
    # PRE-GENERATE STATIC TTS
    # ========================================================

    try:

        static_audio = {}

        static_items = (
            interview_manager.static_tts_items()
        )

        logger.info(
            "[TTS] Found %d static clips",
            len(static_items),
        )

        for item_id, text in static_items:

            logger.info(
                "[TTS] Pre-generating | %s",
                item_id,
            )

            audio, sample_rate = (
                await _get_or_synthesize(
                    tts,
                    item_id,
                    text,
                )
            )

            static_audio[item_id] = (
                audio,
                sample_rate,
            )

        logger.info(
            "[TTS] All static clips ready"
        )

    except Exception as e:

        logger.exception(
            "[TTS] Static pre-generation failed"
        )

        try:

            await websocket.send_json({
                "type": "error",
                "stage": "tts_cache",
                "message": str(e),
            })

        except Exception:
            pass

        return

    # ========================================================
    # VALUE TTS CACHE
    # ========================================================

    value_tasks = {}

    # ========================================================
    # STATE
    # ========================================================

    state = {
        "transcript": "",
        "candidate": {},
        "current_question_index": -1,
        "current_question": None,
        "mode": "interview",
        "current_prompt": [],
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": False,
        "failure_reason": None,
        "interview_finished": False,
        "ended_early": False,
    }

    speech_active = False

    mic_muted_until = 0.0

    # Wall-clock time when the client's sequential TTS queue is
    # expected to finish. Multi-segment turns (summary) send clips
    # back-to-back, but the client plays them one after another —
    # so mute/silence must cover the sum of durations, not only
    # the last clip.
    playback_ends_at = 0.0

    silence_task = None

    silence_level = 0

    # ========================================================
    # SPEAK ONE SEGMENT
    # ========================================================

    async def speak_segment(
        seg: dict,
    ):

        nonlocal speech_active
        nonlocal mic_muted_until
        nonlocal playback_ends_at

        text = seg.get("text")

        if not text:
            return

        mic_muted_until = float("inf")

        speech_active = False

        audio_buffer.clear()

        vad.reset()

        kind = seg.get(
            "kind",
            "static",
        )

        item_id = None

        try:

            if kind == "static":

                item_id = seg["id"]

                cached = static_audio.get(
                    item_id
                )

                if cached is not None:

                    audio, sample_rate = cached

                    logger.info(
                        "[TTS] Using pre-generated "
                        "static clip | %s",
                        item_id,
                    )

                else:

                    logger.info(
                        "[TTS] Static clip wasn't "
                        "pre-generated | %s",
                        item_id,
                    )

                    audio, sample_rate = (
                        await _get_or_synthesize(
                            tts,
                            item_id,
                            text,
                        )
                    )

            else:

                item_id = value_cache_id(
                    text
                )

                task = value_tasks.get(
                    item_id
                )

                if task is None:

                    task = asyncio.create_task(
                        _get_or_synthesize(
                            tts,
                            item_id,
                            text,
                        )
                    )

                    value_tasks[item_id] = task

                audio, sample_rate = (
                    await task
                )

            duration_s = len(audio) / (
                sample_rate * 2
            )

            logger.info(
                "[TTS] Playing | id=%s | "
                "duration=%.2fs | bytes=%d",
                item_id,
                duration_s,
                len(audio),
            )

            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            for offset in range(
                0,
                len(audio),
                TTS_SEND_CHUNK_SIZE,
            ):

                await websocket.send_bytes(
                    audio[
                        offset:
                        offset + TTS_SEND_CHUNK_SIZE
                    ]
                )

            await websocket.send_json({
                "type": "tts_end",
            })

            # Queue this clip after any playback already scheduled.
            # Silence must not start until the whole queue finishes.
            now = time.monotonic()
            queued_start = max(now, playback_ends_at)
            playback_ends_at = (
                queued_start
                + duration_s
                + TTS_TAIL_SECONDS
            )
            mic_muted_until = playback_ends_at

            logger.info(
                "[TTS] Finished | %s | "
                "playback_ends_in=%.2fs",
                item_id,
                max(0.0, playback_ends_at - now),
            )

        except WebSocketDisconnect:

            raise

        except Exception as e:

            mic_muted_until = 0.0
            playback_ends_at = 0.0

            logger.exception(
                "[TTS] Failed to speak | %s",
                item_id,
            )

            try:

                await websocket.send_json({
                    "type": "error",
                    "stage": "tts",
                    "message": str(e),
                })

            except Exception:
                pass

    # ========================================================
    # SPEAK SEGMENTS
    # ========================================================

    async def speak_segments(
        segments: list[dict],
    ):

        for seg in segments:

            await speak_segment(
                seg
            )

    # ========================================================
    # SPEAK CURRENT TURN
    # ========================================================

    async def speak_turn(
        turn_state: dict,
    ):

        # Never let a silence nudge overlap agent speech.
        await stop_silence_timer()

        question = (
            turn_state.get(
                "current_question"
            )
        )

        await websocket.send_json({
            "type": "turn",
            "mode": turn_state.get(
                "mode"
            ),
            "question_id": (
                question["id"]
                if question
                else None
            ),
        })

        intent_prompt = (
            turn_state.get(
                "intent_prompt"
            )
            or []
        )

        if intent_prompt:

            await speak_segments(
                intent_prompt
            )

        current_prompt = (
            turn_state.get(
                "current_prompt"
            )
            or []
        )

        if current_prompt:

            await speak_segments(
                current_prompt
            )

        # Agent has finished *sending*; wait until the client has
        # finished *hearing* before the caller starts silence.
        await wait_until_listening()

    # ========================================================
    # SILENCE
    # ========================================================

    async def stop_silence_timer():

        nonlocal silence_task

        if silence_task is not None:

            silence_task.cancel()

            try:

                await silence_task

            except asyncio.CancelledError:

                pass

            silence_task = None

    async def wait_until_listening():
        """
        Block until TTS playback mute ends.

        Sleep in short slices so a temporary mic_muted_until=inf
        (set while a new clip is being prepared) does not hang
        forever after the real end time is written.
        """

        while True:

            remaining = (
                mic_muted_until
                - time.monotonic()
            )

            if remaining <= 0:

                return

            await asyncio.sleep(
                min(remaining, 0.25)
            )

    async def _finish(
        ended_early: bool,
    ):

        await stop_silence_timer()

        try:

            saved_path = await asyncio.to_thread(
                _write_candidate_json,
                state["candidate"],
                session_id,
            )

            logger.info(
                "[INTERVIEW] Saved candidate data: %s",
                saved_path,
            )

        except Exception:

            logger.exception(
                "[INTERVIEW] Failed to save candidate data"
            )

        try:

            await websocket.send_json({
                "type": "interview_complete",
                "candidate": state["candidate"],
                "ended_early": ended_early,
            })

        except Exception:

            pass

    async def silence_monitor():

        nonlocal silence_level

        step = (
            interview_manager.silence_step_seconds
        )

        while True:

            # Count silence only while the mic is live (after playback).
            await wait_until_listening()

            await asyncio.sleep(
                step
            )

            if speech_active:

                continue

            # User may have spoken (and been muted-out) during the wait;
            # or TTS may have started again — only nudge on true silence.
            if time.monotonic() < mic_muted_until:

                continue

            silence_level += 1

            prompt = (
                interview_manager.get_silence_prompt(
                    silence_level
                )
            )

            if prompt is None:

                return

            logger.info(
                "[INTERVIEW] Silence nudge | level=%d | id=%s",
                silence_level,
                prompt["id"],
            )

            await speak_segment({
                "kind": "static",
                "id": prompt["id"],
                "text": prompt["text"],
            })

            if prompt.get(
                "end_call"
            ):

                logger.info(
                    "[INTERVIEW] Ending call "
                    "after silence escalation"
                )

                await _finish(
                    ended_early=True
                )

                return

    async def start_silence_timer():

        nonlocal silence_task
        nonlocal silence_level

        await stop_silence_timer()

        silence_level = 0

        silence_task = asyncio.create_task(
            silence_monitor()
        )

        logger.info(
            "[INTERVIEW] Silence timer started | "
            "%s seconds after listening begins",
            interview_manager.silence_step_seconds,
        )

    # ========================================================
    # START INTERVIEW
    # ========================================================

    try:

        logger.info(
            "[INTERVIEW] Starting graph"
        )

        state = await agent_graph.ainvoke(
            state
        )

        logger.info(
            "[INTERVIEW] Graph returned"
        )

        logger.info(
            "[INTERVIEW] Current question: %s",
            state.get(
                "current_question"
            ),
        )

        logger.info(
            "[INTERVIEW] Current prompt: %s",
            state.get(
                "current_prompt"
            ),
        )

        # ----------------------------------------------------
        # PLAY FIRST QUESTION
        # ----------------------------------------------------

        await speak_turn(
            state
        )

        logger.info(
            "[INTERVIEW] First question played"
        )

        await start_silence_timer()

        # ====================================================
        # MAIN AUDIO LOOP
        # ====================================================

        while True:

            message = await websocket.receive()

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            if message.get(
                "text"
            ) is not None:

                await websocket.send_json({
                    "type": "message_received",
                    "message": message["text"],
                })

                continue

            # ------------------------------------------------
            # AUDIO
            # ------------------------------------------------

            audio_data = message.get(
                "bytes"
            )

            if not audio_data:

                continue

            now = time.monotonic()

            if now < mic_muted_until:

                continue

            # =================================================
            # AUDIO PROCESSING
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

            vad_events = vad.process(
                normalized_audio
            )

            # =================================================
            # VAD — speech start (must run before buffering)
            # =================================================

            for event in vad_events:

                if event == "speech_start":

                    logger.info(
                        "[VAD] Speech started"
                    )

                    speech_active = True

                    await stop_silence_timer()

                    audio_buffer.clear()

                    await websocket.send_json({
                        "type": "speech_started",
                    })

            # -------------------------------------------------
            # BUFFER CURRENT CHUNK while the user is speaking.
            # Without this, every utterance is empty and the
            # graph retries the same question as no_speech.
            # -------------------------------------------------

            if speech_active:

                audio_buffer.add(
                    normalized_audio
                )

            # =================================================
            # VAD — speech end
            # =================================================

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

                transcript = ""

                # =========================================
                # STT
                # =========================================

                if utterance:

                    try:

                        transcript = (
                            await stt.transcribe(
                                audio_bytes=utterance,
                                sample_rate=TARGET_SAMPLE_RATE,
                            )
                        )

                        transcript = (
                            transcript.strip()
                        )

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

                        audio_buffer.clear()

                        vad.reset()

                        await start_silence_timer()

                        continue

                logger.info(
                    "[STT] Transcript: %s",
                    transcript or "<EMPTY>",
                )

                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                })

                # =========================================
                # RECORD ANSWER
                # =========================================

                question = (
                    state.get(
                        "current_question"
                    )
                )

                question_id = (
                    question["id"]
                    if question
                    else "summary"
                )

                recording_path = (
                    await recorder.save(
                        audio=utterance,
                        sample_rate=TARGET_SAMPLE_RATE,
                        question_id=question_id,
                        transcript=transcript,
                    )
                )

                if recording_path:

                    logger.info(
                        "[RECORD] Saved answer audio: %s",
                        recording_path,
                    )

                # =========================================
                # GRAPH
                # =========================================

                try:

                    state["transcript"] = (
                        transcript
                    )

                    logger.info(
                        "[AGENT] Sending transcript "
                        "to graph"
                    )

                    state = (
                        await agent_graph.ainvoke(
                            state
                        )
                    )

                    logger.info(
                        "[AGENT] Graph returned"
                    )

                    logger.info(
                        "[AGENT] Candidate: %s",
                        state.get(
                            "candidate"
                        ),
                    )

                    # -------------------------------------
                    # FINISHED
                    # -------------------------------------

                    if state.get(
                        "interview_finished"
                    ):

                        logger.info(
                            "[INTERVIEW] Complete | "
                            "ended_early=%s",
                            state.get(
                                "ended_early"
                            ),
                        )

                        await speak_turn(
                            state
                        )

                        await _finish(
                            state.get(
                                "ended_early",
                                False,
                            )
                        )

                        return

                    # -------------------------------------
                    # NEXT QUESTION
                    # -------------------------------------

                    await speak_turn(
                        state
                    )

                    logger.info(
                        "[INTERVIEW] Question played"
                    )

                    await start_silence_timer()

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

                    await start_silence_timer()

                # -----------------------------------------
                # RESET AUDIO STATE
                # -----------------------------------------

                audio_buffer.clear()

                vad.reset()

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

            logger.exception(
                "[WS] Runtime error"
            )

    except Exception:

        logger.exception(
            "[WS] Unexpected error"
        )

    finally:

        await stop_silence_timer()

        for task in value_tasks.values():

            if not task.done():

                task.cancel()

        logger.info(
            "[WS] Connection closed"
        )