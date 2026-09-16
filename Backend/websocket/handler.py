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
from Backend.agent.interview_manager import (
    InterviewManager,
    BEFORE_FOLLOWUPS_ID,
    NO_SPEECH_RETRY_ID,
    WRONG_ANSWER_RETRY_ID,
)
from Backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

QUESTIONS_PATH = "Backend/agent/questions.json"

RESULTS_DIR = Path("Backend/data/interviews")

# Persistent TTS cache, keyed by item_id + speaker.
#
# In one-paragraph mode EVERY spoken clip is static (intro, each
# field's follow-up question, the system messages, completion), so
# after the first run against a given speaker there is never another
# live VoiceTut call. There is no dynamically built text anywhere in
# this flow.
TTS_CACHE_DIR = Path("Backend/data/tts_cache")

# Keep WebSocket audio messages below common message-size limits.
TTS_SEND_CHUNK_SIZE = 32 * 1024

# The opening paragraph is long and the candidate pauses to think
# mid-answer, so the turn must not end on a short pause. Follow-up
# answers are short, so a long threshold there would just add dead
# air before the agent replies.
INTRO_SILENCE_MS = 1500
FOLLOWUP_SILENCE_MS = 500


# ============================================================
# INTERVIEW RESULT STORAGE
# ============================================================

def _write_candidate_json(
    candidate: dict,
    session_id: str,
) -> str:
    """
    Blocking file write — always called via asyncio.to_thread().
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / f"{session_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidate, f, ensure_ascii=False, indent=2)

    return str(output_path)


# ============================================================
# PERSISTENT TTS CACHE
# ============================================================

def _cache_path(item_id: str, speaker: str) -> Path:

    safe_speaker = speaker.replace("/", "_")

    return TTS_CACHE_DIR / f"{item_id}__{safe_speaker}.wav"


def _read_cached_wav(path: Path) -> tuple[bytes, int]:

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())

    return pcm, sample_rate


def _write_cached_wav(
    path: Path,
    audio: bytes,
    sample_rate: int,
) -> None:

    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _get_or_synthesize(
    tts: VoiceTutTTS,
    item_id: str,
    text: str,
) -> tuple[bytes, int]:

    path = _cache_path(item_id, tts.speaker)

    if await asyncio.to_thread(path.exists):

        logger.info("[TTS] Cache hit: %s", item_id)

        return await asyncio.to_thread(_read_cached_wav, path)

    logger.info("[TTS] Cache miss, synthesizing: %s", item_id)

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
    Start loading/synthesizing every clip concurrently, in the
    background, before the intro is even spoken.

    interview_manager.tts_items() is exhaustive for this mode, so
    every speak() below is guaranteed to be an await on one of
    these tasks — never an on-demand synthesis.
    """

    tasks = {
        item_id: asyncio.create_task(
            _get_or_synthesize(tts, item_id, text)
        )
        for item_id, text in interview_manager.tts_items()
    }

    logger.info(
        "[TTS] Started pre-generation for %d static items",
        len(tasks),
    )

    return tasks


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    logger.info("[WS] Client connected")

    session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # ========================================================
    # SESSION COMPONENTS
    # ========================================================

    audio_buffer = AudioBuffer()

    recorder = AnswerRecorder(session_id)

    # Starts in long-pause mode: the very first thing the candidate
    # does is give the long opening paragraph.
    vad = VoiceActivityDetector(
        min_silence_duration_ms=INTRO_SILENCE_MS,
    )

    stt = CohereArabicSTT()

    tts = VoiceTutTTS()

    # Stateful — do NOT reset between utterances.
    resampler = StreamResampler(
        source_rate=INPUT_SAMPLE_RATE,
        target_rate=TARGET_SAMPLE_RATE,
    )

    interview_manager = InterviewManager(QUESTIONS_PATH)

    tts_tasks = _start_tts_pregeneration(interview_manager, tts)

    # ========================================================
    # INITIAL STATE
    # ========================================================

    state = {
        "transcript": "",
        "candidate": {},
        "mode": "intro",
        "current_field": None,
        "current_prompt": None,
        "extraction_success": False,
        "failure_reason": None,
        "interview_finished": False,
        "first_followup": False,
    }

    speech_active = False

    # While monotonic time is below this value, incoming microphone
    # audio is ignored.
    mic_muted_until = 0.0

    # ========================================================
    # SPEAK ONE CLIP
    # ========================================================

    async def speak(item_id: str, text: str):
        """
        Speak exactly ONE cached clip. Never combines two pieces of
        text — that's what would force a live synthesis.
        """

        nonlocal speech_active, mic_muted_until

        if not text:
            return

        # Stop accepting candidate speech immediately, before the
        # cache lookup, so nothing said during the gap is captured.
        mic_muted_until = float("inf")
        speech_active = False
        audio_buffer.clear()
        vad.reset()

        task = tts_tasks.get(item_id)

        try:

            if task is not None:
                audio, sample_rate = await task

            else:
                # Shouldn't happen in this mode — every clip is
                # pre-generated — but don't go silent if it does.
                logger.warning(
                    "[TTS] No pre-generated clip for %s, "
                    "synthesizing on demand",
                    item_id,
                )
                audio, sample_rate = await tts.synthesize(text)

            duration_s = len(audio) / (sample_rate * 2)

            mic_muted_until = time.monotonic() + duration_s

            logger.info(
                "[TTS] Sending %s | %.2fs | %d bytes",
                item_id,
                duration_s,
                len(audio),
            )

            await websocket.send_json({
                "type": "tts_start",
                "sample_rate": sample_rate,
                "channels": 1,
            })

            for offset in range(0, len(audio), TTS_SEND_CHUNK_SIZE):
                await websocket.send_bytes(
                    audio[offset:offset + TTS_SEND_CHUNK_SIZE]
                )

            await websocket.send_json({"type": "tts_end"})

        except WebSocketDisconnect:
            raise

        except Exception as e:

            mic_muted_until = 0.0

            logger.exception("[TTS] Failed to speak %s", item_id)

            try:
                await websocket.send_json({
                    "type": "error",
                    "stage": "tts",
                    "message": str(e),
                })
            except WebSocketDisconnect:
                raise

    # ========================================================
    # SPEAK THE CURRENT PROMPT
    # ========================================================

    async def speak_prompt():
        """
        Speak whatever the graph decided to say next, plus — when
        applicable — a short cached lead-in before it. Each piece is
        its own cached clip, spoken back to back.

        Safe as consecutive speak() calls: nothing between them
        awaits websocket.receive(), so mic muting stays continuous
        (each call pushes mic_muted_until forward before the loop
        can read audio again).
        """

        prompt = state.get("current_prompt")

        if not prompt:
            return

        # One-off "let me ask about what's missing" line, only on
        # the first follow-up after the intro paragraph.
        if state.get("first_followup"):

            lead_in = interview_manager.get_system_message(
                BEFORE_FOLLOWUPS_ID
            )

            if lead_in:
                await speak(BEFORE_FOLLOWUPS_ID, lead_in)

        await websocket.send_json({
            "type": "question",
            "id": prompt["id"],
            "text": prompt["text"],
            "field": state.get("current_field"),
            "repeat": False,
        })

        await speak(prompt["id"], prompt["text"])

    # ========================================================
    # RETRY
    # ========================================================

    async def speak_retry():
        """
        Clarification line + the same prompt again. Both are
        separate cached clips, so a retry costs zero VoiceTut calls.
        """

        prompt = state.get("current_prompt")

        if not prompt:
            return

        failure_reason = state.get("failure_reason")

        retry_id = (
            WRONG_ANSWER_RETRY_ID
            if failure_reason == "wrong_answer"
            else NO_SPEECH_RETRY_ID
        )

        logger.info(
            "[INTERVIEW] Retry | prompt=%s | reason=%s",
            prompt["id"],
            failure_reason,
        )

        await websocket.send_json({
            "type": "answer_not_understood",
            "question_id": prompt["id"],
            "failure_reason": failure_reason,
        })

        retry_text = interview_manager.get_system_message(retry_id)

        if retry_text:
            await speak(retry_id, retry_text)

        await websocket.send_json({
            "type": "question",
            "id": prompt["id"],
            "text": prompt["text"],
            "field": state.get("current_field"),
            "repeat": True,
        })

        await speak(prompt["id"], prompt["text"])

    # ========================================================
    # SESSION
    # ========================================================

    try:

        logger.info("[INTERVIEW] Starting (one-paragraph mode)")

        # ----------------------------------------------------
        # INTRO — one graph call with no transcript sets it up.
        # ----------------------------------------------------

        state = await agent_graph.ainvoke(state)

        await speak_prompt()

        # ====================================================
        # AUDIO LOOP
        # ====================================================

        while True:

            message = await websocket.receive()

            # -------------------------------------------------
            # TEXT MESSAGE
            # -------------------------------------------------

            if message.get("text") is not None:

                await websocket.send_json({
                    "type": "message_received",
                    "message": message["text"],
                })

                continue

            # -------------------------------------------------
            # AUDIO MESSAGE
            # -------------------------------------------------

            audio_data = message.get("bytes")

            if not audio_data:
                continue

            if time.monotonic() < mic_muted_until:
                continue

            audio = pcm16_to_float32(audio_data)
            audio = resampler.process(audio)
            normalized_audio = float32_to_pcm16(audio)

            vad_events = vad.process(normalized_audio)

            for event in vad_events:

                if event == "speech_start":

                    logger.info("[VAD] Speech started")

                    speech_active = True

                    audio_buffer.clear()

                    await websocket.send_json({
                        "type": "speech_started"
                    })

            if speech_active:
                audio_buffer.add(normalized_audio)

            for event in vad_events:

                if event != "speech_end":
                    continue

                logger.info(
                    "[VAD] Speech ended | Audio: %d bytes",
                    audio_buffer.size(),
                )

                speech_active = False

                utterance = audio_buffer.get_audio()

                await websocket.send_json({
                    "type": "speech_ended",
                    "bytes": len(utterance),
                    "sample_rate": TARGET_SAMPLE_RATE,
                })

                # =========================================
                # STT
                # =========================================

                transcript = ""

                if utterance:

                    try:

                        transcript = await stt.transcribe(
                            audio_bytes=utterance,
                            sample_rate=TARGET_SAMPLE_RATE,
                        )

                        transcript = transcript.strip()

                    except WebSocketDisconnect:
                        raise

                    except Exception as e:

                        logger.exception("[STT] Transcription failed")

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

                # =========================================
                # SAVE THE ANSWER RECORDING
                # =========================================
                # Before any branching, so rejected and
                # untranscribable answers are captured too —
                # those are the ones worth listening back to.

                prompt = state.get("current_prompt")

                recording_path = await recorder.save(
                    audio=utterance,
                    sample_rate=TARGET_SAMPLE_RATE,
                    question_id=prompt["id"] if prompt else "unknown",
                    transcript=transcript,
                )

                if recording_path:
                    logger.info(
                        "[RECORD] Saved answer audio: %s",
                        recording_path,
                    )

                # =========================================
                # EMPTY TRANSCRIPT
                # =========================================
                # Nothing usable was heard — always a no_speech
                # retry, handled without invoking the graph (there
                # is nothing for the LLM to reason about, and an
                # empty transcript would route back to "start" and
                # reset the whole interview).

                if not transcript:

                    state["failure_reason"] = "no_speech"
                    state["extraction_success"] = False

                    await speak_retry()

                    audio_buffer.clear()
                    vad.reset()

                    continue

                # =========================================
                # LANGGRAPH
                # =========================================

                try:

                    state["transcript"] = transcript

                    logger.info(
                        "[AGENT] Processing | mode=%s | field=%s",
                        state.get("mode"),
                        state.get("current_field"),
                    )

                    state = await agent_graph.ainvoke(state)

                    logger.info(
                        "[AGENT] Candidate: %s",
                        state["candidate"],
                    )

                    # -------------------------------------
                    # REJECTED ANSWER
                    # -------------------------------------

                    if not state["extraction_success"]:

                        await speak_retry()

                        audio_buffer.clear()
                        vad.reset()

                        continue

                    # -------------------------------------
                    # PAST THE INTRO — shorter answers now,
                    # so end turns on a shorter pause.
                    # -------------------------------------

                    vad.set_min_silence(FOLLOWUP_SILENCE_MS)

                    # -------------------------------------
                    # FINISHED
                    # -------------------------------------

                    if state["interview_finished"]:

                        logger.info(
                            "[INTERVIEW] Complete | candidate: %s",
                            state["candidate"],
                        )

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
                                "[INTERVIEW] Failed to save "
                                "candidate data"
                            )

                        await websocket.send_json({
                            "type": "interview_complete",
                            "candidate": state["candidate"],
                        })

                        # Completion line is a normal cached clip.
                        await speak_prompt()

                        return

                    # -------------------------------------
                    # NEXT MISSING FIELD
                    # -------------------------------------

                    await speak_prompt()

                except WebSocketDisconnect:
                    raise

                except Exception as e:

                    logger.exception("[AGENT] Processing failed")

                    await websocket.send_json({
                        "type": "error",
                        "stage": "agent",
                        "message": str(e),
                    })

                audio_buffer.clear()

                vad.reset()

                # Do NOT reset the resampler — it must stay
                # continuous across the whole microphone stream.

    except WebSocketDisconnect:

        logger.info("[WS] Client disconnected")

    except RuntimeError as e:

        if "disconnect message" in str(e):
            logger.info("[WS] Client disconnected")

        else:
            raise

    finally:

        for task in tts_tasks.values():
            if not task.done():
                task.cancel()

        logger.info("[WS] Connection closed")