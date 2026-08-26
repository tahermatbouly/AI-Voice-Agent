"""
Main AI Voice Agent worker.

Pipeline:

    Caller
      |
      v
    LiveKit
      |
      v
    Silero VAD
      |
      v
    Faster-Whisper CPU
      |
      v
    Groq LLM
      |
      v
    VoiceTut-TTS
      |
      v
    LiveKit audio

NO:
    - DeepFilterNet
    - cloud turn detector
    - cloud STT
    - ElevenLabs
    - EGTTS
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import groq, silero

from agent.stt_plugin import FasterWhisperSTT
from agent.tts_plugin import build_tts
from app import config


load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("voice-agent")


# ============================================================
# AGENT
# ============================================================

class GBCorpAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=config.SYSTEM_PROMPT
        )


# ============================================================
# ENTRYPOINT
# ============================================================

async def entrypoint(ctx: JobContext):

    logger.info("[LIVEKIT] Connecting...")

    await ctx.connect()

    logger.info(
        "[LIVEKIT] Connected: %s",
        ctx.room.name,
    )

    # --------------------------------------------------------
    # STT
    # --------------------------------------------------------

    logger.info(
        "[STT] Loading Faster-Whisper '%s'...",
        config.WHISPER_MODEL_SIZE,
    )

    stt = FasterWhisperSTT(
        model_size=config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        language=config.WHISPER_LANGUAGE,
        beam_size=1,
    )

    logger.info("[STT] Ready.")

    # --------------------------------------------------------
    # TTS
    # --------------------------------------------------------

    logger.info("[TTS] Loading VoiceTut-TTS...")

    tts = build_tts()

    logger.info("[TTS] Ready.")

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    logger.info("[LLM] Loading Groq...")

    llm = groq.LLM(
        model=config.GROQ_LLM_MODEL,
        api_key=config.GROQ_API_KEY,
    )

    logger.info("[LLM] Ready.")

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    logger.info("[VAD] Loading Silero VAD...")

    vad = silero.VAD.load(
        min_silence_duration=0.30,
        prefix_padding_duration=0.25,
    )

    logger.info("[VAD] Ready.")

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        allow_interruptions=False,
        preemptive_generation=False,
    )

    logger.info("[SESSION] Starting...")

    await session.start(
        room=ctx.room,
        agent=GBCorpAgent(),
    )

    logger.info("[SESSION] Started.")

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    logger.info("[AGENT] Sending greeting...")

    await session.generate_reply(
        instructions=(
            'ابدأ المكالمة فورًا وقل فقط: '
            '"أهلاً بيك، معاك قسم الموارد البشرية من GB corp. '
            'ممكن أعرف اسمك بالكامل؟"'
        )
    )

    logger.info("[AGENT] Greeting sent.")

    print()
    print("=" * 60)
    print("AI VOICE AGENT READY")
    print("=" * 60)
    print("Listening for caller...")
    print()


# ============================================================
# WORKER
# ============================================================

if __name__ == "__main__":

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )