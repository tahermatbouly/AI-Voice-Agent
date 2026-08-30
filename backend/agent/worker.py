
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
    Cohere Arabic STT
      |
      v
    Groq LLM
      |
      v
    VoiceTut-TTS
      |
      v
    LiveKit audio

Interview flow:

    question_manager.py
          |
          v
    extraction.py
          |
          v
    Structured candidate record

No additional LLM call is used for question selection.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import groq, silero

from agent.stt_plugin import CohereArabicSTT
from agent.tts_plugin import build_tts
from agent.extraction import CallExtraction, ExtractionTools
from agent.question_manager import QuestionManager

from app import config


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.INFO)

logger = logging.getLogger("voice-agent")


# ============================================================
# FINAL EXTRACTION
# ============================================================

def print_final_extraction(extraction: CallExtraction) -> None:
    """
    Print the final structured candidate record to the terminal.
    """

    record = extraction.get_record()

    print()
    print("=" * 64)
    print("                    FINAL CANDIDATE DATA")
    print("=" * 64)

    fields = [
        ("Candidate Name", "candidate_name"),
        ("Target Domain", "target_domain"),
        ("Years of Experience", "years_of_experience"),
        ("Education Level", "education_level"),
        ("Key Skills", "key_skills"),
        ("Tools / Technologies", "tools_technologies"),
        ("English Proficiency", "english_proficiency"),
        ("Notes", "notes"),
    ]

    for label, key in fields:
        value = record.get(key)

        if value is None or value == "":
            value = "<not provided>"

        print(f"{label:24}: {value}")

    print("=" * 64)
    print("                    INTERVIEW COMPLETE")
    print("=" * 64)
    print()


# ============================================================
# FAREWELL DETECTION
# ============================================================

def is_farewell(text: str) -> bool:
    """
    Detect explicit farewell phrases.
    """

    if not text:
        return False

    text = text.strip().lower()

    farewell_phrases = (
        "مع السلامة",
        "سلام",
        "باي",
        "باى",
        "باي باي",
        "شكرا مع السلامة",
        "شكراً مع السلامة",
    )

    return any(
        phrase in text
        for phrase in farewell_phrases
    )


# ============================================================
# AGENT
# ============================================================

class GBCorpAgent(Agent):

    def __init__(
        self,
        extraction: CallExtraction,
        question_manager: QuestionManager,
    ) -> None:

        self.extraction = extraction
        self.question_manager = question_manager

        # Tools used by the LLM to save caller information.
        self.extraction_tools = ExtractionTools(
            extraction
        )

        super().__init__(
            instructions=config.SYSTEM_PROMPT,
            tools=[
                self.extraction_tools.update_candidate_info,
            ],
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

    # ========================================================
    # CALL STATE
    # ========================================================

    # One extraction object per call.
    extraction = CallExtraction()

    # Track whether the final data has already been printed.
    #
    # This prevents:
    #
    #     interview complete -> print
    #     shutdown -> print again
    #
    extraction_printed = False

    # ========================================================
    # FINAL OUTPUT HELPER
    # ========================================================

    def print_final_once() -> None:
        """
        Print the final candidate data only once.
        """

        nonlocal extraction_printed

        if extraction_printed:
            return

        extraction_printed = True

        logger.info(
            "[CALL] Printing final candidate extraction..."
        )

        print_final_extraction(extraction)

    # ========================================================
    # QUESTION MANAGER
    # ========================================================

    questions_path = (
        Path(__file__).resolve().parent / "questions.json"
    )

    question_manager = QuestionManager(
        questions_path=questions_path
    )

    logger.info(
        "[QUESTIONS] Question manager initialized: %s",
        questions_path,
    )

    # ========================================================
    # EXTRACTION
    # ========================================================

    logger.info(
        "[CALL] Extraction initialized."
    )

    # ========================================================
    # STT
    # ========================================================

    logger.info(
        "[STT] Loading Cohere Transcribe Arabic..."
    )

    stt = CohereArabicSTT(
        api_key=config.COHERE_API_KEY,
        model=config.COHERE_STT_MODEL,
        language=config.COHERE_STT_LANGUAGE,
        sample_rate=config.COHERE_STT_SAMPLE_RATE,
    )

    logger.info(
        "[STT] Cohere STT Ready."
    )

    # ========================================================
    # TTS
    # ========================================================

    logger.info(
        "[TTS] Connecting to VoiceTut API: %s",
        config.VOICETUT_API_URL,
    )

    tts = build_tts()

    logger.info(
        "[TTS] VoiceTut API ready."
    )

    # ========================================================
    # LLM
    # ========================================================

    logger.info(
        "[LLM] Loading Groq..."
    )

    llm = groq.LLM(
        model=config.GROQ_LLM_MODEL,
        api_key=config.GROQ_API_KEY,
    )

    logger.info(
        "[LLM] Ready."
    )

    # ========================================================
    # VAD
    # ========================================================

    logger.info(
        "[VAD] Loading Silero VAD..."
    )

    vad = silero.VAD.load(
        min_silence_duration=0.30,
        prefix_padding_duration=0.25,
    )

    logger.info(
        "[VAD] Ready."
    )

    # ========================================================
    # AGENT
    # ========================================================

    agent = GBCorpAgent(
        extraction=extraction,
        question_manager=question_manager,
    )

    # ========================================================
    # SESSION
    # ========================================================

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,

        # Keep current behavior.
        allow_interruptions=False,
        preemptive_generation=False,
    )

    logger.info(
        "[SESSION] Starting..."
    )

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info(
        "[SESSION] Started."
    )

    # ========================================================
    # SHUTDOWN CALLBACK
    # ========================================================

    @ctx.add_shutdown_callback
    async def on_shutdown(
        reason: str | None = None,
    ):
        logger.info(
            "[CALL] Call ended. Reason: %s",
            reason,
        )

        # Always print whatever information was extracted.
        #
        # This covers:
        #
        # 1. Normal interview completion.
        # 2. Caller hanging up early.
        # 3. LiveKit shutting down.
        # 4. Unexpected call termination.
        #
        print_final_once()

    # ========================================================
    # FIRST QUESTION
    # ========================================================

    first_question = question_manager.get_welcome()

    logger.info(
        "[AGENT] Sending welcome message..."
    )

    await session.generate_reply(
        instructions=(
            "ابدأ المكالمة فورًا.\n"
            "قل الرسالة التالية فقط، بدون إضافة أي كلام:\n\n"
            f"{first_question}"
        )
    )

    logger.info(
        "[AGENT] Welcome message sent."
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print()
    print("=" * 60)
    print("AI VOICE AGENT READY")
    print("=" * 60)
    print("STT       : Cohere Arabic")
    print("LLM       : Groq")
    print("TTS       : VoiceTut")
    print("VAD       : Silero")
    print("Questions : Question Manager")
    print("Extraction: Structured")
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
