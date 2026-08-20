"""
LiveKit voice agent worker.

This is the orchestration layer connecting:

    LiveKit
       |
    VAD / turn detection
       |
    faster-whisper STT
       |
    Groq LLM
       |
    HR extraction tools
       |
    TTS fallback chain

The worker intentionally contains very little business logic.
Extraction lives in extraction.py, STT in stt_plugin.py, and TTS
in tts_plugin.py.

NOTE on turn detection: LiveKit has two different turn-detector systems.
The older, text-based models (MultilingualModel/EnglishModel, from the
livekit-plugins-turn-detector package) do NOT support Arabic. This file
uses the newer audio-based livekit.agents.inference.TurnDetector instead,
which ships directly with livekit-agents (no separate plugin install)
and explicitly supports Arabic among its 14 documented languages. It
requires a one-time model download -- see the setup note before
entrypoint() below.
"""

from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.agents.inference import TurnDetector
from livekit.plugins import groq, silero

from app import config
from agent.stt_plugin import WhisperSTT
from agent.tts_plugin import build_tts
from agent.extraction import CallExtraction, ExtractionTools

# Toggle: the audio TurnDetector officially supports Arabic, but its
# real-world quality specifically on Egyptian *dialect* (vs. MSA) is
# still unverified -- this model's language support was validated on
# Arabic broadly, not confirmed dialect-by-dialect. Set to False to
# fall back to plain VAD-based turn timing if it doesn't behave well
# on Egyptian Arabic in real testing.
USE_TURN_DETECTOR = True


class RecruitmentAgent(Agent):
    """
    Conversational HR recruitment agent.

    The structured extraction state belongs to CallExtraction.
    ExtractionTools exposes that state to the LLM through function tools.
    """

    def __init__(self, extraction: CallExtraction):
        self.extraction = extraction
        self.extraction_tools = ExtractionTools(extraction)

        super().__init__(
            instructions=config.SYSTEM_PROMPT,
            tools=[
                self.extraction_tools.update_candidate_info,
            ],
        )


async def entrypoint(ctx: agents.JobContext):
    """
    Entry point executed for each LiveKit agent job.
    """

    print("[worker] connecting to LiveKit room...")
    await ctx.connect()
    print("[worker] connected.")

    # ---------------------------------------------------------------
    # Per-call extraction state
    # ---------------------------------------------------------------
    extraction = CallExtraction()

    # ---------------------------------------------------------------
    # VAD -- used both to wrap the non-streaming STT (below) and as
    # AgentSession's own top-level vad= parameter for general voice
    # activity / interruption awareness.
    # ---------------------------------------------------------------
    vad = silero.VAD.load(
        min_silence_duration=0.55,
        prefix_padding_duration=0.5,
    )

    # ---------------------------------------------------------------
    # STT -- faster-whisper is non-streaming, so it's wrapped with
    # StreamAdapter + the VAD above, which buffers audio until VAD
    # detects a completed utterance, then hands it over in one batch.
    # ---------------------------------------------------------------
    whisper_stt = WhisperSTT()
    session_stt = agents.stt.StreamAdapter(
        stt=whisper_stt,
        vad=vad,
    )

    # ---------------------------------------------------------------
    # LLM
    # ---------------------------------------------------------------
    llm = groq.LLM(
        model=config.GROQ_LLM_MODEL,
        api_key=config.GROQ_API_KEY,
    )

    # ---------------------------------------------------------------
    # TTS -- the three-tier fallback chain
    # ---------------------------------------------------------------
    tts = build_tts()

    # ---------------------------------------------------------------
    # Turn detection -- audio-based TurnDetector, supports Arabic.
    # Falls back to plain VAD-based timing if disabled below.
    # ---------------------------------------------------------------
    turn_detection = TurnDetector() if USE_TURN_DETECTOR else None

    # ---------------------------------------------------------------
    # Agent session
    # ---------------------------------------------------------------
    session = AgentSession(
        stt=session_stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_detection=turn_detection,  # None falls back to plain VAD timing
    )

    # ---------------------------------------------------------------
    # Start the session
    # ---------------------------------------------------------------
    print("[worker] starting AgentSession...")
    await session.start(
        room=ctx.room,
        agent=RecruitmentAgent(extraction),
    )
    print("[worker] AgentSession started.")

    # ---------------------------------------------------------------
    # Initial greeting
    # ---------------------------------------------------------------
    await session.generate_reply(
        instructions=(
            "ابدأ المكالمة بتحية قصيرة ومهنية باللهجة المصرية، "
            "وعرّف نفسك كمسؤول توظيف واسأل المتقدم عن اسمه."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Default init timeout is too short for heavy CPU-only ML
            # imports (faster-whisper, torch-based TTS, etc). Also
            # reduce idle processes so they don't all compete for CPU
            # at once during import -- that contention is likely why
            # every subprocess was timing out at the same fixed mark.
            initialize_process_timeout=120.0,
            num_idle_processes=1,
        )
    )