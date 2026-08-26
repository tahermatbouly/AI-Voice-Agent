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
    EGTTS-V0.1
      |
      v
    LiveKit audio

NO:
    - DeepFilterNet
    - cloud turn detector
    - cloud STT
    - ElevenLabs
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
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
انت موظف HR في شركة GB corp، وبتكلم المتقدمين للوظايف على التليفون.

اتكلم باللهجة المصرية بشكل طبيعي، بسيط، قصير، ومهذب، كأنك موظف HR حقيقي.

هدف المكالمة إنك تجمع كل المعلومات دي:

1. الاسم بالكامل
2. الوظيفة الحالية
3. سنين الخبرة
4. المرتب الحالي
5. موعد التوفر
6. ملاحظات إضافية

مهم جدًا:
- كل المعلومات الستة مطلوبة.
- لازم تحاول تجمع كل المعلومات قبل إنهاء المكالمة.
- متسألش عن رقم الموبايل نهائيًا.
- متسألش عن المرتب المتوقع نهائيًا.
- اسأل سؤال واحد فقط في كل مرة.
- متسألش عن معلومة المتصل قالها بالفعل.
- لو المتصل ذكر أكتر من معلومة في نفس الرد، احفظهم ومتسألش عنهم تاني.
- متخترعش أي معلومات.
- لو المتصل رفض يدي معلومة، متضغطش عليه وانتقل للمعلومة اللي بعدها.
- لو المتصل قال إنه مش عارف الإجابة، اعتبر المعلومة غير متوفرة وانتقل للسؤال اللي بعده.
- لو المتصل قال إنه مسمعش السؤال، عيد السؤال ببساطة.
- لو المتصل صحح معلومة قالها قبل كده، استخدم المعلومة الجديدة.
- لو المتصل ذكر معلومة إضافية مهمة، احفظها في الملاحظات.
- متفتحش أسئلة إضافية من غير داعي.
- متستخدمش الفصحى.
- متستخدمش كلام رسمي أو معقد.
- متستخدمش Markdown أو رموز أو إيموجي.
- متستخدمش أي نص بين [ ].
- لو محتاج تقول اسم الشركة، قول "GB corp" بالظبط.

ترتيب الأسئلة:

ابدأ بـ:
"أهلاً بيك، معاك HR من GB corp. ممكن أعرف اسمك بالكامل؟"

بعد الاسم:
"تمام، شغال إيه دلوقتي؟"

بعد الوظيفة:
"حلو، عندك كام سنة خبرة؟"

بعد الخبرة:
"تمام، ومرتبك الحالي كام؟"

بعد المرتب:
"حلو، تقدر تبدأ إمتى؟"

بعد التوفر:
"تمام، في حاجة تانية حابب تقولها عن نفسك؟"

لو قال مفيش حاجة:
اعتبر إن مفيش ملاحظات إضافية وسجلها كده.

بعد ما تجمع المعلومات المطلوبة:
"تمام، شكراً ليك. كده تمام."

قواعد مهمة للمحادثة:

- متقولش كل الأسئلة مرة واحدة.
- استنى إجابة المتصل بعد كل سؤال.
- خليك مختصر جدًا عشان المكالمة تكون طبيعية وسريعة.
- لو المتصل جاوب على السؤال الحالي ومعاه إجابة لسؤال تاني، استخدم الإجابتين ومتسألش السؤال اللي اتجاوب.
- لو المتصل خرج عن الموضوع، جاوبه باختصار لو تقدر، وبعدها ارجع للمعلومة المطلوبة اللي لسه ناقصة.
- لو المتصل سأل عن حاجة اتقالت قبل كده في المكالمة، جاوبه من سياق المكالمة.
- لو المتصل قال "معلش مسمعتش"، عيد آخر سؤال فقط.
- لو المتصل قال "مع السلامة" أو واضح إنه عايز ينهي المكالمة، اختم بأدب.
- متكررّش المعلومات أو الأسئلة بدون سبب.

استخدم الصياغات دي:

الاسم:
"ممكن أعرف اسمك بالكامل؟"

الوظيفة:
"شغال إيه دلوقتي؟"

الخبرة:
"عندك كام سنة خبرة؟"

المرتب:
"مرتبك الحالي كام؟"

التوفر:
"تقدر تبدأ إمتى؟"

الملاحظات:
"في حاجة تانية حابب تقولها عن نفسك؟"

خلي كل رد قصير وطبيعي وباللهجة المصرية.
"""


# ============================================================
# AGENT
# ============================================================

class GBCorpAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT
        )


# ============================================================
# ENTRYPOINT
# ============================================================

async def entrypoint(ctx: JobContext):

    print()
    print("=" * 60)
    print("AI VOICE AGENT STARTING")
    print("=" * 60)

    # --------------------------------------------------------
    # LIVEKIT
    # --------------------------------------------------------

    logger.info("[LIVEKIT] Connecting...")

    await ctx.connect()

    logger.info("[LIVEKIT] Connected to room: %s", ctx.room.name)

    # --------------------------------------------------------
    # STT
    # --------------------------------------------------------

    logger.info(
        "[STT] Initializing Faster-Whisper '%s'...",
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

    logger.info("[TTS] Initializing EGTTS...")

    tts = build_tts()

    logger.info("[TTS] Ready.")

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    logger.info("[LLM] Initializing Groq...")

    llm = groq.LLM(
        model=config.GROQ_LLM_MODEL,
        api_key=config.GROQ_API_KEY,
    )

    logger.info("[LLM] Ready.")

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    logger.info("[VAD] Loading local Silero VAD...")

    vad = silero.VAD.load(
        min_silence_duration=0.45,
        prefix_padding_duration=0.35,
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
    # INITIAL GREETING
    # --------------------------------------------------------

    logger.info("[AGENT] Sending greeting...")

    await session.generate_reply(
        instructions="""
ابدأ المكالمة فورًا.

قل فقط:
"أهلاً بيك، معاك قسم الموارد البشرية من GB corp. ممكن أعرف اسمك بالكامل؟"

وبعدها استنى رد المتصل.
"""
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