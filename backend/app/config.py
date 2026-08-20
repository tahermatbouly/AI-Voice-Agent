"""
Central configuration. Every other file in this project reads its
settings from here instead of hardcoding values -- so changing a model,
a voice, or a prompt means editing this one file, not hunting through
logic code.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# LiveKit (real-time transport)
# ---------------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# ---------------------------------------------------------------------
# LLM (Groq, free tier)
# ---------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = "openai/gpt-oss-120b"  # replacement for deprecated
                                          # llama-3.3-70b-versatile

# ---------------------------------------------------------------------
# STT (faster-whisper, local, CPU)
# ---------------------------------------------------------------------
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "ar"

# ---------------------------------------------------------------------
# TTS -- prioritized fallback chain, tried in this order:
#   1. ElevenLabs (primary)
#   2. EGTTS-v0.1 (alternative, local)
#   3. edge-tts (last resort, always available)
# tts_plugin.py reads this list and falls through it in order.
# ---------------------------------------------------------------------
TTS_PROVIDER_ORDER = ["elevenlabs", "egtts", "edge_tts"]

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

EGTTS_REFERENCE_AUDIO_PATH = "reference_voice.wav"

EDGE_TTS_VOICE = "ar-EG-SalmaNeural"

# ---------------------------------------------------------------------
# HR extraction fields -- the canonical set of fields the agent tries to
# capture during a call. Referenced by both db.py (storage schema) and
# extraction.py (the LLM's function-tools), so the two never drift apart.
# ---------------------------------------------------------------------
EXTRACTION_FIELDS = [
    "candidate_name",
    "contact_info",       # phone number or email
    "position",           # job being applied for
    "experience",         # years of experience / qualifications
    "current_salary",
    "expected_salary",
    "availability",       # notice period / when they can start
    "notes",               # free-form additional info
]


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------
DATABASE_URL = "sqlite:///./calls.db"

# ---------------------------------------------------------------------
# Agent system prompt (Egyptian Arabic) -- HR recruitment agent.
# Note: explicitly instructs the agent to answer follow-up/clarification
# questions using prior conversation context, rather than assuming this
# behavior happens automatically.
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """\
انت موظف في قسم الموارد البشرية بيرد على تليفونات المتقدمين للوظايف. اتكلم
باللهجة المصرية بشكل طبيعي ومهذب ومحترف.

هدفك إنك تجمع من المتقدم المعلومات دي أثناء المكالمة:
- الاسم بالكامل وبيانات التواصل (رقم تليفون أو إيميل)
- الوظيفة اللي بيتقدملها
- سنين الخبرة والمؤهلات اللي عنده
- المرتب الحالي والمرتب المتوقع
- إمتى ممكن يبدأ الشغل (فترة الإشعار / التوفر)
- أي ملاحظات إضافية مهمة

اسأل سؤال واحد بس في كل مرة، وابقى مختصر ومحترف.

لو المتصل سأل سؤال عن حاجة انت قلتها قبل كده في المكالمة (زي "قلت كام؟" أو
"معلش مسمعتش" أو "ممكن تعيد؟")، جاوبه من الكلام اللي انت قلته فعلاً قبل كده
في المكالمة -- متطلبش منه يعيد سؤاله، وماتتجاهلش السؤال.

لو حسيت إن المكالمة خلصت (المتصل قال مع السلامة أو خلص كل حاجة عايز يقولها)،
اختم المكالمة بشكل مهذب واشكره على وقته.
"""

# ---------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "caller_memories",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
)
