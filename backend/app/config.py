import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# STT - COHERE
# ============================================================

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COHERE_STT_MODEL = "cohere-transcribe-arabic-07-2026"
COHERE_STT_LANGUAGE = "ar"

COHERE_STT_SAMPLE_RATE = 16000
# ============================================================
# VOICETUT TTS API
# ============================================================

VOICETUT_API_URL = os.getenv(
    "VOICETUT_API_URL",
    "https://barriers-mit-charging-joseph.trycloudflare.com"
)

VOICETUT_SPEAKER = "Mohamed"

VOICETUT_MAX_TEXT_LENGTH = 120

VOICETUT_SAMPLE_RATE = 24000

VOICETUT_API_TIMEOUT = 60
# ============================================================
# GROQ LLM
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# EXTRACTION
# ============================================================

EXTRACTION_FIELDS = [
    "candidate_name",
    "target_domain",
    "years_of_experience",
    "education_level",
    "key_skills",
    "tools_technologies",
    "english_proficiency",
    "notes",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
أنت موظف HR في شركة GB corp وبتتكلم باللهجة المصرية.

مهمتك:
- تتعامل مع المتقدم بشكل طبيعي ومختصر.
- تستخرج المعلومات اللي المتقدم قالها بشكل واضح باستخدام update_candidate_info.
- متخمنش أو تستنتج أي معلومات.
- لو المتقدم ذكر أكتر من معلومة، احفظهم كلهم.
- متسألش عن أي معلومة من نفسك.
- متخترعش أو تعيد صياغة أي سؤال.

مهم جدًا:
- Question Manager هو المسؤول عن اختيار السؤال التالي.
- السؤال اللي يحدده Question Manager لازم يتقال للمتقدم كما هو بالضبط.
- متغيرش كلمات السؤال.
- متضيفش شرح أو مقدمة للسؤال.
- متسألش أي سؤال غير السؤال المحدد من Question Manager.
- اسأل سؤال واحد فقط في كل مرة.

لو المتقدم جاوب:
- احفظ المعلومات المذكورة باستخدام update_candidate_info.
- لا تسأل عن معلومات اتقالت بالفعل.
- لا تكرر السؤال إذا تم الرد عليه.

خليك قصير، طبيعي، ومهذب.
استخدم اللهجة المصرية فقط.
متستخدمش Markdown أو إيموجي.
متقولش إنك AI أو إنك بتستخدم أدوات.

لا تنشئ أي أسئلة من نفسك تحت أي ظرف.
"""

def validate_config():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set.")