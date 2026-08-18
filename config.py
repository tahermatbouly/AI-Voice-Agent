"""
Central configuration for the voice agent prototype.
Edit these values to tune behavior without digging through the other files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq (LLM brain only — STT runs locally via faster-whisper) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = "openai/gpt-oss-120b"  # Groq's recommended replacement for
                                          # llama-3.3-70b-versatile (deprecated
                                          # as of June 17, 2026), still free tier

# --- STT (faster-whisper, local, CPU) ---
WHISPER_MODEL_SIZE = "small"      # "small" or "medium" — test latency on your machine
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"     # int8 = faster on CPU, small accuracy tradeoff
WHISPER_LANGUAGE = "ar"           # Arabic — Whisper doesn't have a separate Egyptian code

# --- TTS (edge-tts, free, no API key) ---
TTS_VOICE = "ar-EG-SalmaNeural"   # Egyptian Arabic female voice
TTS_OUTPUT_FILE = "reply.mp3"

# --- Audio recording ---
SAMPLE_RATE = 16000                # Whisper expects 16kHz
SILENCE_THRESHOLD = 0.01           # rough silence cutoff for auto-stop recording
MAX_RECORD_SECONDS = 15            # safety cap per utterance

# --- Storage ---
RECORDS_DIR = "records"            # one JSON file per call

# --- Agent system prompt ---
# Written in Arabic so the model responds naturally in Egyptian dialect.
SYSTEM_PROMPT = """\
انت موظف استقبال بيرد على تليفونات شركة. اتكلم باللهجة المصرية بشكل طبيعي ومهذب.
هدفك إنك تجمع من المتصل: الاسم، العنوان، الوظيفة أو الجهة اللي بيتكلم منها، والاستفسار بتاعه.
اسأل سؤال واحد بس في كل مرة، وابقى مختصر.

لازم ترد دايمًا بصيغة JSON فقط، من غير أي نص تاني قبلها أو بعدها، بالشكل ده:
{
  "reply_text": "الرد اللي هتقوله للمتصل",
  "extracted": {
    "name": "الاسم لو اتقال",
    "address": "العنوان لو اتقال",
    "position": "الوظيفة لو اتقالت",
    "inquiry": "الاستفسار لو اتقال",
    "notes": "أي معلومة إضافية مهمة"
  },
  "call_done": false
}

لو حسيت إن المكالمة خلصت (المتصل قال مع السلامة أو خلص كل حاجة عايز يقولها)، خلي call_done: true.
سيب أي حقل فاضي "" لو لسه معرفتوش.
"""
