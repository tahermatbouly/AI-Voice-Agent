
import os
from pathlib import Path

import cohere
from dotenv import load_dotenv


# ============================================================
# LOAD PROJECT .ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

env_file = BASE_DIR / ".env"

print(f"[ENV] Loading: {env_file}")

load_dotenv(env_file)


# ============================================================
# CHECK API KEY
# ============================================================

api_key = os.getenv("CO_API_KEY")

if not api_key:
    raise RuntimeError(
        "CO_API_KEY was not found.\n"
        f"Expected .env at: {env_file}"
    )

print("[ENV] CO_API_KEY loaded successfully.")


# ============================================================
# COHERE CLIENT
# ============================================================

client = cohere.ClientV2(
    api_key=api_key
)


# ============================================================
# AUDIO FILE
# ============================================================

audio_file = (
    BASE_DIR
    / "tests"
    / "sample.wav"
)

if not audio_file.exists():
    raise FileNotFoundError(
        f"Audio file not found: {audio_file}"
    )


# ============================================================
# TRANSCRIPTION
# ============================================================

print("[COHERE] Sending audio...")

with open(audio_file, "rb") as f:

    response = client.audio.transcriptions.create(
        model="cohere-transcribe-arabic-07-2026",
        language="ar",
        file=f,
    )


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("COHERE TRANSCRIPTION")
print("=" * 60)

print(response.text)

print("=" * 60)
