from pathlib import Path
import time
import wave

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "EGTTS-V0.1"

MODEL_PATH = MODEL_DIR / "model.pth"
CONFIG_PATH = MODEL_DIR / "config.json"
REFERENCE_PATH = MODEL_DIR / "speaker_reference.wav"

OUTPUT_PATH = PROJECT_ROOT / "output" / "tests" / "local_egtts.wav"


print("=" * 70)
print("Local EGTTS-V0.1 Test")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Config:")
print(CONFIG_PATH)

print()
print("Reference:")
print(REFERENCE_PATH)

assert MODEL_PATH.exists(), "model.pth missing"
assert CONFIG_PATH.exists(), "config.json missing"
assert REFERENCE_PATH.exists(), "speaker_reference.wav missing"

print()
print("All files found.")

# ------------------------------------------------------------
# NumPy compatibility
# ------------------------------------------------------------

aliases = {
    "long": int,
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
}

for name, replacement in aliases.items():
    if not hasattr(np, name):
        setattr(np, name, replacement)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

from TTS.api import TTS

print()
print("[1/2] Loading EGTTS-V0.1...")

start = time.perf_counter()

tts = TTS(
    model_path=str(MODEL_DIR),
    config_path=str(CONFIG_PATH),
    gpu=False,
)

load_time = time.perf_counter() - start

print()
print(f"Model loading time: {load_time:.2f}s")

# ------------------------------------------------------------
# Generate
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

text = (
    "أهلاً وسهلاً بحضرتك، معاك مسؤول التوظيف. "
    "ممكن أعرف اسم حضرتك؟"
)

print()
print("[2/2] Generating speech...")
print(f"Text: {text}")

start = time.perf_counter()

tts.tts_to_file(
    text=text,
    speaker_wav=str(REFERENCE_PATH),
    language="ar",
    file_path=str(OUTPUT_PATH),
)

generation_time = time.perf_counter() - start

# ------------------------------------------------------------
# Inspect output
# ------------------------------------------------------------

with wave.open(str(OUTPUT_PATH), "rb") as wf:

    frames = wf.getnframes()
    sample_rate = wf.getframerate()

    duration = frames / sample_rate

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Model loading time : {load_time:.2f}s")
print(f"TTS generation time: {generation_time:.2f}s")
print(f"Audio duration     : {duration:.2f}s")
print(f"Real-Time Factor   : {generation_time / duration:.2f}")

print()
print(f"Output:")
print(OUTPUT_PATH)

print()
print("=" * 70)