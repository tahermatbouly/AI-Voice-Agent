import subprocess
import time

MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
AUDIO = "test_egyptian_recruitment.wav"
OUTPUT = "coherex_output"


print("=" * 60)
print("CohereX Arabic STT Test")
print("=" * 60)

print(f"Audio : {AUDIO}")
print(f"Model : {MODEL}")

command = [
    "coherex",
    AUDIO,
    "--model",
    MODEL,
    "--language",
    "ar",
    "-f",
    "json",
    "-o",
    OUTPUT,
]

print("\nStarting transcription...")
print("-" * 60)

start = time.perf_counter()

result = subprocess.run(command)

elapsed = time.perf_counter() - start

if result.returncode == 0:
    print("-" * 60)
    print(f"Transcription completed in {elapsed:.2f} seconds")
    print(f"Output saved to: {OUTPUT}/")
    print("=" * 60)
    print("TEST COMPLETED")
else:
    print("-" * 60)
    print("Transcription failed.")
    print("=" * 60)