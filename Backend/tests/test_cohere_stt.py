import asyncio
import os
import time
import wave
from pathlib import Path

import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from cohere import AsyncClientV2


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

MODEL = "cohere-transcribe-arabic-07-2026"

SAMPLE_RATE = 16000
CHANNELS = 1

OUTPUT_DIR = Path("test_audio/cohere")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK CONFIG
# ============================================================

if not COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY is not set.\n"
        "Add it to your .env file."
    )


# ============================================================
# COHERE CLIENT
# ============================================================

client = AsyncClientV2(api_key=COHERE_API_KEY)


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio(filename: Path, duration: float = 8.0):
    print()
    print("=" * 60)
    print(f"Recording for {duration:.1f} seconds...")
    print("Speak naturally.")
    print("=" * 60)

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    sd.wait()

    print("Recording finished.")

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"Saved: {filename}")


# ============================================================
# TRANSCRIBE
# ============================================================
async def transcribe_audio(filename: Path):

    print()
    print("-" * 60)
    print("Sending audio to Cohere...")
    print("-" * 60)

    start = time.perf_counter()

    with open(filename, "rb") as audio_file:

        response = await client.audio.transcriptions.create(
            model=MODEL,
            language="ar",
            file=audio_file,
        )

    elapsed = time.perf_counter() - start

    transcript = getattr(response, "text", None)

    if transcript is None:
        print("Raw Cohere response:")
        print(response)
        transcript = ""

    audio_duration = get_audio_duration(filename)

    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Transcript : {transcript}")
    print(f"Audio      : {audio_duration:.2f}s")
    print(f"STT time   : {elapsed:.2f}s")
    print(f"RTF        : {rtf:.2f}")

    if rtf < 1:
        print("Speed      : FASTER than real-time")
    elif rtf == 1:
        print("Speed      : REAL-TIME")
    else:
        print("Speed      : SLOWER than real-time")

    print("=" * 60)

    return transcript, elapsed, rtf


# ============================================================
# AUDIO DURATION
# ============================================================

def get_audio_duration(filename: Path):

    with wave.open(str(filename), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()

    return frames / rate


# ============================================================
# TEST LOOP
# ============================================================

async def main():

    print()
    print("COHERE ARABIC STT TEST")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Sample rate: {SAMPLE_RATE}")
    print("=" * 60)

    results = []

    test_number = 1

    while True:

        print()
        print("=" * 60)
        print(f"TEST #{test_number}")
        print("=" * 60)

        print()
        print("Press ENTER to start recording.")
        print("Type 'q' to quit.")

        command = input("> ")

        if command.lower() == "q":
            break

        filename = OUTPUT_DIR / f"test_{test_number:03d}.wav"

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        record_audio(
            filename,
            duration=8.0,
        )

        # ----------------------------------------------------
        # TRANSCRIBE
        # ----------------------------------------------------

        transcript, latency, rtf = await transcribe_audio(
            filename
        )

        # ----------------------------------------------------
        # REFERENCE
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("REFERENCE TRANSCRIPT")
        print("-" * 60)

        reference = input(
            "Type exactly what you intended to say:\n> "
        )

        results.append(
            {
                "test": test_number,
                "reference": reference,
                "transcript": transcript,
                "latency": latency,
                "rtf": rtf,
            }
        )

        test_number += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if not results:
        print("No tests performed.")
        return

    total_latency = 0
    total_rtf = 0

    for result in results:

        print()
        print(f"TEST #{result['test']}")
        print("-" * 60)

        print("REFERENCE:")
        print(result["reference"])

        print()
        print("COHERE:")
        print(result["transcript"])

        print()
        print(f"Latency: {result['latency']:.2f}s")
        print(f"RTF:     {result['rtf']:.2f}")

        total_latency += result["latency"]
        total_rtf += result["rtf"]

    print()
    print("=" * 80)
    print("AVERAGES")
    print("=" * 80)

    print(
        f"Average latency: "
        f"{total_latency / len(results):.2f}s"
    )

    print(
        f"Average RTF: "
        f"{total_rtf / len(results):.2f}"
    )

    print()
    print(f"Tests performed: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())