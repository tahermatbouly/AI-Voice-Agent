"""
Standalone test script for Vosk (streaming, offline Arabic STT).

Run this to:
  1. Confirm Vosk loads and transcribes correctly
  2. Hear/read how it handles Egyptian Arabic specifically (expect weaker
     dialect accuracy than faster-whisper -- Vosk's Arabic model is
     generally MSA-trained)
  3. Time transcription, same RTF (real-time factor) metric used in
     test_faster_whisper.py, so results are directly comparable

Unlike faster-whisper, Vosk genuinely supports streaming -- this script
demonstrates that by feeding audio in small chunks (simulating a live
stream) rather than handing over one complete file at once, and prints
partial results as they arrive.

Setup:
  1. pip install vosk sounddevice soundfile
  2. Download an Arabic model from https://alphacephei.com/vosk/models
     (look for an "ar" entry), unzip it, and set VOSK_MODEL_PATH below
     to point at the unzipped folder.
  3. Run: python test_vosk.py
"""

import os
import sys
import time
import json

VOSK_MODEL_PATH = "vosk-model-ar"  # update to your unzipped model's folder name
SAMPLE_RATE = 16000

RECORD_MODE = True
RECORD_COUNT = 3
RECORD_SECONDS = 5

# Below this RTF, transcription is keeping up with real-time speech --
# same threshold used in test_faster_whisper.py for a fair comparison.
RTF_TARGET = 1.0


def check_imports():
    missing = []
    for module_name in ("vosk",):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with: pip install vosk sounddevice soundfile")
        sys.exit(1)

    if not os.path.isdir(VOSK_MODEL_PATH):
        print(f"[test] VOSK_MODEL_PATH not found: {VOSK_MODEL_PATH}")
        print("[test] download an Arabic model from https://alphacephei.com/vosk/models")
        print("[test] unzip it, then set VOSK_MODEL_PATH to the unzipped folder's path.")
        sys.exit(1)


def record_test_clips():
    import sounddevice as sd
    import soundfile as sf

    print(f"[test] recording {RECORD_COUNT} clips -- speak Egyptian Arabic clearly for each.")
    paths = []
    for i in range(1, RECORD_COUNT + 1):
        input(f"\n[test] press Enter, then speak for {RECORD_SECONDS}s (clip {i}/{RECORD_COUNT})...")
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        path = f"vosk_test_input_{i}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        print(f"[test] saved {path}")
        paths.append(path)
    return paths


def load_model():
    from vosk import Model

    print(f"[test] loading Vosk model from '{VOSK_MODEL_PATH}'...")
    t0 = time.time()
    try:
        model = Model(VOSK_MODEL_PATH)
    except Exception as e:
        print(f"[test] FAILED to load model: {e}")
        sys.exit(1)
    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return model


def transcribe_streaming(model, wav_path):
    """Feeds audio in small chunks to simulate a live stream, printing
    partial results as they arrive -- demonstrating Vosk's real
    streaming capability (unlike faster-whisper's batch-only approach)."""
    from vosk import KaldiRecognizer
    import soundfile as sf
    import numpy as np

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    data, sr = sf.read(wav_path, dtype="int16")
    if sr != SAMPLE_RATE:
        print(f"[test] WARNING: file sample rate {sr} != expected {SAMPLE_RATE}, results may be degraded")

    chunk_size = 4000  # simulate small streaming chunks, not one big blob
    partials_seen = 0

    t0 = time.time()
    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size].tobytes()
        if recognizer.AcceptWaveform(chunk):
            pass  # a final segment was recognized internally; full result collected at the end
        else:
            partial = json.loads(recognizer.PartialResult())
            if partial.get("partial"):
                partials_seen += 1
                print(f"    ...partial: {partial['partial']}")

    final = json.loads(recognizer.FinalResult())
    elapsed = time.time() - t0

    return final.get("text", ""), elapsed, partials_seen


def run_tests(model, audio_files):
    import soundfile as sf

    results = []
    for i, path in enumerate(audio_files, start=1):
        if not os.path.exists(path):
            print(f"\n[test] file {i}: NOT FOUND: {path}")
            results.append((i, False, None, None, None))
            continue

        info = sf.info(path)
        duration = info.frames / info.samplerate
        print(f"\n[test] file {i}: {path} ({duration:.1f}s audio)")

        text, elapsed, partials_seen = transcribe_streaming(model, path)
        rtf = elapsed / duration if duration > 0 else float("inf")
        status = "OK" if rtf < RTF_TARGET else "SLOW"

        print(f"[test] transcribed in {elapsed:.2f}s (RTF={rtf:.2f}, {partials_seen} partial updates) [{status}]")
        print(f"[test] final text: \"{text}\"")

        results.append((i, True, elapsed, duration, rtf))

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, ran, elapsed, duration, rtf in results:
        if not ran:
            print(f"  File {i}: FAILED")
        else:
            print(f"  File {i}: {elapsed:.2f}s for {duration:.1f}s audio, RTF={rtf:.2f}")
    print("=" * 60)
    print("Compare these RTF numbers directly against your faster-whisper results.")
    print("Also compare the actual transcribed TEXT above for dialect accuracy --")
    print("Vosk is expected to be faster but less accurate on Egyptian colloquial speech.")


if __name__ == "__main__":
    check_imports()
    audio_files = record_test_clips() if RECORD_MODE else []
    model = load_model()
    results = run_tests(model, audio_files)
    print_summary(results)
