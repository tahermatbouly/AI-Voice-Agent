"""
Standalone test script for faster-whisper (local STT).

Run this on its own to:
  1. Confirm faster-whisper loads correctly on your CPU
  2. Check transcription accuracy on real Egyptian Arabic speech
  3. Time how long transcription takes relative to audio length (important
     for a live agent -- if transcription takes longer than the audio
     itself, the pipeline can't keep up with real-time conversation)

This test transcribes existing .wav files rather than recording live, so
you can test it in isolation without needing a working mic setup yet, and
so you can reuse the same test files to compare model sizes fairly.

Setup:
  1. Record (or find) a few short .wav files of Egyptian Arabic speech,
     ideally the kind of thing a caller might actually say -- a name, an
     address, a short question. 16kHz mono is ideal but not required.
  2. Put their paths in TEST_AUDIO_FILES below.
  3. Run: python test_faster_whisper.py

If you don't have real recordings yet, this script also offers a
--record mode to record a few short mic clips first (see RECORD_MODE).
"""

import os
import sys
import time

# --- Config ---------------------------------------------------------------

WHISPER_MODEL_SIZE = "small"    # try "small" first, then "medium" to compare
# TIP: if "small" is slow to download, temporarily set this to "tiny"
# (~75MB vs ~500MB) to confirm the script/pipeline works end-to-end first,
# then switch back to "small" or "medium" for real accuracy testing.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"   # faster on CPU, small accuracy tradeoff
WHISPER_LANGUAGE = "ar"

# Fill these in with paths to real .wav files of Egyptian Arabic speech.
# Leave empty and use --record mode instead if you don't have files yet.
TEST_AUDIO_FILES = [
    # "samples/sample1.wav",
    # "samples/sample2.wav",
]

# If TEST_AUDIO_FILES is empty, record this many short clips from the mic.
RECORD_MODE = True
RECORD_COUNT = 3
RECORD_SECONDS = 5
SAMPLE_RATE = 16000


def check_imports():
    missing = []
    for module_name in ("faster_whisper",):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with: pip install faster-whisper==1.2.1")
        sys.exit(1)


def record_test_clips():
    """Record a few short clips from the mic to use as test audio."""
    import sounddevice as sd
    import soundfile as sf

    print(f"[test] no TEST_AUDIO_FILES set -- recording {RECORD_COUNT} clips instead.")
    print("[test] speak a short Egyptian Arabic sentence for each (e.g. your name and address).")

    paths = []
    for i in range(1, RECORD_COUNT + 1):
        input(f"\n[test] press Enter, then speak for {RECORD_SECONDS}s (clip {i}/{RECORD_COUNT})...")
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        path = f"whisper_test_input_{i}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        print(f"[test] saved {path}")
        paths.append(path)

    return paths


def load_model():
    from faster_whisper import WhisperModel

    print(f"[test] loading faster-whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
    print("[test] (first load after caching should take a few seconds -- if this hangs")
    print("[test]  for minutes, it's likely a slow Hugging Face Hub 'check for updates'")
    print("[test]  network call, not a broken cache. Ctrl+C and retry with:")
    print("[test]    export HF_HUB_OFFLINE=1")
    t0 = time.time()

    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        )
    except Exception as e:
        print(f"[test] FAILED to load model: {e}")
        if "local_files_only" in str(e) or "Cannot find" in str(e):
            print("[test] the model isn't fully cached yet -- unset HF_HUB_OFFLINE and let it download first.")
        sys.exit(1)

    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return model


def get_audio_duration(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


def run_tests(model, audio_files):
    results = []

    for i, path in enumerate(audio_files, start=1):
        if not os.path.exists(path):
            print(f"\n[test] sentence {i}: FILE NOT FOUND: {path}")
            results.append((i, path, False, None, None, None, ""))
            continue

        audio_duration = get_audio_duration(path)
        print(f"\n[test] file {i}: {path} ({audio_duration:.1f}s audio)")

        t0 = time.time()
        try:
            segments, info = model.transcribe(
                path,
                language=WHISPER_LANGUAGE,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            elapsed = time.time() - t0

            real_time_factor = elapsed / audio_duration if audio_duration > 0 else float("inf")
            keeps_up = real_time_factor < 1.0

            print(f"[test] transcribed in {elapsed:.1f}s (RTF={real_time_factor:.2f}) -> \"{text}\"")
            print(f"[test] {'OK -- faster than real-time' if keeps_up else 'SLOWER than real-time -- may lag in a live call'}")

            results.append((i, path, True, elapsed, audio_duration, real_time_factor, text))

        except Exception as e:
            print(f"[test] FAILED on file {i}: {e}")
            results.append((i, path, False, None, audio_duration, None, ""))

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for i, path, ran, elapsed, duration, rtf, text in results:
        if not ran:
            print(f"  File {i} ({path}): FAILED")
            all_ok = False
        else:
            status = "OK" if rtf < 1.0 else "SLOW (RTF >= 1.0)"
            print(f"  File {i}: {elapsed:.1f}s for {duration:.1f}s audio, RTF={rtf:.2f} [{status}]")
            print(f"           transcript: \"{text}\"")
            if rtf >= 1.0:
                all_ok = False

    print("=" * 60)
    print("RTF = real-time factor = transcription_time / audio_duration.")
    print("RTF < 1.0 means it transcribes faster than the audio plays -- good for live use.")
    print("RTF >= 1.0 means transcription can't keep up with real-time conversation.")
    if not all_ok:
        print("\nIf RTF is consistently >= 1.0: try WHISPER_MODEL_SIZE = \"tiny\" or \"base\",")
        print("or double-check WHISPER_COMPUTE_TYPE is set to \"int8\".")
    print("\nAlso manually check: did it transcribe the Egyptian Arabic correctly,")
    print("or did it slip into MSA-style words/spelling?")


if __name__ == "__main__":
    check_imports()

    audio_files = TEST_AUDIO_FILES
    if not audio_files and RECORD_MODE:
        audio_files = record_test_clips()
    elif not audio_files:
        print("[test] TEST_AUDIO_FILES is empty and RECORD_MODE is False -- nothing to test.")
        sys.exit(1)

    model = load_model()
    results = run_tests(model, audio_files)
    print_summary(results)