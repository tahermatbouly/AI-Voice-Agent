"""
Standalone test script for SILMA TTS v1 (150M params, F5-TTS based).

IMPORTANT: this model's own card states native fluency for MSA/Fusha and
English -- Egyptian dialect is not a stated training target for this
open-source release. Test with realistic expectations: this may sound
correct but formal/MSA-leaning rather than authentically Egyptian. Compare
directly against your NAMAA and EGTTS results before choosing this one.

Like EGTTS, this is a voice-cloning model -- it needs a short reference
audio clip. Unlike EGTTS, it ships with a bundled sample reference voice,
so a first test works even without recording your own.

Run this to:
  1. Confirm the model downloads and loads correctly
  2. Hear how it handles Egyptian Arabic text specifically
  3. Time generation on your CPU (the model card claims very low latency
     on GPU -- CPU timing will be meaningfully different, worth measuring)

Setup:
  1. System ffmpeg required (same as earlier tests tonight).
  2. pip install silma-tts
  3. Run: python test_silma.py

Optional: set REFERENCE_AUDIO_PATH / REFERENCE_AUDIO_TEXT below to use
your own reference_voice.wav (from the EGTTS test) instead of the
bundled sample, for a more consistent voice across your comparisons.
"""

import os
import sys
import time

# Leave both as None to use SILMA's bundled sample reference voice.
# Set both to use your own (e.g. the reference_voice.wav from the EGTTS
# test) -- REFERENCE_AUDIO_TEXT should be None if you want it auto-
# transcribed, or the exact transcription if you have it.
REFERENCE_AUDIO_PATH = None
REFERENCE_AUDIO_TEXT = None

TEST_SENTENCES = [
    "انا سبت الشغل و راجع دلوقتي علي طول",
    "ممكن أعرف اسمك وعنوانك من فضلك؟",
    "شكرا لاتصالك بينا، هنرد عليك في أقرب وقت ممكن",
]

MIN_EXPECTED_SECONDS = 0.5


def check_imports():
    missing = []
    try:
        import silma_tts  # noqa: F401
    except ImportError:
        missing.append("silma-tts")

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with: pip install silma-tts")
        print("[test] (also requires system ffmpeg -- see requirements-silma-test.txt)")
        sys.exit(1)


def load_model():
    from silma_tts.api import SilmaTTS

    print("[test] loading SILMA TTS v1...")
    print("[test] (first run downloads the model checkpoint -- much smaller")
    print("[test]  than NAMAA/EGTTS at 150M params, should be faster to fetch)")
    t0 = time.time()

    try:
        model = SilmaTTS()
    except Exception as e:
        print(f"[test] FAILED to load model: {e}")
        print("[test] check your internet connection and that dependencies installed correctly.")
        sys.exit(1)

    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return model


def run_tests(model):
    results = []

    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] sentence {i}: {text}")
        out_path = f"silma_test_{i}.wav"
        t0 = time.time()

        try:
            wav, sr, spec = model.infer(
                ref_file=REFERENCE_AUDIO_PATH,   # None = use bundled sample
                ref_text=REFERENCE_AUDIO_TEXT,   # None = auto-transcribe
                gen_text=text,
                file_wave=out_path,
                seed=None,
                speed=1,
            )
            elapsed = time.time() - t0

            duration = len(wav) / sr if sr else 0
            file_size = os.path.getsize(out_path)
            passed = duration >= MIN_EXPECTED_SECONDS and file_size > 0
            status = "PASS" if passed else "SUSPICIOUS (too short/empty)"

            print(f"[test] generated in {elapsed:.1f}s, audio duration {duration:.1f}s -> {out_path} [{status}]")
            results.append((i, True, elapsed, duration, passed))

        except Exception as e:
            print(f"[test] FAILED on sentence {i}: {e}")
            results.append((i, False, None, None, False))

    return results


def print_summary(results):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for i, ran, elapsed, duration, passed in results:
        if not ran:
            print(f"  Sentence {i}: FAILED TO RUN")
            all_passed = False
        elif not passed:
            print(f"  Sentence {i}: SUSPICIOUS OUTPUT ({elapsed:.1f}s gen, {duration:.1f}s audio)")
            all_passed = False
        else:
            print(f"  Sentence {i}: OK ({elapsed:.1f}s gen, {duration:.1f}s audio)")

    print("=" * 50)
    if all_passed:
        print("All tests passed. Listen to silma_test_*.wav and judge:")
        print("  - Does it sound Egyptian, or more MSA/formal (likely, per the model card)?")
        print("  - How does generation speed compare to NAMAA/EGTTS on your CPU?")
        print("  - The model's small size (150M) may make it noticeably faster --")
        print("    worth weighing against any dialect authenticity trade-off.")
    else:
        print("Some tests failed or looked suspicious -- see details above.")


if __name__ == "__main__":
    check_imports()
    model = load_model()
    results = run_tests(model)
    print_summary(results)
