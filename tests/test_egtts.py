"""
Standalone test script for EGTTS-V0.1 (XTTS v2 fine-tuned for Egyptian Arabic).

IMPORTANT -- this model works differently from NAMAA-Egyptian-TTS:
it's a VOICE-CLONING model (built on XTTS v2), not a model with a single
built-in voice. It needs a short REFERENCE AUDIO CLIP of a speaker's voice
(a few seconds is enough) -- the model then generates new Egyptian Arabic
speech that mimics that voice.

Run this on its own first to:
  1. Confirm the model downloads and loads correctly
  2. Hear what it actually sounds like with your chosen reference voice
  3. Time how long generation takes on your CPU

Setup before running:
  1. Get a short (5-10s) clean audio clip of someone speaking (ideally
     Egyptian Arabic, but any clear speech works reasonably for testing).
     A short recording of yourself works fine for an initial test --
     see the note in load_reference_audio() below if you want to record
     one using the same recording approach as the whisper test.
  2. Set REFERENCE_AUDIO_PATH below to point at that file.
  3. Run: python test_egtts.py

If this works and the latency/quality are good, we'll wire it into tts.py
next. If it's too slow or the cloned voice sounds off, NAMAA or edge-tts
remain the alternatives already tested/available.
"""

import os
import sys
import time

# --- Config ---------------------------------------------------------------

DEVICE = "cpu"  # no GPU available

# REQUIRED: path to a short reference audio clip (wav) of a voice to clone.
# The model's own examples use Egyptian Arabic reference audio for best
# results, but any clear speech sample works for an initial smoke test.
REFERENCE_AUDIO_PATH = "reference_voice.wav"

TEST_SENTENCES = [
    "انا سبت الشغل و راجع دلوقتي علي طول",
    "ممكن أعرف اسمك وعنوانك من فضلك؟",
    "شكرا لاتصالك بينا، هنرد عليك في أقرب وقت ممكن",
]

MIN_EXPECTED_SECONDS = 0.5


def check_imports():
    missing = []
    for module_name in ("torch", "torchaudio", "TTS", "huggingface_hub"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install -r requirements-egtts-test.txt")
        sys.exit(1)


def check_reference_audio():
    if not os.path.exists(REFERENCE_AUDIO_PATH):
        print(f"[test] REFERENCE_AUDIO_PATH not found: {REFERENCE_AUDIO_PATH}")
        print("[test] EGTTS needs a short (5-10s) reference voice clip to clone.")
        print("[test] Quick way to record one using your mic:")
        print("")
        print("  python -c \"")
        print("import sounddevice as sd, soundfile as sf")
        print("print('Recording 8s -- speak clearly...')")
        print("audio = sd.rec(int(8*16000), samplerate=16000, channels=1, dtype='float32')")
        print("sd.wait()")
        print("sf.write('reference_voice.wav', audio, 16000)")
        print("print('Saved reference_voice.wav')")
        print("\"")
        print("")
        print(f"[test] then rerun this script (make sure REFERENCE_AUDIO_PATH = '{REFERENCE_AUDIO_PATH}').")
        sys.exit(1)


def load_model():
    from huggingface_hub import snapshot_download
    from TTS.api import TTS

    print(f"[test] loading EGTTS-V0.1 on device='{DEVICE}'...")
    print("[test] (first run downloads the checkpoint -- this can take a while")
    print("[test]  on a slow connection, same as the other TTS models we tested)")
    t0 = time.time()

    try:
        ckpt_dir = snapshot_download(
            repo_id="OmarSamir/EGTTS-V0.1",
            repo_type="model",
            revision="main",
        )

        tts = TTS(
            model_path=ckpt_dir,
            config_path=os.path.join(ckpt_dir, "config.json"),
        ).to(DEVICE)
    except Exception as e:
        print(f"[test] FAILED to load model: {e}")
        print("[test] check your internet connection, disk space, and that dependencies installed correctly.")
        sys.exit(1)

    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return tts


def run_tests(tts):
    results = []

    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] sentence {i}: {text}")
        t0 = time.time()
        out_path = f"egtts_test_{i}.wav"

        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=REFERENCE_AUDIO_PATH,
                language="ar",
                file_path=out_path,
            )
            elapsed = time.time() - t0

            import soundfile as sf
            info = sf.info(out_path)
            duration = info.frames / info.samplerate
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
        print("All tests passed. Listen to egtts_test_*.wav to judge quality and how")
        print("well it cloned your reference voice speaking Egyptian Arabic.")
        print("Compare generation times against the NAMAA and edge-tts results.")
    else:
        print("Some tests failed or looked suspicious -- see details above.")


if __name__ == "__main__":
    check_imports()
    check_reference_audio()
    tts = load_model()
    results = run_tests(tts)
    print_summary(results)