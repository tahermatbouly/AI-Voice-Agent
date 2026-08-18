"""
Standalone test script for NAMAA-Egyptian-TTS.

Run this on its own first, separate from the rest of the pipeline, to:
  1. Confirm the model downloads and loads correctly
  2. Hear what it actually sounds like
  3. Time how long generation takes on your CPU (important for a live agent)
  4. Sanity-check that real audio was actually produced, not empty/broken output

If this works and the latency is acceptable, we'll wire it into tts.py next.
If it's too slow on CPU, edge-tts (ar-EG-SalmaNeural) remains the fallback.

Run:
    python test_namaa_tts.py
"""

import os
import sys
import time

# A few Egyptian Arabic test sentences covering different lengths/content,
# so you can judge quality and timing across realistic call-agent phrases.
TEST_SENTENCES = [
    "انا سبت الشغل و راجع دلوقتي علي طول",
    "ممكن أعرف اسمك وعنوانك من فضلك؟",
    "شكرا لاتصالك بينا، هنرد عليك في أقرب وقت ممكن",
]

# Change to "cuda" if you ever get access to a GPU. "cpu" for now.
DEVICE = "cpu"

# Below this duration, treat the output as suspicious (likely empty/broken)
# rather than a real spoken sentence.
MIN_EXPECTED_SECONDS = 0.5


def check_imports():
    """Fail fast with a clear message if dependencies aren't installed,
    instead of a raw traceback halfway through."""
    missing = []
    for module_name in ("torch", "torchaudio", "chatterbox", "huggingface_hub", "safetensors"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with:")
        print("  pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install -r requirements-namaa-test.txt")
        sys.exit(1)


def load_model():
    import torchaudio as ta  # noqa: F401 (imported here so check_imports runs first)
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    from chatterbox import mtl_tts

    print(f"[test] loading NAMAA-Egyptian-TTS on device='{DEVICE}'...")
    t0 = time.time()

    try:
        ckpt_dir = snapshot_download(
            repo_id="NAMAA-Space/NAMAA-Egyptian-TTS",
            repo_type="model",
            revision="main",
        )

        model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

        t3_state = load_safetensors(
            f"{ckpt_dir}/t3_mtl23ls_v2.safetensors",
            device=DEVICE,
        )
        model.t3.load_state_dict(t3_state)
        model.t3.to(DEVICE).eval()
    except Exception as e:
        print(f"[test] FAILED to load model: {e}")
        print("[test] check your internet connection and that dependencies installed correctly.")
        sys.exit(1)

    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return model


def run_tests(model):
    import torchaudio as ta

    results = []

    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] sentence {i}: {text}")
        t0 = time.time()

        try:
            wav = model.generate(text, language_id="ar")
            elapsed = time.time() - t0

            out_path = f"namaa_test_{i}.wav"
            ta.save(out_path, wav, model.sr)

            duration = wav.shape[-1] / model.sr
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
        print("All tests passed. Listen to the namaa_test_*.wav files to judge quality.")
        print("If generation time feels too slow for a live call, edge-tts is the fallback.")
    else:
        print("Some tests failed or looked suspicious — see details above.")


if __name__ == "__main__":
    check_imports()
    model = load_model()
    results = run_tests(model)
    print_summary(results)
