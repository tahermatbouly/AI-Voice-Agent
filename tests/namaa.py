"""
Standalone test script for NAMAA-Egyptian-TTS.

Run this on its own first, separate from the rest of the pipeline, to:
  1. Confirm the model downloads and loads correctly
  2. Hear what it actually sounds like
  3. Time how long generation takes on your CPU (important for a live agent)

If this works and the latency is acceptable, we'll wire it into tts.py next.
If it's too slow on CPU, edge-tts (ar-EG-SalmaNeural) remains the fallback.
"""

import time
import torchaudio as ta
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from chatterbox import mtl_tts

# Change to "cuda" if you ever get access to a GPU. "cpu" for now.
DEVICE = "cpu"

# A few Egyptian Arabic test sentences covering different lengths/content,
# so you can judge quality and timing across realistic call-agent phrases.
TEST_SENTENCES = [
    "انا سبت الشغل و راجع دلوقتي علي طول",
    "ممكن أعرف اسمك وعنوانك من فضلك؟",
    "شكرا لاتصالك بينا، هنرد عليك في أقرب وقت ممكن",
]


def load_model():
    print(f"[test] loading NAMAA-Egyptian-TTS on device='{DEVICE}'...")
    t0 = time.time()

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

    print(f"[test] model loaded in {time.time() - t0:.1f}s")
    return model


def run_tests(model):
    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] sentence {i}: {text}")
        t0 = time.time()

        wav = model.generate(text, language_id="ar")

        elapsed = time.time() - t0
        out_path = f"namaa_test_{i}.wav"
        ta.save(out_path, wav, model.sr)

        print(f"[test] generated in {elapsed:.1f}s -> saved to {out_path}")


if __name__ == "__main__":
    model = load_model()
    run_tests(model)
    print("\n[test] done. Play the namaa_test_*.wav files to listen to the results.")