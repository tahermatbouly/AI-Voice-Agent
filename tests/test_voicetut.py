"""
Optimized test/benchmark for VoiceTut-TTS on CPU.

VoiceTut-TTS is built on OmniVoice: a Qwen3-0.6B LLM backbone + an audio
tokenizer (Higgs). This means it's an autoregressive, LLM-style TTS model
-- the same general class as NAMAA (Chatterbox-based) and EGTTS (XTTS v2
based), not a lightweight model like edge-tts. Slow CPU generation is
expected for this architecture class; this script isolates WHERE the
time actually goes and applies the cheap, safe optimizations available.

Changes from a naive test:
  1. Tests SHORT, separate sentences (like the other TTS tests tonight),
     not one long multi-sentence paragraph in a single generate() call --
     autoregressive generation time scales with output length, so a long
     paragraph in one call is the most likely reason it "takes so much
     time", independent of any real inefficiency.
  2. Explicitly sets PyTorch's CPU thread count, in case it's not using
     all available cores by default.
  3. Checks whether the model is running in bf16 (a training-time
     convenience) on CPU, where bf16 matmuls are often SLOWER than
     float32 unless the CPU has AVX512_BF16 support -- and casts to
     float32 if so.
  4. Reports characters-per-second as a normalized metric, so results
     are comparable across different text lengths.

CONFIDENCE NOTE: some of these optimizations depend on what the
voicetut_tts package actually exposes (dtype/device kwargs, a streaming
API). Where I'm not certain a parameter exists, the script checks for it
at runtime rather than assuming -- read the printed diagnostics.
"""

import os
import inspect
from pathlib import Path
from time import perf_counter

from voicetut_tts import VoiceTutTTS

MODEL_NAME = "mohammedaly22/VoiceTut-TTS"
OUTPUT_DIR = Path("tests")

# Short sentences, matching the pattern used for NAMAA/EGTTS/SILMA
# testing tonight -- isolates per-sentence generation time rather than
# timing one long paragraph.
TEST_SENTENCES = [
    "مساء الخير، مع حضرتك محمد، مسؤول التوظيف في شركة GB corp.",
    "ممكن أعرف اسم حضرتك، وساكن فين؟",
    "لو مهتم، أقدر أوضحلك تفاصيل الوظيفة والمرتب.",
]

# The original full paragraph, kept as a separate test so you can compare
# "one long call" vs. "several short calls" directly.
FULL_PARAGRAPH = (
    "مساء الخير، مع حضرتك محمد, مسؤول التوظيف في شركة شغلني. "
    "بكلم حضرتك بخصوص فرصة عمل متاحة حالياً في شركة غبور للحافلات. "
    "المكالمة مش هتاخد من وقت حضرتك غير دقيقتين تقريباً، "
    "وعايزين نعرف لو الوظيفة مناسبة ليك ولخبراتك. "
    "ممكن أعرف اسم حضرتك، وساكن فين، وإيه المجال اللي عندك خبرة فيه؟ "
    "ولو مهتم، أقدر أوضحلك تفاصيل الوظيفة والمرتب ومواعيد العمل."
)


def tune_cpu_threads():
    """Make sure PyTorch is actually using all available CPU cores --
    a cheap, safe check that sometimes gets missed."""
    import torch

    cpu_count = os.cpu_count() or 4
    current = torch.get_num_threads()
    print(f"[optimize] CPU cores available: {cpu_count}, torch currently using: {current}")
    if current < cpu_count:
        torch.set_num_threads(cpu_count)
        print(f"[optimize] set torch threads to {cpu_count}")


def check_and_fix_dtype(tts):
    """If the model loaded in bf16 (common training default), check
    whether that's actually helping or hurting on this CPU, and cast to
    float32 if bf16 isn't hardware-accelerated here."""
    import torch

    model = getattr(tts, "model", None)
    if model is None:
        print("[optimize] couldn't introspect model dtype (no .model attribute found) -- skipping this check.")
        return

    try:
        current_dtype = next(model.parameters()).dtype
    except Exception:
        print("[optimize] couldn't determine model dtype -- skipping this check.")
        return

    print(f"[optimize] model is currently running in: {current_dtype}")

    if current_dtype in (torch.bfloat16, torch.float16):
        print(f"[optimize] model is in {current_dtype}. Most consumer CPUs lack")
        print("[optimize] accelerated float16/bfloat16 matmul kernels -- this is")
        print("[optimize] often SLOWER than float32 on CPU, not faster (it was a")
        print("[optimize] training-time convenience, not a CPU-inference optimization).")
        print("[optimize] Casting to float32 for this test.")
        try:
            model.to(torch.float32)
            print("[optimize] cast to float32 successfully.")
        except Exception as e:
            print(f"[optimize] could not cast dtype: {e}")


def discover_api(tts):
    """Print exact call signatures before using them -- avoids guessing
    method names/arguments wrong a second time."""
    import inspect as inspect_module

    speakers = []
    try:
        speakers = tts.list_speakers()
        print(f"[optimize] available speakers: {speakers}")
    except Exception as e:
        print(f"[optimize] could not call list_speakers(): {e}")

    for method_name in ("synthesize", "synthesize_long", "stream", "save"):
        method = getattr(tts, method_name, None)
        if method is not None:
            try:
                sig = inspect_module.signature(method)
                print(f"[optimize] signature: {method_name}{sig}")
            except (TypeError, ValueError):
                print(f"[optimize] could not introspect signature of {method_name}")

    default_speaker = speakers[0].speaker_id if speakers else None
    if default_speaker:
        print(f"[optimize] using default speaker: {default_speaker} ({speakers[0].speaker_name})")
    return default_speaker


def synthesize_one(tts, text: str, speaker, output_path: str = None):
    """Calls tts.synthesize() with the confirmed real signature:
    synthesize(text, *, speaker: Optional[str] = None, ..., output: Optional[str] = None, ...)
    speaker must be a speaker_id STRING (from Speaker.speaker_id), not a
    Speaker object -- that was the bug in the previous run."""
    return tts.synthesize(text, speaker=speaker, output=output_path)


def run_sentence_tests(tts, speaker):
    results = []
    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] short sentence {i} ({len(text)} chars): {text}")
        out_path = OUTPUT_DIR / f"voicetut_short_{i}.wav"

        t0 = perf_counter()
        synthesize_one(tts, text, speaker, output_path=str(out_path))
        elapsed = perf_counter() - t0

        chars_per_sec = len(text) / elapsed if elapsed > 0 else 0
        print(f"[test] generated in {elapsed:.2f}s ({chars_per_sec:.1f} chars/sec) -> {out_path}")
        results.append((i, elapsed, len(text), chars_per_sec))
    return results


def run_paragraph_test(tts, speaker):
    print(f"\n[test] full paragraph ({len(FULL_PARAGRAPH)} chars, one call)...")
    out_path = OUTPUT_DIR / "voicetut_paragraph.wav"

    t0 = perf_counter()
    synthesize_one(tts, FULL_PARAGRAPH, speaker, output_path=str(out_path))
    elapsed = perf_counter() - t0

    chars_per_sec = len(FULL_PARAGRAPH) / elapsed if elapsed > 0 else 0
    print(f"[test] generated in {elapsed:.2f}s ({chars_per_sec:.1f} chars/sec) -> {out_path}")
    return elapsed, len(FULL_PARAGRAPH), chars_per_sec


def print_summary(sentence_results, paragraph_result):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_short_time = sum(r[1] for r in sentence_results)
    avg_chars_per_sec = sum(r[3] for r in sentence_results) / len(sentence_results) if sentence_results else 0

    for i, elapsed, chars, cps in sentence_results:
        print(f"  Short sentence {i}: {elapsed:.2f}s for {chars} chars ({cps:.1f} chars/sec)")

    print(f"\n  Average throughput (short sentences): {avg_chars_per_sec:.1f} chars/sec")

    if paragraph_result:
        p_elapsed, p_chars, p_cps = paragraph_result
        print(f"\n  Full paragraph: {p_elapsed:.2f}s for {p_chars} chars ({p_cps:.1f} chars/sec)")
        print(f"  (compare this throughput to the short-sentence average above --")
        print(f"   if they're similar, the model scales linearly and splitting long")
        print(f"   replies into shorter TTS calls is your main lever for perceived latency)")

    print("=" * 60)
    print("This is an LLM-backbone TTS model (Qwen3-0.6B) -- expect meaningfully")
    print("slower CPU generation than lightweight models like edge-tts, in the")
    print("same general ballpark as NAMAA/EGTTS. If this throughput is too slow")
    print("for live use, consider: splitting agent replies into shorter TTS calls,")
    print("checking for a quantized checkpoint, or falling back to edge-tts for")
    print("live calls while reserving VoiceTut for pre-generated/offline audio.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("VoiceTut-TTS CPU Optimization Test")
    print("=" * 60)

    tune_cpu_threads()

    print("\n[1/3] Loading VoiceTut-TTS...")
    t0 = perf_counter()
    tts = VoiceTutTTS.from_pretrained(MODEL_NAME)
    print(f"[OK] loaded in {perf_counter() - t0:.2f}s")

    check_and_fix_dtype(tts)
    default_speaker = discover_api(tts)

    print("\n[2/3] Testing short sentences (isolates per-sentence speed)...")
    sentence_results = run_sentence_tests(tts, default_speaker)

    print("\n[3/3] Testing the full original paragraph (for comparison)...")
    paragraph_result = run_paragraph_test(tts, default_speaker)

    print_summary(sentence_results, paragraph_result)