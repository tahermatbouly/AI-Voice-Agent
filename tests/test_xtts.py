from pathlib import Path
import os
import time
import warnings

import torch
import sounddevice as sd
import soundfile as sf


# ============================================================
# CPU CONFIGURATION
# ============================================================

CPU_COUNT = os.cpu_count() or 4

# For CPU inference, using every logical CPU thread is not
# always optimal. Allow manual override:
#
# XTTS_THREADS=8 python tests/test_xtts.py
#
CPU_THREADS = int(
    os.environ.get(
        "XTTS_THREADS",
        CPU_COUNT,
    )
)

CPU_THREADS = max(1, CPU_THREADS)


# PyTorch CPU optimizations
torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(
        max(1, min(4, CPU_THREADS))
    )
except RuntimeError:
    # Can happen if another part of PyTorch has already
    # initialized the thread pool.
    pass

try:
    torch.backends.mkldnn.enabled = True
except Exception:
    pass


# ============================================================
# PYTORCH / XTTS COMPATIBILITY
# ============================================================

_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault(
        "weights_only",
        False,
    )
    return _original_torch_load(
        *args,
        **kwargs,
    )


torch.load = _torch_load_compat


# ============================================================
# IMPORT XTTS
# ============================================================

from TTS.api import TTS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "tts_models/multilingual/multi-dataset/xtts_v2"
)

REFERENCE_DIR = Path(
    "data/masri30/wavs"
)

OUTPUT_DIR = Path(
    "output/tests"
)

OUTPUT_AUDIO = (
    OUTPUT_DIR /
    "xtts_optimized.wav"
)

LANGUAGE = "ar"


# ============================================================
# REFERENCE CONFIGURATION
# ============================================================

# Default: ONE reference.
#
# Override:
#
# XTTS_REFERENCES=3
# XTTS_REFERENCES=5
#
NUM_REFERENCES = int(
    os.environ.get(
        "XTTS_REFERENCES",
        "1",
    )
)


# ============================================================
# TEXT
# ============================================================

TEXT = (
    "أهلاً بيك، معاك المساعد الصوتي. "
    "ممكن أعرف اسم حضرتك؟"
)


# ============================================================
# REFERENCE DISCOVERY
# ============================================================

def get_reference_files():

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Reference directory does not exist:\n"
            f"{REFERENCE_DIR}"
        )

    files = list(
        REFERENCE_DIR.glob("*.wav")
    )

    if not files:
        raise FileNotFoundError(
            f"No WAV files found in:\n"
            f"{REFERENCE_DIR}"
        )

    # --------------------------------------------------------
    # Sort by file size.
    #
    # Smaller WAVs generally mean shorter clips, which makes
    # speaker conditioning considerably cheaper.
    # --------------------------------------------------------

    files.sort(
        key=lambda p: p.stat().st_size
    )

    selected = files[:NUM_REFERENCES]

    return selected


# ============================================================
# PRINT AUDIO INFORMATION
# ============================================================

def inspect_reference_files(files):

    print()
    print("[REFERENCE AUDIO]")

    total_duration = 0.0

    for index, path in enumerate(
        files,
        start=1,
    ):

        try:

            info = sf.info(path)

            duration = info.frames / info.samplerate

            total_duration += duration

            print(
                f"{index:02d}. "
                f"{path.name}"
                f" | {duration:.2f}s"
                f" | {info.samplerate} Hz"
                f" | {info.channels} ch"
            )

        except Exception as exc:

            print(
                f"{index:02d}. "
                f"{path.name}"
                f" | ERROR: {exc}"
            )

    print()
    print(
        f"Total reference duration: "
        f"{total_duration:.2f}s"
    )

    return total_duration


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING XTTS-v2")
    print("=" * 70)

    start = time.perf_counter()

    tts = TTS(
        model_name=MODEL_NAME,
        progress_bar=True,
        gpu=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"[OK] Model loaded in "
        f"{elapsed:.2f}s"
    )

    return tts, elapsed


# ============================================================
# TTS GENERATION
# ============================================================

def generate(
    tts,
    reference_audio,
    output_path,
):

    start = time.perf_counter()

    # --------------------------------------------------------
    # Important:
    #
    # Coqui's TTS API performs the actual model inference
    # internally. We cannot wrap its internal operations
    # completely with inference_mode(), but this keeps any
    # surrounding PyTorch work from tracking gradients.
    # --------------------------------------------------------

    with torch.inference_mode():

        tts.tts_to_file(
            text=TEXT,
            speaker_wav=reference_audio,
            language=LANGUAGE,
            file_path=str(output_path),
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    return elapsed


# ============================================================
# AUDIO METRICS
# ============================================================

def get_audio_duration(path):

    info = sf.info(path)

    return (
        info.frames /
        info.samplerate
    )


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    print()
    print("=" * 70)
    print("XTTS-v2 MAXIMUM CPU OPTIMIZATION TEST")
    print("=" * 70)

    print()
    print("[SYSTEM]")
    print(
        f"Detected CPU threads : "
        f"{CPU_COUNT}"
    )

    print(
        f"PyTorch CPU threads  : "
        f"{CPU_THREADS}"
    )

    print(
        f"Reference count      : "
        f"{NUM_REFERENCES}"
    )

    print()
    print("[TEXT]")
    print(TEXT)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTING REFERENCE AUDIO")
    print("=" * 70)

    reference_files = (
        get_reference_files()
    )

    reference_audio = [
        str(
            path.resolve()
        )
        for path in reference_files
    ]

    total_reference_duration = (
        inspect_reference_files(
            reference_files
        )
    )

    print()
    print(
        f"[OK] Selected "
        f"{len(reference_audio)} "
        f"reference WAV file(s)."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tts, load_time = load_model()

    # --------------------------------------------------------
    # FIRST GENERATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST GENERATION")
    print("=" * 70)

    print()
    print(
        "The first generation is NOT a clean "
        "real-time benchmark."
    )

    first_output = (
        OUTPUT_DIR /
        "xtts_first.wav"
    )

    first_time = generate(
        tts,
        reference_audio,
        first_output,
    )

    first_duration = (
        get_audio_duration(
            first_output
        )
    )

    first_rtf = (
        first_time /
        first_duration
    )

    print()
    print(
        f"Generation time : "
        f"{first_time:.2f}s"
    )

    print(
        f"Audio duration  : "
        f"{first_duration:.2f}s"
    )

    print(
        f"RTF             : "
        f"{first_rtf:.2f}"
    )

    # --------------------------------------------------------
    # WARM GENERATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WARM GENERATION BENCHMARK")
    print("=" * 70)

    print()
    print(
        "Running the same model again to "
        "measure warm inference."
    )

    warm_output = (
        OUTPUT_DIR /
        "xtts_warm.wav"
    )

    warm_start = time.perf_counter()

    warm_time = generate(
        tts,
        reference_audio,
        warm_output,
    )

    warm_total = (
        time.perf_counter()
        - warm_start
    )

    warm_duration = (
        get_audio_duration(
            warm_output
        )
    )

    warm_rtf = (
        warm_time /
        warm_duration
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)

    print()
    print(
        f"Reference files       : "
        f"{len(reference_audio)}"
    )

    print(
        f"Reference duration    : "
        f"{total_reference_duration:.2f}s"
    )

    print(
        f"Model load            : "
        f"{load_time:.2f}s"
    )

    print()
    print(
        f"First generation      : "
        f"{first_time:.2f}s"
    )

    print(
        f"First audio duration  : "
        f"{first_duration:.2f}s"
    )

    print(
        f"First RTF             : "
        f"{first_rtf:.2f}"
    )

    print()
    print(
        f"Warm generation       : "
        f"{warm_time:.2f}s"
    )

    print(
        f"Warm audio duration   : "
        f"{warm_duration:.2f}s"
    )

    print(
        f"Warm RTF              : "
        f"{warm_rtf:.2f}"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if warm_rtf < 1:

        print(
            "🟢 REAL-TIME"
        )

        print(
            "XTTS generates speech faster "
            "than it is played."
        )

    elif warm_rtf < 2:

        print(
            "🟡 NEAR REAL-TIME"
        )

        print(
            "Potentially usable with "
            "short response chunks."
        )

    elif warm_rtf < 5:

        print(
            "🟠 SLOW"
        )

        print(
            "Not ideal for real-time "
            "conversation."
        )

    else:

        print(
            "🔴 VERY SLOW"
        )

        print(
            "CPU XTTS is currently "
            "not suitable for direct "
            "real-time conversation."
        )

    # --------------------------------------------------------
    # Playback warm result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PLAYBACK")
    print("=" * 70)

    audio, sample_rate = sf.read(
        warm_output
    )

    print()
    print(
        "Playing warm inference result..."
    )

    sd.play(
        audio,
        sample_rate,
    )

    sd.wait()

    print()
    print(
        f"Output: {warm_output}"
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()