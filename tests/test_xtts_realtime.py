from pathlib import Path
import time

import sounddevice as sd
import soundfile as sf
import torch


# ============================================================
# PyTorch / XTTS compatibility
# ============================================================
#
# PyTorch 2.6+ changed torch.load() behavior.
# XTTS-v2 / older Coqui checkpoints may require
# weights_only=False.
#
# Only use this with trusted model files.
# ============================================================

_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


from TTS.api import TTS


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Directory containing your Egyptian Arabic WAV files
REFERENCE_DIR = Path("data/masri30/wavs")

# Output
OUTPUT_AUDIO = Path(
    "output/tests/realtime_xtts.wav"
)

# XTTS language
LANGUAGE = "ar"


# ============================================================
# Test text
# ============================================================

TEXT = (
    "أهلاً بيك، معاك المساعد الصوتي. "
    "ممكن أعرف اسم حضرتك والعنوان والوظيفة؟ "
    "ولو عندك أي استفسار، أنا جاهز أساعدك."
)


# ============================================================
# Get all reference WAV files
# ============================================================

def get_reference_audio():
    """
    Find all WAV files in the Egyptian Arabic dataset directory.

    Returns:
        list[str]: Absolute paths to WAV files.
    """

    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(
            f"Reference directory not found:\n"
            f"{REFERENCE_DIR}"
        )

    reference_files = sorted(
        REFERENCE_DIR.glob("*.wav")
    )

    if not reference_files:
        raise FileNotFoundError(
            f"No WAV files found in:\n"
            f"{REFERENCE_DIR}"
        )

    return [
        str(path.resolve())
        for path in reference_files
    ]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("XTTS-v2 EGYPTIAN ARABIC REAL-TIME CPU TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Find reference WAV files
    # --------------------------------------------------------

    print()
    print("[INFO] Searching for reference audio...")

    reference_audio = get_reference_audio()

    print(
        f"[OK] Found {len(reference_audio)} reference WAV files."
    )

    print()
    print("[INFO] Reference directory:")
    print(f"       {REFERENCE_DIR}")

    print()
    print("[INFO] First 10 reference files:")

    for i, path in enumerate(reference_audio[:10], start=1):
        print(f"       {i:03d}: {Path(path).name}")

    if len(reference_audio) > 10:
        print(
            f"       ... and {len(reference_audio) - 10} more"
        )

    print()
    print(f"[INFO] Text:")
    print(f"       {TEXT}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_AUDIO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load XTTS
    # --------------------------------------------------------

    print()
    print("[1/3] Loading XTTS-v2...")
    print("[INFO] Running on CPU")

    load_start = time.perf_counter()

    tts = TTS(
        model_name=MODEL_NAME,
        progress_bar=True,
        gpu=False,
    )

    load_time = time.perf_counter() - load_start

    print()
    print(
        f"[OK] Model loaded in {load_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Generate speech
    # --------------------------------------------------------

    print()
    print("[2/3] Generating Egyptian Arabic speech...")

    print()
    print(
        f"[INFO] Using {len(reference_audio)} "
        f"WAV files as speaker references."
    )

    print(
        "[INFO] This may take longer because XTTS "
        "has to process the reference audio."
    )

    print()

    generation_start = time.perf_counter()

    tts.tts_to_file(
        text=TEXT,

        # IMPORTANT:
        # Pass the LIST directly.
        #
        # Do NOT use:
        # str(reference_audio)
        #
        # because that would turn the list into one string.
        speaker_wav=reference_audio,

        language=LANGUAGE,

        file_path=str(OUTPUT_AUDIO),
    )

    generation_time = (
        time.perf_counter() - generation_start
    )

    # --------------------------------------------------------
    # Read generated audio
    # --------------------------------------------------------

    audio, sample_rate = sf.read(
        OUTPUT_AUDIO
    )

    audio_duration = (
        len(audio) / sample_rate
    )

    # --------------------------------------------------------
    # Calculate RTF
    # --------------------------------------------------------

    rtf = generation_time / audio_duration

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("[3/3] RESULT")
    print("=" * 70)

    print(
        f"Reference WAV files : {len(reference_audio)}"
    )

    print(
        f"Model loading time  : {load_time:.2f}s"
    )

    print(
        f"TTS generation time : {generation_time:.2f}s"
    )

    print(
        f"Generated audio     : {audio_duration:.2f}s"
    )

    print(
        f"Real-Time Factor    : {rtf:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Interpret RTF
    # --------------------------------------------------------

    if rtf < 1:
        print(
            "Performance         : REAL-TIME CAPABLE"
        )
        print(
            "                    XTTS generated audio "
            "faster than playback."
        )

    elif rtf < 2:
        print(
            "Performance         : NEAR REAL-TIME"
        )
        print(
            "                    Suitable for experimentation."
        )

    elif rtf < 5:
        print(
            "Performance         : SLOW"
        )
        print(
            "                    Optimization will be needed "
            "for a real-time phone agent."
        )

    else:
        print(
            "Performance         : VERY SLOW"
        )
        print(
            "                    Not suitable for real-time "
            "conversation in the current configuration."
        )

    print()
    print(
        f"Output audio        : {OUTPUT_AUDIO}"
    )

    # --------------------------------------------------------
    # Play generated audio
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PLAYING GENERATED AUDIO")
    print("=" * 70)

    sd.play(
        audio,
        sample_rate
    )

    sd.wait()

    print()
    print("[OK] Playback finished.")

    print()
    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()