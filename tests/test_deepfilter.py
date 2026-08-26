import time
from pathlib import Path

import torch
from df.enhance import init_df, enhance, load_audio, save_audio


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = Path("samples/egyptian_test.wav")
OUTPUT_FILE = Path("samples/egyptian_test_denoised.wav")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("DeepFilterNet CPU Noise Reduction Test")
    print("=" * 60)

    # Make sure the input exists
    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:")
        print(f"  {INPUT_FILE}")
        print("\nPut your noisy WAV file at:")
        print("  samples/egyptian_test.wav")
        return

    # Check PyTorch
    print(f"\nPyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("WARNING: CUDA is available, but this test is intended for CPU.")

    print("\nLoading DeepFilterNet model...")

    start = time.perf_counter()

    # Load pretrained DeepFilterNet model
    model, df_state, _ = init_df()

    load_time = time.perf_counter() - start

    print(f"Model loaded in : {load_time:.2f} seconds")
    print(f"Sample rate     : {df_state.sr()} Hz")

    # -----------------------------------------------------
    # Load audio
    # -----------------------------------------------------

    print(f"\nLoading audio:")
    print(f"  {INPUT_FILE}")

    audio, _ = load_audio(
        str(INPUT_FILE),
        sr=df_state.sr(),
    )

    print(f"Audio shape     : {audio.shape}")

    # -----------------------------------------------------
    # Noise reduction
    # -----------------------------------------------------

    print("\nRunning DeepFilterNet...")
    print("This may take a little while on CPU.\n")

    start = time.perf_counter()

    enhanced = enhance(
        model,
        df_state,
        audio,
    )

    processing_time = time.perf_counter() - start

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_audio(
        str(OUTPUT_FILE),
        enhanced,
        df_state.sr(),
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    duration_seconds = audio.shape[-1] / df_state.sr()

    realtime_factor = processing_time / duration_seconds

    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"Input duration  : {duration_seconds:.2f} seconds")
    print(f"Processing time  : {processing_time:.2f} seconds")
    print(f"Real-time factor : {realtime_factor:.2f}x")
    print(f"Output file      : {OUTPUT_FILE}")

    if realtime_factor < 1:
        print("\n✓ Fast enough for real-time processing.")
    else:
        print("\n⚠ Slower than real-time on this CPU.")

    print("\nYou can now compare:")
    print(f"  Original : {INPUT_FILE}")
    print(f"  Cleaned  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()