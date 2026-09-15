import wave
from pathlib import Path

from Backend.audio.background_mixer import BackgroundMixer


BASE_DIR = Path("Backend/data/sounds/processed")

AMBIENCE_PATH = BASE_DIR / "call_center.wav"
RING_PATH = BASE_DIR / "phone_ring.wav"

OUTPUT_PATH = Path("Backend/data/sounds/test_mixed_background.wav")


def main():
    print("Loading background mixer...")

    mixer = BackgroundMixer(
        ambience_path=str(AMBIENCE_PATH),
        ring_path=str(RING_PATH),
        ambience_volume=0.750,
        ring_volume=0.750,
        ring_min_interval=2.0,
        ring_max_interval=4.0,
    )

    print("Mixer loaded successfully.")

    # Create 10 seconds of silence.
    silence = b"\x00\x00" * (24_000 * 10)

    print("Mixing 10 seconds of silence with background...")

    mixed = mixer.mix(silence)

    print(f"Input bytes : {len(silence)}")
    print(f"Output bytes: {len(mixed)}")

    with wave.open(str(OUTPUT_PATH), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(mixed)

    print()
    print(f"Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()