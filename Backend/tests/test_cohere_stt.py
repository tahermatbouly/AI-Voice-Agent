
import asyncio
import wave

from Backend.stt_plugin import CohereArabicSTT


AUDIO_FILE = "egyptian_reference.wav"


async def main():

    # --------------------------------------------------------
    # Load WAV
    # --------------------------------------------------------

    with wave.open(AUDIO_FILE, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        audio = wav.readframes(
            wav.getnframes()
        )

    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {channels}")
    print(f"Sample width: {sample_width}")
    print(f"Audio bytes: {len(audio)}")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if channels != 1:
        raise ValueError(
            "Test audio must be mono."
        )

    if sample_width != 2:
        raise ValueError(
            "Test audio must be PCM16."
        )

    # --------------------------------------------------------
    # Create STT
    # --------------------------------------------------------

    stt = CohereArabicSTT()

    # --------------------------------------------------------
    # Transcribe
    # --------------------------------------------------------

    text = await stt.transcribe(
        audio_bytes=audio,
        sample_rate=sample_rate,
    )

    print()
    print("=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
