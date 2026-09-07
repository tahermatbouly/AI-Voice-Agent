import asyncio
import wave

from Backend.tts_plugin import VoiceTutTTS


async def main():
    tts = VoiceTutTTS()

    text = "ممكن أعرف اسم حضرتك بالكامل,لو سمحت؟"

    audio, sample_rate = await tts.synthesize(text)

    output_file = "data/test_voice-tut_tts.wav"

    with wave.open(output_file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)

    print()
    print("========== TTS TEST ==========")
    print("Audio bytes:", len(audio))
    print("Sample rate:", sample_rate)
    print("Saved to:", output_file)
    print("================================")


if __name__ == "__main__":
    asyncio.run(main())