from voicetut_tts import VoiceTutTTS


def main():
    print("Loading VoiceTut-TTS...")

    tts = VoiceTutTTS.from_pretrained(
        "mohammedaly22/VoiceTut-TTS"
    )

    print("Model loaded.")

    text = "ازيك يا باشا، عامل ايه؟ النهارده الجو حلو اوي."

    print(f"Generating speech for:\n{text}")

    tts.synthesize(
        text,
        speaker="Mohamed",
        output="test_egyptian.wav"
    )

    print("Done!")
    print("Audio saved as: test_egyptian.wav")


if __name__ == "__main__":
    main()