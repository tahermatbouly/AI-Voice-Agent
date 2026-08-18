from faster_whisper import WhisperModel

MODEL_SIZE = "small"

print("Loading Whisper model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully.")

audio_file = "hello_in_arabic.mp3"

segments, info = model.transcribe(
    audio_file,
    language="ar",
    beam_size=5
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")

print("\nTranscription:")
print("-" * 50)

for segment in segments:
    print(segment.text)

print("-" * 50)