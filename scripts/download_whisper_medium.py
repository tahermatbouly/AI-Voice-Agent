from faster_whisper import WhisperModel

print("=" * 60)
print("Downloading Faster-Whisper Medium")
print("=" * 60)

print()
print("Model: medium")
print("Device: CPU")
print("Compute type: int8")
print()

print("Loading model...")
print("This downloads the model if it is not already cached.")
print()

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
)

print()
print("=" * 60)
print("Faster-Whisper medium is ready.")
print("=" * 60)
