import time
import torch
import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice


MODEL_ID = "ehabnegm/lahgtna-omnivoice-egyptian-v3"

TEXT = "أهلاً بحضرتك، معاك خدمة العملاء، ممكن أساعدك إزاي؟"


print("=" * 60)
print("Lahgtna Egyptian TTS Test")
print("=" * 60)

print("\nLoading model...")

start = time.perf_counter()

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    dtype=torch.float32,
)

load_time = time.perf_counter() - start

print(f"Model loaded in: {load_time:.2f} seconds")

print("\nGenerating speech...")
print(f"Text: {TEXT}")

start = time.perf_counter()

audio = model.generate(
    text=TEXT,
    language="arz",
)

generation_time = time.perf_counter() - start

print(f"Generation time: {generation_time:.2f} seconds")

output_file = "lahgtna_test.wav"

sf.write(
    output_file,
    audio[0],
    24000,
)

print(f"\nAudio saved to: {output_file}")

print("=" * 60)
print("DONE")
print("=" * 60)