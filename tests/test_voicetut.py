import time

from voicetut_tts import VoiceTutTTS


MODEL = "mohammedaly22/VoiceTut-TTS"

TEXT = "أهلاً بيك، معاك قسم الموارد البشرية من GB corp. ممكن أعرف اسمك بالكامل؟"


print("=" * 60)
print("VoiceTut CPU Test")
print("=" * 60)

print("\n[1] Loading model...")

start = time.perf_counter()

tts = VoiceTutTTS.from_pretrained(MODEL)

load_time = time.perf_counter() - start

print(f"Model loading time: {load_time:.2f}s")


print("\n[2] Generating speech...")

start = time.perf_counter()

tts.synthesize(
    TEXT,
    speaker="Mohamed",
    num_step=32,
    output="tests/voicetut_test.wav",
)

generation_time = time.perf_counter() - start

print(f"Generation time: {generation_time:.2f}s")


print("\n[3] Done!")
print("Output: tests/voicetut_test.wav")
print(f"Generation time: {generation_time:.2f}s")