import torchaudio
import torch
from Backend.audio.audio_utils import resample_audio
from Backend.audio.vad import VoiceActivityDetector
import time

INPUT_FILE = "test_egyptian_recruitment.wav"


# -------------------------
# Load audio
# -------------------------

waveform, sample_rate = torchaudio.load(INPUT_FILE)

print(f"Original sample rate: {sample_rate}")

# Mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio = waveform.squeeze(0)


# -------------------------
# Resample
# -------------------------

audio = resample_audio(
    audio,
    source_rate=sample_rate,
    target_rate=16000,
)

# Add 1 second of silence so VAD can detect the final speech_end.
silence = torch.zeros(16000)

audio = torch.cat(
    [audio, silence]
)
print(
    f"Resampled: "
    f"{audio.numel()} samples @ 16000 Hz"
)


# -------------------------
# Convert to PCM16
# -------------------------

pcm_audio = (
    audio.clamp(-1.0, 1.0)
    * 32767
).to(
    dtype=__import__("torch").int16
).numpy().tobytes()


# -------------------------
# VAD
# -------------------------

vad = VoiceActivityDetector()

chunk_size = 3200

start_time = time.monotonic()

for i in range(0, len(pcm_audio), chunk_size):

    chunk = pcm_audio[i:i + chunk_size]

    event = vad.process(chunk)
    if event:
        elapsed = time.monotonic() - start_time
    
        for vad_event in event:
            print(
                f"[VAD] {vad_event} "
                f"| chunk={i // chunk_size + 1} "
                f"| audio_time={(i / 2) / 16000:.2f}s "
                f"| elapsed={elapsed:.2f}s"
            )