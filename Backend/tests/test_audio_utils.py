import torch
import torchaudio

from Backend.audio.audio_utils import (
    float32_to_pcm16,
    resample_audio,
)


INPUT_FILE = "test_egyptian_recruitment.wav"


waveform, sample_rate = torchaudio.load(INPUT_FILE)

print(f"Original sample rate: {sample_rate}")
print(f"Original shape: {waveform.shape}")

# Make sure audio is mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Remove channel dimension
audio = waveform.squeeze(0)

# Resample to 16 kHz
audio = resample_audio(
    audio,
    source_rate=sample_rate,
    target_rate=16000,
)

print(f"New sample rate: 16000")
print(f"New samples: {audio.numel()}")

# Convert back to PCM16
pcm_audio = float32_to_pcm16(audio)

print(f"PCM16 bytes: {len(pcm_audio)}")
print(f"Duration: {audio.numel() / 16000:.2f} seconds")