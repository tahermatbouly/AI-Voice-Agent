# understands the diffrent audio types and formates 

import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def pcm16_to_float32(audio_bytes: bytes) -> torch.Tensor:
    """
    Convert PCM16 bytes to float32 audio in [-1, 1].
    """

    audio = torch.frombuffer(
        bytearray(audio_bytes),
        dtype=torch.int16,
    ).float()

    return audio / 32768.0


def float32_to_pcm16(audio: torch.Tensor) -> bytes:
    """
    Convert float32 audio in [-1, 1] to PCM16 bytes.
    """

    audio = torch.clamp(audio, -1.0, 1.0)

    audio = (audio * 32767.0).to(torch.int16)

    return audio.cpu().numpy().tobytes()


def resample_audio(
    audio: torch.Tensor,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Resample mono float32 audio.
    """

    if source_rate == target_rate:
        return audio

    return torchaudio.functional.resample(
        audio,
        source_rate,
        target_rate,
    )