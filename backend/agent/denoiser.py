"""
DeepFilterNet3 CPU audio denoiser.

Loads the model once and reuses it for every utterance.
"""

import torch

from df.enhance import init_df, enhance


class DeepFilterDenoiser:
    """CPU-based noise suppression using DeepFilterNet3."""

    def __init__(self):
        print("[denoiser] loading DeepFilterNet3...")

        self.model, self.df_state, _ = init_df(
            log_level="INFO"
        )

        self.sample_rate = self.df_state.sr()

        print(
            f"[denoiser] DeepFilterNet3 loaded "
            f"(sample rate: {self.sample_rate} Hz)"
        )

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Denoise an audio waveform.

        Args:
            audio:
                Float32 torch tensor with shape [channels, samples].
                Sample rate must equal DeepFilterNet's sample rate.

        Returns:
            Denoised float32 tensor with the same shape.
        """

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # DeepFilterNet runs on CPU in our setup.
        audio = audio.to("cpu")

        enhanced = enhance(
            self.model,
            self.df_state,
            audio,
        )

        return enhanced