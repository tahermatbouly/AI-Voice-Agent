from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from pathlib import Path
from typing import Optional

import cohere
import librosa
import numpy as np

from app import config


logger = logging.getLogger("cohere_stt")


class CohereArabicSTT:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "cohere-transcribe-arabic-07-2026",
        language: str = "ar",
        sample_rate: int = 16000,
    ) -> None:

        self._api_key = api_key or config.COHERE_API_KEY
        self._model = model
        self._language = language
        self._sample_rate = sample_rate

        if not self._api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set."
            )

        self._client = cohere.ClientV2(
            api_key=self._api_key
        )

        logger.info(
            "[STT] Cohere configured "
            "model=%s language=%s sample_rate=%s",
            self._model,
            self._language,
            self._sample_rate,
        )

    # ============================================================
    # AUDIO CONVERSION
    # ============================================================

    @staticmethod
    def pcm_to_numpy(
        pcm: bytes,
        sample_rate: int,
        num_channels: int = 1,
    ) -> tuple[np.ndarray, int]:

        if not pcm:
            raise ValueError(
                "[STT] Empty PCM audio."
            )

        audio = np.frombuffer(
            pcm,
            dtype=np.int16,
        )

        if audio.size == 0:
            raise ValueError(
                "[STT] PCM contains no samples."
            )

        # --------------------------------------------------------
        # Stereo / multichannel → mono
        # --------------------------------------------------------

        if num_channels > 1:

            if audio.size % num_channels != 0:
                raise ValueError(
                    "[STT] PCM sample count is not "
                    "divisible by channel count."
                )

            audio = audio.reshape(
                -1,
                num_channels,
            )

            audio = audio.astype(
                np.float32
            ).mean(axis=1)

        else:

            audio = audio.astype(
                np.float32
            )

        # --------------------------------------------------------
        # int16 → float32
        # --------------------------------------------------------

        audio /= 32768.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        duration = len(audio) / sample_rate

        logger.info(
            "[STT] Received %.3fs @ %d Hz",
            duration,
            sample_rate,
        )

        return audio, sample_rate

    # ============================================================
    # RESAMPLING
    # ============================================================

    @staticmethod
    def resample_to_16khz(
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:

        if sample_rate == 16000:
            return audio

        logger.info(
            "[STT] Resampling %d Hz -> 16000 Hz",
            sample_rate,
        )

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )

        return np.asarray(
            audio,
            dtype=np.float32,
        )

    # ============================================================
    # WAV
    # ============================================================

    @staticmethod
    def write_wav(
        audio: np.ndarray,
        path: str,
        sample_rate: int = 16000,
    ) -> None:

        audio_int16 = np.clip(
            audio * 32768.0,
            -32768,
            32767,
        ).astype(np.int16)

        with wave.open(path, "wb") as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            wav.writeframes(
                audio_int16.tobytes()
            )

    # ============================================================
    # COHERE
    # ============================================================

    async def _transcribe_wav(
        self,
        wav_path: str,
    ) -> str:

        logger.info(
            "[STT] Sending audio to Cohere..."
        )

        def transcribe():

            with open(
                wav_path,
                "rb",
            ) as audio_file:

                response = (
                    self._client
                    .audio
                    .transcriptions
                    .create(
                        model=self._model,
                        language=self._language,
                        file=audio_file,
                    )
                )

            return response.text.strip()

        text = await asyncio.to_thread(
            transcribe
        )

        logger.info(
            "[STT] Transcript: %s",
            text if text else "<EMPTY>",
        )

        return text

    # ============================================================
    # MAIN TRANSCRIBE METHOD
    # ============================================================

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
        num_channels: int = 1,
    ) -> str:

        logger.info(
            "[STT] ========================================"
        )

        logger.info(
            "[STT] transcribe()"
        )

        # --------------------------------------------------------
        # PCM → numpy
        # --------------------------------------------------------

        audio, input_sample_rate = await asyncio.to_thread(
            self.pcm_to_numpy,
            pcm,
            sample_rate,
            num_channels,
        )

        # --------------------------------------------------------
        # Resample
        # --------------------------------------------------------

        audio = await asyncio.to_thread(
            self.resample_to_16khz,
            audio,
            input_sample_rate,
        )

        duration = len(audio) / 16000.0

        logger.info(
            "[STT] Final audio: %.3fs @ 16000 Hz",
            duration,
        )

        if len(audio) == 0:
            return ""

        # --------------------------------------------------------
        # Temporary WAV
        # --------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:

            wav_path = temp_file.name

        try:

            await asyncio.to_thread(
                self.write_wav,
                audio,
                wav_path,
                16000,
            )

            logger.info(
                "[STT] WAV created: %s",
                wav_path,
            )

            text = await self._transcribe_wav(
                wav_path
            )

            return text

        finally:

            try:
                Path(wav_path).unlink(
                    missing_ok=True
                )

            except Exception as e:

                logger.warning(
                    "[STT] Failed to remove "
                    "temporary WAV: %s",
                    e,
                )

    # ============================================================
    # FILE TRANSCRIPTION
    # ============================================================

    async def transcribe_file(
        self,
        wav_path: str,
    ) -> str:

        wav_path = str(
            Path(wav_path).resolve()
        )

        if not Path(wav_path).exists():
            raise FileNotFoundError(
                wav_path
            )

        return await self._transcribe_wav(
            wav_path
        )

    # ============================================================
    # CLEANUP
    # ============================================================

    async def aclose(self) -> None:

        logger.info(
            "[STT] Closing Cohere STT..."
        )

        self._client = None

        logger.info(
            "[STT] Cohere STT closed."
        )