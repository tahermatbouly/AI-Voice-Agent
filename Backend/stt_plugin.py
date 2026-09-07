
from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from pathlib import Path

import cohere
import librosa
import numpy as np

from Backend import config


logger = logging.getLogger("cohere_stt")


class CohereArabicSTT:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "cohere-transcribe-arabic-07-2026",
        language: str = "ar",
    ):
        self.api_key = api_key or config.COHERE_API_KEY
        self.model = model
        self.language = language

        if not self.api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set."
            )

        self.client = cohere.ClientV2(
            api_key=self.api_key
        )

        logger.info(
            "[STT] Cohere configured "
            "model=%s language=%s",
            self.model,
            self.language,
        )

    # ========================================================
    # PCM16 → FLOAT32
    # ========================================================

    @staticmethod
    def _pcm16_to_float32(
        audio_bytes: bytes,
    ) -> np.ndarray:

        if not audio_bytes:
            raise ValueError(
                "[STT] Empty audio received."
            )

        audio = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        )

        if audio.size == 0:
            raise ValueError(
                "[STT] Audio contains no samples."
            )

        audio = audio.astype(
            np.float32
        )

        audio /= 32768.0

        return audio

    # ========================================================
    # RESAMPLE
    # ========================================================

    @staticmethod
    def _resample_to_16khz(
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

    # ========================================================
    # WAV CREATION
    # ========================================================

    @staticmethod
    def _write_wav(
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

    # ========================================================
    # COHERE API
    # ========================================================

    async def _transcribe_with_cohere(
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

                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    language=self.language,
                    file=audio_file,
                )

            return response.text.strip()

        try:

            text = await asyncio.to_thread(
                transcribe
            )

        except Exception as e:

            logger.error(
                "[STT] Cohere transcription failed: "
                "%s: %s",
                type(e).__name__,
                e,
            )

            raise

        logger.info(
            "[STT] Transcript: %s",
            text if text else "<EMPTY>",
        )

        return text

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int,
    ) -> str:

        logger.info(
            "[STT] ========================================"
        )

        logger.info(
            "[STT] Transcription started"
        )

        # ----------------------------------------------------
        # PCM16 → float32
        # ----------------------------------------------------

        audio = await asyncio.to_thread(
            self._pcm16_to_float32,
            audio_bytes,
        )

        duration = len(audio) / sample_rate

        logger.info(
            "[STT] Input audio: %.3fs @ %d Hz",
            duration,
            sample_rate,
        )

        # ----------------------------------------------------
        # Resample → 16 kHz
        # ----------------------------------------------------

        audio = await asyncio.to_thread(
            self._resample_to_16khz,
            audio,
            sample_rate,
        )

        final_duration = len(audio) / 16000.0

        logger.info(
            "[STT] Final audio: %.3fs @ 16000 Hz",
            final_duration,
        )

        # ----------------------------------------------------
        # Create temporary WAV
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:

            wav_path = temp_file.name

        try:

            await asyncio.to_thread(
                self._write_wav,
                audio,
                wav_path,
                16000,
            )

            file_size = Path(
                wav_path
            ).stat().st_size

            logger.info(
                "[STT] WAV created: %d bytes",
                file_size,
            )

            # ------------------------------------------------
            # Send to Cohere
            # ------------------------------------------------

            text = await self._transcribe_with_cohere(
                wav_path
            )

        finally:

            try:
                Path(wav_path).unlink(
                    missing_ok=True
                )

            except Exception as e:

                logger.warning(
                    "[STT] Failed to remove temporary WAV: %s",
                    e,
                )

        logger.info(
            "[STT] TRANSCRIPT: %s",
            text if text else "<EMPTY>",
        )

        logger.info(
            "[STT] ========================================"
        )

        return text
