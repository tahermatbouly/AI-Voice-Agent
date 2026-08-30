"""
Cohere Transcribe Arabic STT plugin for LiveKit Agents 1.6.10.

Pipeline:

    LiveKit AudioFrame
          ↓
    AudioBuffer
          ↓
    PCM int16
          ↓
    mono float32
          ↓
    16 kHz
          ↓
    temporary WAV
          ↓
    Cohere Transcribe Arabic
          ↓
    Arabic transcript
          ↓
    LiveKit SpeechEvent

Cloud STT:
    Cohere Transcribe Arabic

Model:
    cohere-transcribe-arabic-07-2026
"""

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

from livekit.agents import stt
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer

from app import config


logger = logging.getLogger("cohere_stt")


class CohereArabicSTT(stt.STT):

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "cohere-transcribe-arabic-07-2026",
        language: str = "ar",
        sample_rate: int = 16000,
    ) -> None:

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

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

    # ========================================================
    # METADATA
    # ========================================================

    @property
    def label(self) -> str:
        return "cohere-transcribe"

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "cohere"

    # ========================================================
    # AUDIO BUFFER
    # ========================================================

    @staticmethod
    def _audio_buffer_to_numpy(
        buffer: AudioBuffer,
    ) -> tuple[np.ndarray, int]:

        logger.info(
            "[STT] received AudioBuffer type=%s",
            type(buffer).__name__,
        )

        if isinstance(buffer, list):
            frames = buffer
        else:
            frames = [buffer]

        if not frames:
            raise ValueError(
                "[STT] AudioBuffer contains no AudioFrames."
            )

        first_frame = frames[0]

        sample_rate = first_frame.sample_rate
        num_channels = first_frame.num_channels

        logger.info(
            "[STT] input audio: "
            "sample_rate=%s channels=%s frames=%s",
            sample_rate,
            num_channels,
            len(frames),
        )

        chunks: list[np.ndarray] = []

        for frame in frames:

            if frame.sample_rate != sample_rate:
                raise ValueError(
                    "[STT] AudioFrames have different "
                    "sample rates."
                )

            data = np.frombuffer(
                frame.data,
                dtype=np.int16,
            )

            if data.size == 0:
                continue

            # ------------------------------------------------
            # Convert stereo/multichannel → mono
            # ------------------------------------------------

            if num_channels > 1:

                if data.size % num_channels != 0:
                    raise ValueError(
                        "[STT] PCM sample count is not "
                        "divisible by channel count."
                    )

                data = data.reshape(
                    -1,
                    num_channels,
                )

                data = data.astype(
                    np.float32
                ).mean(axis=1)

            else:

                data = data.astype(
                    np.float32
                )

            chunks.append(data)

        if not chunks:
            raise ValueError(
                "[STT] AudioBuffer contained no samples."
            )

        audio = np.concatenate(chunks)

        # int16 → float32
        audio = audio / 32768.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        duration = len(audio) / sample_rate

        logger.info(
            "[STT] extracted %.3fs @ %d Hz",
            duration,
            sample_rate,
        )

        return audio, sample_rate

    # ========================================================
    # RESAMPLING
    # ========================================================

    @staticmethod
    def _resample_to_16khz(
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:

        if sample_rate == 16000:
            return audio

        logger.info(
            "[STT] resampling %d Hz -> 16000 Hz",
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

        # float32 [-1, 1] → int16
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
    # COHERE TRANSCRIPTION
    # ========================================================

    async def _transcribe_with_cohere(
        self,
        wav_path: str,
    ) -> str:

        logger.info(
            "[STT] Sending audio to Cohere..."
        )

        logger.info(
            "[STT] model=%s language=%s file=%s",
            self._model,
            self._language,
            wav_path,
        )

        def transcribe():

            with open(
                wav_path,
                "rb",
            ) as audio_file:

                response = self._client.audio.transcriptions.create(
                    model=self._model,
                    language=self._language,
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
            "[STT] Cohere transcript: %s",
            text if text else "<EMPTY>",
        )

        return text

    # ========================================================
    # RECOGNITION
    # ========================================================

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:

        logger.info(
            "[STT] ========================================"
        )

        logger.info(
            "[STT] Cohere recognize() called"
        )

        # ----------------------------------------------------
        # Extract LiveKit audio
        # ----------------------------------------------------

        audio, sample_rate = await asyncio.to_thread(
            self._audio_buffer_to_numpy,
            buffer,
        )

        # ----------------------------------------------------
        # Resample to 16 kHz
        # ----------------------------------------------------

        audio = await asyncio.to_thread(
            self._resample_to_16khz,
            audio,
            sample_rate,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        duration = len(audio) / 16000.0

        logger.info(
            "[STT] final audio: %.3fs @ 16000 Hz",
            duration,
        )

        # ----------------------------------------------------
        # Empty audio
        # ----------------------------------------------------

        if len(audio) == 0:

            logger.warning(
                "[STT] empty audio received"
            )

            text = ""

        else:

            # ------------------------------------------------
            # Create temporary WAV
            # ------------------------------------------------

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
                    "[STT] WAV created: "
                    "%s bytes",
                    file_size,
                )

                # ------------------------------------------------
                # Cohere
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
                        "[STT] Failed to remove "
                        "temporary WAV: %s",
                        e,
                    )

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        language_code = self._language

        if language is not NOT_GIVEN and language:
            language_code = str(language)

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if text:

            logger.info(
                "[STT] TRANSCRIPT: %s",
                text,
            )

        else:

            logger.warning(
                "[STT] TRANSCRIPT: <EMPTY>"
            )

        logger.info(
            "[STT] ========================================"
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=language_code,
                    text=text,
                    confidence=1.0,
                )
            ],
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    async def aclose(self) -> None:

        logger.info(
            "[STT] closing Cohere STT..."
        )

        self._client = None

        logger.info(
            "[STT] Cohere STT closed."
        )