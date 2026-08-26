"""
Local Faster-Whisper STT plugin for LiveKit Agents 1.6.10.

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
    Faster-Whisper
          ↓
    Arabic transcript

CPU-only.
No DeepFilterNet.
No cloud STT.
No cloud turn detector.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from livekit.agents import stt
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer


logger = logging.getLogger("stt_plugin")


class FasterWhisperSTT(stt.STT):

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ar",
        beam_size: int = 1,
    ) -> None:

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size

        self._model: Optional[WhisperModel] = None
        self._load_lock = asyncio.Lock()

        logger.info(
            "[STT] configured Faster-Whisper "
            "model=%s device=%s compute_type=%s language=%s",
            model_size,
            device,
            compute_type,
            language,
        )

    # ========================================================
    # METADATA
    # ========================================================

    @property
    def label(self) -> str:
        return "faster-whisper"

    @property
    def model(self) -> str:
        return f"faster-whisper-{self._model_size}"

    @property
    def provider(self) -> str:
        return "local"

    # ========================================================
    # MODEL
    # ========================================================

    async def _ensure_model(self) -> None:

        if self._model is not None:
            return

        async with self._load_lock:

            if self._model is not None:
                return

            logger.info(
                "[STT] loading Faster-Whisper '%s' on %s...",
                self._model_size,
                self._device,
            )

            self._model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

            logger.info("[STT] Faster-Whisper model loaded.")

    # ========================================================
    # AUDIO BUFFER
    # ========================================================

    @staticmethod
    def _audio_buffer_to_numpy(
        buffer: AudioBuffer,
    ) -> tuple[np.ndarray, int]:
        """
        Convert LiveKit AudioBuffer into:

            mono float32 NumPy array
            original sample rate

        AudioBuffer in LiveKit Agents 1.6.10 is:

            AudioFrame | list[AudioFrame]
        """

        logger.info(
            "[STT] received AudioBuffer type=%s",
            type(buffer).__name__,
        )

        # ----------------------------------------------------
        # AudioBuffer can be:
        #
        #   AudioFrame
        #   list[AudioFrame]
        # ----------------------------------------------------

        if isinstance(buffer, list):
            frames = buffer

        else:
            frames = [buffer]

        if not frames:
            raise ValueError(
                "[STT] AudioBuffer contains no AudioFrames."
            )

        # ----------------------------------------------------
        # Inspect first frame
        # ----------------------------------------------------

        first_frame = frames[0]

        sample_rate = first_frame.sample_rate
        num_channels = first_frame.num_channels

        logger.info(
            "[STT] audio format: "
            "sample_rate=%s channels=%s frames=%s",
            sample_rate,
            num_channels,
            len(frames),
        )

        # ----------------------------------------------------
        # Extract PCM from every AudioFrame
        # ----------------------------------------------------

        chunks: list[np.ndarray] = []

        for index, frame in enumerate(frames):

            if frame.sample_rate != sample_rate:
                raise ValueError(
                    "[STT] AudioFrames have different sample rates: "
                    f"{sample_rate} vs {frame.sample_rate}"
                )

            data = np.frombuffer(
                frame.data,
                dtype=np.int16,
            )

            if data.size == 0:
                continue

            # ------------------------------------------------
            # Convert interleaved multi-channel PCM to mono.
            #
            # Example stereo:
            #
            # L R L R L R
            #
            # becomes:
            #
            # (L+R)/2 ...
            # ------------------------------------------------

            if num_channels > 1:

                if data.size % num_channels != 0:
                    raise ValueError(
                        "[STT] PCM sample count is not divisible "
                        f"by channel count: "
                        f"{data.size} / {num_channels}"
                    )

                data = data.reshape(
                    -1,
                    num_channels,
                )

                data = data.astype(
                    np.float32
                ).mean(axis=1)

            else:
                data = data.astype(np.float32)

            chunks.append(data)

        if not chunks:
            raise ValueError(
                "[STT] AudioBuffer contained no PCM samples."
            )

        # ----------------------------------------------------
        # Concatenate frames
        # ----------------------------------------------------

        audio = np.concatenate(chunks)

        # ----------------------------------------------------
        # Normalize int16 → float32 [-1, 1]
        # ----------------------------------------------------

        if num_channels == 1:
            audio = audio / 32768.0

        else:
            audio = audio / 32768.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Calculate duration
        # ----------------------------------------------------

        duration = len(audio) / sample_rate

        logger.info(
            "[STT] extracted %.3fs of audio "
            "(%d samples @ %d Hz)",
            duration,
            len(audio),
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

        import librosa

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
            "[STT] recognize() called"
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        await self._ensure_model()

        if self._model is None:
            raise RuntimeError(
                "[STT] Faster-Whisper model failed to load."
            )

        # ----------------------------------------------------
        # Extract audio
        # ----------------------------------------------------

        audio, sample_rate = await asyncio.to_thread(
            self._audio_buffer_to_numpy,
            buffer,
        )

        # ----------------------------------------------------
        # Resample
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
        # Ignore completely empty audio
        # ----------------------------------------------------

        if len(audio) == 0:

            logger.warning(
                "[STT] empty audio received"
            )

            text = ""

        else:

            # ------------------------------------------------
            # Faster-Whisper inference
            # ------------------------------------------------

            logger.info(
                "[STT] running Faster-Whisper..."
            )

            language_code = self._language

            if language is not NOT_GIVEN and language:
                language_code = str(language)

            def transcribe():

                segments, info = self._model.transcribe(
                    audio,
                    language=language_code,
                    beam_size=self._beam_size,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                )

                parts: list[str] = []

                for segment in segments:

                    segment_text = segment.text.strip()

                    if segment_text:
                        parts.append(segment_text)

                return " ".join(parts).strip()

            text = await asyncio.to_thread(
                transcribe
            )

        # ----------------------------------------------------
        # RESULT
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

        # ----------------------------------------------------
        # Return LiveKit SpeechEvent
        # ----------------------------------------------------

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
            "[STT] releasing Faster-Whisper..."
        )

        self._model = None

        logger.info(
            "[STT] Faster-Whisper released."
        )