from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


logger = logging.getLogger("voice-agent.audio")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"


class AudioManager:

    def __init__(
        self,
        config: AudioConfig | None = None,
    ) -> None:

        self.config = (
            config
            or AudioConfig()
        )

        self.sample_rate = (
            self.config.sample_rate
        )

        self.channels = (
            self.config.channels
        )

        self.dtype = (
            self.config.dtype
        )

        logger.info(
            "[AUDIO] Configured "
            "sample_rate=%s channels=%s dtype=%s",
            self.sample_rate,
            self.channels,
            self.dtype,
        )

    # ==========================================================
    # DEVICE INFORMATION
    # ==========================================================

    def list_devices(self) -> None:

        logger.info(
            "[AUDIO] Available devices:"
        )

        print(sd.query_devices())

    # ==========================================================
    # RECORD
    # ==========================================================

    async def record(
        self,
        duration: float,
    ) -> bytes:

        if duration <= 0:
            raise ValueError(
                "Recording duration must "
                "be greater than zero."
            )

        logger.info(
            "[AUDIO] Recording for %.2f seconds...",
            duration,
        )

        frames = int(
            duration * self.sample_rate
        )

        def capture():

            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )

            sd.wait()

            return audio.copy()

        audio = await asyncio.to_thread(
            capture
        )

        pcm = audio.tobytes()

        logger.info(
            "[AUDIO] Recorded %.2f seconds "
            "(%d bytes)",
            duration,
            len(pcm),
        )

        return pcm

    # ==========================================================
    # PLAYBACK
    # ==========================================================

    async def play(
        self,
        pcm: bytes,
        sample_rate: int,
        channels: int = 1,
    ) -> None:

        if not pcm:
            logger.warning(
                "[AUDIO] Empty audio. "
                "Nothing to play."
            )
            return

        logger.info(
            "[AUDIO] Playing %.2f KB "
            "at %d Hz...",
            len(pcm) / 1024,
            sample_rate,
        )

        audio = np.frombuffer(
            pcm,
            dtype=np.int16,
        )

        if channels > 1:

            audio = audio.reshape(
                -1,
                channels,
            )

        def playback():

            sd.play(
                audio,
                samplerate=sample_rate,
                blocking=True,
            )

        await asyncio.to_thread(
            playback
        )

        logger.info(
            "[AUDIO] Playback complete."
        )

    # ==========================================================
    # RECORD TO FILE
    # ==========================================================

    async def record_to_file(
        self,
        path: str,
        duration: float,
    ) -> str:

        import wave

        pcm = await self.record(
            duration
        )

        def write():

            with wave.open(
                path,
                "wb",
            ) as wav:

                wav.setnchannels(
                    self.channels
                )

                wav.setsampwidth(2)

                wav.setframerate(
                    self.sample_rate
                )

                wav.writeframes(
                    pcm
                )

        await asyncio.to_thread(
            write
        )

        logger.info(
            "[AUDIO] Saved recording: %s",
            path,
        )

        return path