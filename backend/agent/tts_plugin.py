from __future__ import annotations

import asyncio
import io
import logging
import wave

import requests

from app import config


logger = logging.getLogger("voice-agent.tts")


class VoiceTut:

    def __init__(self):

        self.api_url = (
            config.VOICETUT_API_URL.rstrip("/")
        )

        self.speaker = (
            config.VOICETUT_SPEAKER
        )

        logger.info(
            "[TTS] VoiceTut API configured: %s",
            self.api_url,
        )

        logger.info(
            "[TTS] Speaker: %s",
            self.speaker,
        )

    # ============================================================
    # SYNTHESIZE
    # ============================================================

    async def synthesize(
        self,
        text: str,
    ) -> tuple[bytes, int]:

        text = " ".join(
            text.split()
        )

        if not text:
            return b"", config.VOICETUT_SAMPLE_RATE

        if len(text) > config.VOICETUT_MAX_TEXT_LENGTH:

            text = text[
                :config.VOICETUT_MAX_TEXT_LENGTH
            ]

        logger.info(
            "[TTS] text: %s",
            text,
        )

        pcm, sample_rate = await asyncio.to_thread(
            self._generate,
            text,
        )

        return pcm, sample_rate

    # ============================================================
    # API
    # ============================================================

    def _generate(
        self,
        text: str,
    ) -> tuple[bytes, int]:

        url = (
            f"{self.api_url}/tts"
        )

        payload = {
            "text": text,
            "speaker": self.speaker,
        }

        logger.info(
            "[TTS] POST %s",
            url,
        )

        response = requests.post(
            url,
            json=payload,
            timeout=config.VOICETUT_API_TIMEOUT,
        )

        logger.info(
            "[TTS] API status: %s",
            response.status_code,
        )

        if response.status_code != 200:

            raise RuntimeError(
                f"VoiceTut API returned "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            )

        if not response.content:

            raise RuntimeError(
                "VoiceTut API returned empty audio."
            )

        logger.info(
            "[TTS] Downloaded %.2f KB",
            len(response.content) / 1024,
        )

        return self._parse_wav(
            response.content
        )

    # ============================================================
    # WAV PARSER
    # ============================================================

    @staticmethod
    def _parse_wav(
        data: bytes,
    ) -> tuple[bytes, int]:

        with wave.open(
            io.BytesIO(data),
            "rb",
        ) as wav:

            sample_rate = (
                wav.getframerate()
            )

            channels = (
                wav.getnchannels()
            )

            sample_width = (
                wav.getsampwidth()
            )

            frame_count = (
                wav.getnframes()
            )

            pcm = wav.readframes(
                frame_count
            )

        logger.info(
            "[TTS] WAV: "
            "%d Hz, "
            "%d channels, "
            "%d bytes/sample, "
            "%d frames",
            sample_rate,
            channels,
            sample_width,
            frame_count,
        )

        if channels != 1:

            raise RuntimeError(
                f"VoiceTut returned "
                f"{channels} channels. "
                f"Expected mono audio."
            )

        return pcm, sample_rate

    # ============================================================
    # SYNTHESIZE TO WAV FILE
    # ============================================================

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str,
    ) -> str:

        pcm, sample_rate = (
            await self.synthesize(text)
        )

        if not pcm:
            return output_path

        def write():

            with wave.open(
                output_path,
                "wb",
            ) as wav:

                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(
                    sample_rate
                )

                wav.writeframes(pcm)

        await asyncio.to_thread(
            write
        )

        logger.info(
            "[TTS] Saved: %s",
            output_path,
        )

        return output_path