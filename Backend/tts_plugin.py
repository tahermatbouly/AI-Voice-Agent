import asyncio
import io
import logging
import wave

import requests

from Backend import config


logger = logging.getLogger("voice-agent.tts")


class VoiceTutTTS:

    def __init__(self):
        self.api_url = config.VOICETUT_API_URL.rstrip("/")
        self.speaker = config.VOICETUT_SPEAKER

        logger.info(
            "[TTS] VoiceTut API configured: %s",
            self.api_url,
        )

        logger.info(
            "[TTS] Speaker: %s",
            self.speaker,
        )

    async def synthesize(self, text: str):
        text = " ".join(text.split())

        if not text:
            return b"", 0

        if len(text) > config.VOICETUT_MAX_TEXT_LENGTH:
            text = text[:config.VOICETUT_MAX_TEXT_LENGTH]

        logger.info("[TTS] Synthesizing: %s", text)

        return await asyncio.to_thread(
            self._generate,
            text,
        )

    def _generate(self, text: str):

        url = f"{self.api_url}/tts"

        payload = {
            "text": text,
            "speaker": self.speaker,
        }

        logger.info("[TTS] POST %s", url)

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

        return self._parse_wav(response.content)

    @staticmethod
    def _parse_wav(data: bytes):

        with wave.open(
            io.BytesIO(data),
            "rb",
        ) as wav:

            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()

            pcm = wav.readframes(frame_count)

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
                f"VoiceTut returned {channels} channels. "
                f"Expected mono audio."
            )

        return pcm, sample_rate