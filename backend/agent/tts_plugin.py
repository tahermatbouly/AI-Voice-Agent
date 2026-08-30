import asyncio
import io
import logging
import wave

import requests

from livekit.agents import tts

from app import config


logger = logging.getLogger("voice-agent.tts")


class VoiceTut(tts.TTS):

    def __init__(self):

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=config.VOICETUT_SAMPLE_RATE,
            num_channels=1,
        )

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

    def synthesize(
        self,
        text,
        *,
        conn_options=None,
    ):

        text = " ".join(text.split())

        if not text:
            return VoiceTutStream(
                tts_instance=self,
                input_text="",
                conn_options=conn_options,
            )

        if len(text) > config.VOICETUT_MAX_TEXT_LENGTH:
            text = text[:config.VOICETUT_MAX_TEXT_LENGTH]

        logger.info("[TTS] text: %s", text)

        return VoiceTutStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
        )


class VoiceTutStream(tts.ChunkedStream):

    def __init__(
        self,
        tts_instance,
        input_text,
        conn_options,
    ):

        super().__init__(
            tts=tts_instance,
            input_text=input_text,
            conn_options=conn_options,
        )

        self.voicetut = tts_instance

    async def _run(self, output_emitter):

        if not self._input_text:
            return

        logger.info(
            "[TTS] Requesting VoiceTut API: %s",
            self._input_text,
        )

        loop = asyncio.get_running_loop()

        try:

            pcm, sample_rate = await loop.run_in_executor(
                None,
                self._generate,
            )

            logger.info(
                "[TTS] Received %d bytes at %d Hz",
                len(pcm),
                sample_rate,
            )

            output_emitter.initialize(
                request_id="voicetut-api",
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            output_emitter.push(pcm)

            output_emitter.flush()

            logger.info(
                "[TTS] Audio sent to LiveKit."
            )

        except Exception as e:

            logger.exception(
                "[TTS] VoiceTut API error: %s",
                e,
            )

            raise

    def _generate(self):

        url = f"{self.voicetut.api_url}/tts"

        payload = {
            "text": self._input_text,
            "speaker": self.voicetut.speaker,
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

        return self._parse_wav(response.content)

    @staticmethod
    def _parse_wav(data):

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


def build_tts():

    return VoiceTut()