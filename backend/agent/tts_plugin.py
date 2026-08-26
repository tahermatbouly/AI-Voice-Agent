import asyncio
import tempfile
import wave
from pathlib import Path

from livekit.agents import tts

from app import config


class VoiceTut(tts.TTS):

    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=config.VOICETUT_SAMPLE_RATE,
            num_channels=1,
        )

        print("[TTS] Loading VoiceTut-TTS...")

        from voicetut_tts import VoiceTutTTS

        self.tts_engine = VoiceTutTTS.from_pretrained(
            config.VOICETUT_MODEL
        )

        self.speaker = config.VOICETUT_SPEAKER

        print("[TTS] VoiceTut-TTS loaded.")
        print(f"[TTS] Speaker: {self.speaker}")

    def synthesize(self, text, *, conn_options=None):

        text = " ".join(text.split())

        if not text:
            return VoiceTutStream(
                tts_instance=self,
                input_text="",
                conn_options=conn_options,
            )

        if len(text) > config.VOICETUT_MAX_TEXT_LENGTH:
            text = text[:config.VOICETUT_MAX_TEXT_LENGTH]

        print(f"[TTS] text: {text}")

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

        print(
            f"[TTS] Generating VoiceTut audio -> "
            f"{self._input_text}"
        )

        loop = asyncio.get_running_loop()

        try:
            pcm, sample_rate = await loop.run_in_executor(
                None,
                self._generate,
            )

            print(
                f"[TTS] Generated {len(pcm)} bytes "
                f"at {sample_rate} Hz"
            )

            output_emitter.initialize(
                request_id="voicetut",
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            output_emitter.push(pcm)
            output_emitter.flush()

            print("[TTS] Audio sent to LiveKit.")

        except Exception as e:
            print(
                f"[TTS] ERROR: "
                f"{type(e).__name__}: {e}"
            )
            raise

    def _generate(self):

        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as f:

            print("[TTS] Running VoiceTut inference...")

            self.voicetut.tts_engine.synthesize(
                self._input_text,
                speaker=self.voicetut.speaker,
                num_step=config.VOICETUT_NUM_STEPS,
                output=f.name,
            )

            with wave.open(f.name, "rb") as wav:

                sample_rate = wav.getframerate()

                pcm = wav.readframes(
                    wav.getnframes()
                )

                print(
                    f"[TTS] WAV info: "
                    f"{sample_rate} Hz, "
                    f"{wav.getnchannels()} channels, "
                    f"{len(pcm)} bytes"
                )

                return pcm, sample_rate


def build_tts():
    return VoiceTut()