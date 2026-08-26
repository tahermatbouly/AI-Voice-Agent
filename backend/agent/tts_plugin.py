import asyncio
import tempfile
import wave
from pathlib import Path

from livekit.agents import tts

from app import config


MODEL_DIR = Path(config.EGTTS_MODEL_PATH)
REFERENCE_AUDIO = Path(config.EGTTS_REFERENCE_AUDIO_PATH)


def _patch_numpy_legacy_aliases():
    """NumPy >= 1.24 removed several deprecated aliases (np.long,
    np.ulong, np.int, np.float, np.bool, np.object). Coqui-tts's
    dependency chain (scipy internals) still references these directly
    and crashes with "module 'numpy' has no attribute '...'" the moment
    synthesis actually runs. This restores them before TTS.api is
    imported, so nothing downstream can hit the missing attribute."""
    import numpy as np
    for name, replacement in (
        ("long", int), ("ulong", int), ("int", int),
        ("float", float), ("bool", bool), ("object", object),
    ):
        if not hasattr(np, name):
            setattr(np, name, replacement)


_patch_numpy_legacy_aliases()

from TTS.api import TTS

def normalize_egyptian_text(text: str) -> str:
    """
    Prepare Arabic text for EGTTS.

    The goal is not to translate the text.
    We only make the text easier for the TTS model
    to pronounce naturally.
    """

    replacements = {
        # Common formal → Egyptian forms
        "هل يمكنك": "ممكن",
        "هل تستطيع": "ممكن",
        "يرجى": "ممكن",
        "نود معرفة": "عايزين نعرف",
        "أود معرفة": "عايز أعرف",
        "تزويدي": "تقولّي",
        "المتقدم": "حضرتك",
        "المتقدمة": "حضرتك",

        # Common HR phrases
        "الوظيفة التي تتقدم لها": "الوظيفة اللي بتقدم عليها",
        "الوظيفة المتقدم لها": "الوظيفة اللي بتقدم عليها",
        "سنوات الخبرة": "سنين الخبرة",
        "الراتب الحالي": "المرتب الحالي",
        "الراتب المتوقع": "المرتب المتوقع",
        "موعد التفرغ": "ممكن تبدأ إمتى",
        "بيانات التواصل": "رقم الموبايل",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove characters that can hurt pronunciation.
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("_", "")

    # Normalize repeated whitespace.
    text = " ".join(text.split())

    return text.strip()

class EGTTS(tts.TTS):

    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
                
            ),
            sample_rate=24000,
            num_channels=1,
        )

        print("[TTS] Loading EGTTS-V0.1...")

        self.tts_engine = TTS(
            model_path=str(MODEL_DIR),
            config_path=str(MODEL_DIR / "config.json"),
            gpu=False,
            progress_bar=False,
        )

        print("[TTS] EGTTS-V0.1 loaded.")

    def synthesize(self, text, *, conn_options=None):

        original_text = text

        text = normalize_egyptian_text(text)

        print(f"[TTS] original: {original_text}")
        print(f"[TTS] normalized: {text}")

        return EGTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
        )


class EGTTSStream(tts.ChunkedStream):

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

        self.egtts = tts_instance

    async def _run(self, output_emitter):

        print(f"[TTS] _run(): generating -> {self._input_text}")

        loop = asyncio.get_running_loop()

        try:
            pcm, sample_rate = await loop.run_in_executor(
                None,
                self._generate,
            )

            print(
                f"[TTS] generated {len(pcm)} bytes "
                f"at {sample_rate} Hz"
            )

            output_emitter.initialize(
                request_id="egtts",
                sample_rate=sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            output_emitter.push(pcm)
            output_emitter.flush()

            print("[TTS] audio sent to LiveKit.")

        except Exception as e:
            print(f"[TTS] ERROR: {type(e).__name__}: {e}")
            raise

    def _generate(self):

        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as f:

            print("[TTS] Running EGTTS inference...")

            self.egtts.tts_engine.tts_to_file(
                text=self._input_text,
                speaker_wav=str(REFERENCE_AUDIO),
                language="ar",
                file_path=f.name,
                split_sentences=False,
            )

            print(f"[TTS] WAV generated: {f.name}")

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
    return EGTTS()