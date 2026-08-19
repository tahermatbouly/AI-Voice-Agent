"""
Text-to-speech for the agent, as a prioritized fallback chain:
  1. ElevenLabs (primary) -- official LiveKit plugin, cloud, fast
  2. EGTTS-v0.1 (alternative) -- local, Egyptian-dialect-tuned, custom plugin
  3. edge-tts (last resort) -- free, cloud, always available, custom plugin

Uses LiveKit's built-in tts.FallbackAdapter to manage the actual
fallback behavior (retries, partial-output guards so it won't swap
voices mid-utterance, automatic recovery checks) rather than
hand-rolling that logic here.

CONFIDENCE NOTE: the ElevenLabs section uses LiveKit's official,
documented plugin -- high confidence. The EdgeTTS and EGTTS custom
plugin classes below implement LiveKit's TTS/ChunkedStream interface
based on the confirmed pattern (synthesize() -> ChunkedStream, with
_run(output_emitter) doing the actual work), but the EXACT AudioEmitter
push method calls should be verified against your installed version:

    python -c "from livekit.agents import tts; import inspect; print(inspect.getsource(tts.AudioEmitter))"

If push()/flush() aren't the right method names in your installed
version, adjust _run() below to match what that introspection shows.
"""

import asyncio
import io
import subprocess

from livekit.agents import tts
from livekit.plugins import elevenlabs

from app import config


# ---------------------------------------------------------------------
# 1. ElevenLabs -- official plugin, no custom code needed.
# ---------------------------------------------------------------------
def _build_elevenlabs_tts() -> tts.TTS:
    return elevenlabs.TTS(
        api_key=config.ELEVENLABS_API_KEY,
        voice_id=config.ELEVENLABS_VOICE_ID,
        model=config.ELEVENLABS_MODEL_ID,
    )


# ---------------------------------------------------------------------
# 2. edge-tts -- custom plugin (no official LiveKit plugin exists).
# ---------------------------------------------------------------------
class EdgeTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: "EdgeTTS", input_text: str, conn_options):
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import edge_tts

        # edge-tts outputs mp3; decode to raw PCM via ffmpeg (already
        # required elsewhere in this project) so it can be pushed as
        # audio frames.
        communicate = edge_tts.Communicate(self._input_text, config.EDGE_TTS_VOICE)
        mp3_chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.extend(chunk["data"])

        pcm_bytes = _mp3_bytes_to_pcm(bytes(mp3_chunks), sample_rate=self._tts.sample_rate)

        output_emitter.initialize(
            request_id="edge-tts",
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_bytes)
        output_emitter.flush()


class EdgeTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

    def synthesize(self, text: str, *, conn_options=None) -> tts.ChunkedStream:
        return EdgeTTSChunkedStream(tts_instance=self, input_text=text, conn_options=conn_options)


def _mp3_bytes_to_pcm(mp3_bytes: bytes, sample_rate: int) -> bytes:
    """Decode mp3 bytes to raw 16-bit PCM mono via ffmpeg (subprocess)."""
    process = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3->pcm conversion failed: {process.stderr.decode(errors='ignore')}")
    return process.stdout


# ---------------------------------------------------------------------
# 3. EGTTS-v0.1 -- custom plugin (no official LiveKit plugin exists).
# ---------------------------------------------------------------------
class EGTTSChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: "EGTTS", input_text: str, conn_options):
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # Reuses the same model-loading pattern validated in
        # tests/test_egtts.py -- loaded once and cached on the TTS
        # instance, not reloaded per utterance.
        model = self._tts._get_model()

        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, self._synthesize_sync, model)

        output_emitter.initialize(
            request_id="egtts",
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(wav_bytes)
        output_emitter.flush()

    def _synthesize_sync(self, model) -> bytes:
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            model.tts_to_file(
                text=self._input_text,
                speaker_wav=config.EGTTS_REFERENCE_AUDIO_PATH,
                language="ar",
                file_path=tmp.name,
            )
            with wave.open(tmp.name, "rb") as wf:
                return wf.readframes(wf.getnframes())


class EGTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            from TTS.api import TTS as CoquiTTS
            from huggingface_hub import snapshot_download
            import os

            print("[tts_plugin] loading EGTTS-v0.1...")
            ckpt_dir = snapshot_download(repo_id="OmarSamir/EGTTS-V0.1", repo_type="model")
            self._model = CoquiTTS(
                model_path=ckpt_dir,
                config_path=os.path.join(ckpt_dir, "config.json"),
            ).to("cpu")
            print("[tts_plugin] EGTTS-v0.1 loaded.")
        return self._model

    def synthesize(self, text: str, *, conn_options=None) -> tts.ChunkedStream:
        return EGTTSChunkedStream(tts_instance=self, input_text=text, conn_options=conn_options)


# ---------------------------------------------------------------------
# Public entry point: builds the full fallback chain, ordered per
# config.TTS_PROVIDER_ORDER. This is what worker.py imports and uses.
# ---------------------------------------------------------------------
def build_tts() -> tts.TTS:
    provider_builders = {
        "elevenlabs": _build_elevenlabs_tts,
        "egtts": EGTTS,
        "edge_tts": EdgeTTS,
    }

    ordered_instances = [
        provider_builders[name]()
        for name in config.TTS_PROVIDER_ORDER
        if name in provider_builders
    ]

    if not ordered_instances:
        raise RuntimeError("No TTS providers configured in config.TTS_PROVIDER_ORDER")

    return tts.FallbackAdapter(ordered_instances)