"""
Wraps faster-whisper as a LiveKit Agents STT plugin.

IMPORTANT: faster-whisper does not support streaming input -- it needs a
complete utterance's audio before it can transcribe. This plugin
implements LiveKit's standard non-streaming STT interface
(_recognize_impl). To actually use it inside an AgentSession, it must be
wrapped with LiveKit's StreamAdapter + a VAD instance -- that wrapping
happens in worker.py, not here:

    from livekit.agents import stt as stt_module
    from livekit.plugins import silero

    whisper_stt = WhisperSTT()
    session_stt = stt_module.StreamAdapter(
        stt=whisper_stt,
        vad=silero.VAD.load(),
    )

This file only defines the plugin itself -- the actual transcription
logic, reusing the same faster-whisper setup from the original
console-prototype stt.py.

NOTE: livekit-agents' exact STT base-class method signature can shift
slightly between versions. If this fails to load with an error about
_recognize_impl's signature, check the installed version's source:
    python -c "import livekit.agents.stt as m; import inspect; print(inspect.getsource(m.STT))"
and adjust the signature below to match.
"""

from faster_whisper import WhisperModel

from livekit.agents import stt
from livekit.agents.utils import AudioBuffer

from app import config

_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    """Load the Whisper model once and reuse it across every call --
    loading is slow, so this must not happen per-utterance."""
    global _model
    if _model is None:
        print(f"[stt_plugin] loading faster-whisper '{config.WHISPER_MODEL_SIZE}' on {config.WHISPER_DEVICE}...")
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("[stt_plugin] model loaded.")
    return _model


class WhisperSTT(stt.STT):
    """LiveKit STT plugin backed by local faster-whisper.
    Must be wrapped with stt.StreamAdapter + a VAD before use in an
    AgentSession -- see module docstring."""

    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        model = _load_model()

        # AudioBuffer -> a temp wav file, since faster-whisper's
        # transcribe() expects a file path or raw samples, not LiveKit's
        # internal buffer type directly.
        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(buffer.num_channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(buffer.sample_rate)
            wf.writeframes(buffer.data)
        wav_buffer.seek(0)

        segments, info = model.transcribe(
            wav_buffer,
            language=language or config.WHISPER_LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or config.WHISPER_LANGUAGE)],
        )