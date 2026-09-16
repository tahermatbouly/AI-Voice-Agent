import torch
from silero_vad import load_silero_vad, VADIterator


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        print("[VAD] Loading Silero VAD...")

        self.model = load_silero_vad()

        self._build_iterator()

        # Silero requires exactly 512 samples at 16 kHz.
        self.frame_size = 512

        # Incoming WebSocket chunks don't necessarily
        # contain exactly 512 samples.
        self.audio_buffer = torch.empty(
            0,
            dtype=torch.float32,
        )

        print("[VAD] Ready")

    def _build_iterator(self):
        self.iterator = VADIterator(
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def set_min_silence(self, min_silence_duration_ms: int):
        """
        Change how much trailing silence ends a turn.

        Needed because the two kinds of answer in this interview are
        very different: the opening self-introduction is a long
        paragraph where the candidate pauses to think mid-sentence,
        so a short threshold cuts them off and we end up extracting
        from half an answer. A one-field follow-up ("عندي 3 سنين")
        is short, and a long threshold there just adds dead air
        before the agent replies.

        VADIterator takes this at construction, so changing it means
        rebuilding the iterator. That also resets its internal
        speech/silence state, which is why this should only be
        called between turns, never mid-utterance.
        """

        if min_silence_duration_ms == self.min_silence_duration_ms:
            return

        print(
            "[VAD] min_silence_duration_ms:",
            self.min_silence_duration_ms,
            "->",
            min_silence_duration_ms,
        )

        self.min_silence_duration_ms = min_silence_duration_ms

        self._build_iterator()

        self.audio_buffer = torch.empty(
            0,
            dtype=torch.float32,
        )

    def process(self, audio_chunk: bytes) -> list[str]:
        """
        Process PCM16 audio.

        The incoming chunk can have any number of samples.
        Internally we split it into 512-sample frames.

        Returns:
            A list containing zero or more events:
                ["speech_start"]
                ["speech_end"]
                ["speech_start", "speech_end"]
        """

        if not audio_chunk:
            return []

        # Make a writable copy to avoid the PyTorch warning.
        audio = torch.frombuffer(
            bytearray(audio_chunk),
            dtype=torch.int16,
        ).float()

        audio = audio / 32768.0

        # Add incoming samples to our internal frame buffer.
        self.audio_buffer = torch.cat(
            (
                self.audio_buffer,
                audio,
            )
        )

        events = []

        while self.audio_buffer.numel() >= self.frame_size:

            frame = self.audio_buffer[:self.frame_size]

            self.audio_buffer = self.audio_buffer[
                self.frame_size:
            ]

            event = self.iterator(frame)

            if event is None:
                continue

            if "start" in event:
                events.append("speech_start")

            elif "end" in event:
                events.append("speech_end")

        return events

    def reset(self):
        self.iterator.reset_states()

        self.audio_buffer = torch.empty(
            0,
            dtype=torch.float32,
        )