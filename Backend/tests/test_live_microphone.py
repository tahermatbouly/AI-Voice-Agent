import asyncio
import json
import queue
import time

import numpy as np
import sounddevice as sd
import websockets


WS_URL = "ws://127.0.0.1:8000/ws/voice"

SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"

# How long to keep the mic gated after playback finishes, to absorb
# room reverb / acoustic tail (speaker sound bouncing around the
# room doesn't stop the instant the audio buffer ends). Tune this up
# if you're still getting echo, down if turn-taking feels sluggish.
TTS_TAIL_SECONDS = 0.4

audio_queue = queue.Queue()

mic_chunks = 0
sent_chunks = 0
last_mic_log = 0
last_send_log = 0

# Plain bool is fine to share between the sounddevice callback thread
# and the asyncio loop here — CPython bool assignment is atomic, and
# we only ever need "read the latest value", not a strict handshake.
bot_speaking = False


def audio_callback(indata, frames, time_info, status):
    global mic_chunks
    global last_mic_log

    if status:
        print("[MIC ERROR]", status)

    if bot_speaking:
        # Don't capture at all while the bot is talking (or during
        # the acoustic tail after it stops) — on laptop speakers +
        # open mic, this is what was being picked up and transcribed
        # as if it were the candidate speaking.
        return

    audio = indata.copy().tobytes()

    audio_queue.put(audio)

    mic_chunks += 1

    now = time.monotonic()

    if now - last_mic_log >= 1:
        print(
            f"[MIC] Receiving audio | "
            f"chunks={mic_chunks} | "
            f"bytes={len(audio)}"
        )
        last_mic_log = now


async def send_audio(websocket):
    global sent_chunks
    global last_send_log

    while True:
        audio = await asyncio.to_thread(
            audio_queue.get
        )

        await websocket.send(audio)

        sent_chunks += 1

        now = time.monotonic()

        if now - last_send_log >= 1:
            print(
                f"[WS] Sending microphone audio | "
                f"chunks={sent_chunks} | "
                f"bytes={len(audio)}"
            )
            last_send_log = now


async def play_tts_audio(
    audio_bytes: bytes,
    sample_rate: int,
):
    global bot_speaking

    print(
        f"[TTS] Starting playback at "
        f"{sample_rate} Hz"
    )

    print(
        f"[TTS] Playing "
        f"{len(audio_bytes)} bytes at "
        f"{sample_rate} Hz"
    )

    audio = np.frombuffer(
        audio_bytes,
        dtype=np.int16,
    )

    await asyncio.to_thread(
        sd.play,
        audio,
        samplerate=sample_rate,
    )

    await asyncio.to_thread(
        sd.wait,
    )

    print("[TTS] Playback finished")

    # Keep the mic gated a bit longer to cover room echo, then
    # re-open it for the candidate to speak.
    await asyncio.sleep(TTS_TAIL_SECONDS)

    bot_speaking = False

    print("[MIC] Re-armed (bot finished speaking)")


def print_candidate_summary(candidate: dict):
    print()
    print("=" * 50)
    print("INTERVIEW COMPLETE — DATA COLLECTED")
    print("=" * 50)

    if not candidate:
        print("(no data extracted)")
    else:
        for key, value in candidate.items():
            print(f"  {key}: {value}")

    print("=" * 50)
    print()


async def receive_messages(websocket):
    global bot_speaking

    tts_sample_rate = SAMPLE_RATE

    # Accumulates chunks between tts_start and tts_end — the server
    # now sends audio as multiple smaller frames instead of one big
    # message (to stay under a WebSocket per-message size limit that
    # the long end-of-interview summary was hitting), so we assemble
    # them back into one clip before playing, same as before.
    tts_audio_chunks = []

    while True:
        message = await websocket.recv()

        if isinstance(message, str):

            print()
            print("[SERVER]", message)
            print()

            try:
                data = json.loads(message)

                message_type = data.get("type")

                if message_type == "tts_start":
                    tts_sample_rate = data.get(
                        "sample_rate",
                        SAMPLE_RATE,
                    )

                    tts_audio_chunks = []

                    # Gate the mic as soon as we know audio is
                    # coming — don't wait for the bytes to arrive,
                    # there can be a little network delay between
                    # this message and the actual audio.
                    bot_speaking = True

                    print(
                        f"[MIC] Gated (bot about to speak) | "
                        f"sample rate: {tts_sample_rate} Hz"
                    )

                elif message_type == "tts_end":
                    print(
                        "[TTS] Server finished sending audio"
                    )

                    full_audio = b"".join(tts_audio_chunks)
                    tts_audio_chunks = []

                    if full_audio:
                        await play_tts_audio(
                            full_audio,
                            tts_sample_rate,
                        )
                    else:
                        # No audio arrived at all (e.g. empty text) —
                        # nothing to play, so re-arm the mic directly
                        # instead of waiting on play_tts_audio to do
                        # it.
                        bot_speaking = False

                elif message_type == "interview_complete":
                    print_candidate_summary(
                        data.get("candidate", {})
                    )

            except Exception:
                pass

        elif isinstance(message, bytes):

            # Just buffer it — actual playback happens once, on
            # tts_end, after every frame for this clip has arrived.
            tts_audio_chunks.append(message)


async def main():

    print("Connecting to server...")

    print()
    print("[AUDIO] Default devices:")
    print(sd.query_devices())
    print()

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=120,
    ) as websocket:

        print("Connected!")

        print()
        print(
            f"[MIC] Opening microphone at "
            f"{SAMPLE_RATE} Hz..."
        )

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
            blocksize=960,
        ):

            print()
            print("=" * 50)
            print("MICROPHONE ACTIVE")
            print("Speak normally.")
            print("Press Ctrl+C to stop.")
            print("=" * 50)
            print()

            sender = asyncio.create_task(
                send_audio(websocket)
            )

            receiver = asyncio.create_task(
                receive_messages(websocket)
            )

            try:
                await asyncio.gather(
                    sender,
                    receiver,
                )

            except asyncio.CancelledError:
                pass

            except websockets.exceptions.ConnectionClosed:
                # Expected: the server closes the connection right
                # after sending the goodbye message.
                print()
                print("Interview finished. Connection closed. Goodbye!")

            finally:
                sender.cancel()
                receiver.cancel()


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print()
        print("Stopped.")