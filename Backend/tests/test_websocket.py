import asyncio
import wave

import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws/voice"

    with wave.open("test_egyptian_recruitment.wav", "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        audio = wav.readframes(wav.getnframes())

    # Add 1 second of silence at the end.
    # This gives Silero VAD enough silence to detect
    # the final speech_end event.
    silence = b"\x00\x00" * sample_rate

    audio += silence

    print(f"Sample rate: {sample_rate}")
    print(f"Channels: {channels}")
    print(f"Sample width: {sample_width}")
    print(f"Audio bytes: {len(audio)}")

    async with websockets.connect(uri) as websocket:
        print("Connected")

        await websocket.send("hello")

        response = await websocket.recv()
        print("Server:", response)

        # Send the audio in small WebSocket chunks.
        chunk_size = 3200

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]

            await websocket.send(chunk)

            print(
                f"Sent audio chunk "
                f"{i // chunk_size + 1}"
            )

            # Simulate real-time audio transmission.
            await asyncio.sleep(0.1)

        print("Finished sending audio")

        # Wait for the server's final events.
        try:
            while True:
                response = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=2,
                )

                print("Server:", response)

        except asyncio.TimeoutError:
            print("No more server messages")


if __name__ == "__main__":
    asyncio.run(main())
