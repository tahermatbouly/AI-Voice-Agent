import sys
import time
from pathlib import Path

import requests


API_URL = "https://entered-educated-breaking-commission.trycloudflare.com"


def test_tts(text: str):
    print("=" * 60)
    print("VoiceTut API Test")
    print("=" * 60)

    print(f"Text: {text}")
    print(f"API:  {API_URL}")

    start = time.perf_counter()

    try:
        response = requests.post(
            f"{API_URL}/tts",
            json={
                "text": text,
                "speaker": "Mohamed",
            },
            timeout=120,
        )

        elapsed = time.perf_counter() - start

        print(f"\nHTTP status: {response.status_code}")
        print(f"Latency:     {elapsed:.2f}s")

        if response.status_code != 200:
            print("\nAPI Error:")
            print(response.text)
            response.raise_for_status()

        output_path = Path("tests/voicetut_api_test.wav")

        output_path.write_bytes(response.content)

        print(f"Audio size:  {len(response.content) / 1024:.2f} KB")
        print(f"Saved to:    {output_path}")

        return output_path

    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out.")
        sys.exit(1)

    except requests.exceptions.ConnectionError as e:
        print("\nERROR: Could not connect to VoiceTut API.")
        print(e)
        sys.exit(1)

    except requests.exceptions.RequestException as e:
        print("\nERROR:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":

    text = (
        "أهلاً بيك، معاك محمد من GB corp. "
        
    )

    test_tts(text)