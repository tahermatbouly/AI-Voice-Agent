"""
Standalone test script for ElevenLabs Text-to-Speech API (Arabic).

Unlike NAMAA/EGTTS (local models) and edge-tts (unofficial, no key needed),
ElevenLabs is a real paid-by-default API with a free tier -- this test
needs a valid API key and will consume your free monthly character quota
(10,000 chars/month on the free tier).

Run this to:
  1. Confirm your API key and voice ID work
  2. Hear how ElevenLabs handles Egyptian Arabic specifically
  3. Time how long generation takes (should be fast -- it's cloud-hosted,
     not running on your CPU like NAMAA/EGTTS)
  4. Track how many characters this test used against your free quota

Setup before running:
  1. Sign up free at https://elevenlabs.io/app/sign-up
  2. Get an API key at https://elevenlabs.io/app/settings/api-keys
  3. Pick a voice at https://elevenlabs.io/app/voice-library and copy its Voice ID
  4. Set ELEVENLABS_API_KEY and VOICE_ID below (or via a .env file -- see note)
  5. pip install elevenlabs python-dotenv
  6. python test_elevenlabs_tts.py
"""

import os
import sys
import time

# --- Config -----------------------------------------------------------

# Reads from environment first (recommended -- put these in a .env file:
#   ELEVENLABS_API_KEY=your_key_here
#   ELEVENLABS_VOICE_ID=your_chosen_voice_id
# so the key never gets hardcoded/committed). Falls back to the literal
# strings below only if env vars aren't set -- fill these in directly if
# you'd rather not bother with .env for a quick test.
API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")

MODEL_ID = "eleven_multilingual_v2"  # ElevenLabs' multilingual model, supports Arabic

TEST_SENTENCES = [
    "انا سبت الشغل و راجع دلوقتي علي طول",
    "ممكن أعرف اسمك وعنوانك من فضلك؟",
    "شكرا لاتصالك بينا، هنرد عليك في أقرب وقت ممكن",
]

MIN_EXPECTED_SECONDS = 0.5


def check_setup():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # optional -- only needed if using a .env file

    global API_KEY, VOICE_ID
    API_KEY = os.environ.get("ELEVENLABS_API_KEY", API_KEY)
    VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", VOICE_ID)

    missing = []
    try:
        import elevenlabs  # noqa: F401
    except ImportError:
        missing.append("elevenlabs")

    if missing:
        print(f"[test] MISSING DEPENDENCIES: {', '.join(missing)}")
        print("[test] install with: pip install elevenlabs python-dotenv")
        sys.exit(1)

    if not API_KEY:
        print("[test] MISSING API KEY.")
        print("[test] get one at https://elevenlabs.io/app/settings/api-keys")
        print("[test] then set ELEVENLABS_API_KEY in a .env file, or edit this script directly.")
        sys.exit(1)

    if not VOICE_ID:
        print("[test] MISSING VOICE_ID.")
        print("[test] pick a voice at https://elevenlabs.io/app/voice-library")
        print("[test] then set ELEVENLABS_VOICE_ID in a .env file, or edit this script directly.")
        sys.exit(1)


def run_tests():
    from elevenlabs.client import ElevenLabs

    print(f"[test] using VOICE_ID: {VOICE_ID}")
    client = ElevenLabs(api_key=API_KEY)
    results = []
    total_chars = 0

    for i, text in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[test] sentence {i}: {text}")
        t0 = time.time()

        try:
            audio = client.text_to_speech.convert(
                voice_id=VOICE_ID,
                text=text,
                model_id=MODEL_ID,
            )

            out_path = f"elevenlabs_test_{i}.mp3"
            with open(out_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            elapsed = time.time() - t0
            file_size = os.path.getsize(out_path)
            passed = file_size > 0
            status = "PASS" if passed else "SUSPICIOUS (empty file)"

            total_chars += len(text)
            print(f"[test] generated in {elapsed:.1f}s, {file_size} bytes -> {out_path} [{status}]")
            results.append((i, True, elapsed, file_size, passed))

        except Exception as e:
            print(f"[test] FAILED on sentence {i}: {e}")
            if "quota" in str(e).lower() or "credit" in str(e).lower():
                print("[test] this looks like a free-tier quota issue -- check your usage at")
                print("[test] https://elevenlabs.io/app/usage")
            results.append((i, False, None, None, False))

    return results, total_chars


def print_summary(results, total_chars):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for i, ran, elapsed, size, passed in results:
        if not ran:
            print(f"  Sentence {i}: FAILED TO RUN")
            all_passed = False
        elif not passed:
            print(f"  Sentence {i}: SUSPICIOUS OUTPUT")
            all_passed = False
        else:
            print(f"  Sentence {i}: OK ({elapsed:.1f}s, {size} bytes)")

    print("=" * 50)
    print(f"Approx characters used this run: {total_chars} (free tier: 10,000/month)")
    if all_passed:
        print("All tests passed. Listen to elevenlabs_test_*.mp3 and judge:")
        print("  - Does it sound Egyptian, or more generic/MSA-leaning?")
        print("  - How does generation speed compare to NAMAA/EGTTS on your CPU?")
        print("Compare against your NAMAA, EGTTS, and edge-tts results.")
    else:
        print("Some tests failed -- see details above.")


def list_available_voices():
    """Print voices actually usable on your current plan, so you don't
    pick a tier-gated one by accident (a common free-tier gotcha)."""
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=API_KEY)
    print("[test] fetching voices available on your plan...")
    try:
        response = client.voices.get_all()
        for voice in response.voices:
            print(f"  {voice.voice_id}  --  {voice.name}")
    except Exception as e:
        print(f"[test] could not list voices: {e}")


if __name__ == "__main__":
    check_setup()
    if "--list-voices" in sys.argv:
        list_available_voices()
        sys.exit(0)
    results, total_chars = run_tests()
    print_summary(results, total_chars)
