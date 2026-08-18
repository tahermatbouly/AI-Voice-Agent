"""
Test script for Groq's free tier LLM API -- verifies:
  1. Your API key actually works (no billing/card required)
  2. Real conversational latency for your actual system prompt + model
     (the number that matters for a live voice agent, not marketing claims)
  3. Rate limit headroom, read directly from Groq's response headers
     (the actual source of truth, not third-party blog estimates)

Run:
    python test_groq_llm.py

Setup:
  1. pip install groq python-dotenv
  2. Get a free key at https://console.groq.com/ (no card needed)
  3. Set GROQ_API_KEY in a .env file
"""

import os
import sys
import time

MODEL = "openai/gpt-oss-120b"  # Groq's recommended replacement for the
                                 # now-deprecated llama-3.3-70b-versatile
                                 # (announced deprecated June 17, 2026)

# Simulates a few turns of the actual call-agent conversation, in Egyptian
# Arabic, so the latency numbers reflect real usage, not a generic "hello".
SYSTEM_PROMPT = """\
انت موظف استقبال بيرد على تليفونات شركة. اتكلم باللهجة المصرية بشكل طبيعي ومهذب.
هدفك إنك تجمع من المتصل: الاسم، العنوان، الوظيفة، والاستفسار بتاعه.
اسأل سؤال واحد بس في كل مرة، وابقى مختصر.
"""

TEST_TURNS = [
    "ابدأ المكالمة بترحيب قصير.",
    "اسمي أحمد وعايز أسأل عن عروض الصيانة.",
    "أنا ساكن في مدينة نصر، شارع مصطفى النحاس.",
]

# A response under this many seconds feels responsive in a live call.
# Above it, the pause becomes noticeable/awkward.
LATENCY_TARGET_SECONDS = 1.0


def check_setup():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        import groq  # noqa: F401
    except ImportError:
        print("[test] MISSING DEPENDENCY: groq")
        print("[test] install with: pip install groq python-dotenv")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[test] MISSING GROQ_API_KEY.")
        print("[test] get a free key at https://console.groq.com/ (no card required)")
        print("[test] then set GROQ_API_KEY in a .env file.")
        sys.exit(1)

    return api_key


def run_test(api_key):
    from groq import Groq

    client = Groq(api_key=api_key)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    last_headers = None

    for i, turn in enumerate(TEST_TURNS, start=1):
        history.append({"role": "user", "content": turn})
        print(f"\n[test] turn {i}: {turn}")

        t0 = time.time()
        try:
            # with_raw_response lets us read rate-limit headers directly
            # from Groq's actual response, rather than guessing from blogs.
            raw = client.chat.completions.with_raw_response.create(
                model=MODEL,
                messages=history,
                temperature=0.4,
            )
            elapsed = time.time() - t0

            completion = raw.parse()
            reply = completion.choices[0].message.content
            history.append({"role": "assistant", "content": reply})

            last_headers = raw.headers
            within_target = elapsed <= LATENCY_TARGET_SECONDS
            status = "OK" if within_target else f"SLOW (target: {LATENCY_TARGET_SECONDS}s)"

            print(f"[test] reply ({elapsed:.2f}s) [{status}]: {reply[:80]}...")
            results.append((i, True, elapsed, within_target))

        except Exception as e:
            elapsed = time.time() - t0
            print(f"[test] FAILED on turn {i} after {elapsed:.2f}s: {e}")
            err_str = str(e).lower()
            if "credit" in err_str or "billing" in err_str or "payment" in err_str:
                print("[test] this looks like a BILLING issue -- free tier should never hit this.")
            elif "rate" in err_str or "429" in err_str:
                print("[test] this looks like a RATE LIMIT issue -- expected occasionally on free tier under load.")
            results.append((i, False, elapsed, False))

    return results, last_headers


def print_summary(results, headers):
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    all_ok = True
    total_time = 0
    for i, ran, elapsed, within_target in results:
        if not ran:
            print(f"  Turn {i}: FAILED")
            all_ok = False
        else:
            print(f"  Turn {i}: {elapsed:.2f}s {'OK' if within_target else 'SLOW'}")
            total_time += elapsed
            if not within_target:
                all_ok = False

    if results:
        avg = total_time / len([r for r in results if r[1]])
        print(f"\n  Average response time: {avg:.2f}s (target: <= {LATENCY_TARGET_SECONDS}s)")

    if headers:
        print("\n  Rate limit info from Groq's actual response headers:")
        for key in headers.keys():
            if "ratelimit" in key.lower() or "rate-limit" in key.lower():
                print(f"    {key}: {headers[key]}")

    print("=" * 55)
    if all_ok:
        print("Groq's free tier is working and fast enough for a live voice agent.")
        print("No billing issues -- this is genuinely free at your current usage.")
    else:
        print("Some turns failed or were slower than the live-call target --")
        print("review details above before committing to this as the production LLM.")
    print("\nFor your exact, current rate limits: https://console.groq.com/ (dashboard)")


if __name__ == "__main__":
    api_key = check_setup()
    results, headers = run_test(api_key)
    print_summary(results, headers)
