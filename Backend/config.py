import os

from dotenv import load_dotenv


load_dotenv()


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


# STT
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# TTS
VOICETUT_API_URL = os.getenv(
    "VOICETUT_API_URL",
)

VOICETUT_SPEAKER = os.getenv(
    "VOICETUT_SPEAKER",
    "Mohamed",
)

VOICETUT_SAMPLE_RATE = int(
    os.getenv(
        "VOICETUT_SAMPLE_RATE",
        "24000",
    )
)

VOICETUT_MAX_TEXT_LENGTH = int(
    os.getenv(
        "VOICETUT_MAX_TEXT_LENGTH",
        "500",
    )
)

VOICETUT_API_TIMEOUT = int(
    os.getenv(
        "VOICETUT_API_TIMEOUT",
        "60",
    )
)