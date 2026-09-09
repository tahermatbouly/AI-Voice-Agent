# AI Voice Agent — Dependencies & Run Guide

This document explains how to install, configure, and run the current AI Voice Agent.

## 1. Current Architecture

The current agent is built around:

- **Python**
- **FastAPI + Uvicorn** — backend HTTP/WebSocket server
- **WebSockets** — real-time communication between terminal client and backend
- **Silero VAD** — speech start/end detection
- **Cohere Arabic STT** — `cohere-transcribe-arabic-07-2026`
- **LangChain / LangGraph** — conversation flow and agent state
- **Groq / ChatGroq** — LLM used by the extraction/agent layer
- **VoiceTut TTS** — Arabic text-to-speech through the VoiceTut API
- **PyTorch / TorchAudio** — audio processing and VAD
- **Librosa** — audio resampling in the Cohere STT layer
- **SoundDevice** — microphone capture and terminal playback

The current flow is:

```text
Microphone
   ↓
Terminal WebSocket Client
   ↓
FastAPI WebSocket Server
   ↓
Audio processing / resampling
   ↓
Silero VAD
   ↓
Audio Buffer
   ↓
Cohere Arabic STT
   ↓
LangGraph / LangChain
   ↓
Extraction + validation + question management
   ↓
VoiceTut TTS
   ↓
WebSocket
   ↓
Terminal speaker
```

---

# 2. Requirements

## Hardware

The application is designed to run on CPU.

Recommended/current development machine:

- Intel i7 11th generation or better
- 16 GB RAM minimum
- 32 GB+ recommended
- 64 GB RAM is more than sufficient
- Working microphone
- Working speakers/headphones
- Internet connection for Cohere, Groq, and VoiceTut APIs

A dedicated NVIDIA GPU is **not required**.

---

# 3. Operating System

The current development environment is:

```text
Fedora Linux 42
```

The commands below assume Fedora/Linux.

---

# 4. System Dependencies

Update the system:

```bash
sudo dnf update -y
```

Install Python, development tools, Git, audio libraries, and FFmpeg:

```bash
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    git \
    ffmpeg \
    portaudio \
    portaudio-devel \
    libsndfile \
    libsndfile-devel
```

Check the installations:

```bash
python3 --version
pip3 --version
git --version
ffmpeg -version
```

---

# 5. Clone / Open the Project

If the repository has already been cloned:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
```

Otherwise:

```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-Voice-Agent
```

> Replace `<YOUR_REPOSITORY_URL>` with the repository URL.

---

# 6. Create the Python Virtual Environment

Use Python 3.11 for the project.

Check that Python 3.11 is available:

```bash
python3.11 --version
```

Create the virtual environment:

```bash
python3.11 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, verify:

```bash
which python
python --version
```

The output of `which python` should point to something similar to:

```text
.../AI-Voice-Agent/venv/bin/python
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

# 7. IMPORTANT: Always Use the Project Virtual Environment

Before running the backend, activate the environment:

```bash
source venv/bin/activate
```

Then verify:

```bash
which python
which pip
python --version
python -m pip --version
```

Do **not** rely on a globally installed `uvicorn`.

Use:

```bash
python -m uvicorn
```

instead of:

```bash
uvicorn
```

This prevents problems where packages are installed in the virtual environment but Uvicorn is running from another Python installation.

For example, if you see:

```text
/home/<user>/.local/lib/python3.13/site-packages/uvicorn/
```

while your project is supposed to use Python 3.11, you are probably running the wrong environment.

---

# 8. Python Dependencies

Install the main application dependencies:

```bash
python -m pip install \
    fastapi \
    uvicorn[standard] \
    websockets \
    numpy \
    torch \
    torchaudio \
    librosa \
    sounddevice \
    cohere \
    silero-vad \
    langchain \
    langgraph \
    langchain-groq \
    python-dotenv
```

The package:

```text
langchain-groq
```

is imported in Python as:

```python
from langchain_groq import ChatGroq
```

If you get:

```text
ModuleNotFoundError: No module named 'langchain_groq'
```

install it with:

```bash
python -m pip install -U langchain-groq
```

---

# 9. PyTorch

The application is CPU-only.

If the project requires the CPU wheels explicitly, install them from the PyTorch CPU index:

```bash
python -m pip install \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected:

```text
CUDA available: False
```

This is correct for the CPU-only setup.

---

# 10. Verify Important Python Packages

Run:

```bash
python -c "import fastapi; print('FastAPI OK')"
```

```bash
python -c "import uvicorn; print('Uvicorn OK')"
```

```bash
python -c "import websockets; print('WebSockets OK')"
```

```bash
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```

```bash
python -c "import torchaudio; print('TorchAudio OK:', torchaudio.__version__)"
```

```bash
python -c "import librosa; print('Librosa OK')"
```

```bash
python -c "import cohere; print('Cohere OK')"
```

```bash
python -c "from silero_vad import load_silero_vad; print('Silero VAD OK')"
```

```bash
python -c "import langchain; print('LangChain OK')"
```

```bash
python -c "import langgraph; print('LangGraph OK')"
```

```bash
python -c "from langchain_groq import ChatGroq; print('LangChain Groq OK')"
```

```bash
python -c "import sounddevice; print('SoundDevice OK')"
```

---

# 11. Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Edit it:

```bash
nano .env
```

Add the required API keys/configuration.

Example:

```env
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key

COHERE_STT_MODEL=cohere-transcribe-arabic-07-2026

VOICETUT_API_URL=your_voicetut_api_url
VOICETUT_SPEAKER=Mohamed
```

Use the actual values required by the current `Backend/config.py`.

**Do not commit `.env` to Git.**

Add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

If `.gitignore` already contains `.env`, do not add it twice.

---

# 12. Check API Keys

After activating the virtual environment:

```bash
python -c "from Backend import config; print('Cohere key configured:', bool(config.COHERE_API_KEY)); print('Groq key configured:', bool(config.GROQ_API_KEY))"
```

The output should show:

```text
Cohere key configured: True
Groq key configured: True
```

Do not print the actual API keys.

---

# 13. VoiceTut TTS

The current architecture uses VoiceTut TTS through its API.

The backend needs the VoiceTut server/API to be reachable.

The configuration is currently based on:

```env
VOICETUT_API_URL=...
VOICETUT_SPEAKER=Mohamed
```

If VoiceTut is running locally:

```text
VoiceTut server
       ↓
VoiceTut API
       ↓
AI Voice Agent
```

If VoiceTut is running remotely through a tunnel/API:

```text
AI Voice Agent
       ↓
Internet
       ↓
VoiceTut API / tunnel
       ↓
VoiceTut TTS
```

Before starting the complete agent, make sure the VoiceTut endpoint is available.

---

# 14. Start VoiceTut First

If VoiceTut is hosted separately, start it first according to its own server setup.

For example, if it exposes an HTTP API:

```text
VoiceTut API
http://<host>:<port>
```

Then configure:

```env
VOICETUT_API_URL=http://<host>:<port>
```

If using a Cloudflare tunnel, make sure the tunnel is running before starting the agent.

---

# 15. Start the Backend

From the project root:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
```

Activate the environment:

```bash
source venv/bin/activate
```

Verify the environment:

```bash
which python
python --version
```

Start FastAPI/Uvicorn:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

The backend should then be available at:

```text
http://127.0.0.1:8000
```

The WebSocket endpoint used by the terminal client is:

```text
ws://127.0.0.1:8000/ws/voice
```

---

# 16. Check the Backend

Open another terminal.

Activate the environment:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

If the health endpoint is implemented as expected, it should return a healthy/status response.

---

# 17. Run the Terminal Client

Open another terminal.

Go to the project:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
```

Activate the environment:

```bash
source venv/bin/activate
```

Run the client:

```bash
python client.py
```

If the client is inside another directory, run it using its actual module/path.

For example:

```bash
python Backend/client.py
```

or:

```bash
python -m Backend.client
```

Use whichever matches the actual location of the current `client.py`.

---

# 18. Expected Startup Sequence

The recommended order is:

### Terminal 1 — VoiceTut

Start the VoiceTut TTS API/server.

```bash
# Start VoiceTut according to its server configuration
```

Make sure its API is reachable.

---

### Terminal 2 — Backend

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate

python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

---

### Terminal 3 — Terminal Voice Client

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate

python client.py
```

Then speak into the microphone.

---

# 19. Microphone Check

The terminal client uses:

```python
sounddevice
```

To list audio devices:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

You can also run:

```bash
python -c "import sounddevice as sd; print(sd.default.device)"
```

If the wrong microphone is selected, change the input device in the client.

---

# 20. Audio Configuration

The current terminal client captures:

```text
Sample rate: 24000 Hz
Channels: 1
Format: PCM16
Block size: 960 samples
```

At 24 kHz:

```text
960 samples / 24000 samples/sec
= 0.04 sec
= 40 ms
```

So the client sends approximately:

```text
40 ms audio chunks
```

The backend converts the audio to:

```text
16000 Hz
```

for VAD/STT processing.

---

# 21. Current STT Model

The current Cohere Arabic STT model is:

```text
cohere-transcribe-arabic-07-2026
```

Configuration:

```env
COHERE_STT_MODEL=cohere-transcribe-arabic-07-2026
```

The STT pipeline is:

```text
24 kHz PCM16
      ↓
Backend
      ↓
Convert PCM16 → float32
      ↓
Resample 24 kHz → 16 kHz
      ↓
Create temporary WAV
      ↓
Cohere Arabic STT
      ↓
Arabic transcript
```

The temporary WAV is deleted after transcription.

---

# 22. VAD

The application uses:

```text
Silero VAD
```

The current VAD configuration includes:

```text
sample_rate = 16000
threshold = 0.5
min_silence_duration_ms = 500
speech_pad_ms = 100
frame_size = 512
```

The VAD detects:

```text
speech_start
speech_end
```

The utterance between these events is sent to STT.

---

# 23. LangGraph / Extraction

After STT produces the transcript:

```text
Transcript
    ↓
LangGraph
    ↓
LangChain / LLM
    ↓
Extraction / validation
    ↓
Candidate state
    ↓
Next question
```

The extraction layer is responsible for turning natural-language answers into structured candidate information.

Typical fields include:

```text
candidate_name
contact_info
position
experience
current_salary
expected_salary
availability
education
skills/tools
english
notes
```

The exact fields should always match the schema currently defined in the project.

---

# 24. Common Import Error

### Error

```text
ModuleNotFoundError: No module named 'langchain_groq'
```

### Fix

Activate the correct virtual environment:

```bash
source venv/bin/activate
```

Then:

```bash
python -m pip install -U langchain-groq
```

Verify:

```bash
python -c "from langchain_groq import ChatGroq; print('langchain-groq OK')"
```

Then start:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

---

# 25. Common Wrong-Python Problem

If the traceback contains something like:

```text
/home/<user>/.local/lib/python3.13/site-packages/uvicorn/
```

but the project uses Python 3.11, check:

```bash
which python
which uvicorn
python --version
python -m pip --version
```

The safest approach is:

```bash
source venv/bin/activate
```

and then:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

Avoid:

```bash
uvicorn Backend.main:app
```

unless you have verified that `uvicorn` belongs to the project virtual environment.

---

# 26. Check All Installed Packages

Useful command:

```bash
python -m pip list
```

Save the current environment:

```bash
python -m pip freeze > requirements.txt
```

Then another developer can recreate the environment with:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is already maintained manually, update it carefully rather than blindly overwriting it.

---

# 27. Recommended requirements.txt

A minimal application-level dependency list is:

```text
fastapi
uvicorn[standard]
websockets
numpy
torch
torchaudio
librosa
sounddevice
cohere
silero-vad
langchain
langgraph
langchain-groq
python-dotenv
```

For deployment/reproducibility, pin exact versions after the environment has been tested successfully:

```bash
python -m pip freeze > requirements.txt
```

---

# 28. Complete Fresh Installation

For a completely new machine:

```bash
sudo dnf update -y

sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make \
    git \
    ffmpeg \
    portaudio \
    portaudio-devel \
    libsndfile \
    libsndfile-devel
```

Then:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
```

Create the environment:

```bash
python3.11 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch CPU:

```bash
python -m pip install \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

Install application dependencies:

```bash
python -m pip install \
    fastapi \
    uvicorn[standard] \
    websockets \
    numpy \
    librosa \
    sounddevice \
    cohere \
    silero-vad \
    langchain \
    langgraph \
    langchain-groq \
    python-dotenv
```

Create environment configuration:

```bash
nano .env
```

Configure the API keys and VoiceTut settings.

Then verify:

```bash
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import cohere; print('Cohere OK')"
python -c "from langchain_groq import ChatGroq; print('Groq OK')"
python -c "from silero_vad import load_silero_vad; print('VAD OK')"
python -c "import sounddevice; print('Audio OK')"
```

Start the backend:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
python client.py
```

---

# 29. Daily Startup — Short Version

Once everything is installed, the normal workflow is simply:

### Terminal 1 — Backend

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

### Terminal 2 — Client

```bash
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
python client.py
```

### If VoiceTut is external

Make sure the VoiceTut API/tunnel is running before starting the backend.

---

# 30. Troubleshooting Checklist

## Backend does not start

Run:

```bash
source venv/bin/activate
which python
python --version
python -m pip --version
```

Then test imports:

```bash
python -c "from langchain_groq import ChatGroq; print('Groq OK')"
python -c "import cohere; print('Cohere OK')"
python -c "from silero_vad import load_silero_vad; print('VAD OK')"
```

---

## Port 8000 is already in use

Check:

```bash
ss -ltnp | grep :8000
```

Find the process:

```bash
lsof -i :8000
```

Stop it if necessary:

```bash
kill <PID>
```

Then restart:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

---

## Cannot connect to WebSocket

Check that the backend is running:

```bash
curl http://127.0.0.1:8000/health
```

Check that the client uses:

```text
ws://127.0.0.1:8000/ws/voice
```

---

## Microphone is not working

Run:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Check that the correct input device is available.

---

## Cohere fails

Check:

```bash
python -c "from Backend import config; print(bool(config.COHERE_API_KEY))"
```

Make sure the `.env` file contains:

```env
COHERE_API_KEY=...
```

---

## Groq fails

Check:

```bash
python -c "from Backend import config; print(bool(config.GROQ_API_KEY))"
```

Make sure:

```env
GROQ_API_KEY=...
```

is configured.

---

## VoiceTut fails

Check:

1. VoiceTut server/API is running.
2. `VOICETUT_API_URL` is correct.
3. The API/tunnel is reachable.
4. The configured speaker exists.

---

# 31. Security

Never commit API keys.

Check:

```bash
git status
```

Make sure `.env` is ignored:

```bash
git check-ignore .env
```

If `.env` is not ignored:

```bash
echo ".env" >> .gitignore
```

Never put real API keys directly into Python source files.

---

# 32. Development Architecture Goal

The current implementation is being optimized toward:

```text
                ┌── Current answer → STT → validation ──┐
User speaking ─┤                                         ├→ Commit
                └── Next question → LLM → TTS → cache ─┘
```

The authoritative interview state remains sequential, while independent preparation can run asynchronously.

Target behavior:

```text
Q1 answering
      │
      ├──────────────→ Prepare Q2
      │                  ├→ Generate Q2
      │                  └→ Generate Q2 TTS
      │
      ↓
Q1 validated
      │
      ↓
Q2 immediately available
      │
      ├──────────────→ Prepare Q3
      │
      ↓
Q3 ...
```

Speculative data must not modify the authoritative LangGraph state until the current answer has been validated and committed.

---

# 33. Git Workflow

Check the current branch:

```bash
git branch --show-current
```

Check changes:

```bash
git status
```

Create a new branch:

```bash
git switch -c <new-branch-name>
```

Add changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Update AI voice agent"
```

Push the new branch:

```bash
git push -u origin <new-branch-name>
```

---

# 34. Final Startup Checklist

Before testing the agent:

```text
[ ] Python 3.11 installed
[ ] Virtual environment activated
[ ] `which python` points to venv/bin/python
[ ] PyTorch installed
[ ] TorchAudio installed
[ ] FastAPI installed
[ ] Uvicorn installed
[ ] WebSockets installed
[ ] Cohere installed
[ ] Silero VAD installed
[ ] LangChain installed
[ ] LangGraph installed
[ ] langchain-groq installed
[ ] SoundDevice installed
[ ] FFmpeg installed
[ ] PortAudio installed
[ ] .env configured
[ ] Cohere API key configured
[ ] Groq API key configured
[ ] VoiceTut API available
[ ] Microphone detected
[ ] Backend starts on port 8000
[ ] WebSocket client connects
```

## Normal Run

```bash
# Terminal 1
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

```bash
# Terminal 2
cd "/run/media/taher27elmatbouly/New Volume/Taher/GB/AI-Voice-Agent"
source venv/bin/activate
python client.py
```

If VoiceTut is hosted separately, start it before Terminal 1.
