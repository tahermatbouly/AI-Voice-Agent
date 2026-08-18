# Technical Specification: AI Voice Call Agent

**Project:** GB Corp Digital Transformation — AI Intern Task
**Author:** [Your name]
**Status:** Draft — Phase 1 planning

---

## 1. Overview

An AI-powered voice agent that answers company phone calls, holds a natural
real-time conversation in Egyptian Arabic, and automatically extracts and
stores structured caller information (name, address, position, inquiry,
notes) for every call. A companion web dashboard lets staff review call
history and extracted data.

The reference model for the intended experience is **Olimi AI**'s "Preview
Agent" feature — a browser-based live call with a configurable AI agent.

---

## 2. Goals

- Hold a live, real-time spoken conversation with a caller in Egyptian
  Arabic (not Modern Standard Arabic).
- Extract structured data from the conversation as it happens: name,
  address, position/organization, inquiry, and free-form notes.
- Detect when a call has naturally ended and close it gracefully.
- Persist every call's extracted record and full transcript.
- Provide a clean web dashboard to browse call history and view individual
  call details.
- Keep the entire stack free to run — no paid APIs required.

### Out of scope for v1 (stretch goals)
- Real inbound/outbound phone number handling (e.g. via Twilio).
- Barge-in (caller interrupting the agent mid-reply).
- Multi-language support beyond Egyptian Arabic.

---

## 3. Architecture

```
Browser (React)                          FastAPI backend
+----------------------+                 +-------------------------------+
|  Call page             |  WebSocket    |  WebSocket handler              |
|  - mic capture          |<------------->|  - session_manager.py           |
|  - audio playback         |  chunks     |  - vad.py (turn detection)      |
|  - waveform UI              |           |  - stt.py (faster-whisper)      |
+----------------------+                 |  - agent.py (Groq LLM)          |
                                          |  - tts.py (edge-tts)            |
+----------------------+   REST (HTTP)   |  - db.py (SQLite, SQLAlchemy)   |
|  Dashboard page         |<------------->|  - api/ (GET /calls, etc.)      |
|  - call list              |             +-------------------------------+
|  - call detail view         |
+----------------------+
```

**Data flow per conversation turn:**
1. Browser streams mic audio to the backend over the open WebSocket.
2. `vad.py` detects when the caller stops speaking and buffers the
   utterance.
3. `stt.py` transcribes the buffered audio (local faster-whisper, Arabic).
4. `agent.py` sends the transcript + conversation history to Groq's LLM,
   which returns structured JSON: `reply_text`, `extracted` fields, and
   `call_done`.
5. `tts.py` synthesizes `reply_text` as Egyptian Arabic speech and streams
   it back to the browser for playback.
6. The agent's running record is updated with any newly extracted fields.
7. When `call_done` is true, the final record + transcript are saved to
   SQLite via `db.py`, and the WebSocket connection closes.

---

## 4. Technology stack

| Layer | Choice | Reason |
|---|---|---|
| Backend framework | FastAPI + uvicorn | Native async + WebSocket support |
| STT | faster-whisper (local, CPU) | Free, unlimited, no API dependency |
| LLM | Groq API (Llama 3.3 70B) | Free tier, fast inference, no cost |
| TTS | edge-tts | Free, includes native Egyptian Arabic voice |
| VAD | webrtcvad | Lightweight turn-detection for streaming audio |
| Database | SQLite + SQLAlchemy | Simple, file-based, easy querying for dashboard |
| Frontend framework | React + Vite | Fast dev experience, suits an interactive dashboard |
| Styling | Tailwind CSS | Clean UI without heavy custom design work |
| Frontend routing | react-router-dom | Navigate between call page and dashboard |

**Cost:** $0 for the core pipeline. Only a future move to real phone lines
(Twilio) would introduce cost, and is not part of this scope.

---

## 5. Folder structure

```
call_agent_project/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, WebSocket + REST routes
│   │   ├── config.py          # settings: model sizes, voice, prompt, paths
│   │   ├── stt.py             # faster-whisper wrapper
│   │   ├── tts.py             # edge-tts wrapper
│   │   ├── agent.py           # CallAgent: conversation + extraction
│   │   ├── session_manager.py # tracks active calls, one CallAgent per session
│   │   ├── vad.py             # voice activity detection / turn boundaries
│   │   └── db.py              # SQLAlchemy models + queries
│   ├── api/
│   │   └── calls.py           # GET /calls, GET /calls/{id}
│   ├── requirements.txt
│   └── .env                   # GROQ_API_KEY
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── CallPage.jsx     # live call UI
│   │   │   ├── Dashboard.jsx    # call list
│   │   │   └── CallDetail.jsx   # single call record + transcript
│   │   ├── components/          # WaveformIndicator, CallCard, etc.
│   │   ├── api.js               # REST helper functions
│   │   └── App.jsx              # routing
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
└── README.md
```

---

## 6. Data model

**Call record** (stored per completed call):

| Field | Type | Description |
|---|---|---|
| id | integer | Primary key |
| timestamp | datetime | When the call started |
| name | text | Caller's name, if captured |
| address | text | Caller's address, if captured |
| position | text | Caller's role/organization, if captured |
| inquiry | text | Reason for the call |
| notes | text | Any additional relevant info |
| transcript | JSON | Full turn-by-turn conversation log |

---

## 7. API design

**REST endpoints (dashboard):**
- `GET /calls` — list all calls (id, timestamp, name, inquiry summary)
- `GET /calls/{id}` — full record + transcript for one call

**WebSocket endpoint (live call):**
- `WS /ws/call` — opens a session; streams audio chunks in both
  directions for the duration of the call

---

## 8. Known constraints and risks

- **CPU-only STT**: no GPU available, so Whisper model size is capped at
  roughly `small`-`medium` for acceptable live latency; larger models are
  likely impractical for real-time use on this hardware.
- **Egyptian Arabic dialect accuracy**: base Whisper is weaker on dialect
  than MSA; may need to test Hugging Face Egyptian-fine-tuned Whisper
  checkpoints if base accuracy proves insufficient.
- **Groq free-tier rate limits**: sufficient for prototyping and demos,
  but not production call volume — worth monitoring via the Groq console
  during testing.
- **No barge-in in v1**: caller cannot interrupt the agent mid-reply;
  flagged as a stretch goal.

---

## 9. Phased roadmap

**Phase 1 — Core pipeline (console prototype)**
Mic → local Whisper → Groq LLM agent → edge-tts, fixed-turn console loop.
Completed as an initial prototype.

**Phase 2 — Web backend**
FastAPI + WebSocket real-time flow, session management, VAD-based turn
detection, SQLite storage.

**Phase 3 — Frontend**
React call page (mic, waveform, playback) + dashboard (call list, call
detail view).

**Phase 4 — Polish / stretch goals**
Barge-in handling, Twilio integration (if confirmed required), richer
dashboard filtering/analytics.

---

## 10. Open questions for supervisor

- Is real inbound phone call handling (Twilio) required for this
  deliverable, or is a browser-based demo sufficient?
- Is there a specific set of fields beyond name/address/position/inquiry
  that should be captured?
- What does "done" look like for this task — a working demo, a deployed
  internal tool, or a written report?
