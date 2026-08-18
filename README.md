# AI Voice Call Agent

## Project Overview & Technical Architecture

## 1. Project Overview

This project is an **AI-powered voice agent** that answers company phone calls, holds a natural real-time conversation with callers in **Egyptian Arabic**, and automatically extracts and stores structured information from the call.

The agent is designed to capture information such as:

* Caller name
* Address
* Position or organization
* Inquiry
* Relevant notes
* Questions and answers

A companion web dashboard allows staff to:

* Browse previous calls
* Review extracted caller information
* Access complete call transcripts
* Review call history

The reference model for the intended user experience is **Olimi AI's “Preview Agent”**, which provides a browser-based, real-time voice conversation with a configurable AI agent.

### Primary Goals

1. Real-time conversation in Egyptian Arabic
2. Accurate structured data extraction
3. Low conversational latency
4. Natural interruption handling
5. Persistent caller memory across calls
6. Clean web interface for making calls and reviewing call history
7. Free or open-source components wherever possible

### Project Constraints

The architecture is shaped by two major constraints:

* **CPU-only development environment** — no GPU is available.
* **Strong preference for free tools and models** throughout the stack.

---

# 2. Architecture

The system is built around a **real-time streaming voice pipeline** rather than a traditional request/response API.

The goal is to provide a natural live conversation instead of a turn-based chatbot with audio added on top.

## 2.1 High-Level Components

```text
┌───────────────────────────────────────────────────────────┐
│                     Browser Frontend                      │
│                                                           │
│       React + Vite + LiveKit Client + Tailwind CSS       │
│                                                           │
│              Live Call     │     Dashboard                │
└────────────────────────────┬──────────────────────────────┘
                             │
                             │ Real-Time Audio
                             ▼
┌───────────────────────────────────────────────────────────┐
│                  LiveKit Server                           │
│                  Self-Hosted                              │
│                                                           │
│        Real-Time Transport + VAD + Turn Detection         │
│              + Interruption Handling                      │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                    Agent Worker                           │
│                      Python                               │
│                                                           │
│  ┌──────────────┐   ┌────────────────┐   ┌─────────────┐ │
│  │ Turn         │ → │ Speech-to-Text │ → │ LLM / Agent │ │
│  │ Detection    │   │ faster-whisper │   │   Brain     │ │
│  └──────────────┘   └────────────────┘   └──────┬──────┘ │
│                                                   │        │
│                                                   ▼        │
│                                          ┌─────────────┐   │
│                                          │ Structured  │   │
│                                          │ Extraction  │   │
│                                          └─────────────┘   │
│                                                   │        │
│                                                   ▼        │
│                                          ┌─────────────┐   │
│                                          │     TTS     │   │
│                                          │ ElevenLabs  │   │
│                                          │ / EGTTS     │   │
│                                          │ / edge-tts  │   │
│                                          └─────────────┘   │
└───────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │       Memory Layer          │
              │            Mem0             │
              └─────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │          Database           │
              │      SQLite + SQLAlchemy    │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │          FastAPI            │
              │          REST API           │
              └─────────────────────────────┘
                             │
                             ▼
                    Dashboard / Frontend
```

### Main Components

| Component           | Technology                    | Responsibility                    |
| ------------------- | ----------------------------- | --------------------------------- |
| Browser Frontend    | React + Vite                  | Live call interface and dashboard |
| Real-Time Transport | LiveKit                       | Real-time audio communication     |
| Agent Worker        | Python                        | Runs the complete voice pipeline  |
| Turn Detection      | LiveKit Agents                | Detects natural end of user turns |
| STT                 | faster-whisper                | Converts speech to text           |
| Conversational AI   | Groq + `openai/gpt-oss-120b`  | Conversation and reasoning        |
| TTS                 | ElevenLabs / EGTTS / edge-tts | Generates spoken responses        |
| Memory              | Mem0                          | Cross-call caller memory          |
| API                 | FastAPI                       | Dashboard REST API                |
| Database            | SQLite + SQLAlchemy           | Call records and transcripts      |
| Deployment          | Docker                        | Runs self-hosted infrastructure   |

---

## 2.2 Why a Real-Time Framework Instead of a Custom Build?

An early version of the project used a custom WebSocket server with manually implemented:

* Session tracking
* Silence-based turn detection
* Audio handling
* Conversation state
* Interruption handling

This approach was replaced with **LiveKit Agents**.

LiveKit provides important real-time voice functionality as built-in, tested functionality, including:

* Voice activity detection
* Model-based turn detection
* Interruption handling
* Real-time audio streaming
* Agent session management

This avoids rebuilding and debugging complex real-time communication infrastructure from scratch.

More importantly, it directly supports the project's **low-latency conversational goal**, because LiveKit is designed specifically for real-time voice and streaming workloads.

---

# 3. Conversation Flow

The following describes what happens during a single call.

### Step 1 — Start the Call

The caller opens the web application and starts a call.

The browser joins a LiveKit room, and the AI agent worker joins the same room.

### Step 2 — Check Caller Memory

Before the conversation starts, the memory layer is queried for previous information associated with the caller.

If the caller has interacted with the system before, relevant information can be retrieved.

### Step 3 — Agent Greeting

The agent starts the conversation with a short spoken greeting.

The greeting is generated by the TTS engine and streamed to the caller.

### Step 4 — Caller Speaks

The caller's audio is continuously streamed to the agent worker rather than being recorded into fixed audio chunks.

### Step 5 — Turn Detection

The turn-detection system determines when the caller has naturally finished speaking.

Instead of relying only on silence duration, model-based turn detection can determine whether the speaker has actually completed their thought.

### Step 6 — Speech-to-Text

The audio belonging to the completed turn is processed by the STT engine.

**faster-whisper** converts the Egyptian Arabic speech into text.

### Step 7 — Conversational AI

The transcribed text is sent to the LLM.

The LLM:

* Understands the caller's response
* Determines the appropriate next response
* Maintains the conversation context
* Extracts relevant information
* Calls structured tools when information needs to be stored

For example, if the caller says their name, the agent can immediately invoke a structured function to save the caller's name.

### Step 8 — Text-to-Speech

The LLM's response is sent to the selected TTS engine.

The generated Egyptian Arabic audio is streamed back to the caller.

### Step 9 — Interruption Handling

If the caller begins speaking while the agent is still talking, the interruption mechanism detects the new speech and stops the current agent response.

This allows the conversation to behave naturally instead of forcing the caller to wait until the agent finishes speaking.

### Step 10 — Continue Conversation

Steps 4–9 repeat until the conversation naturally concludes.

### Step 11 — Persist the Call

At the end of the call:

* The final extracted record is saved
* The complete transcript is stored
* Relevant durable caller information is saved to the memory layer

### Step 12 — Dashboard Review

Staff can later access the dashboard to:

* View previous calls
* Open individual call records
* Review extracted information
* Read the complete transcript

---

# 4. Tools & Technology Stack

## 4.1 Backend

| Technology         | Purpose                                    |
| ------------------ | ------------------------------------------ |
| **Python**         | Primary implementation language            |
| **LiveKit Agents** | Real-time voice pipeline                   |
| **FastAPI**        | REST API for the dashboard                 |
| **SQLAlchemy**     | Database ORM                               |
| **SQLite**         | Call and transcript storage                |
| **Docker**         | Running self-hosted LiveKit infrastructure |

## 4.2 Frontend

| Technology         | Purpose                              |
| ------------------ | ------------------------------------ |
| **React**          | Web application framework            |
| **Vite**           | Frontend development/build tooling   |
| **livekit-client** | Connects the browser to LiveKit      |
| **Tailwind CSS**   | Styling                              |
| **React Router**   | Navigation between application pages |

---

# 5. Models & APIs — Chosen Options and Alternatives

Every major model and API was evaluated according to three main criteria:

1. Quality for **Egyptian Arabic**, rather than only Modern Standard Arabic
2. Whether it is genuinely free or has a usable free tier
3. Whether it is practical on **CPU-only hardware** where applicable

---

## 5.1 Speech-to-Text (STT)

| Component      | Chosen                                 | Alternatives                                                     | Why Chosen                                                                                                                                                |
| -------------- | -------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speech-to-Text | **faster-whisper** — self-hosted/local | OpenAI Whisper API — paid; Groq-hosted Whisper — free tier/cloud | Free and effectively unlimited because it runs locally. faster-whisper is a CPU-optimized reimplementation of Whisper, making it practical without a GPU. |

### Selected Solution

**faster-whisper**

The model runs locally and therefore avoids API costs and usage limits.

Its CPU-oriented implementation makes it a suitable choice for the project's hardware constraints.

---

# 5.2 Conversational LLM

| Component         | Chosen                               | Alternatives                                        | Why Chosen                                                                                                                                                       |
| ----------------- | ------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM / Agent Brain | **Groq API — `openai/gpt-oss-120b`** | Self-hosted local LLM; OpenAI/Anthropic APIs — paid | Provides a free tier with no card requirement. Groq's inference infrastructure is optimized for low latency, which is important for real-time voice interaction. |

### Why a Cloud LLM?

A local LLM was considered, but running a sufficiently capable model on CPU-only hardware introduces significant inference latency.

Since the agent needs to respond quickly during a live conversation, a cloud inference provider is currently the more practical option.

### Model Selection Note

The LLM originally selected for this project was:

```text
llama-3.3-70b-versatile
```

This model was subsequently announced as deprecated by Groq.

The project therefore moved to:

```text
openai/gpt-oss-120b
```

which is Groq's recommended replacement and remains available on the free tier.

---

# 5.3 Text-to-Speech (TTS)

TTS is implemented as a **prioritized fallback chain** rather than a single fixed model.

This is necessary because the available options involve different trade-offs between:

* Voice quality
* Egyptian dialect authenticity
* Latency
* Reliability
* Hardware requirements
* Cost

**SILMA TTS v1** was also evaluated but was not selected because its documentation targets Modern Standard Arabic fluency rather than Egyptian dialect speech.

## TTS Priority Chain

| Priority            | Option             | Type                       | Notes                                                                                                                  |
| ------------------- | ------------------ | -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **1 — Primary**     | **ElevenLabs API** | Commercial API / Free tier | Cloud-hosted, so there is no local CPU inference cost. Strong general voice quality.                                   |
| **2 — Alternative** | **EGTTS-v0.1**     | Local / Self-hosted / Free | Designed specifically for colloquial Egyptian Arabic. XTTS v2-based, but heavier on CPU.                               |
| **3 — Last Resort** | **edge-tts**       | Free / Cloud               | Requires no API key and provides reliable Egyptian Arabic voices such as `ar-EG-SalmaNeural` and `ar-EG-ShakirNeural`. |

### 1. ElevenLabs

**Primary TTS option**

Advantages:

* Cloud-hosted
* No local CPU inference cost
* High general voice quality
* Low implementation complexity

#### Important Limitation

During evaluation, the free tier was found to restrict API access to Voice Library voices.

This means Egyptian-accented Voice Library voices may not be available on the free tier.

If a specific Egyptian-accented Voice Library voice is required, a paid tier may be necessary.

---

### 2. EGTTS-v0.1

**Local fallback**

EGTTS-v0.1 is designed specifically for **colloquial Egyptian Arabic** and is based on XTTS v2.

Advantages:

* Egyptian Arabic focused
* Self-hosted
* No API cost
* No external service dependency

Disadvantage:

* Considerably heavier than a cloud TTS service
* Multi-second generation time per sentence on CPU-only hardware

Therefore, it is kept as a fallback rather than the primary TTS engine.

---

### 3. edge-tts

**Final fallback**

The project can use Egyptian Arabic voices such as:

```text
ar-EG-SalmaNeural
ar-EG-ShakirNeural
```

Advantages:

* Free
* No API key required
* Cloud-hosted
* Fast
* Reliable
* Easy to integrate

Trade-off:

The voices tend to lean toward a more formal, Modern Standard Arabic-influenced pronunciation rather than fully colloquial Egyptian speech.

---

# 5.4 Real-Time Voice Infrastructure

| Component                            | Chosen                    | Alternative                                              | Why Chosen                                                                                                                  |
| ------------------------------------ | ------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Real-Time Transport & Turn Detection | **LiveKit — self-hosted** | Custom WebSocket server with manual session/VAD handling | Free and open-source when self-hosted. Provides tested real-time transport, VAD, turn detection, and interruption handling. |

LiveKit removes the need to manually implement and maintain critical real-time functionality.

This makes it particularly suitable for achieving the project's low-latency requirements.

---

# 5.5 Cross-Call Memory

| Component     | Chosen   | Alternative                          | Why Chosen                                                                                                  |
| ------------- | -------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Caller Memory | **Mem0** | Stateless calls with no memory layer | Allows the agent to recognize returning callers and recall relevant information from previous interactions. |

The memory layer enables conversations to continue across multiple calls rather than treating every call as completely independent.

---

# 6. End-to-End Data Flow

The complete system can be summarized as:

```text
Caller
  │
  │ Voice
  ▼
Browser
  │
  │ WebRTC / LiveKit
  ▼
LiveKit Server
  │
  ▼
Agent Worker
  │
  ├── Turn Detection
  │
  ├── faster-whisper
  │       │
  │       ▼
  │   Transcribed Text
  │       │
  │       ▼
  │   Groq LLM
  │   openai/gpt-oss-120b
  │       │
  │       ├── Conversation
  │       │
  │       └── Structured Extraction
  │               │
  │               ▼
  │           Call Information
  │
  ▼
TTS Fallback Chain
  │
  ├── ElevenLabs
  ├── EGTTS-v0.1
  └── edge-tts
  │
  ▼
Spoken Response
  │
  ▼
Caller
```

At the end of the call:

```text
                    Completed Call
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        SQLite Database              Mem0
             │                         │
             │                         └── Durable caller facts
             │
       ┌─────┴─────┐
       ▼           ▼
 Extracted     Transcript
   Record
       │
       ▼
    FastAPI
       │
       ▼
   Dashboard
```

---

# 7. Known Constraints & Open Questions

## 7.1 CPU-Only Development

No GPU is available on the development machine.

Therefore, all local model choices are evaluated specifically for CPU feasibility.

This particularly affects:

* STT inference speed
* Local TTS generation
* Local LLM inference

---

## 7.2 TTS Quality vs. Latency

There is an inherent trade-off between:

* Voice quality
* Egyptian dialect authenticity
* Generation latency
* Reliability
* Cost

The current fallback chain is therefore:

```text
ElevenLabs
     ↓
EGTTS-v0.1
     ↓
edge-tts
```

Each option can be swapped without changing the overall architecture.

---

## 7.3 Browser-Based Calls vs. Real Phone Calls

Real inbound telephone number handling, for example through **Twilio**, is not currently part of the system design.

The current scope is a **browser-based live call**, which matches the browser-based preview experience of the reference product.

---

## 7.4 Open Question: Phone-Line Integration

A key question for the supervising team is whether the final system must support **real telephone calls**.

If required, a future version could introduce a telephony provider such as Twilio between the telephone network and the real-time voice infrastructure.

This would change the call entry point but would not necessarily require replacing the core conversational agent architecture.

---

# 8. Design Summary

The project follows a **real-time, streaming, modular architecture** designed specifically for conversational Egyptian Arabic voice interaction.

The current design prioritizes:

* **LiveKit** for real-time communication and turn handling
* **faster-whisper** for local, CPU-friendly speech recognition
* **Groq + `openai/gpt-oss-120b`** for low-latency conversational reasoning
* **ElevenLabs → EGTTS → edge-tts** as a TTS fallback chain
* **Mem0** for persistent caller memory
* **SQLite + SQLAlchemy** for call records and transcripts
* **FastAPI** for backend APIs
* **React + Vite** for the web interface
* **Docker** for self-hosted infrastructure

The architecture is intentionally modular so that individual models or services can be replaced without redesigning the entire system.

---

## 9. Project Status

### Current Scope

* [x] Browser-based voice interaction
* [x] Real-time audio transport architecture
* [x] Egyptian Arabic conversation target
* [x] Local CPU-compatible STT option
* [x] Cloud-based low-latency LLM
* [x] TTS fallback architecture
* [x] Cross-call memory architecture
* [x] Call persistence architecture
* [x] Dashboard/API architecture

### Future Considerations

* [ ] Real telephone number integration
* [ ] Twilio or equivalent telephony integration
* [ ] Further optimization of CPU-based TTS
* [ ] More extensive Egyptian Arabic voice evaluation
* [ ] Production deployment
* [ ] Authentication and role-based dashboard access
* [ ] Advanced analytics for call outcomes

---

# 10. Conclusion

The AI Voice Call Agent is designed as a **real-time conversational voice system rather than a traditional chatbot**.

The combination of LiveKit's real-time infrastructure, local speech recognition, low-latency cloud inference, modular TTS, persistent memory, and a dedicated dashboard provides a foundation for building a natural Egyptian Arabic voice agent while maintaining the project's CPU-only and cost-conscious constraints.

The architecture also leaves clear paths for future expansion, particularly the addition of real telephone calls and production-scale deployment.
