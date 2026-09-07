# Architecture

## Overview

A fully self-hosted, real-time conversational avatar system. Zero paid APIs — every component runs on your own GPU server. Users interact via browser; all AI inference happens on the GPU server.

---

## System Diagram

```
┌─────────────────────────────────────────────────────┐
│                  USER BROWSER                       │
│                                                     │
│   Microphone ──► WebRTC Audio ──► [LiveTalking]     │
│   Speaker    ◄── WebRTC Audio ◄── [LiveTalking]     │
│   Avatar     ◄── WebRTC Video ◄── [LiveTalking]     │
└─────────────────────────────────────────────────────┘
                         ▕
                  WebRTC / HTTP
                         ▕
┌─────────────────────────────────────────────────────┐
│              GPU SERVER (RunPod)                    │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         LiveTalking  (app.py :8010)          │   │
│  │                                              │   │
│  │  1. STT   → Whisper (local model, GPU)       │   │
│  │  2. LLM   → Ollama  (Llama 3.1 8B, local)   │   │
│  │  3. TTS   → Kokoro  (local model)            │   │
│  │  4. Avatar→ MuseTalk (lip-sync, GPU)         │   │
│  │  5. Stream→ WebRTC  (aiortc, to browser)     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  GPU: RTX 4090 / A100 (40GB VRAM recommended)       │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow (one conversation turn)

```
User speaks
    │
    ▼
[WebRTC audio] ──► LiveTalking receives raw audio
    │
    ▼
[Whisper STT] ──► converts audio → text
    │
    ▼
[Ollama / Llama 3.1] ──► text → LLM response (streamed)
    │
    ▼
[Kokoro TTS] ──► response text → audio waveform (streamed)
    │
    ▼
[MuseTalk] ──► audio waveform → lip-synced video frames (GPU)
    │
    ▼
[WebRTC] ──► video frames + audio streamed back to browser
    │
    ▼
User sees avatar speaking + hears voice
```

---

## Component Responsibilities

### LiveTalking (`app.py`)
- Main aiohttp server, port 8010
- Handles WebRTC offer/answer signaling
- Orchestrates the STT → LLM → TTS → avatar pipeline
- Session management (multiple users supported)

### STT — Whisper (local)
- Model: `whisper-base` or `whisper-small` (trade speed vs accuracy)
- Runs on same GPU
- Receives audio from browser mic via WebRTC
- Outputs text → fed into LLM

### LLM — Ollama (local)
- Default model: `llama3.1:8b` (fast, good quality)
- Better quality: `llama3.1:70b` (needs A100)
- System prompt configurable per avatar/persona
- Streams tokens → fed into TTS as they arrive

### TTS — Kokoro (local)
- Fast, natural quality, runs on CPU or GPU
- No API key, no internet needed
- Voice cloning supported (give it a reference audio)
- Streams audio chunks → fed into MuseTalk

### Avatar — MuseTalk (GPU)
- Takes audio chunks → generates lip-synced video frames
- Avatar identity = one reference photo/video you upload
- Runs at 45fps on RTX 3090, 72fps on RTX 4090
- Upgrade path: swap to Hallo-Live for higher quality (needs 2x A100)

### Transport — WebRTC (aiortc)
- Browser-native, no plugins needed
- Delivers avatar video + voice audio to browser in real-time
- STUN server: `stun:stun.freeswitch.org:3478` (free, public)

---

## Deployment

### Development / MVP
- **RunPod**: rent RTX 4090 (~$0.74/hr) or A100 (~$1.50/hr)
- SSH into pod, clone repo, run `python app.py`
- Access via RunPod's public URL from any browser

### Production
- Dedicated GPU server (own hardware or cloud)
- Docker Compose for one-command deploy
- Nginx reverse proxy for HTTPS (required for browser mic access)
- Domain + SSL cert (Let's Encrypt, free)

---

## Avatar Identity

Each avatar = one folder in `avatars/` containing:
- Reference photo or short video clip of the person
- Optional: reference audio for voice cloning
- Pre-processed MuseTalk frames (generated once on first run)

Multiple avatars can be loaded; user selects via API parameter.

---

## Scalability

- Single GPU server handles up to 5 concurrent sessions (config: `max_session`)
- Each session = independent avatar + conversation context
- Scale horizontally: multiple GPU pods behind a load balancer
- No shared state between sessions (fully isolated)

---

## What this system is NOT

- Not using any paid APIs (no OpenAI, no ElevenLabs, no Anam, no Tavus)
- Not streaming pre-recorded video — every frame is generated in real-time
- Not a cloud service — you own and control everything
