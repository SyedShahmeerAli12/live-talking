# Tech Stack

## Guiding Principle

**Zero paid APIs. Everything runs locally on our GPU server.**  
No OpenAI. No ElevenLabs. No Anam. No Tavus. No DashScope.  
One GPU server → full product.

---

## Stack

### Runtime & Server
| Layer | Technology | Version | Why |
|---|---|---|---|
| Language | Python | 3.12 | LiveTalking native |
| Web framework | aiohttp | latest | async, handles WebRTC concurrency |
| WebRTC | aiortc | latest | Python-native WebRTC, no extra server needed |
| Process management | Python multiprocessing | built-in | GPU isolation per model |

### STT (Speech → Text)
| Choice | Model | Why |
|---|---|---|
| **Whisper (local)** | `whisper-base` or `whisper-small` | Open source, runs on GPU, no API, fast |
| Library | `faster-whisper` | 4x faster than original Whisper, same quality |
| Fallback | `FunASR` (SenseVoice) | Better for non-English, already in LiveTalking |

### LLM (Text → Response)
| Choice | Model | Why |
|---|---|---|
| **Ollama** | `llama3.1:8b` | Local, fast, no API key, OpenAI-compatible API format |
| Quality upgrade | `llama3.1:70b` | Better responses, needs A100 40GB |
| Alternative | `qwen2.5:7b` | Better for multilingual (Arabic/English) |
| Interface | Ollama REST API (OpenAI-compatible) | Drop-in replacement, LiveTalking already uses OpenAI client format |

### TTS (Text → Voice Audio)
| Choice | Model | Why |
|---|---|---|
| **Kokoro** | kokoro-82M | Best open source TTS, fast, natural, MIT license |
| Alternative | CosyVoice | Voice cloning, already built into LiveTalking |
| Alternative | GPT-SoVITS | Best voice cloning quality, more GPU memory |
| Avoid | EdgeTTS | Requires internet (Microsoft servers) |

### Avatar / Lip-Sync (Audio → Video frames)
| Choice | Model | GPU needed | FPS | Why |
|---|---|---|---|---|
| **MuseTalk v1.5** | musetalk | RTX 3090+ | 45fps | Best balance quality/speed, in LiveTalking |
| Quality upgrade | **Hallo-Live** | 2x A100 | 20fps | Near-cinematic, Fudan University |
| Fast/cheap | Wav2Lip256 | RTX 3060 | 60fps | Lower quality but runs anywhere |

### Streaming Transport
| Layer | Technology | Why |
|---|---|---|
| Protocol | WebRTC | Browser-native, no plugins, real-time |
| Library | aiortc | Python WebRTC implementation |
| STUN | stun.freeswitch.org | Free public STUN server |
| TURN (production) | coturn (self-hosted) | Needed when behind strict NAT/firewall |

### Infrastructure
| Layer | Technology | Why |
|---|---|---|
| GPU server (MVP) | RunPod | Cheapest hourly GPU rental, easy setup |
| GPU server (prod) | Dedicated A100/H100 | Own hardware, no hourly cost |
| Container | Docker + Docker Compose | One-command deploy |
| Reverse proxy | Nginx | HTTPS termination (required for browser mic) |
| SSL | Let's Encrypt | Free certificates |
| OS | Ubuntu 22.04 | Best CUDA support |

---

## GPU Requirements

| Config | GPU | VRAM | Cost (RunPod) | Use case |
|---|---|---|---|---|
| MVP / testing | RTX 4090 | 24GB | ~$0.74/hr | MuseTalk + Llama 8B + Whisper |
| Production | A100 40GB | 40GB | ~$1.50/hr | Everything + headroom |
| High quality | 2x A100 | 80GB | ~$3.00/hr | Hallo-Live quality |

### VRAM breakdown (RTX 4090, 24GB)
| Component | VRAM |
|---|---|
| MuseTalk | ~6 GB |
| Llama 3.1 8B (Ollama) | ~8 GB |
| Whisper-small | ~1 GB |
| Kokoro TTS | ~1 GB |
| System overhead | ~2 GB |
| **Total** | **~18 GB** ✅ fits in 24GB |

---

## What We Are NOT Using

| Removed | Replaced with | Reason |
|---|---|---|
| OpenAI API | Ollama (local) | Paid, external dependency |
| DashScope (Alibaba) | Ollama (local) | Paid, external dependency |
| ElevenLabs TTS | Kokoro (local) | Paid, external dependency |
| Anam SDK | MuseTalk (local) | Paid, proprietary avatar |
| Tavus | MuseTalk (local) | Paid, proprietary avatar |
| EdgeTTS | Kokoro (local) | Requires internet (Microsoft) |
| Azure Speech | Whisper (local) | Paid, external dependency |

---

## Frontend

| Layer | Technology | Why |
|---|---|---|
| Framework | Plain HTML/JS (LiveTalking built-in) | Already works, WebRTC wired up |
| Upgrade path | Next.js (same as lilly/ai-doctor) | If custom UI needed |
| Video element | HTML `<video>` with WebRTC stream | Receives avatar video from server |
| Audio | WebRTC audio track | Mic → server, server audio → speaker |

---

## Configuration

All config lives in two files:
- `config.yaml` — model selection, TTS engine, transport, ports
- `.env` — secrets only (none needed for fully local setup)

Never hardcode API keys in source files. Always use `.env`.

---

## Upgrade Path (quality ladder)

```
Phase 1 (MVP):    Wav2Lip  + Llama 8B  + Kokoro   → RTX 4090, ~$0.74/hr
Phase 2 (Good):   MuseTalk + Llama 8B  + CosyVoice → RTX 4090, ~$0.74/hr  
Phase 3 (Great):  MuseTalk + Llama 70B + GPT-SoVITS → A100,    ~$1.50/hr
Phase 4 (Best):   Hallo-Live + Llama 70B + GPT-SoVITS → 2x A100, ~$3.00/hr
```
