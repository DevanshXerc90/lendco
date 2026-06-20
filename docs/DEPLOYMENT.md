# Deployment Instructions

## A. Local (recommended for the demo)

### Prerequisites
- Python 3.10+ (tested on 3.14). `pip`.
- Everything else is optional and degrades gracefully.

### Steps
```bash
pip install -r requirements.txt          # core deps; heavy/voice deps are optional
python -m data.generate                  # create datasets under data/generated/
python -m scripts.run_platforms          # terminal 1: 5 services on :8101-:8105
python -m eval.evaluator                 # (optional) -> eval/results.json
python -m dashboard.build_report         # -> dashboard/report.html (open in browser)
python -m app.cli --scenario ptp_memory  # terminal 2: talk to the agent
```

### Enabling real LLM reasoning
```bash
cp .env.example .env
# set ANTHROPIC_API_KEY=sk-ant-...   and optionally LLM_MODEL=claude-opus-4-8
```
With a key, the Voice Agent runs a full tool-use loop; without it, the deterministic
planner is used (same tools, same data, template phrasing).

### Enabling local voice
```bash
pip install faster-whisper pyttsx3 sounddevice scipy
# in .env:
STT_BACKEND=whisper
TTS_BACKEND=pyttsx3
python -m app.cli --borrower BRW00010 --voice
```
- TTS uses the OS speech engine (Windows SAPI5 / macOS NSSpeech / Linux espeak).
- STT is push-to-talk: press Enter, speak ~6s, it transcribes.

### Reset between demos
```bash
python -m scripts.reset      # clears runtime writes + memory.db (keeps datasets)
```

## B. Docker / docker-compose
```bash
docker compose up --build    # starts all 5 platforms; agent runs from the `agent` container
```
- `docker-compose.yml` defines one service per platform plus a tooling container.
- Mount a `.env` for the LLM key.

## C. Production notes (what changes)
| Concern | Local prototype | Production |
|---|---|---|
| Platforms | Mock FastAPI + JSON | Real CRM/Payments/Support/KB/Workflow APIs behind the same client interface |
| Telephony | Local mic/speaker | SIP/Twilio Media Streams + streaming STT/TTS with barge-in |
| Vector store | In-process TF-IDF | pgvector / managed vector DB + reranker |
| Memory | SQLite | Postgres + Redis (hot recall), per-borrower encryption |
| Workflow | Mock triggers | Real n8n/Temporal with retries + idempotency |
| Security | Modelled | Identity verification before disclosure, PII redaction, per-tool authz, audit log |
| Observability | Tool trace in summary | Distributed tracing of every tool call; eval run on live transcripts |

The client interface (`core/clients.py`) and tool registry (`core/tools.py`) are the
seams: swapping a mock for a real system means changing the client base URL/auth, not
the agents.

## D. Repository layout
```
config/        settings (env-driven, safe fallbacks)
data/          generate.py + generated datasets + runtime writes + memory.db
platforms/     crm, payments, support, knowledge, workflow  (5 services)
core/          clients, context_engine, diagnosis, rag, memory, tools, llm
agents/        context/diagnosis/knowledge/memory/voice agents + orchestrator
voice/         stt, tts (local, pluggable)
app/           cli (inbound-call simulator) + scenarios
eval/          evaluator + results.json
dashboard/     analytics + build_report (HTML) + streamlit_app (optional)
scripts/       run_platforms, reset, record_demos
docs/          architecture, system design, deployment, demo script
tests/         pytest suite
```
