# Deployment Instructions

Open demo mode is for local development only. A Vercel deployment must set
`API_SECRET`, `DEMO_API_KEY`, and `CORS_ORIGINS`; the application refuses to
start in production when either secret is missing. `/api/chat` and
`/api/scenario` require `X-API-Key`, enforce a 16 KiB request limit, and apply
an in-memory per-IP rate limit. Use a shared gateway limiter for a
multi-instance production deployment.

## A. Local (recommended for the demo)

### Prerequisites
- Python 3.11 or 3.12. `pip`.
- Platform services are optional and degrade gracefully.

### Steps
```bash
pip install -r requirements.txt
python -m data.generate                  # create datasets under data/generated/
python -m scripts.run_platforms          # terminal 1: 5 services on :8101-:8105
python -m eval.evaluator                 # (optional) -> eval/results.json
python -m dashboard.build_report         # -> dashboard/report.html (open in browser)
python -m app.cli --scenario ptp_memory  # terminal 2: talk to the agent
```

### Reset between demos
```bash
python -m scripts.reset      # clears runtime writes + memory.db (keeps datasets)
```

## B. Docker / docker-compose
```bash
docker compose up --build    # starts all 5 platforms; agent runs from the `agent` container
```
- `docker-compose.yml` defines one service per platform plus a tooling container.
- Mount a `.env` for deployment settings.

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
app/           cli (inbound-call simulator) + scenarios
eval/          evaluator + results.json
dashboard/     analytics + build_report (HTML) + streamlit_app (optional)
scripts/       run_platforms, reset, record_demos
docs/          architecture, system design, deployment, demo script
tests/         pytest suite
```

### Vercel environment variables

Required:

- `API_SECRET`: server-side secret used to authorize protected API requests.
- `DEMO_API_KEY`: key supplied in the `X-API-Key` header by the demo client.
- `CORS_ORIGINS`: comma-separated exact browser origins.

Optional: `CRM_URL`, `PAYMENTS_URL`, `SUPPORT_URL`, `KNOWLEDGE_URL`, and
`WORKFLOW_URL`.
Never commit any of these values.
