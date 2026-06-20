# LendCo — Agentic Inbound Voice Agent for Borrower Support

An AI voice agent that behaves like a knowledgeable loan-servicing
representative: it identifies a borrower, builds a unified context across five
independent systems, diagnoses what's known and what to ask, reasons with tools,
grounds every policy statement in the knowledge base (RAG), and **remembers** —
so the second call is faster, more personal, and more accurate than the first.

```
   Phone call ─► STT ─► ┌─────────────── Orchestrator (multi-agent) ───────────────┐ ─► TTS ─► Borrower
                        │ Context · Diagnosis · Knowledge(RAG) · Memory · Voice    │
                        └───┬─────────┬──────────┬───────────┬──────────┬──────────┘
                            ▼         ▼          ▼           ▼          ▼
                          CRM     Payments    Support    Knowledge   Workflow      (5 independent services)
```

## Why this is agentic, not an IVR
- **No decision tree.** The Diagnosis Layer computes information *gaps* per turn and
  asks only what it cannot look up; the Voice Agent decides actions via tool-calling.
- **Grounded.** Every policy claim is retrieved from company docs (RAG) and cited.
- **It learns.** A memory architecture (borrower / conversation / agent tiers) makes
  repeat interactions measurably better.

---

## Quickstart (Windows / macOS / Linux)

```bash
# 1. install (all heavy deps are optional — see note below)
pip install -r requirements.txt

# 2. generate synthetic data (120 borrowers, 1.8k payments, 130 tickets, 130 convos, 22 KB docs)
python -m data.generate

# 3. start the 5 independent platform services (own ports 8101-8105)
python -m scripts.run_platforms          # leave running in one terminal

# 4. (optional) run the evaluation harness -> eval/results.json
python -m eval.evaluator

# 5. build the analytics + evaluation dashboard -> dashboard/report.html
python -m dashboard.build_report

# 6. talk to the agent
python -m app.cli --scenario waiver          # scripted demo
python -m app.cli --scenario ptp_memory      # the two-call MEMORY demo (scenario 6)
python -m app.cli --borrower BRW00010        # interactive (type to the agent)
python -m app.cli --borrower BRW00010 --voice  # local mic + speaker (set STT/TTS backends)
```

> **Runs with zero API keys.** Without `ANTHROPIC_API_KEY` the agent uses a
> deterministic planner that still calls tools and grounds answers; with a key it
> runs a full LLM tool-use loop. Without `faster-whisper`/`pyttsx3` it uses
> typed input / printed output. Copy `.env.example` to `.env` to configure.

---

## Demo scenarios (all implemented)
| Cmd | Scenario |
|---|---|
| `--scenario remaining` | Remaining EMI inquiry — tenure math + next due date |
| `--scenario interest` | Interest paid — amortization split + outstanding |
| `--scenario penalty` | Penalty inquiry — timeline + RAG policy |
| `--scenario payment_failure` | Bank-glitch failure — gateway logs + root cause |
| `--scenario waiver` | Penalty waiver — eligibility from root cause + auto-reversal/ticket |
| `--scenario ptp_memory` | **Follow-up call with memory** (two calls) |

## How the assignment maps to the code
| Requirement | Where |
|---|---|
| Multi-system data aggregation (5) | [`platforms/`](platforms) — CRM, Payments, Support, Knowledge, Workflow |
| Sample data setup | [`data/generate.py`](data/generate.py) |
| Context Engine | [`core/context_engine.py`](core/context_engine.py) |
| Diagnosis Layer | [`core/diagnosis.py`](core/diagnosis.py) |
| Agentic voice app | [`agents/`](agents), [`voice/`](voice), [`app/cli.py`](app/cli.py) |
| RAG | [`core/rag.py`](core/rag.py) + [`platforms/knowledge`](platforms/knowledge) |
| Memory & continuous learning | [`core/memory.py`](core/memory.py), [`agents/memory_agent.py`](agents/memory_agent.py) |
| Multi-agent architecture (bonus) | [`agents/`](agents) — 5 explicit agents + orchestrator |
| Sentiment / risk / analytics / eval (bonus) | [`agents/orchestrator.py`](agents/orchestrator.py), [`eval/`](eval), [`dashboard/`](dashboard) |

## Documentation
- [Architecture](docs/ARCHITECTURE.md) · [System Design](docs/SYSTEM_DESIGN.md) · [Deployment](docs/DEPLOYMENT.md) · [Demo script](docs/DEMO_SCRIPT.md)

## Tests
```bash
python -m pytest -q        # core engines, scenarios, memory-improvement
```
