# LendCo — Agentic Inbound Voice Agent for Borrower Support

An AI voice agent that behaves like a knowledgeable loan-servicing representative: it identifies a borrower, builds a unified context across five independent systems, diagnoses what's known and what to ask, reasons with tools, grounds every policy statement in the knowledge base (RAG), and **remembers** — so the second call is faster, more personal, and more accurate than the first.

```
   Phone call ─► STT ─► ┌─────────────── Orchestrator (multi-agent) ───────────────┐ ─► TTS ─► Borrower
                        │ Context · Diagnosis · Knowledge(RAG) · Memory · Voice    │
                        └───┬─────────┬──────────┬───────────┬──────────┬──────────┘
                            ▼         ▼          ▼           ▼          ▼
                          CRM     Payments    Support    Knowledge   Workflow      (5 independent services)
```

## Why this is agentic, not an IVR
- **No decision tree.** The Diagnosis Layer computes information *gaps* per turn and asks only what it cannot look up; the Voice Agent decides actions via tool-calling.
- **Grounded.** Every policy claim is retrieved from company docs (RAG) and cited.
- **It learns.** A memory architecture (borrower / conversation / agent tiers) makes repeat interactions measurably better.

---

## Resume Claims & Verified Metrics

| Resume Claim | Status | Measured Benchmark (Synthetic Test Set) | Evidence / Command |
|---|:---:|---|---|
| **Claim 1:** *5 agents across 5 FastAPI services, orchestrated through an Anthropic tool use loop, tested on a synthetic set of 120 borrowers and 1,813 payments.* | **Verified** | **5 agents**, **5 services**, **120 borrowers**, **1,813 payments**; tool-use loop tested across 4 edge cases + deterministic fallback for offline evaluation. | [`core/llm.py`](core/llm.py), [`tests/test_anthropic_loop.py`](tests/test_anthropic_loop.py)<br>`python -m pytest tests/test_anthropic_loop.py` |
| **Claim 2:** *Grounded every policy answer in citations from a 22 document knowledge base, and ported the Knowledge Agent to LangChain LCEL with identical citations across 3 policy queries.* | **Verified** | **100.0% policy grounding rate** across all 64 evaluation runs. Ported to LCEL with **100% citation parity and identical ordering** on benchmark queries. | [`agents/knowledge_agent_lcel.py`](agents/knowledge_agent_lcel.py), [`eval/lcel_parity.py`](eval/lcel_parity.py)<br>`python -m eval.lcel_parity` |
| **Claim 3:** *Designed 3 tier SQLite memory with slot based context derivation. On the synthetic test set, resolution turns dropped from 3 to 1 and the redundant question rate was 0%.* | **Verified** | **0.0% redundant questions**; Repeat promise-to-pay turns dropped from **3.0 ➔ 1.0 turns** across 20 borrowers with **100.0% memory-aware openers**; 87.5% resolved without escalation (12.5% escalated by design). | [`core/memory.py`](core/memory.py), [`eval/evaluator.py`](eval/evaluator.py)<br>`python -m eval.evaluator` |

---

## Quickstart (Windows / macOS / Linux)

```bash
# 1. install (all heavy deps are optional — see note below)
pip install -r requirements.txt

# 2. generate synthetic data (120 borrowers, 1.8k payments, 130 tickets, 130 convos, 22 KB docs)
python -m data.generate

# 3. start the 5 independent platform services (own ports 8101-8105)
python -m scripts.run_platforms          # leave running in one terminal

# 4. run the evaluation harness -> eval/results.json
python -m eval.evaluator

# 5. verify LangChain LCEL citation parity
python -m eval.lcel_parity

# 6. run full test suite (121 tests, all offline — no API keys needed)
python -m pytest

# 7. build the analytics + evaluation dashboard -> dashboard/report.html
python -m dashboard.build_report

# 8. talk to the agent
python -m app.cli --scenario waiver          # scripted demo
python -m app.cli --scenario ptp_memory      # the two-call MEMORY demo (scenario 6)
python -m app.cli --borrower BRW00010        # interactive (type to the agent)
python -m app.cli --borrower BRW00010 --voice  # local mic + speaker (set STT/TTS backends)

# 9. serve the HTTP API (optional; the demo console pages are static files in public/)
uvicorn api.index:app --port 8000   # API + Swagger docs on /api/docs
```

> **Runs with zero API keys.** Without `ANTHROPIC_API_KEY` the agent uses a deterministic planner that still calls tools and grounds answers; with a key it runs a full LLM tool-use loop (`--llm` mode). Without `faster-whisper`/`pyttsx3` it uses typed input / printed output. Copy `.env.example` to `.env` to configure.

---

## Architecture & Systems

### 1. 5 Agents Across 5 FastAPI Services
- **Agents** ([`agents/`](agents)):
  - **Context Agent** ([`context_agent.py`](agents/context_agent.py)): Aggregates customer data into a unified profile across services.
  - **Diagnosis Agent** ([`diagnosis_agent.py`](agents/diagnosis_agent.py)): Computes information gaps, missing slots, and prioritized tool recommendations.
  - **Knowledge Agent** ([`knowledge_agent.py`](agents/knowledge_agent.py) & [`knowledge_agent_lcel.py`](agents/knowledge_agent_lcel.py)): Retrieves and cites policy documentation using RAG.
  - **Memory Agent** ([`memory_agent.py`](agents/memory_agent.py)): Manages short-term session recall, open commitment tracking, and continuous learning.
  - **Voice Agent** ([`voice_agent.py`](agents/voice_agent.py)): Generates borrower-facing spoken responses via tool-calling with policy grounding guardrails.
  - **Orchestrator** ([`orchestrator.py`](agents/orchestrator.py)): Manages the end-to-end conversation turn loop.
- **Platform Microservices** ([`platforms/`](platforms)):
  - `CRM` (port 8101) — Customer profile, contact info, KYC, credit limit.
  - `Payments` (port 8102) — Payment schedule, gateway attempt logs, payment links.
  - `Support` (port 8103) — Ticket lifecycle, supervisor escalations.
  - `Knowledge` (port 8104) — 22-document policy knowledge base.
  - `Workflow` (port 8105) — Penalty reversals, payoff statements, scheduled callbacks.

### 2. LangChain LCEL Integration
The **Knowledge Agent** is ported to LangChain Expression Language in [`agents/knowledge_agent_lcel.py`](agents/knowledge_agent_lcel.py):
- Composed declaratively with `RunnablePassthrough` and `RunnableLambda` piped to the `core.rag` retriever and citation formatters.
- Exposes the exact same public interface (`retrieve`, `grounding_block`, `cite`).
- Uses `langchain-core` as an optional dependency; the repo remains fully functional offline without it.
- **Parity Verification**: Verified via `python -m eval.lcel_parity` and [`tests/test_lcel.py`](tests/test_lcel.py), confirming 100% identical document IDs and citation order across the 3 policy benchmark queries (`late payment penalty policy`, `payment failure handling`, `penalty waiver eligibility`).

### 3. Policy Grounding Guardrail
To ensure zero hallucinations on company policies:
- In `VoiceAgent.respond_llm` and `AnthropicLLM.run_agent`, if an utterance matches a policy intent (`penalty_inquiry`, `penalty_waiver`, `payment_failure`, `settlement_request`, `foreclosure`, `interest_paid`), the response must cite a document that **actually exists in the knowledge base and was retrieved this turn** (`core.rag.RAG.verify_citations`). A citation-shaped string naming a missing document (e.g. `99_fake_policy`) counts as fabricated, not grounded.
- An ungrounded answer forces one `search_knowledge_base` retry. If retrieval returns nothing, the agent says the policy still needs checking instead of guessing; if the retry still cannot be grounded, it raises exactly one support ticket (`ticket_created`) and delivers the safe fallback.
- Tool arguments are validated before execution and side effects are de-duplicated within a turn (`core/tools.py`); retrieved text and tool outputs are treated as data, never as instructions.
- Unit tested in [`tests/test_grounding_retry.py`](tests/test_grounding_retry.py) and [`tests/test_rag_and_tools.py`](tests/test_rag_and_tools.py).

### 4. 3-Tier SQLite Memory Architecture
- **Tier 1 (Borrower Memory)**: Key-value facts, preferred contact times, and commitments across calls (e.g. promise-to-pay date and amount).
- **Tier 2 (Conversation History)**: Complete turn transcripts, sentiment trajectories, and resolved statuses.
- **Tier 3 (Agent Memory / Continuous Learning)**: Tracks success/failure counts and average turns per intent and resolution path, allowing the orchestrator to dynamically prefer the most efficient paths over time.
- **Slot-Based Derivation**: Categorizes slots as either *derivable* from unified context (e.g., installments remaining, overdue balance) or *ask-only* (e.g., date of future payment). By never asking for derivable values, the redundant question rate is strictly **0.0%**.
- **PII hygiene**: free text before it is persisted (FAQ questions, slot answers, commitment reasons, transcripts) is scrubbed of PAN, phone number and email (`core.memory.scrub`).

### 5. HTTP API & Authentication
The serverless API (`api/index.py`) serves the demo console and dashboards:

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/auth/login` | borrower knows the registered phone | issues an 8-hour HMAC-signed bearer token |
| `GET /api/borrowers/{id}`, `/api/payments`, `/api/gateway-logs`, `/api/tickets`, `/api/analytics/{id}` | borrower token **or** operator key | a token scoped to another borrower gets **403** |
| `GET /api/borrowers`, `POST /api/scenario` | operator key (`X-API-Key`) | unfiltered lists / side-effecting runner |
| `POST /api/chat` | borrower token or operator key | one turn of the agent |
| `/api/documents`, `/api/scenarios`, `/api/health`, `/api/eval` | none | public reference data |

Bad input returns **400** (malformed ids), **401** (missing/invalid credential), **403** (wrong borrower or missing operator key), **404** (no such record/workflow) or **422** (invalid amount, field, priority). Mutating endpoints on the internal platform services additionally expect a shared `X-Platform-Key` when `PLATFORM_API_KEY` is configured. See "Known limitations" for what this deliberately does *not* do.

---

## Demo Scenarios
| Cmd | Scenario | Key Features Tested |
|---|---|---|
| `--scenario remaining` | Remaining EMI inquiry | Tenure math + next due date |
| `--scenario interest` | Interest paid | Reducing balance amortization split + outstanding principal |
| `--scenario penalty` | Penalty inquiry | Timeline + RAG policy grounding |
| `--scenario payment_failure` | Payment failure handling | Gateway logs + system vs customer root cause |
| `--scenario waiver` | Penalty waiver | Eligibility from root cause + auto-reversal / ticket |
| `--scenario foreclosure` | Loan foreclosure | 6-EMI lock-in verification + workflow payoff statement |
| `--scenario settlement` | One-time settlement | Hardship recording + escalation by design to credit committee |
| `--scenario account` | Account summary | Full profile + loan analytics derivation |
| `--scenario ptp_memory` | **Follow-up call with memory** | Two-call promise-to-pay resolution (turns drop from 3 to 1) |

---

## Evaluation Results & Reproducibility

Results generated directly from [`eval/results.json`](eval/results.json) across the synthetic test set (120 borrowers, 1,813 payments, 22 KB docs):

### Overall Performance

| Metric | Measured Value | Notes |
|---|---|---|
| **Intent Accuracy** | **100.0%** (1.0) | 12/12 labelled intents matching ground truth |
| **Policy Grounding Rate** | **100.0%** (1.0) | 64/64 answers grounded in 22-doc KB citations |
| **Resolved without Escalation** | **87.5%** (0.875) | 56/64 scenarios resolved autonomously |
| **Escalated by Design** | **12.5%** (0.125) | 8/8 settlement requests routed to human committee |
| **Redundant Question Rate** | **0.0%** (0.0) | Zero re-asking of derivable/known slots |
| **Root-Cause Accuracy** | **100.0%** (1.0) | 25/25 system vs customer failure classifications |
| **Average Turns / Call** | **1.25 turns** | Across all 8 scenario scripts |
| **Memory Turn Reduction** | **3.0 ➔ 1.0 turns** | Evaluated on 20 borrowers (min=1, max=1 on repeat call) |
| **Memory-Aware Opener Share** | **100.0%** (1.0) | 20/20 repeat calls proactively reference prior commitments |

### Per-Scenario Breakdown

| Scenario | Runs | Resolved (No Esc.) | Escalated by Design | Grounding Rate | Intent Accuracy | Avg Turns |
|---|---|---|---|---|---|---|
| `remaining` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |
| `interest` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |
| `penalty` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |
| `payment_failure` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |
| `waiver` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 2.00 |
| `foreclosure` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |
| `settlement` | 8 | 0.0% | 100.0% (by design) | 100.0% | 100.0% | 1.00 |
| `account` | 8 | 100.0% | 0.0% | 100.0% | 100.0% | 1.00 |

### Reproduction Commands

```bash
# 1. Start the 5 mock platform services (terminal 1)
python -m scripts.run_platforms

# 2. Run the evaluation suite (terminal 2) -> writes eval/results.json
python -m eval.evaluator

# 3. Verify LangChain LCEL citation parity -> validates identical KB citations
python -m eval.lcel_parity

# 4. Optional: Run LLM evaluation (requires ANTHROPIC_API_KEY) -> writes eval/results_llm.json
python -m eval.evaluator --llm

# 5. Run complete test suite (all 121 tests pass offline)
python -m pytest
```

---

## Test Suite
The repository includes 121 unit and integration tests, all passing offline with zero external API dependencies:
- `tests/test_system.py` (13): Context engine, amortization, intent diagnosis, slot derivation, end-to-end scenarios.
- `tests/test_dates_and_calcs.py` (26): Controllable clock, calendar-month due dates, overdue/partial/duplicate payment math (Decimal).
- `tests/test_api_validation.py` (24): Login, cross-borrower 403s, operator key, malformed ids, invalid amounts/fields, platform key.
- `tests/test_api_flows.py` (7): End-to-end chat, analytics, citations and scenario runner through the HTTP API.
- `tests/test_rag_and_tools.py` (28): Tool-argument validation, one-side-effect-per-turn, citation verification, empty retrieval, model failure fallback, PII scrubbing.
- `tests/test_service_failures.py` (13): Downstream outages, marked fallback writes, malformed responses, timeouts, config validation.
- `tests/test_grounding_retry.py` (2): Forced KB search on ungrounded replies and the safe ticket fallback.
- `tests/test_anthropic_loop.py` (5): Mocked Anthropic client tool-use loop, multi-tool turns, iteration bounds, truncation.
- `tests/test_lcel.py` (2): LCEL Knowledge Agent parity and interface compatibility.
- `tests/test_memory_eval.py` (1): Multi-borrower memory evaluation across 20 borrowers.

```bash
python -m pytest -q
```

There is no configured linter in this repository; syntax is checked with `python -m compileall`.

---

## Known Limitations
Honest boundaries of this portfolio project — none of these are hidden behind marketing claims:

- **Demo mode is open.** `POST /api/auth/login` issues real HMAC-signed borrower tokens and token scope is enforced (a token for another borrower gets 403), but with `DEMO_API_KEY` unset the operator endpoints and unauthenticated reads are open, by design, and the API prints a warning at startup. Set `API_SECRET` + `DEMO_API_KEY` to lock it down.
- **No user accounts, refresh tokens, rate limiting or audit log.** Auth is two static keys plus one signed token — deliberately small, not a banking identity system.
- **Synthetic data.** All 120 borrowers, 1,813 payments and 22 KB documents are generated (`python -m data.generate`), and the five "platforms" are local mock services backed by JSON files. Metrics in this README are measured on that synthetic set, not on real customers.
- **Fallback writes are marked, not audited.** When a platform service is down, tickets/payment links/workflow runs are stored in `data/runtime/` with `fallback: True`; nothing reconciles them later.
- **Memory stores plaintext SQLite** on local disk (with PAN/phone/email scrubbed), not encrypted at rest.
- **LLM path is optional.** Without `ANTHROPIC_API_KEY` everything runs the deterministic planner; the LLM tool-use loop is covered by mocked tests, not by a live-model test suite.
- **One process, no horizontal scale.** In-process TF-IDF index, in-memory singletons and SQLite are chosen for clarity, not throughput.

---

## Documentation
- [Architecture](docs/ARCHITECTURE.md) · [System Design](docs/SYSTEM_DESIGN.md) · [Deployment](docs/DEPLOYMENT.md) · [Demo script](docs/DEMO_SCRIPT.md)
