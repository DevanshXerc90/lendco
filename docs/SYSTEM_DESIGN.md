# System Design Document

## 1. Goal & non-goals
**Goal:** an inbound voice agent that resolves borrower loan-servicing requests like a
knowledgeable human rep — understanding context, retrieving across systems, diagnosing
gaps, conducting dynamic conversation, using tools, and learning across calls.

**Non-goals:** real telephony/PSTN integration (the voice layer is local + pluggable);
production-grade auth/PII handling (modelled but not hardened); a real LLM is optional.

## 2. Data model (synthetic, reproducible — `data/generate.py`, seed=42)
- **Borrowers (120):** id, name, phone, loan id/type/amount, interest rate, EMI, tenure,
  start/end, delinquency status + DPD, auto-debit flag, preferred language, KYC block.
- **Payments (1,813):** installment-level rows with status
  (`success|failed|partial|missed|auto_debit_failed`), method, gateway, **response code**,
  **failure_reason**, **root_cause** (`system|customer`), penalty charged.
- **Tickets (130):** category, subject, status, priority, channel, resolution.
- **Conversations (130):** transcript, intent, sentiment, resolved, and **promise-to-pay**
  records that seed memory.
- **Knowledge base (22 docs):** loan FAQs, foreclosure, settlement, late-payment, penalty
  waiver, payment-failure handling, NACH, interest calc, hardship, PTP, privacy, etc.

The `root_cause` field is the backbone of the payment-failure and waiver scenarios: it
lets the agent distinguish a bank/gateway glitch (not the borrower's fault → waiver) from
a customer-side decline (insufficient funds → not waivable).

## 3. Five independent platforms
Each is a standalone FastAPI service with its own port and data slice — satisfying
"5 independent business platforms" while remaining free and offline.

| Platform | Emulates | Key endpoints |
|---|---|---|
| CRM `:8101` | Zoho/HubSpot | `GET /borrowers/{id}`, `?phone=`, `/kyc`, `PATCH /borrowers/{id}` |
| Payments `:8102` | Razorpay/Stripe | `GET /payments`, `/gateway-logs`, `POST /payment-links` |
| Support `:8103` | Freshdesk | `GET/POST /tickets` |
| Knowledge `:8104` | Notion/Confluence | `GET /documents`, `/search` |
| Workflow `:8105` | n8n/Make | `POST /callbacks`, `/escalations`, `/workflows/{name}/trigger` |

Writes persist to `data/runtime/` so the dashboard and memory can observe side-effects.

## 4. Context Engine
Produces a single `UnifiedContext` object with three sub-contexts:
- **Borrower:** profile, KYC, full payment history.
- **Interaction:** prior conversations + prior commitments (from seeded history *and* live memory).
- **Operational:** delinquency status/DPD, open tickets, recent failures.

It also derives analytics by walking the amortization schedule for the number of
installments actually paid: `interest = outstanding × monthly_rate`, principal = `EMI −
interest`. This yields interest-paid, principal-paid, outstanding, EMIs remaining, next
due date, overdue amount and total penalties — the numbers every scenario needs.

## 5. Diagnosis Layer
Slot-based gap analysis per intent:
- Each intent declares slots; each slot is **derivable** (a resolver pulls it from context)
  or **ask-only** (only the borrower can answer).
- Known = resolvable slots; Unknown = the rest. Questions are generated **only** for
  missing ask-only slots, ranked by priority and filtered against `already_asked`
  (de-duplication / no redundant questioning).
- Emits `kb_query` (for RAG grounding) and `suggested_tools` (for the agent).

Example (spec's own example): EMI overdue + auto-debit failed are *derived* (known); the
*reason* is ask-only and missing → question: "Could you tell me what you saw when the
payment failed…". For `remaining_emi`, everything is derivable → zero questions.

## 6. Voice Agent (reasoning)
Two interchangeable paths, identical tools:
- **LLM path** (`ANTHROPIC_API_KEY` set): a tool-use loop. System prompt carries the
  context summary, memory recall, persona and diagnosis hints; the model reasons, calls
  tools, and speaks. Prompt caching on the stable system+tools blocks.
- **Deterministic planner** (offline): runs the relevant read tools, retrieves policy via
  RAG, takes the correct action, and composes a grounded, cited answer per intent.

Action policy examples:
- *Payment failure:* read gateway logs → classify system vs customer → reassure / explain →
  generate payment link.
- *Penalty waiver:* if root cause = system → eligible → trigger reversal workflow; else →
  create a high-priority ticket for human review (human-in-the-loop).
- *Promise-to-pay:* record commitment (date/amount/reason) → schedule callback.

## 7. Memory & continuous learning
Three tiers in SQLite:
- **Borrower memory:** preferences, communication style, FAQs, commitments.
- **Conversation memory:** Q&A pairs, asked-slots, outstanding issues, resolution history.
- **Agent memory:** per-intent resolution-path success/fail counts and avg turns →
  `best_resolution_path(intent)`.

`recall()` runs before every call. Concretely, the second interaction is:
- **Faster** — previously-asked slots are skipped; known root cause avoids re-discovery.
- **More personalized** — language/style + a memory-aware opener referencing commitments.
- **More accurate** — verified failure causes and prior resolutions are reused.
- **More outcome-oriented** — the agent steers toward the path that historically resolves.

Measured by the evaluator (`memory_learning`: call-1 vs call-2 turns, memory-aware flag).

## 8. Failure modes & resilience
| Failure | Behaviour |
|---|---|
| A platform service down | Client falls back to local data for reads; writes surface a tool error, agent continues. |
| No `ANTHROPIC_API_KEY` | Deterministic planner (same tools, template phrasing). |
| No voice deps | Text STT/TTS. |
| Unknown borrower | Agent asks for phone/loan id instead of crashing. |
| Tool exception | Caught in `tools.execute`, returned as `{error}`, never crashes the loop. |

## 9. Evaluation
`eval/evaluator.py` scores intent accuracy, resolution rate, grounding rate, avg turns,
redundant-question rate, root-cause classification accuracy, and the memory-improvement
delta; writes `eval/results.json` rendered by the dashboard.

## 10. Production hardening (next steps)
Real STT/TTS streaming + barge-in; identity verification before disclosure; PII redaction
in logs; vector DB (pgvector) + reranker for RAG; durable queue for workflow triggers;
per-tool authz; observability (tracing every tool call); A/B prompts measured by agent
memory; supervisor approval UI for waivers/settlements.
