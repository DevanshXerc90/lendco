# Demo Script (10–15 min video)

A suggested running order. Reset first: `python -m scripts.reset`, and have
`python -m scripts.run_platforms` running in another terminal.

## 0. (1 min) Architecture
Open `docs/ARCHITECTURE.md` — walk the system context + sequence diagram. Emphasise:
5 independent platforms, multi-agent orchestrator, diagnosis-driven questioning, RAG
grounding, memory-based learning.

## 1. (2 min) The data + platforms
```bash
python -m data.generate
curl http://127.0.0.1:8102/gateway-logs?borrower_id=BRW00009   # raw gateway evidence
```
Show 120 borrowers, 1.8k payments with `root_cause`, 22 KB docs.

## 2. (2 min) Context Engine + Diagnosis
```bash
python -c "from core.context_engine import ContextEngine as C; import json; print(json.dumps(C().build(borrower_id='BRW00009').analytics, indent=2))"
```
Show derived analytics (interest/principal split, remaining EMIs). Then show the diagnosis
example: payment_failure → knows the failed record, asks only the reason.

## 3. (3 min) Scenario 4+5 — payment failure → penalty waiver (RAG + tools + action)
```bash
python -m app.cli --scenario waiver
```
Narrate: the agent reads gateway logs, classifies the failure as **system-side**, cites
the **penalty-waiver policy** (RAG), and **auto-initiates a reversal** — distinguishing a
bank glitch from a customer fault. Point at the `[diagnosis]` trace line (intent, tools,
action).

## 4. (3 min) Scenario 6 — follow-up call with MEMORY (the headline)
```bash
python -m app.cli --scenario ptp_memory
```
- **Call 1:** borrower says salary is delayed → agent records the promise-to-pay
  (date/amount/reason) and schedules a callback.
- **Call 2 (new process, memory persisted):** the agent opens with
  *"During our previous conversation, you mentioned … were you able to complete that
  payment?"* — and on "no", smoothly moves to a new commitment.
Show the call summaries: call 1 = 3 turns, call 2 = 1 turn (memory made it faster).

## 5. (2 min) Evaluation + dashboard
```bash
python -m eval.evaluator
python -m dashboard.build_report   # open dashboard/report.html
```
Walk the metrics: intent accuracy, resolution rate, **grounding rate**, redundant-question
rate (0%), **root-cause accuracy**, and the **memory-improvement** delta. Show the
"Agent Memory — learned resolution paths" table.

## 6. (1 min) Production stance
Point at `core/clients.py` + `core/tools.py` as the seams to swap mocks for real systems,
and the graceful-degradation table in the System Design doc.

---

### Other one-liners
```bash
python -m app.cli --scenario remaining     # remaining EMI
python -m app.cli --scenario interest       # interest paid
python -m app.cli --scenario penalty        # penalty inquiry
python -m app.cli --scenario foreclosure    # foreclosure payoff
python -m app.cli --borrower BRW00010        # free-form interactive
python -m scripts.record_demos               # write transcripts to docs/sample_calls/
```
