# LendCo

LendCo is an offline-first borrower-support voice-agent prototype: Python modules are coordinated by a deterministic orchestrator, with an Anthropic tool-use loop available in the voice and agent path. Five mock FastAPI services use JSON files, and policy replies use an in-process TF-IDF index.

![Demo placeholder](docs/demo.gif)

## Results

These values are read from [`eval/results.json`](eval/results.json), produced by `python -m eval.evaluator` (deterministic planner). The summary test recomputes this table from that artifact.

| Metric | Result | Reproduce |
|---|---:|---:|---|
| Intent accuracy | 100.0% (30/30) | `python -m eval.evaluator` |
| Policy grounding rate | 100.0% (64/64) | `python -m eval.evaluator` |
| Resolved without escalation | 87.5% (56/64) | `python -m eval.evaluator` |
| Escalated by design | 12.5% (8/64) | `python -m eval.evaluator` |
| Redundant question rate | 0.0% (0) | `python -m eval.evaluator` |
| Root-cause accuracy | 100.0% (40/40) | `python -m eval.evaluator` |
| Average turns / call | 1.25 (weighted 80 turns / 64 calls) | `python -m eval.evaluator` |
| Memory turns | 3.0 -> 1.0 (20 borrowers) | `python -m eval.evaluator` |
| Memory-aware opener share | 100.0% (20/20) | `python -m eval.evaluator` |
| LCEL citation parity | 100.0% (13/13) | `python -m eval.lcel_parity` |
| Adversarial cases | 26 of 30 passed | `python -m eval.adversarial` |

All results come from the deterministic planner. The Anthropic tool use loop is covered by tests
with a mocked client and has not been evaluated live. Run
`python -m eval.evaluator --llm` with an API key to measure it.

`Average turns / call` is the weighted overall value in the JSON (`total turns / total runs`). The unweighted mean of the eight scenario averages is different; the earlier discrepancy came from a stale scenario table, including an outdated settlement row. The scenario table in the generated artifact is authoritative.

The 0% redundant-question rate holds by construction: derivable slots are never asked. The 3-to-1 reduction comes from a scripted first call plus stored slots. Both are design properties covered by unit tests, not learned improvement.

### Adversarial known failures

The adversarial runner reports failures in [`eval/results_adversarial.json`](eval/results_adversarial.json) rather than hiding them. It covers 30 misspellings/paraphrases, missing data, conflicting services, outages, unknown policies, and grounding attacks. The deterministic keyword planner is not tuned only to pass this set; ambiguous and novel language can still be misclassified.

| Result | Count |
|---|---:|
| Cases passed | 26 |
| Known failures | 4 |

Known failures are four misspelled/paraphrased intent cases recorded in the
JSON artifact. This is a result of the current scripted inputs, not evidence
that arbitrary language or service failures are solved.

The labelled intent set contains 30 cases and the root-cause set runs 40 samples. These are generated from the reproducible synthetic data created by `python -m data.generate`.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m data.generate
python -m eval.evaluator
python -m eval.adversarial
python -m pytest -q
```

For the local typed demo:

```powershell
python -m app.cli --scenario ptp_memory
```

Copy `.env.example` to `.env` for optional Anthropic, voice, or service settings. To measure Anthropic separately, set `ANTHROPIC_API_KEY` and run `python -m eval.evaluator --llm`; never put that key in the repository.

## Security and deployment

Open demo mode is only for local development. In Vercel/production, set `API_SECRET` and `DEMO_API_KEY`; `/api/chat` and `/api/scenario` require `X-API-Key`, have a 16 KiB body limit, and use an in-memory per-IP rate limit. Set `CORS_ORIGINS` to the exact browser origin(s).

Required Vercel variables: `API_SECRET`, `DEMO_API_KEY`, and `CORS_ORIGINS`. Optional: `ANTHROPIC_API_KEY`, `LLM_MODEL`, `CRM_URL`, `PAYMENTS_URL`, `SUPPORT_URL`, `KNOWLEDGE_URL`, `WORKFLOW_URL`, `EMBEDDINGS_BACKEND`.

See [deployment](docs/DEPLOYMENT.md), [API](docs/API.md), [testing](docs/TESTING.md), and [agents](docs/AGENTS.md).

## Known limitations

- Retrieval is an in-process TF-IDF index; it is not a hosted vector database.
- Voice uses an optional `faster-whisper` path; typed input is the default.
- The five platforms are mock FastAPI services backed by JSON files.
- The LLM tool loop is covered by mocked Anthropic-client tests, not a live-service guarantee.
- Production still needs identity verification, PII controls, durable shared rate limiting, and authenticated real integrations.

## License

MIT; see [LICENSE](LICENSE).
