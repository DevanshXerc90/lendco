# Testing

Run `python -m pytest -q` for the offline suite. It covers the context engine,
diagnosis and slot derivation, RAG, mocked Anthropic tool loops, LCEL parity,
memory evaluation, README metric consistency, and API limits.

Evaluation commands:

- `python -m eval.evaluator` writes `eval/results.json` using the deterministic planner.
- `python -m eval.evaluator --llm` writes `eval/results_llm.json` only when `ANTHROPIC_API_KEY` is set.
- `python -m eval.adversarial` writes `eval/results_adversarial.json` and preserves known failures.
- `python -m eval.lcel_parity` checks 13 citation queries.

Tests do not loosen assertions to improve reported results.
