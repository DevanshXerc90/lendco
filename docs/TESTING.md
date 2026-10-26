# Testing

Run `python -m pytest -q` for the offline suite. It covers the context engine,
diagnosis and slot derivation, RAG, memory evaluation, README metric consistency,
and API limits.

Evaluation commands:

- `python -m eval.evaluator` writes `eval/results.json` using the deterministic planner.
- `python -m eval.adversarial` writes `eval/results_adversarial.json` and preserves known failures.

Tests do not loosen assertions to improve reported results.
