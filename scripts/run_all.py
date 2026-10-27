"""Run the complete offline build, evaluation, demo, and test flow."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


def run(module: str, *args: str) -> None:
    env = os.environ.copy()
    env.update({"CRM_BACKEND": "mock", "PAYMENTS_BACKEND": "mock", "LENDCO_LOCAL_ONLY": "1"})
    subprocess.run([sys.executable, "-m", module, *args], check=True, env=env)


def main() -> None:
    run("data.generate")
    run("eval.evaluator")
    run("eval.adversarial")
    run("scripts.update_readme")
    run("app.cli", "--scenario", "waiver", "--quiet")
    run("pytest", "-q")
    results = json.loads((Path(__file__).resolve().parents[1] / "eval" / "results.json").read_text())
    adversarial = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "results_adversarial.json").read_text()
    )
    o = results["scenarios"]["overall"]
    rows = [
        ("Dev intent accuracy", results["intents"]["intent_accuracy"], results["intents"]["samples"]),
        ("Held-out intent accuracy", results["intent_heldout"]["intent_accuracy"],
         results["intent_heldout"]["samples"]),
        ("Policy grounding rate", o["grounding_rate"], o["total_runs"]),
        ("Resolved without escalation", o["resolved_without_escalation"], o["total_runs"]),
        ("Escalated by design", o["escalated_by_design"], o["total_runs"]),
        ("Root-cause accuracy", results["root_cause"]["root_cause_accuracy"],
         results["root_cause"]["samples"]),
    ]
    print("\n=== FINAL RESULTS TABLE ===")
    print("| Metric | Result |")
    print("|---|---:|")
    for label, rate, samples in rows:
        print(f"| {label} | {rate:.1%} ({round(rate * samples)}/{samples}) |")
    print(f"| Average turns / call | {o['avg_turns']:.2f} |")
    print(f"| Naive baseline turns / call | {results['baseline']['scenario_avg_turns']:.2f} |")
    print(f"| Adversarial cases passed | {adversarial['passed']} of {adversarial['cases']} |")


if __name__ == "__main__":
    main()
