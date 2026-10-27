"""Render measured evaluator values into the README results table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    adversarial = json.loads(
        (ROOT / "eval" / "results_adversarial.json").read_text(encoding="utf-8")
    )
    overall = results["scenarios"]["overall"]
    root = results["root_cause"]
    dev = results["intents"]
    heldout = results["intent_heldout"]
    baseline = results["baseline"]
    values = {
        "Dev intent accuracy": (
            f"{dev['intent_accuracy']:.1%} "
            f"({round(dev['intent_accuracy'] * dev['samples'])}/{dev['samples']})"
        ),
        "Held-out intent accuracy": (
            f"{heldout['intent_accuracy']:.1%} "
            f"({round(heldout['intent_accuracy'] * heldout['samples'])}/{heldout['samples']})"
        ),
        "Policy grounding rate": (
            f"{overall['grounding_rate']:.1%} "
            f"({round(overall['grounding_rate'] * overall['total_runs'])}/{overall['total_runs']})"
        ),
        "Resolved without escalation": (
            f"{overall['resolved_without_escalation']:.1%} "
            f"({round(overall['resolved_without_escalation'] * overall['total_runs'])}/{overall['total_runs']})"
        ),
        "Escalated by design": (
            f"{overall['escalated_by_design']:.1%} "
            f"({round(overall['escalated_by_design'] * overall['total_runs'])}/{overall['total_runs']})"
        ),
        "Root-cause accuracy": (
            f"{root['root_cause_accuracy']:.1%} "
            f"({round(root['root_cause_accuracy'] * root['samples'])}/{root['samples']})"
        ),
        "Average turns / call": f"{overall['avg_turns']:.2f}",
        "Naive baseline turns / call": f"{baseline['scenario_avg_turns']:.2f}",
        "Adversarial cases passed": f"{adversarial['passed']} of {adversarial['cases']}",
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    rendered = []
    for line in readme:
        for label, value in values.items():
            prefix = f"| {label} |"
            if line.startswith(prefix):
                line = f"| {label} | {value} |"
        rendered.append(line)
    (ROOT / "README.md").write_text("\n".join(rendered) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
