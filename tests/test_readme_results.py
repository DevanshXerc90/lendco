from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_summary_is_recomputed_from_results():
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    adversarial = json.loads(
        (ROOT / "eval" / "results_adversarial.json").read_text(encoding="utf-8")
    )
    overall = results["scenarios"]["overall"]
    intents = results["intents"]
    heldout = results["intent_heldout"]
    root = results["root_cause"]
    baseline = results["baseline"]
    readme = _readme()
    per_scenario = results["scenarios"]["per_scenario"]
    scenario_turns = sum(
        scenario["runs"] * scenario["avg_turns"] for scenario in per_scenario.values()
    )
    overall_turns = overall["total_runs"] * overall["avg_turns"]
    assert scenario_turns == overall_turns

    checks = {
        "Dev intent accuracy": f"{intents['intent_accuracy']:.1%} ({round(intents['intent_accuracy'] * intents['samples'])}/{intents['samples']})",
        "Held-out intent accuracy": f"{heldout['intent_accuracy']:.1%} ({round(heldout['intent_accuracy'] * heldout['samples'])}/{heldout['samples']})",
        "Policy grounding rate": f"{overall['grounding_rate']:.1%} ({round(overall['grounding_rate'] * overall['total_runs'])}/{overall['total_runs']})",
        "Resolved without escalation": f"{overall['resolved_without_escalation']:.1%} ({round(overall['resolved_without_escalation'] * overall['total_runs'])}/{overall['total_runs']})",
        "Escalated by design": f"{overall['escalated_by_design']:.1%} ({round(overall['escalated_by_design'] * overall['total_runs'])}/{overall['total_runs']})",
        "Root-cause accuracy": f"{root['root_cause_accuracy']:.1%} ({round(root['root_cause_accuracy'] * root['samples'])}/{root['samples']})",
        "Average turns / call": f"{overall['avg_turns']:.2f}",
        "Naive baseline turns / call": f"{baseline['scenario_avg_turns']:.2f}",
        "Adversarial cases passed": f"{adversarial['passed']} of {adversarial['cases']}",
    }
    for label, value in checks.items():
        row = next(line for line in readme.splitlines() if line.startswith(f"| {label} |"))
        assert value in row, (label, value, row)

    header = next(line for line in readme.splitlines() if line.startswith("| Metric |"))
    assert header == "| Metric | Result |"
    assert "not run" not in readme.lower()


def test_readme_has_no_resume_claim_language():
    text = _readme().lower()
    assert "resume" not in text
    assert "claim 1" not in text
    assert "verified" not in text
    assert "zero hallucinations" not in text
