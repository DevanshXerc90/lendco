"""Agent evaluation framework.

Runs the agent across many borrowers and scores it on the dimensions that
matter for this assignment, then writes eval/results.json (consumed by the
dashboard). Metrics:

  intent_accuracy        : classifier vs labelled utterances
  grounding_rate         : % of policy answers that cite a KB document
  resolution_rate        : % of scenario calls resolved without escalation
  avg_turns              : conversational efficiency
  redundant_question_rate: % of turns that re-ask an already-answered slot
  root_cause_accuracy    : payment-failure system/customer classification correct
  memory_improvement     : turns saved on a repeat interaction (continuous learning)

Run:  python -m eval.evaluator
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from core.diagnosis import classify_intent  # noqa: E402
from core.memory import MemoryManager  # noqa: E402
from core.llm import NullLLM  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from app import scenarios  # noqa: E402

CITE_RE = re.compile(r"\d{2}_[a-z_]+")  # KB doc-id pattern, e.g. 05_penalty_waiver_policy

LABELLED = [
    ("how many emis are remaining", "remaining_emi"),
    ("how many installments are left on my loan", "remaining_emi"),
    ("how much interest have i paid so far", "interest_paid"),
    ("why was a penalty charged on my account", "penalty_inquiry"),
    ("my payment failed even though i had enough balance", "payment_failure"),
    ("my auto debit failed", "payment_failure"),
    ("can my penalty be waived because the bank failed", "penalty_waiver"),
    ("please reverse the penalty", "penalty_waiver"),
    ("my salary is delayed i will pay next friday", "promise_to_pay"),
    ("i want a one time settlement", "settlement_request"),
    ("i want to foreclose my loan", "foreclosure"),
    ("give me a summary of my account", "account_info"),
]

POLICY_INTENTS = {"penalty_inquiry", "penalty_waiver", "payment_failure", "settlement_request",
                  "foreclosure", "interest_paid"}


def _tmp_mem() -> MemoryManager:
    # a fresh, isolated on-disk DB (":memory:" won't persist across our per-op connections)
    return MemoryManager(db_path=tempfile.mktemp(suffix=".db"))


def _fresh_orc(mem):
    return Orchestrator(memory=mem, llm=NullLLM())


def eval_intents() -> dict:
    correct = sum(1 for u, exp in LABELLED if classify_intent(u)[0] == exp)
    return {"intent_accuracy": round(correct / len(LABELLED), 3), "labelled_n": len(LABELLED)}


def eval_scenarios(n_borrowers: int = 8) -> dict:
    borrowers = json.loads((settings.DATA_DIR / "borrowers.json").read_text(encoding="utf-8"))
    per_scenario = defaultdict(lambda: {"runs": 0, "resolved": 0, "grounded": 0, "turns": 0,
                                        "intent_ok": 0, "tools": defaultdict(int)})
    redundant_turns = total_turns = 0

    for name, script in scenarios.SCRIPTS.items():
        expected = {"remaining": "remaining_emi", "interest": "interest_paid", "penalty": "penalty_inquiry",
                    "payment_failure": "payment_failure", "waiver": "penalty_waiver",
                    "foreclosure": "foreclosure", "settlement": "settlement_request",
                    "account": "account_info"}.get(name)
        sample = borrowers[:n_borrowers]
        for b in sample:
            mem = _tmp_mem()
            orc = _fresh_orc(mem)
            orc.start(borrower_id=b["borrower_id"])
            last = None
            asked = set()
            for utt in script:
                res = orc.handle(utt)
                last = res
                total_turns += 1
                if res.get("asking"):
                    if res["asking"] in asked:
                        redundant_turns += 1
                    asked.add(res["asking"])
            summary = orc.end()
            s = per_scenario[name]
            s["runs"] += 1
            s["resolved"] += 1 if summary["resolved"] else 0
            s["turns"] += summary["turns"]
            if expected and last and last["intent"] == expected:
                s["intent_ok"] += 1
            if last and (last["intent"] not in POLICY_INTENTS or CITE_RE.search(last["text"] or "")):
                s["grounded"] += 1
            for t in summary["tools_used"]:
                s["tools"][t] += 1

    # finalize per-scenario
    out = {}
    tot_runs = tot_res = tot_ground = tot_intent = 0
    sum_turns = 0
    for name, s in per_scenario.items():
        runs = s["runs"] or 1
        out[name] = {
            "runs": s["runs"],
            "resolution_rate": round(s["resolved"] / runs, 3),
            "grounding_rate": round(s["grounded"] / runs, 3),
            "intent_accuracy": round(s["intent_ok"] / runs, 3),
            "avg_turns": round(s["turns"] / runs, 2),
            "top_tools": sorted(s["tools"], key=s["tools"].get, reverse=True)[:4],
        }
        tot_runs += s["runs"]; tot_res += s["resolved"]; tot_ground += s["grounded"]
        tot_intent += s["intent_ok"]; sum_turns += s["turns"]
    overall = {
        "total_runs": tot_runs,
        "resolution_rate": round(tot_res / (tot_runs or 1), 3),
        "grounding_rate": round(tot_ground / (tot_runs or 1), 3),
        "avg_turns": round(sum_turns / (tot_runs or 1), 2),
        "redundant_question_rate": round(redundant_turns / (total_turns or 1), 3),
    }
    return {"overall": overall, "per_scenario": out}


def eval_root_cause(n: int = 25) -> dict:
    """Does the agent's failure classification match the ground-truth root cause?"""
    payments = json.loads((settings.DATA_DIR / "payments.json").read_text(encoding="utf-8"))
    byb = defaultdict(list)
    for p in payments:
        byb[p["borrower_id"]].append(p)
    correct = total = 0
    for b, ps in list(byb.items()):
        fails = [p for p in ps if p["status"] != "success"]
        if not fails:
            continue
        truth = fails[-1]["root_cause"]
        mem = _tmp_mem()
        orc = _fresh_orc(mem)
        orc.start(borrower_id=b)
        res = orc.handle("My payment failed, what happened?")
        text = (res["text"] or "").lower()
        predicted = "system" if ("bank/gateway" in text or "not anything wrong" in text or "system-side" in text) else "customer"
        correct += 1 if predicted == truth else 0
        total += 1
        if total >= n:
            break
    return {"root_cause_accuracy": round(correct / (total or 1), 3), "samples": total}


def eval_memory() -> dict:
    """Continuous learning: a repeat interaction should be faster + memory-aware."""
    mem = _tmp_mem()
    bid = scenarios.pick_borrower("ptp_memory")

    orc1 = _fresh_orc(mem)
    orc1.start(borrower_id=bid)
    for utt in scenarios.PTP_MEMORY["call1"]:
        orc1.handle(utt)
    s1 = orc1.end()

    orc2 = _fresh_orc(mem)
    start2 = orc2.start(borrower_id=bid)
    for utt in scenarios.PTP_MEMORY["call2"]:
        orc2.handle(utt)
    s2 = orc2.end()

    best = mem.best_resolution_path("promise_to_pay")
    return {
        "call1_turns": s1["turns"],
        "call2_memory_aware": start2["memory_aware"],
        "call2_turns": s2["turns"],
        "turns_saved": s1["turns"] - s2["turns"],
        "learned_resolution_path": best,
    }


def run_all() -> dict:
    results = {
        "intents": eval_intents(),
        "scenarios": eval_scenarios(),
        "root_cause": eval_root_cause(),
        "memory_learning": eval_memory(),
    }
    out_path = settings.ROOT / "eval" / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    r = run_all()
    o = r["scenarios"]["overall"]
    print("=== EVALUATION RESULTS ===")
    print(f"intent accuracy        : {r['intents']['intent_accuracy']}")
    print(f"resolution rate        : {o['resolution_rate']}")
    print(f"grounding rate         : {o['grounding_rate']}")
    print(f"avg turns / call       : {o['avg_turns']}")
    print(f"redundant question rate: {o['redundant_question_rate']}")
    print(f"root-cause accuracy    : {r['root_cause']['root_cause_accuracy']} (n={r['root_cause']['samples']})")
    m = r["memory_learning"]
    print(f"memory: call1={m['call1_turns']} turns -> call2={m['call2_turns']} turns "
          f"(saved {m['turns_saved']}, memory_aware={m['call2_memory_aware']})")
    print("\nwrote eval/results.json")
