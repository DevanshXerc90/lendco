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


def _fresh_orc(mem, llm=None):
    return Orchestrator(memory=mem, llm=llm or NullLLM())


def eval_intents() -> dict:
    correct = sum(1 for u, exp in LABELLED if classify_intent(u)[0] == exp)
    return {"intent_accuracy": round(correct / len(LABELLED), 3), "labelled_n": len(LABELLED)}


def eval_scenarios(n_borrowers: int = 8, llm=None) -> dict:
    borrowers = json.loads((settings.DATA_DIR / "borrowers.json").read_text(encoding="utf-8"))
    per_scenario = defaultdict(lambda: {"runs": 0, "resolved": 0, "grounded": 0, "turns": 0,
                                        "intent_ok": 0, "tools": defaultdict(int)})
    redundant_turns = total_turns = 0

    payments = json.loads((settings.DATA_DIR / "payments.json").read_text(encoding="utf-8"))
    borrowers_with_failures = {p["borrower_id"] for p in payments if p["status"] != "success"}

    for name, script in scenarios.SCRIPTS.items():
        expected = {"remaining": "remaining_emi", "interest": "interest_paid", "penalty": "penalty_inquiry",
                    "payment_failure": "payment_failure", "waiver": "penalty_waiver",
                    "foreclosure": "foreclosure", "settlement": "settlement_request",
                    "account": "account_info"}.get(name)
        if name == "payment_failure":
            sample = [b for b in borrowers if b["borrower_id"] in borrowers_with_failures][:n_borrowers]
        else:
            sample = borrowers[:n_borrowers]
        for b in sample:
            mem = _tmp_mem()
            orc = _fresh_orc(mem, llm=llm)
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
            makes_policy_statement = (
                last["intent"] in POLICY_INTENTS
                and "don't see a failed payment" not in (last["text"] or "").lower()
            ) if last else False
            if last and (not makes_policy_statement or CITE_RE.search(last["text"] or "")):
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


def eval_root_cause(n: int = 25, llm=None) -> dict:
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
        orc = _fresh_orc(mem, llm=llm)
        orc.start(borrower_id=b)
        res = orc.handle("My payment failed, what happened?")
        text = (res["text"] or "").lower()
        predicted = "system" if ("bank/gateway" in text or "not anything wrong" in text or "system-side" in text) else "customer"
        correct += 1 if predicted == truth else 0
        total += 1
        if total >= n:
            break
    return {"root_cause_accuracy": round(correct / (total or 1), 3), "samples": total}


def eval_memory(n_borrowers: int = 20, llm=None) -> dict:
    """Continuous learning: a repeat interaction should be faster + memory-aware across borrowers."""
    borrowers = json.loads((settings.DATA_DIR / "borrowers.json").read_text(encoding="utf-8"))
    sample = borrowers[:n_borrowers]

    c1_turns = []
    c2_turns = []
    mem_aware = []
    last_best = None

    for b in sample:
        mem = _tmp_mem()
        orc1 = _fresh_orc(mem, llm=llm)
        orc1.start(borrower_id=b["borrower_id"])
        for utt in scenarios.PTP_MEMORY["call1"]:
            orc1.handle(utt)
        s1 = orc1.end()
        c1_turns.append(s1["turns"])

        orc2 = _fresh_orc(mem, llm=llm)
        start2 = orc2.start(borrower_id=b["borrower_id"])
        mem_aware.append(bool(start2.get("memory_aware")))
        for utt in scenarios.PTP_MEMORY["call2"]:
            orc2.handle(utt)
        s2 = orc2.end()
        c2_turns.append(s2["turns"])

        last_best = mem.best_resolution_path("promise_to_pay")

    mean_c1 = round(sum(c1_turns) / len(c1_turns), 2)
    mean_c2 = round(sum(c2_turns) / len(c2_turns), 2)
    share_aware = round(sum(mem_aware) / len(mem_aware), 3)

    return {
        "borrowers_tested": len(sample),
        "call1_turns": mean_c1,
        "call2_turns": mean_c2,
        "call1_mean": mean_c1,
        "call1_min": min(c1_turns),
        "call1_max": max(c1_turns),
        "call2_mean": mean_c2,
        "call2_min": min(c2_turns),
        "call2_max": max(c2_turns),
        "share_memory_aware": share_aware,
        "call2_memory_aware": all(mem_aware),
        "turns_saved": round(mean_c1 - mean_c2, 2),
        "learned_resolution_path": last_best,
    }



def run_all(use_llm: bool = False) -> dict:
    if use_llm:
        if not settings.ANTHROPIC_API_KEY:
            print("[ERROR] ANTHROPIC_API_KEY is not set. Refusing to run LLM evaluation without an API key.", file=sys.stderr)
            sys.exit(1)
        from core.llm import AnthropicLLM
        llm = AnthropicLLM()
        out_path = settings.ROOT / "eval" / "results_llm.json"
    else:
        llm = NullLLM()
        out_path = settings.ROOT / "eval" / "results.json"

    results = {
        "intents": eval_intents(),
        "scenarios": eval_scenarios(llm=llm),
        "root_cause": eval_root_cause(llm=llm),
        "memory_learning": eval_memory(llm=llm),
    }
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agent performance")
    parser.add_argument("--llm", action="store_true", help="Run evaluation with real AnthropicLLM (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    r = run_all(use_llm=args.llm)
    o = r["scenarios"]["overall"]
    dest_file = "eval/results_llm.json" if args.llm else "eval/results.json"
    print("=== EVALUATION RESULTS ===")
    print(f"intent accuracy        : {r['intents']['intent_accuracy']}")
    print(f"resolution rate        : {o['resolution_rate']}")
    print(f"grounding rate         : {o['grounding_rate']}")
    print(f"avg turns / call       : {o['avg_turns']}")
    print(f"redundant question rate: {o['redundant_question_rate']}")
    print(f"root-cause accuracy    : {r['root_cause']['root_cause_accuracy']} (n={r['root_cause']['samples']})")
    m = r["memory_learning"]
    print(f"memory (n={m.get('borrowers_tested', 1)}): call1 mean={m['call1_mean']} (min={m['call1_min']}, max={m['call1_max']}) -> "
          f"call2 mean={m['call2_mean']} (min={m['call2_min']}, max={m['call2_max']}), "
          f"share_memory_aware={m['share_memory_aware']*100:.1f}%, saved={m['turns_saved']}")
    print(f"\nwrote {dest_file}")


