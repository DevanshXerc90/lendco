"""Inbound borrower-support CLI.

Usage:
  python -m app.cli --borrower BRW00010                 # interactive
  python -m app.cli --phone +9188...                    # identify by phone
  python -m app.cli --scenario waiver                   # scripted single-call demo
  python -m app.cli --scenario ptp_memory               # two-call memory demo
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.orchestrator import Orchestrator  # noqa: E402
from config import settings  # noqa: E402
from app import scenarios  # noqa: E402


def _trace_line(res: dict) -> str:
    tools = ",".join(dict.fromkeys(t["tool"] for t in res.get("tool_trace", []))) or "-"
    bits = [f"intent={res['intent']}", f"sentiment={res['sentiment']}", f"tools=[{tools}]"]
    if res.get("asking"):
        bits.append(f"asking={res['asking']}")
    if res.get("action"):
        bits.append(f"action={res['action']}")
    return "   [diagnosis] " + "  ".join(bits)


def run_interactive(borrower=None, phone=None, debug=True):
    orc = Orchestrator()

    print("\n=== LendCo Inbound Borrower Support ===")
    start = orc.start(borrower_id=borrower, phone=phone)
    if not start["found"]:
        print("AGENT  >", start["text"]); return
    _say(start["text"])

    while True:
        utt = input("BORROWER> ").strip()
        if not utt or utt.lower() in ("bye", "goodbye", "exit", "quit", "that's all", "no thanks"):
            _say("Thank you for calling LendCo. Have a great day!")
            break
        res = orc.handle(utt)
        _say(res["text"])
        if debug:
            print(_trace_line(res))
    print("\n--- CALL SUMMARY ---")
    _pp(orc.end())


def _say(text):
    print(f"\nAGENT  > {text}\n")


def _pp(obj):
    import json
    print(json.dumps(obj, indent=2, default=str))


def run_scenario(name, debug=True):
    if name in ("ptp_memory", "memory"):
        return run_ptp_memory(debug)
    bid = scenarios.pick_borrower(name)
    script = scenarios.SCRIPTS.get(name)
    if not script:
        print(f"unknown scenario '{name}'. options: {list(scenarios.SCRIPTS) + ['ptp_memory']}"); return
    orc = Orchestrator()
    print(f"\n=== SCENARIO: {name}  (borrower {bid}) ===")
    print("AGENT  >", orc.start(borrower_id=bid)["text"], "\n")
    for utt in script:
        print("BORROWER>", utt)
        res = orc.handle(utt)
        print("AGENT  >", res["text"])
        if debug:
            print(_trace_line(res))
        print()
    print("--- CALL SUMMARY ---"); _pp(orc.end())


def run_ptp_memory(debug=True):
    from core.memory import get_memory
    mem = get_memory()
    bid = scenarios.pick_borrower("ptp_memory")
    print(f"\n=== SCENARIO: ptp_memory (borrower {bid}) ===")
    print("\n********** CALL 1: borrower makes a promise-to-pay **********")
    orc = Orchestrator(memory=mem)
    print("AGENT  >", orc.start(borrower_id=bid)["text"], "\n")
    for utt in scenarios.PTP_MEMORY["call1"]:
        print("BORROWER>", utt)
        res = orc.handle(utt)
        print("AGENT  >", res["text"])
        if debug:
            print(_trace_line(res))
        print()
    print("--- CALL 1 SUMMARY ---"); _pp(orc.end())

    print("\n********** CALL 2: days later - the agent REMEMBERS **********")
    orc2 = Orchestrator(memory=mem)
    start = orc2.start(borrower_id=bid)
    print("AGENT  >", start["text"], f"   [memory_aware={start['memory_aware']}]\n")
    for utt in scenarios.PTP_MEMORY["call2"]:
        print("BORROWER>", utt)
        res = orc2.handle(utt)
        print("AGENT  >", res["text"])
        if debug:
            print(_trace_line(res))
        print()
    print("--- CALL 2 SUMMARY ---"); _pp(orc2.end())


def main():
    settings.validate()  # clear startup error instead of a confusing failure later
    ap = argparse.ArgumentParser(description="LendCo inbound borrower support")
    ap.add_argument("--borrower", help="borrower id, e.g. BRW00010")
    ap.add_argument("--phone", help="registered phone, e.g. +9188...")
    ap.add_argument("--scenario", help="scripted demo: " + ", ".join(list(scenarios.SCRIPTS) + ["ptp_memory"]))
    ap.add_argument("--quiet", action="store_true", help="hide the diagnosis/tool trace")
    args = ap.parse_args()

    if args.scenario:
        run_scenario(args.scenario, debug=not args.quiet)
    else:
        run_interactive(borrower=args.borrower, phone=args.phone, debug=not args.quiet)


if __name__ == "__main__":
    main()
