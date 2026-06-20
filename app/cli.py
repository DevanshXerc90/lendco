"""Inbound voice-agent CLI — the borrower 'calls in'.

Usage:
  python -m app.cli --borrower BRW00010                 # interactive (text or voice)
  python -m app.cli --phone +9188...                    # identify by phone
  python -m app.cli --scenario waiver                   # scripted single-call demo
  python -m app.cli --scenario ptp_memory               # two-call memory demo
  python -m app.cli --scenario remaining --voice        # speak via local STT/TTS

Env STT_BACKEND/TTS_BACKEND control voice; defaults are text/print so it runs
anywhere. The agent uses the LLM if ANTHROPIC_API_KEY is set, else the
deterministic planner.
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.orchestrator import Orchestrator  # noqa: E402
from core.llm import get_llm  # noqa: E402
from app import scenarios  # noqa: E402


def _trace_line(res: dict) -> str:
    tools = ",".join(dict.fromkeys(t["tool"] for t in res.get("tool_trace", []))) or "-"
    bits = [f"intent={res['intent']}", f"sentiment={res['sentiment']}", f"tools=[{tools}]"]
    if res.get("asking"):
        bits.append(f"asking={res['asking']}")
    if res.get("action"):
        bits.append(f"action={res['action']}")
    return "   [diagnosis] " + "  ".join(bits)


def run_interactive(borrower=None, phone=None, voice=False, debug=True):
    from voice.stt import get_stt
    from voice.tts import get_tts
    stt = get_stt() if voice else None
    tts = get_tts() if voice else None
    orc = Orchestrator(llm=get_llm())

    print(f"\n=== LendCo Inbound Voice Agent (LLM: {orc.llm.name}) ===")
    start = orc.start(borrower_id=borrower, phone=phone)
    if not start["found"]:
        print("AGENT  >", start["text"]); return
    _say(start["text"], tts)

    while True:
        utt = stt.listen() if voice else input("BORROWER> ").strip()
        if not utt or utt.lower() in ("bye", "goodbye", "exit", "quit", "that's all", "no thanks"):
            _say("Thank you for calling LendCo. Have a great day!", tts)
            break
        res = orc.handle(utt)
        _say(res["text"], tts)
        if debug:
            print(_trace_line(res))
    print("\n--- CALL SUMMARY ---")
    _pp(orc.end())


def _say(text, tts):
    if tts:
        tts.speak(text)
    else:
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
    orc = Orchestrator(llm=get_llm())
    print(f"\n=== SCENARIO: {name}  (borrower {bid}, LLM: {orc.llm.name}) ===")
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
    orc = Orchestrator(memory=mem, llm=get_llm())
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
    orc2 = Orchestrator(memory=mem, llm=get_llm())
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
    ap = argparse.ArgumentParser(description="LendCo inbound voice agent")
    ap.add_argument("--borrower", help="borrower id, e.g. BRW00010")
    ap.add_argument("--phone", help="registered phone, e.g. +9188...")
    ap.add_argument("--scenario", help="scripted demo: " + ", ".join(list(scenarios.SCRIPTS) + ["ptp_memory"]))
    ap.add_argument("--voice", action="store_true", help="use local STT/TTS (set STT_BACKEND/TTS_BACKEND)")
    ap.add_argument("--quiet", action="store_true", help="hide the diagnosis/tool trace")
    args = ap.parse_args()

    if args.scenario:
        run_scenario(args.scenario, debug=not args.quiet)
    else:
        run_interactive(borrower=args.borrower, phone=args.phone, voice=args.voice, debug=not args.quiet)


if __name__ == "__main__":
    main()
