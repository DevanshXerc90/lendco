"""Record scenario transcripts to docs/sample_calls/*.txt (Sample Call Recordings).

With voice backends enabled these would be audio; here we capture the full
turn-by-turn transcript plus the diagnosis/tool trace and call summary.

Run:  python -m scripts.record_demos
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from core.llm import NullLLM  # noqa: E402
from core.memory import MemoryManager  # noqa: E402
from app import scenarios  # noqa: E402

OUT = settings.ROOT / "docs" / "sample_calls"


def _trace(res):
    tools = ",".join(dict.fromkeys(t["tool"] for t in res.get("tool_trace", []))) or "-"
    extra = f" action={res['action']}" if res.get("action") else ""
    ask = f" asking={res['asking']}" if res.get("asking") else ""
    return f"      [intent={res['intent']} sentiment={res['sentiment']} tools=[{tools}]{ask}{extra}]"


def record_single(name: str, lines: list[str]) -> str:
    import tempfile
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    bid = scenarios.pick_borrower(name)
    orc = Orchestrator(memory=mem, llm=NullLLM())
    start = orc.start(borrower_id=bid)
    lines.append(f"=== SCENARIO: {name}  (borrower {bid}) ===")
    lines.append(f"AGENT   : {start['text']}")
    for utt in scenarios.SCRIPTS[name]:
        lines.append(f"BORROWER: {utt}")
        res = orc.handle(utt)
        lines.append(f"AGENT   : {res['text']}")
        lines.append(_trace(res))
    lines.append("--- SUMMARY ---")
    lines.append(json.dumps(orc.end(), indent=2, default=str))
    return "\n".join(lines)


def record_memory() -> str:
    import tempfile
    mem = MemoryManager(db_path=tempfile.mktemp(suffix=".db"))
    bid = scenarios.pick_borrower("ptp_memory")
    lines = [f"=== SCENARIO: ptp_memory (borrower {bid}) ===", "", "***** CALL 1 *****"]
    orc = Orchestrator(memory=mem, llm=NullLLM())
    lines.append(f"AGENT   : {orc.start(borrower_id=bid)['text']}")
    for utt in scenarios.PTP_MEMORY["call1"]:
        lines.append(f"BORROWER: {utt}")
        res = orc.handle(utt)
        lines.append(f"AGENT   : {res['text']}")
        lines.append(_trace(res))
    lines.append("--- CALL 1 SUMMARY ---")
    lines.append(json.dumps(orc.end(), indent=2, default=str))

    lines += ["", "***** CALL 2 (days later, memory persisted) *****"]
    orc2 = Orchestrator(memory=mem, llm=NullLLM())
    start2 = orc2.start(borrower_id=bid)
    lines.append(f"AGENT   : {start2['text']}   [memory_aware={start2['memory_aware']}]")
    for utt in scenarios.PTP_MEMORY["call2"]:
        lines.append(f"BORROWER: {utt}")
        res = orc2.handle(utt)
        lines.append(f"AGENT   : {res['text']}")
        lines.append(_trace(res))
    lines.append("--- CALL 2 SUMMARY ---")
    lines.append(json.dumps(orc2.end(), indent=2, default=str))
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in scenarios.SCRIPTS:
        (OUT / f"{name}.txt").write_text(record_single(name, []), encoding="utf-8")
        print(f"  wrote docs/sample_calls/{name}.txt")
    (OUT / "ptp_memory.txt").write_text(record_memory(), encoding="utf-8")
    print("  wrote docs/sample_calls/ptp_memory.txt")


if __name__ == "__main__":
    main()
