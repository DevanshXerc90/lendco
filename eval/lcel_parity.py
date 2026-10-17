"""LCEL Parity Check — verifies identical citations across original and LCEL Knowledge Agents.

Run: python -m eval.lcel_parity
Exits with 0 on parity, 1 on mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.knowledge_agent import KnowledgeAgent as OriginalKnowledgeAgent  # noqa: E402
from agents.knowledge_agent_lcel import KnowledgeAgentLCEL  # noqa: E402

FIXED_QUERIES = [
    "late payment penalty policy",
    "payment failure handling",
    "penalty waiver eligibility",
]


def extract_doc_ids(citations: list[str]) -> list[str]:
    """Extract doc_id tokens from citation strings 'doc_id (Title)'."""
    return [c.split()[0] for c in citations]


def check_lcel_parity() -> bool:
    orig_agent = OriginalKnowledgeAgent()
    lcel_agent = KnowledgeAgentLCEL()

    all_matched = True
    rows = []

    for q in FIXED_QUERIES:
        _, orig_cites = orig_agent.cite(q, k=2)
        _, lcel_cites = lcel_agent.cite(q, k=2)

        orig_ids = extract_doc_ids(orig_cites)
        lcel_ids = extract_doc_ids(lcel_cites)

        matched = (orig_ids == lcel_ids) and (len(orig_ids) > 0)
        if not matched:
            all_matched = False

        rows.append((q, orig_ids, lcel_ids, "PASS" if matched else "FAIL"))

    # Print parity table
    print("\n" + "=" * 90)
    print("LCEL vs ORIGINAL KNOWLEDGE AGENT CITATION PARITY EVALUATION")
    print("=" * 90)
    col_w = [30, 26, 26, 8]
    header = f"{'Query':<{col_w[0]}} | {'Original Doc IDs':<{col_w[1]}} | {'LCEL Doc IDs':<{col_w[2]}} | {'Status'}"
    print(header)
    print("-" * len(header))
    for q, orig_ids, lcel_ids, status in rows:
        print(f"{q:<{col_w[0]}} | {str(orig_ids):<{col_w[1]}} | {str(lcel_ids):<{col_w[2]}} | {status}")
    print("=" * 90)

    return all_matched


def main():
    success = check_lcel_parity()
    if success:
        print("\n[SUCCESS] Citations and their order are 100% identical across all queries.\n")
        sys.exit(0)
    else:
        print("\n[ERROR] Parity mismatch detected between Original and LCEL Knowledge Agents!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
