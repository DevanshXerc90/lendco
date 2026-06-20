"""Optional interactive dashboard (bonus).

Run:  streamlit run dashboard/streamlit_app.py
If streamlit isn't installed, use the dependency-free HTML report instead:
      python -m dashboard.build_report   ->  dashboard/report.html
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
    import pandas as pd
except Exception:  # pragma: no cover
    print("streamlit/pandas not installed. Run: python -m dashboard.build_report")
    raise SystemExit(1)

from dashboard.analytics import all_analytics  # noqa: E402

st.set_page_config(page_title="LendCo Voice Agent", layout="wide")
a = all_analytics()
pf, pay, ops, conv, ev = a["portfolio"], a["payments"], a["operations"], a["conversations"], a["eval"]

st.title("LendCo Voice Agent — Analytics & Evaluation")

c = st.columns(5)
c[0].metric("Borrowers", pf["borrowers"])
c[1].metric("Loan book", f"Rs.{pf['loan_book']:,.0f}")
c[2].metric("Avg interest", f"{pf['avg_interest']}%")
c[3].metric("Payment records", pay["total"])
c[4].metric("Conversations", conv["total"])

st.subheader("Portfolio")
c1, c2 = st.columns(2)
c1.bar_chart(pd.Series(pf["delinquency"]))
c2.bar_chart(pd.Series(pay["by_status"]))

st.subheader("Payment failure root cause / modes")
c3, c4 = st.columns(2)
c3.bar_chart(pd.Series(pay["failure_root_cause"]))
c4.bar_chart(pd.Series(pay["failure_modes"]))

st.subheader("Live agent operations (this session)")
st.json(ops)

st.subheader("Agent memory — learned resolution paths")
if a["agent_learning"]:
    st.dataframe(pd.DataFrame(a["agent_learning"]))
else:
    st.info("Run some calls to populate agent memory.")

st.subheader("Evaluation")
if ev:
    o = ev.get("scenarios", {}).get("overall", {})
    m = ev.get("memory_learning", {})
    e = st.columns(5)
    e[0].metric("Intent accuracy", f"{ev.get('intents',{}).get('intent_accuracy',0)*100:.0f}%")
    e[1].metric("Resolution", f"{o.get('resolution_rate',0)*100:.0f}%")
    e[2].metric("Grounding", f"{o.get('grounding_rate',0)*100:.0f}%")
    e[3].metric("Avg turns", o.get("avg_turns", "-"))
    e[4].metric("Root-cause acc", f"{ev.get('root_cause',{}).get('root_cause_accuracy',0)*100:.0f}%")
    st.write(f"Memory: call 1 = {m.get('call1_turns')} turns -> call 2 = {m.get('call2_turns')} turns "
             f"(saved {m.get('turns_saved')}, memory-aware = {m.get('call2_memory_aware')})")
    st.dataframe(pd.DataFrame(ev.get("scenarios", {}).get("per_scenario", {})).T)
else:
    st.info("Run: python -m eval.evaluator")
