"""Generate a self-contained HTML evaluation/analytics dashboard.

Run:  python -m dashboard.build_report   ->  dashboard/report.html (open in a browser)

No external dependencies — inline CSS + CSS-bar charts, so it works anywhere.
"""
from __future__ import annotations

import html

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from dashboard.analytics import all_analytics  # noqa: E402

PALETTE = ["#4f8cff", "#34c98b", "#ffb020", "#ff6b6b", "#9b6bff", "#22b8cf", "#f06595"]


def _rupee(x):
    try:
        return f"Rs.{float(x):,.0f}"
    except Exception:
        return str(x)


def card(label, value, sub=""):
    return (f'<div class="card"><div class="card-v">{value}</div>'
            f'<div class="card-l">{html.escape(str(label))}</div>'
            f'<div class="card-s">{html.escape(str(sub))}</div></div>')


def bars(title, data: dict, fmt=str):
    if not data:
        return f'<div class="panel"><h3>{html.escape(title)}</h3><p class="muted">no data</p></div>'
    mx = max(data.values()) or 1
    rows = ""
    for i, (k, v) in enumerate(sorted(data.items(), key=lambda x: -x[1])):
        pct = 100 * v / mx
        color = PALETTE[i % len(PALETTE)]
        rows += (f'<div class="bar-row"><div class="bar-k">{html.escape(str(k))}</div>'
                 f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'
                 f'<div class="bar-v">{fmt(v)}</div></div>')
    return f'<div class="panel"><h3>{html.escape(title)}</h3>{rows}</div>'


def table(title, rows: list[dict], cols: list[tuple[str, str]]):
    if not rows:
        return f'<div class="panel"><h3>{html.escape(title)}</h3><p class="muted">no data yet — run some calls</p></div>'
    head = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(c[0], '')))}</td>" for c in cols) + "</tr>"
    return f'<div class="panel"><h3>{html.escape(title)}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def eval_section(ev: dict) -> str:
    if not ev:
        return '<div class="panel"><h3>Evaluation Metrics</h3><p class="muted">run <code>python -m eval.evaluator</code></p></div>'
    o = ev.get("scenarios", {}).get("overall", {})
    m = ev.get("memory_learning", {})
    cards = (
        card("Intent accuracy", f"{ev.get('intents',{}).get('intent_accuracy',0)*100:.0f}%") +
        card("Resolution rate", f"{o.get('resolution_rate',0)*100:.0f}%") +
        card("Grounding rate", f"{o.get('grounding_rate',0)*100:.0f}%", "answers citing policy") +
        card("Avg turns / call", o.get("avg_turns", "-")) +
        card("Redundant questions", f"{o.get('redundant_question_rate',0)*100:.0f}%", "lower is better") +
        card("Root-cause accuracy", f"{ev.get('root_cause',{}).get('root_cause_accuracy',0)*100:.0f}%")
    )
    mem = (f"<p>Continuous learning: first call <b>{m.get('call1_turns','-')}</b> turns "
           f"&rarr; repeat call <b>{m.get('call2_turns','-')}</b> turns "
           f"(saved {m.get('turns_saved','-')}, memory-aware opener = {m.get('call2_memory_aware','-')}).</p>")
    per = ev.get("scenarios", {}).get("per_scenario", {})
    per_rows = [{"scenario": k, "runs": v["runs"], "resolution": f"{v['resolution_rate']*100:.0f}%",
                 "grounding": f"{v['grounding_rate']*100:.0f}%", "intent": f"{v['intent_accuracy']*100:.0f}%",
                 "avg_turns": v["avg_turns"]} for k, v in per.items()]
    tbl = table("Per-scenario evaluation", per_rows,
                [("scenario", "Scenario"), ("runs", "Runs"), ("resolution", "Resolved"),
                 ("grounding", "Grounded"), ("intent", "Intent acc"), ("avg_turns", "Avg turns")])
    return f'<div class="panel"><h3>Evaluation Metrics</h3><div class="cards">{cards}</div>{mem}</div>{tbl}'


def build() -> Path:
    a = all_analytics()
    pf, pay, ops, conv = a["portfolio"], a["payments"], a["operations"], a["conversations"]

    top_cards = (
        card("Borrowers", pf["borrowers"]) +
        card("Loan book", _rupee(pf["loan_book"])) +
        card("Avg interest", f"{pf['avg_interest']}%") +
        card("Auto-debit", f"{pf['auto_debit_pct']}%") +
        card("Payment records", pay["total"]) +
        card("Conversations", conv["total"], f"{conv['promises_to_pay']} promises-to-pay")
    )

    ops_cards = (
        card("Tickets created", ops["tickets_created"]) +
        card("Payment links", ops["payment_links"]) +
        card("Callbacks", ops["callbacks"]) +
        card("Escalations", ops["escalations"]) +
        card("Workflow runs", ops["workflow_runs"]) +
        card("CRM updates", ops["crm_updates"])
    )

    learn_rows = a["agent_learning"]
    learn_tbl = table("Agent Memory — learned resolution paths", learn_rows,
                      [("intent", "Intent"), ("resolution_path", "Resolution path"),
                       ("runs", "Runs"), ("success_rate", "Success"), ("avg_turns", "Avg turns")])

    body = f"""
    <h1>LendCo Voice Agent — Analytics & Evaluation</h1>
    <p class="muted">Unified view across CRM, Payments, Support, Knowledge & Workflow platforms,
    agent memory and the evaluation harness.</p>

    <h2>Portfolio</h2>
    <div class="cards">{top_cards}</div>
    <div class="grid">
      {bars("Delinquency distribution", pf["delinquency"])}
      {bars("Payment status breakdown", pay["by_status"])}
    </div>
    <div class="grid">
      {bars("Payment-failure root cause", pay["failure_root_cause"])}
      {bars("Failure modes", pay["failure_modes"])}
    </div>

    <h2>Conversation Analytics</h2>
    <div class="grid">
      {bars("Intent distribution", conv["intents"])}
      {bars("Sentiment distribution", conv["sentiment"])}
    </div>

    <h2>Live Agent Operations (this session)</h2>
    <div class="cards">{ops_cards}</div>
    {learn_tbl}

    <h2>Evaluation</h2>
    {eval_section(a["eval"])}
    """

    htmldoc = f"""<!doctype html><html><head><meta charset="utf-8"><title>LendCo Voice Agent Dashboard</title>
<style>
:root{{--bg:#0e1525;--panel:#162033;--ink:#e6edf7;--muted:#8aa0bf;--line:#243349}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);padding:32px;max-width:1200px;margin:auto}}
h1{{font-size:26px;margin:0 0 4px}} h2{{margin:34px 0 14px;font-size:19px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:14px;margin:0 0 12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
.muted{{color:var(--muted)}} code{{background:#0b1020;padding:2px 6px;border-radius:4px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:150px;flex:1}}
.card-v{{font-size:24px;font-weight:700}} .card-l{{font-size:13px;color:var(--muted);margin-top:2px}}
.card-s{{font-size:11px;color:#62789a}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}
.bar-row{{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}}
.bar-k{{width:150px;color:var(--muted);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar-track{{flex:1;background:#0b1020;border-radius:6px;height:16px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:6px}} .bar-v{{width:60px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}} th{{color:var(--muted);font-weight:600}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>{body}
<p class="muted" style="margin-top:30px">Generated by dashboard/build_report.py</p></body></html>"""

    out = settings.ROOT / "dashboard" / "report.html"
    out.write_text(htmldoc, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build()
    print(f"dashboard written to {path}")
