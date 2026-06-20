"""Voice Agent — the conversational brain.

Two interchangeable reasoning paths produce the borrower-facing reply:

  * LLM path (ANTHROPIC_API_KEY set): a tool-use loop. The model is given the
    unified context, memory recall, RAG grounding hints, a persona, and the
    tool subset the Diagnosis Agent suggested. It reasons, calls tools, and
    speaks — fully dynamic, no decision tree.

  * Deterministic path (offline fallback): an evidence-driven planner that runs
    the relevant read tools, retrieves policy via RAG, takes the right action
    (ticket / payment link / reversal / commitment), and composes a grounded,
    cited answer. Driven by the diagnosis result, not a static script.

Both return {"text", "tool_trace", "action"} and both ground policy claims in
retrieved documents.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import tools as toolkit  # noqa: E402
from core.tools import ToolContext  # noqa: E402


def rupee(x) -> str:
    try:
        return f"Rs.{float(x):,.0f}"
    except Exception:
        return str(x)


PERSONA = (
    "You are Aarav, an inbound voice support agent for LendCo, a lending company. "
    "You speak like a knowledgeable, empathetic loan-servicing representative on a phone call: "
    "warm, concise, natural spoken sentences (no bullet points, no markdown). "
    "You ground every policy statement in retrieved company policy and never invent numbers — "
    "use the tools to fetch real figures. When a borrower is distressed, acknowledge it first. "
    "Ask at most one focused follow-up question when you genuinely need information you cannot look up."
)


def build_system_prompt(context: dict, recall: dict, diagnosis, persona: str = PERSONA) -> str:
    p = context.get("profile", {})
    a = context.get("analytics", {})
    commitments = recall.get("open_commitments", [])
    commit_line = ""
    if commitments:
        c = commitments[0]
        commit_line = (f"\nOPEN COMMITMENT FROM A PRIOR CALL: the borrower promised to pay {rupee(c['amount'])} "
                       f"by {c['date']} (reason: {c['reason']}). Proactively reference this and ask if it was completed.")
    prefs = recall.get("preferences", {})
    pref_line = f"\nKnown preferences: {prefs}." if prefs else ""
    return (
        f"{persona}\n\n"
        f"=== BORROWER CONTEXT ===\n"
        f"Name: {p.get('name')} | Loan {p.get('loan_id')} ({p.get('loan_type')}) | "
        f"Amount {rupee(p.get('loan_amount'))} @ {p.get('interest_rate')}% | EMI {rupee(a.get('emi_amount'))}\n"
        f"EMIs paid {a.get('installments_paid')}/{a.get('tenure_months')} | "
        f"Outstanding {rupee(a.get('outstanding_principal'))} | Next due {a.get('next_due_date')}\n"
        f"Delinquency {context.get('delinquency_status')} ({context.get('dpd_days')} DPD) | "
        f"Overdue {rupee(a.get('overdue_amount'))} | Penalties {rupee(a.get('total_penalty_charged'))}"
        f"{commit_line}{pref_line}\n\n"
        f"Likely intent: {diagnosis.intent} (confidence {diagnosis.confidence}). "
        f"Suggested tools for this intent: {diagnosis.suggested_tools}. "
        f"If you state any policy, first call search_knowledge_base and cite it."
    )


class VoiceAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

    # ---------------------------------------------------------------- LLM path
    def respond_llm(self, context, recall, diagnosis, history, tool_ctx: ToolContext) -> dict:
        system = build_system_prompt(context, recall, diagnosis)
        tools = [toolkit.SCHEMA_BY_NAME[n] for n in diagnosis.suggested_tools if n in toolkit.SCHEMA_BY_NAME]
        # always allow KB + analytics
        for must in ("search_knowledge_base", "get_loan_analytics"):
            if toolkit.SCHEMA_BY_NAME[must] not in tools:
                tools.append(toolkit.SCHEMA_BY_NAME[must])
        messages = [{"role": "user" if m["role"] == "borrower" else "assistant", "content": m["text"]} for m in history]

        result = self.llm.run_agent(
            system=system, messages=messages, tools=tools,
            tool_executor=lambda name, args: toolkit.execute(name, args, tool_ctx),
        )
        return {"text": result.text, "tool_trace": tool_ctx.trace, "action": None}

    # ------------------------------------------------------- deterministic path
    def respond_rule_based(self, context, recall, diagnosis, collected, tool_ctx: ToolContext) -> dict:
        intent = diagnosis.intent
        handler = getattr(self, f"_h_{intent}", self._h_general_faq)
        text, action = handler(context, recall, diagnosis, collected, tool_ctx)
        return {"text": text, "tool_trace": tool_ctx.trace, "action": action}

    # -- per-intent handlers --------------------------------------------------
    def _analytics(self, tool_ctx):
        return toolkit.execute("get_loan_analytics", {}, tool_ctx)

    def _kb(self, tool_ctx, query):
        res = toolkit.execute("search_knowledge_base", {"query": query, "k": 2}, tool_ctx)
        hits = res.get("results", [])
        cite = ", ".join(dict.fromkeys(h["doc_id"] for h in hits))  # dedupe doc ids, keep order
        return hits, cite

    def _h_remaining_emi(self, ctx, recall, dx, collected, tc):
        a = self._analytics(tc)
        end = ctx.get("profile", {}).get("loan_end_date")
        text = (f"You've paid {a['installments_paid']} of {a['tenure_months']} EMIs, "
                f"so {a['installments_remaining']} EMIs remain on your {a['loan_type']}. "
                f"Your next EMI of {rupee(a['emi_amount'])} is due on {a['next_due_date']}, "
                f"and at the current schedule the loan is set to close around {end}.")
        if a["overdue_amount"] > 0:
            text += f" I do see an overdue amount of {rupee(a['overdue_amount'])} - would you like a payment link to clear it?"
        return text, None

    def _h_interest_paid(self, ctx, recall, dx, collected, tc):
        a = self._analytics(tc)
        _, cite = self._kb(tc, "how interest is calculated reducing balance")
        text = (f"So far you've paid {rupee(a['interest_paid'])} towards interest and "
                f"{rupee(a['principal_paid'])} towards principal, out of {a['installments_paid']} EMIs paid. "
                f"Your outstanding principal is {rupee(a['outstanding_principal'])}. "
                f"Interest is charged on the reducing balance, so early EMIs carry more interest "
                f"(per our policy {cite}).")
        return text, None

    def _h_penalty_inquiry(self, ctx, recall, dx, collected, tc):
        a = self._analytics(tc)
        logs = toolkit.execute("get_gateway_logs", {}, tc)
        latest = logs.get("latest_failure")
        hits, cite = self._kb(tc, "late payment penalty policy grace period charge")
        text = (f"A total of {rupee(a['total_penalty_charged'])} in penalties has been charged on your account. ")
        if latest:
            text += (f"The most recent relates to the EMI due on {latest['due_date']}, which was "
                     f"'{latest['status']}' ({latest['response_message']}). ")
        text += (f"As per our late-payment policy ({cite}), a 2% penalty applies once a payment crosses the "
                 f"3-day grace period after the due date.")
        if latest and latest.get("root_cause") == "system":
            text += (" However, this failure looks bank/gateway-caused, which is not your liability - "
                     "I can raise a reversal for you. Shall I?")
        return text, None

    def _h_payment_failure(self, ctx, recall, dx, collected, tc):
        logs = toolkit.execute("get_gateway_logs", {}, tc)
        latest = logs.get("latest_failure")
        hits, cite = self._kb(tc, "payment failure handling gateway NACH bounce root cause")
        if not latest:
            return ("I don't see a failed payment on your most recent installments. "
                    "Could you tell me the date you attempted the payment?"), None
        rc = latest.get("root_cause")
        if rc == "system":
            link = toolkit.execute("create_payment_link", {"purpose": "EMI retry"}, tc)
            text = (f"I checked the gateway logs. Your payment on {latest['due_date']} failed with code "
                    f"{latest['response_code']} - '{latest['response_message']}'. That's a bank/gateway-side issue, "
                    f"not anything wrong on your end, so no penalty should stick (policy {cite}). "
                    f"I've generated a fresh payment link for you: {link['url']}. "
                    f"You can also wait for the automatic re-presentation in this cycle.")
            return text, "payment_link"
        else:
            link = toolkit.execute("create_payment_link", {"purpose": "EMI retry"}, tc)
            text = (f"I looked at the gateway logs. The payment on {latest['due_date']} was declined with code "
                    f"{latest['response_code']} - '{latest['response_message']}', which is an account-side reason. "
                    f"Per our payment-failure process ({cite}), please ensure sufficient balance or a valid mandate. "
                    f"Here's a payment link to complete it now: {link['url']}.")
            return text, "payment_link"

    def _h_penalty_waiver(self, ctx, recall, dx, collected, tc):
        logs = toolkit.execute("get_gateway_logs", {}, tc)
        latest = logs.get("latest_failure")
        hits, cite = self._kb(tc, "penalty waiver eligibility policy bank failure root cause")
        if latest and latest.get("root_cause") == "system":
            wf = toolkit.execute("trigger_workflow",
                                 {"name": "initiate_penalty_reversal",
                                  "params": {"payment_id": latest["payment_id"]}}, tc)
            text = (f"I've verified the failure on {latest['due_date']}: code {latest['response_code']} "
                    f"('{latest['response_message']}'), which our system classifies as a bank/gateway failure. "
                    f"Under our penalty-waiver policy ({cite}), penalties from system-side failures are not your "
                    f"liability, so you're eligible. I've initiated the reversal (ref {wf.get('run_id')}); "
                    f"it will reflect within a couple of working days.")
            return text, "penalty_reversed"
        else:
            tk = toolkit.execute("create_support_ticket",
                                 {"category": "penalty_waiver", "subject": "Penalty waiver request",
                                  "description": "Borrower requests waiver; latest failure is customer-side or unverified - needs human review.",
                                  "priority": "high"}, tc)
            reason = latest['response_message'] if latest else "no recent failure on record"
            text = (f"I've checked your records - the most recent charge appears to be account-side ({reason}). "
                    f"Our waiver policy ({cite}) covers failures caused by the bank or gateway, not account-side ones, "
                    f"so I can't auto-approve this. I've raised a request (ticket {tk.get('ticket_id')}) for a "
                    f"supervisor to review your case, and we'll get back to you.")
            return text, "ticket_created"

    def _h_promise_to_pay(self, ctx, recall, dx, collected, tc):
        date = collected.get("ptp_date")
        reason = collected.get("ptp_reason", "not stated")
        amount = ctx.get("analytics", {}).get("emi_amount")
        rec = toolkit.execute("record_commitment", {"date": date, "amount": amount, "reason": reason}, tc)
        cb = toolkit.execute("schedule_callback",
                             {"reason": "promise-to-pay follow-up", "preferred_window": f"around {date}"}, tc)
        text = (f"Thank you for letting me know. I've noted that you'll pay {rupee(amount)} by {date} "
                f"(reason: {reason}). There's no penalty discussion needed while this commitment is active. "
                f"I've also scheduled a gentle follow-up around {date} (ref {cb.get('callback_id')}). "
                f"Is there anything else I can help with?")
        return text, "commitment_recorded"

    def _h_settlement_request(self, ctx, recall, dx, collected, tc):
        hits, cite = self._kb(tc, "one time settlement policy eligibility credit bureau impact")
        tk = toolkit.execute("create_support_ticket",
                             {"category": "settlement_request", "subject": "One-time settlement request",
                              "description": f"Borrower at {ctx.get('dpd_days')} DPD requesting OTS. Hardship: {collected.get('hardship_reason','not provided')}.",
                              "priority": "high"}, tc)
        text = (f"I understand, and I've recorded your settlement request (ticket {tk.get('ticket_id')}). "
                f"One important thing to know per our policy ({cite}): a settlement closes the loan as 'settled' "
                f"rather than 'closed', which is reported to credit bureaus and can lower your credit score. "
                f"The settlement amount needs credit-committee approval, so I can't quote a figure now, "
                f"but our team will reach out with the next steps.")
        return text, "ticket_created"

    def _h_foreclosure(self, ctx, recall, dx, collected, tc):
        a = self._analytics(tc)
        hits, cite = self._kb(tc, "foreclosure policy lock-in charges payoff floating fixed")
        eligible = a["installments_paid"] >= 6
        if eligible:
            wf = toolkit.execute("trigger_workflow", {"name": "send_foreclosure_statement"}, tc)
            text = (f"You're eligible to foreclose - you've paid {a['installments_paid']} EMIs, past the 6-EMI lock-in. "
                    f"Your outstanding principal is {rupee(a['outstanding_principal'])}; the payoff is this plus accrued "
                    f"interest to the foreclosure date and any applicable charges. For floating-rate individual loans "
                    f"foreclosure charges are nil (policy {cite}). I've requested a foreclosure statement for you "
                    f"(ref {wf.get('run_id')}); it's valid for 7 days.")
        else:
            text = (f"I see you've paid {a['installments_paid']} EMIs. Foreclosure is permitted after the 6-EMI "
                    f"lock-in per our policy ({cite}), so you'd be eligible after a few more installments. "
                    f"Your current outstanding principal is {rupee(a['outstanding_principal'])}.")
        return text, "foreclosure_statement" if eligible else None

    def _h_account_info(self, ctx, recall, dx, collected, tc):
        a = self._analytics(tc)
        p = ctx.get("profile", {})
        text = (f"Here's your loan summary: {p.get('loan_type')} {p.get('loan_id')}, sanctioned for "
                f"{rupee(p.get('loan_amount'))} at {p.get('interest_rate')}%. Your EMI is {rupee(a['emi_amount'])}, "
                f"next due {a['next_due_date']}. You've paid {a['installments_paid']} of {a['tenure_months']} EMIs "
                f"with {rupee(a['outstanding_principal'])} outstanding.")
        return text, None

    def _h_general_faq(self, ctx, recall, dx, collected, tc):
        query = dx.kb_query or "loan FAQs"
        hits, cite = self._kb(tc, query)
        snippet = hits[0]["text"].replace("\n", " ") if hits else "I'll need to check on that."
        text = f"Here's what our policy says ({cite}): {snippet[:280]}"
        return text, None
