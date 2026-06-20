"""Synthetic data generator for the borrower-support voice agent.

Produces reproducible datasets (seeded) under data/generated/:
  - borrowers.json        (>=100)  loan + KYC profile
  - payments.json         (>=500)  installment-level payment history
  - tickets.json          (>=100)  support tickets
  - conversations.json    (>=100)  prior call/chat transcripts (seeds memory)
  - kb/*.md               (>=20)   knowledge-base policy documents

Run:  python -m data.generate
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

fake = Faker("en_IN")

OUT = settings.DATA_DIR
KB_OUT = settings.KB_DIR

LOAN_TYPES = ["Personal Loan", "Home Loan", "Auto Loan", "Business Loan", "Education Loan"]
LANGS = ["English", "Hindi", "Tamil", "Telugu", "Marathi", "Bengali"]
EMPLOYMENT = ["Salaried", "Self-Employed", "Business Owner"]

GATEWAYS = ["Razorpay", "Cashfree", "Stripe"]
PAY_METHODS = ["auto_debit", "upi", "netbanking", "debit_card", "nach"]

# Gateway failure taxonomy -> (code, human message, root cause bucket)
FAILURE_MODES = {
    "bank_timeout": ("GW-504", "Issuing bank did not respond within timeout window", "system"),
    "nach_failure": ("NACH-RJ", "NACH mandate debit rejected by sponsor bank", "system"),
    "insufficient_funds": ("R01", "Insufficient funds in account", "customer"),
    "gateway_error": ("GW-500", "Payment gateway internal error", "system"),
    "network_issue": ("GW-CONN", "Network connectivity error during authorization", "system"),
    "technical_decline": ("DECL-T", "Technical decline by acquiring bank", "system"),
    "mandate_revoked": ("NACH-REV", "Auto-debit mandate revoked by customer", "customer"),
}


def _seed() -> None:
    random.seed(settings.SEED)
    Faker.seed(settings.SEED)


def _monthly_emi(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 12 / 100
    if r == 0:
        return round(principal / months, 2)
    factor = (1 + r) ** months
    return round(principal * r * factor / (factor - 1), 2)


def gen_borrowers(n: int = 120) -> list[dict]:
    borrowers = []
    today = date(2026, 6, 9)
    for i in range(1, n + 1):
        bid = f"BRW{i:05d}"
        loan_type = random.choice(LOAN_TYPES)
        amount = random.choice([100000, 200000, 350000, 500000, 750000, 1000000, 1500000, 2500000, 4000000])
        rate = round(random.uniform(8.5, 18.0), 2)
        tenure = random.choice([12, 24, 36, 48, 60, 84, 120])
        # Started somewhere in the past so there is real payment history.
        months_elapsed = random.randint(3, min(tenure - 1, 30))
        start = today - timedelta(days=months_elapsed * 30 + random.randint(0, 20))
        end = start + timedelta(days=tenure * 30)
        emi = _monthly_emi(amount, rate, tenure)

        # Delinquency buckets weighted toward "current".
        dpd_days = random.choices(
            [0, random.randint(1, 30), random.randint(31, 60), random.randint(61, 90), random.randint(91, 180)],
            weights=[55, 20, 12, 8, 5],
        )[0]
        status = (
            "current" if dpd_days == 0
            else "dpd_1_30" if dpd_days <= 30
            else "dpd_31_60" if dpd_days <= 60
            else "dpd_61_90" if dpd_days <= 90
            else "dpd_90_plus"
        )

        name = fake.name()
        slug = "".join(ch for ch in name.lower() if ch.isalnum() or ch == " ").strip().replace(" ", ".")
        borrowers.append({
            "borrower_id": bid,
            "name": name,
            "phone": f"+91{random.randint(7000000000, 9999999999)}",
            "email": f"{slug}{i}@example.com",
            "loan_id": f"LN{i:06d}",
            "loan_type": loan_type,
            "loan_amount": amount,
            "interest_rate": rate,
            "emi_amount": emi,
            "tenure_months": tenure,
            "months_elapsed": months_elapsed,
            "loan_start_date": start.isoformat(),
            "loan_end_date": end.isoformat(),
            "delinquency_status": status,
            "dpd_days": dpd_days,
            "auto_debit_enabled": random.random() < 0.7,
            "preferred_language": random.choices(LANGS, weights=[50, 25, 6, 6, 7, 6])[0],
            "kyc": {
                "pan": f"{fake.lexify('?????').upper()}{random.randint(1000,9999)}{fake.lexify('?').upper()}",
                "aadhaar_masked": f"XXXX-XXXX-{random.randint(1000,9999)}",
                "address": fake.address().replace("\n", ", "),
                "city": fake.city(),
                "employment_type": random.choice(EMPLOYMENT),
                "monthly_income": random.choice([35000, 50000, 75000, 100000, 150000, 250000]),
                "kyc_verified": random.random() < 0.92,
            },
        })
    return borrowers


def gen_payments(borrowers: list[dict]) -> list[dict]:
    payments = []
    pid = 1
    today = date(2026, 6, 9)
    for b in borrowers:
        start = date.fromisoformat(b["loan_start_date"])
        emi = b["emi_amount"]
        n_installments = b["months_elapsed"]
        # Borrowers in delinquency miss / fail more recent installments.
        delinquent = b["dpd_days"] > 0
        for k in range(n_installments):
            due = start + timedelta(days=(k + 1) * 30)
            if due > today:
                break
            is_recent = k >= n_installments - max(1, b["dpd_days"] // 30 + 1)

            roll = random.random()
            if delinquent and is_recent and roll < 0.6:
                # A problematic installment.
                if b["auto_debit_enabled"] and random.random() < 0.6:
                    reason = random.choice(["bank_timeout", "nach_failure", "gateway_error", "network_issue", "insufficient_funds"])
                    status = "auto_debit_failed"
                else:
                    reason = random.choice(["insufficient_funds", "technical_decline", "mandate_revoked"])
                    status = random.choice(["failed", "missed"])
            elif roll < 0.05:
                reason = random.choice(list(FAILURE_MODES.keys()))
                status = random.choice(["failed", "partial"])
            else:
                reason = None
                status = "success"

            code, msg, bucket = (FAILURE_MODES[reason] if reason else ("00", "Approved", "none"))
            method = "auto_debit" if b["auto_debit_enabled"] and random.random() < 0.7 else random.choice(PAY_METHODS)

            if status == "success":
                paid, paid_date, penalty = emi, due + timedelta(days=random.randint(0, 2)), 0.0
            elif status == "partial":
                paid, paid_date, penalty = round(emi * random.uniform(0.2, 0.7), 2), due + timedelta(days=random.randint(1, 5)), round(emi * 0.02, 2)
            elif status == "missed":
                paid, paid_date, penalty = 0.0, None, round(emi * 0.02, 2)
            else:  # failed / auto_debit_failed
                paid, paid_date, penalty = 0.0, None, round(emi * 0.02, 2) if bucket != "system" else 0.0

            payments.append({
                "payment_id": f"PAY{pid:07d}",
                "borrower_id": b["borrower_id"],
                "loan_id": b["loan_id"],
                "installment_no": k + 1,
                "due_date": due.isoformat(),
                "paid_date": paid_date.isoformat() if paid_date else None,
                "amount_due": emi,
                "amount_paid": paid,
                "status": status,
                "method": method,
                "gateway": random.choice(GATEWAYS),
                "gateway_response_code": code,
                "gateway_response_message": msg,
                "failure_reason": reason,
                "root_cause": bucket,
                "penalty_charged": penalty,
            })
            pid += 1
    return payments


TICKET_TEMPLATES = [
    ("emi_dispute", "EMI amount looks higher than agreed", "Borrower disputes the EMI debited this month versus sanction letter."),
    ("settlement_request", "Request for one-time settlement", "Borrower facing financial hardship, requesting a settlement quote."),
    ("loan_closure", "Foreclosure / loan closure enquiry", "Borrower wants to foreclose the loan and asks for the outstanding payoff amount."),
    ("payment_failure", "Auto-debit failed despite sufficient balance", "Borrower reports a failed payment though funds were available."),
    ("penalty_waiver", "Request to waive late-payment penalty", "Borrower requests a penalty waiver citing a bank-side failure."),
    ("foreclosure", "Foreclosure charges clarification", "Borrower asks about applicable foreclosure charges and lock-in."),
    ("general", "Update registered mobile number", "Borrower requests KYC contact update."),
]


def gen_tickets(borrowers: list[dict], n: int = 130) -> list[dict]:
    tickets = []
    today = datetime(2026, 6, 9, 10, 0, 0)
    for i in range(1, n + 1):
        b = random.choice(borrowers)
        cat, subj, desc = random.choice(TICKET_TEMPLATES)
        created = today - timedelta(days=random.randint(0, 120), hours=random.randint(0, 23))
        status = random.choices(["open", "in_progress", "resolved", "closed"], weights=[25, 20, 30, 25])[0]
        resolved_states = {"resolved", "closed"}
        tickets.append({
            "ticket_id": f"TKT{i:05d}",
            "borrower_id": b["borrower_id"],
            "loan_id": b["loan_id"],
            "category": cat,
            "subject": subj,
            "description": desc,
            "priority": random.choices(["low", "medium", "high", "urgent"], weights=[30, 40, 20, 10])[0],
            "status": status,
            "channel": random.choice(["voice", "email", "chat", "app"]),
            "created_at": created.isoformat(),
            "updated_at": (created + timedelta(days=random.randint(0, 10))).isoformat(),
            "resolution": (
                random.choice([
                    "Penalty reversed after verifying bank-side NACH failure.",
                    "Settlement quote shared; borrower to confirm.",
                    "Explained EMI breakup; no discrepancy found.",
                    "Foreclosure statement emailed to borrower.",
                    "Auto-debit mandate re-registered.",
                ]) if status in resolved_states else None
            ),
        })
    return tickets


INTENTS = [
    "remaining_emi", "interest_paid", "penalty_inquiry", "payment_failure",
    "penalty_waiver", "promise_to_pay", "settlement_request", "foreclosure",
    "account_info", "general_faq",
]
SENTIMENTS = ["positive", "neutral", "frustrated", "anxious", "angry"]


def gen_conversations(borrowers: list[dict], n: int = 130) -> list[dict]:
    convos = []
    today = datetime(2026, 6, 9, 10, 0, 0)
    for i in range(1, n + 1):
        b = random.choice(borrowers)
        intent = random.choice(INTENTS)
        ts = today - timedelta(days=random.randint(1, 180), hours=random.randint(0, 23))
        sentiment = random.choices(SENTIMENTS, weights=[20, 35, 25, 12, 8])[0]
        ptp = None
        transcript = [{"role": "agent", "text": "Thank you for calling. How can I help you today?"}]

        if intent == "promise_to_pay":
            days_ahead = random.randint(3, 20)
            ptp_date = (ts + timedelta(days=days_ahead)).date().isoformat()
            reason = random.choice(["salary delay", "medical emergency", "business cash-flow gap", "awaiting client payment"])
            transcript += [
                {"role": "borrower", "text": f"My {reason} means I cannot pay right now. I'll pay after my salary credits."},
                {"role": "agent", "text": "I understand. When do you expect to make the payment?"},
                {"role": "borrower", "text": f"I should be able to pay the EMI of Rs.{b['emi_amount']:.0f} by {ptp_date}."},
                {"role": "agent", "text": "Noted. I've recorded your commitment and we'll follow up gently."},
            ]
            ptp = {"date": ptp_date, "amount": b["emi_amount"], "reason": reason, "kept": None}
        elif intent == "remaining_emi":
            transcript += [
                {"role": "borrower", "text": "How many EMIs are remaining on my loan?"},
                {"role": "agent", "text": "Let me check your loan tenure and due schedule."},
            ]
        elif intent == "penalty_inquiry":
            transcript += [
                {"role": "borrower", "text": "Why was a penalty charged on my account?"},
                {"role": "agent", "text": "I'll review your payment timeline and the late-payment policy."},
            ]
        else:
            transcript += [
                {"role": "borrower", "text": f"I have a question about my {b['loan_type'].lower()}."},
                {"role": "agent", "text": "Sure, I can help with that."},
            ]

        resolved = random.random() < 0.7
        convos.append({
            "conversation_id": f"CONV{i:05d}",
            "borrower_id": b["borrower_id"],
            "channel": random.choice(["voice", "chat"]),
            "timestamp": ts.isoformat(),
            "intent": intent,
            "sentiment": sentiment,
            "transcript": transcript,
            "resolved": resolved,
            "outcome": "resolved" if resolved else "escalated" if random.random() < 0.5 else "follow_up_needed",
            "promise_to_pay": ptp,
            "preferred_language": b["preferred_language"],
        })
    return convos


# ---------------------------------------------------------------------------
# Knowledge base documents (markdown). >=20 policy/FAQ docs.
# ---------------------------------------------------------------------------
KB_DOCS: dict[str, str] = {
    "01_loan_faqs.md": """# Loan FAQs

**What is an EMI?** An Equated Monthly Installment is the fixed amount paid every month, comprising principal and interest. Early EMIs are interest-heavy; later EMIs are principal-heavy.

**How is my EMI calculated?** EMI = P x r x (1+r)^n / ((1+r)^n - 1), where P is principal, r is the monthly interest rate, and n is the tenure in months.

**When is my EMI due?** On the due date in your repayment schedule, typically the same day each month.

**How can I view remaining EMIs?** Remaining EMIs = total tenure minus the number of installments already paid. The agent can compute this from your repayment schedule.
""",
    "02_foreclosure_policy.md": """# Foreclosure Policy

Foreclosure (preclosure) is full repayment of the outstanding principal before the scheduled end date.

- **Lock-in period:** Foreclosure is permitted after 6 EMIs have been paid.
- **Foreclosure charges:** For floating-rate retail loans to individuals, NIL charges as per RBI guidelines. Fixed-rate loans: up to 4% of outstanding principal.
- **Payoff amount** = outstanding principal + accrued interest till the foreclosure date + applicable charges - any unadjusted credits.
- A foreclosure statement is generated on request and is valid for 7 days.
""",
    "03_settlement_policy.md": """# Settlement Policy

A one-time settlement (OTS) is a negotiated closure for borrowers in financial distress, typically for accounts in advanced delinquency (90+ DPD).

- Eligibility is assessed case by case based on hardship evidence and recovery viability.
- A settlement closes the loan as "settled" (not "closed"), which is reported to credit bureaus and may affect the credit score.
- Settlement requests must be approved by the credit committee; the agent should create a ticket and set expectations rather than promise an amount.
""",
    "04_late_payment_policy.md": """# Late Payment Policy

- A payment is **overdue** the day after the due date.
- A **grace period of 3 days** applies before a late-payment penalty is levied.
- **Late payment penalty:** 2% of the overdue EMI amount per occurrence, plus applicable taxes.
- Bounce charges of Rs. 500 apply if an auto-debit / NACH presentation is returned for customer-side reasons (e.g., insufficient funds).
- Penalties for **bank-side or gateway-side failures are not the borrower's liability** and are reversed on verification.
""",
    "05_penalty_waiver_policy.md": """# Penalty Waiver Policy

Penalties may be waived when the failure was **not attributable to the borrower**.

**Eligible (waiver granted):**
- Bank timeout, NACH technical rejection, gateway error, or network failure (root cause = system).
- First-time genuine delay with a documented reason, at supervisor discretion.

**Not eligible (waiver denied):**
- Insufficient funds, revoked mandate, or repeated customer-side defaults (root cause = customer).

**Process:** Verify the gateway response code and root-cause bucket from payment logs. If system-caused, the agent may auto-approve reversal and confirm. If borderline/customer-caused, create a penalty-waiver ticket for human review. Always cite the payment record and this policy.
""",
    "06_payment_failure_handling.md": """# Payment Failure Handling Process

1. Retrieve the payment log for the failed installment.
2. Read the **gateway response code** and **root-cause bucket** (system vs customer).
3. Classify:
   - **System** (bank_timeout / nach_failure / gateway_error / network_issue): not the borrower's fault. Reassure, no penalty, offer a fresh payment link or auto-retry.
   - **Customer** (insufficient_funds / mandate_revoked): explain, offer to pay now via link, advise on balance/mandate.
4. Recommend next action: retry now, generate a payment link, re-register mandate, or schedule a callback.
5. If a wrongful penalty was charged due to a system failure, initiate reversal.
""",
    "07_emi_bounce.md": """# EMI Bounce / Auto-Debit Failure

An EMI "bounce" occurs when an auto-debit or NACH presentation is returned. Common return reasons:
- R01 Insufficient funds (customer-side, bounce charge applies)
- NACH-RJ NACH rejected by sponsor bank (system-side, no charge)
- NACH-REV Mandate revoked (customer-side)
Re-presentation is attempted once within the cycle. Borrowers can also pay manually via a payment link.
""",
    "08_nach_mandate.md": """# NACH Mandate / Auto-Debit Setup

NACH (National Automated Clearing House) lets us auto-debit your EMI from your bank account. To set up or re-register a mandate, the borrower authorizes a maximum debit amount and frequency. A revoked or expired mandate causes auto-debit failures and must be re-registered; the agent can trigger a re-registration workflow.
""",
    "09_prepayment_policy.md": """# Part-Prepayment Policy

Part-prepayment reduces outstanding principal. For floating-rate individual loans there are no part-prepayment charges. Borrowers may choose to reduce EMI or reduce tenure after a prepayment. Minimum prepayment is one EMI; a revised amortization schedule is issued.
""",
    "10_interest_calculation.md": """# How Interest Is Calculated

Interest accrues on the reducing outstanding principal. For each EMI, interest = outstanding principal x monthly rate; the remainder of the EMI reduces principal. Therefore total interest paid so far = sum of EMIs paid - reduction in principal. The agent computes interest-paid-to-date from the payment history and amortization schedule.
""",
    "11_auto_debit_setup.md": """# Managing Auto-Debit

Borrowers can enable, pause, or cancel auto-debit. If auto-debit is disabled, EMIs must be paid manually before the due date to avoid penalties. Changes take effect from the next billing cycle.
""",
    "12_moratorium_policy.md": """# Moratorium / EMI Holiday

A moratorium temporarily defers EMI payments. Interest continues to accrue during the moratorium and is added to the outstanding principal. Moratoriums are granted only under specific hardship programs and require approval; they are not automatic.
""",
    "13_kyc_update.md": """# KYC Update Policy

To update KYC details (address, phone, email), the borrower must verify identity. Mobile number updates require OTP verification on both old and new numbers where possible. The agent can raise a KYC-update ticket and trigger the verification workflow.
""",
    "14_grievance_redressal.md": """# Grievance Redressal

Level 1: Customer support / voice agent. Level 2: Nodal officer (response within 7 working days). Level 3: Principal Nodal Officer. Unresolved grievances beyond 30 days may be escalated to the RBI Banking Ombudsman. The agent should escalate to a human and create a high-priority ticket for formal grievances.
""",
    "15_credit_bureau_reporting.md": """# Credit Bureau Reporting

Repayment behavior is reported monthly to credit bureaus (CIBIL, Experian). Missed/late payments and settlements negatively affect the score. Foreclosure / full closure is reported positively as "closed". Timely payments improve the score over time.
""",
    "16_payment_methods.md": """# Accepted Payment Methods

EMIs can be paid via auto-debit (NACH), UPI, net-banking, debit card, or a payment link shared by the agent. Payment links are valid for 24 hours. Successful payments reflect within 1 working day; the agent can confirm receipt from the payment platform.
""",
    "17_penalty_structure.md": """# Penalty & Charge Structure

- Late payment penalty: 2% of overdue EMI per occurrence + taxes.
- Bounce / NACH return charge (customer-side): Rs. 500.
- Foreclosure charge: NIL (floating individual) / up to 4% (fixed).
- Cheque/mandate swap charge: Rs. 500.
All charges are exclusive of applicable GST.
""",
    "18_hardship_assistance.md": """# Hardship Assistance

Borrowers facing genuine hardship (job loss, medical emergency, salary delay) may request: a revised due date, a short promise-to-pay extension, EMI restructuring, or a settlement. The agent should capture the reason, the committed pay date, and the amount, then schedule a follow-up rather than escalating immediately.
""",
    "19_promise_to_pay.md": """# Promise-to-Pay (PTP) Handling

When a borrower commits to pay by a future date, the agent records: the committed date, the amount, and the reason for delay. No penalty discussion is forced during an active, reasonable PTP. On the next interaction the agent must reference the prior commitment and confirm whether it was kept before discussing escalation.
""",
    "20_data_privacy.md": """# Data Privacy & Verification

Before sharing account details, the agent verifies the caller's identity using registered phone number and at least one secondary detail (loan ID, date of birth, or PAN last 4). Sensitive identifiers (full Aadhaar, full PAN) are never read aloud. Calls may be recorded for quality and training.
""",
    "21_settlement_vs_closure.md": """# Settlement vs. Closure (Important Distinction)

- **Closed:** Loan fully repaid as per contract (or foreclosed). Reported positively.
- **Settled:** Lender accepted less than the full dues. Reported as "settled", which lowers the credit score and may affect future eligibility.
Always explain this trade-off when a borrower asks about settlement.
""",
    "22_callback_scheduling.md": """# Callback Scheduling

The agent can schedule a callback at a borrower-preferred time. A callback is logged in the workflow system with the borrower ID, reason, and requested window. Borrowers on an active promise-to-pay are offered a courtesy callback shortly after their committed date.
""",
}


def write_kb() -> int:
    KB_OUT.mkdir(parents=True, exist_ok=True)
    for fname, body in KB_DOCS.items():
        (KB_OUT / fname).write_text(body, encoding="utf-8")
    return len(KB_DOCS)


def main() -> None:
    _seed()
    OUT.mkdir(parents=True, exist_ok=True)

    borrowers = gen_borrowers(120)
    payments = gen_payments(borrowers)
    tickets = gen_tickets(borrowers, 130)
    conversations = gen_conversations(borrowers, 130)

    (OUT / "borrowers.json").write_text(json.dumps(borrowers, indent=2), encoding="utf-8")
    (OUT / "payments.json").write_text(json.dumps(payments, indent=2), encoding="utf-8")
    (OUT / "tickets.json").write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    (OUT / "conversations.json").write_text(json.dumps(conversations, indent=2), encoding="utf-8")
    n_kb = write_kb()

    print("Synthetic data generated:")
    print(f"  borrowers      : {len(borrowers):>5}  -> data/generated/borrowers.json")
    print(f"  payments       : {len(payments):>5}  -> data/generated/payments.json")
    print(f"  tickets        : {len(tickets):>5}  -> data/generated/tickets.json")
    print(f"  conversations  : {len(conversations):>5}  -> data/generated/conversations.json")
    print(f"  kb documents   : {n_kb:>5}  -> data/generated/kb/*.md")


if __name__ == "__main__":
    main()
