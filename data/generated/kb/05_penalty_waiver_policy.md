# Penalty Waiver Policy

Penalties may be waived when the failure was **not attributable to the borrower**.

**Eligible (waiver granted):**
- Bank timeout, NACH technical rejection, gateway error, or network failure (root cause = system).
- First-time genuine delay with a documented reason, at supervisor discretion.

**Not eligible (waiver denied):**
- Insufficient funds, revoked mandate, or repeated customer-side defaults (root cause = customer).

**Process:** Verify the gateway response code and root-cause bucket from payment logs. If system-caused, the agent may auto-approve reversal and confirm. If borderline/customer-caused, create a penalty-waiver ticket for human review. Always cite the payment record and this policy.
