# Payment Failure Handling Process

1. Retrieve the payment log for the failed installment.
2. Read the **gateway response code** and **root-cause bucket** (system vs customer).
3. Classify:
   - **System** (bank_timeout / nach_failure / gateway_error / network_issue): not the borrower's fault. Reassure, no penalty, offer a fresh payment link or auto-retry.
   - **Customer** (insufficient_funds / mandate_revoked): explain, offer to pay now via link, advise on balance/mandate.
4. Recommend next action: retry now, generate a payment link, re-register mandate, or schedule a callback.
5. If a wrongful penalty was charged due to a system failure, initiate reversal.
