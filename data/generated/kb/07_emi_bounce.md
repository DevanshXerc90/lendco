# EMI Bounce / Auto-Debit Failure

An EMI "bounce" occurs when an auto-debit or NACH presentation is returned. Common return reasons:
- R01 Insufficient funds (customer-side, bounce charge applies)
- NACH-RJ NACH rejected by sponsor bank (system-side, no charge)
- NACH-REV Mandate revoked (customer-side)
Re-presentation is attempted once within the cycle. Borrowers can also pay manually via a payment link.
