"""Razorpay test-mode client — real integration with Razorpay for payment links.

Uses the Razorpay REST API in test mode. Payment history and gateway logs 
remain on the mock/local database to avoid creating massive amounts of dummy live records.
"""
from __future__ import annotations

import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402


class RazorpayPaymentsClient:
    """Razorpay-backed payment link generation."""

    def __init__(self):
        self.key_id = settings.RAZORPAY_KEY_ID
        self.key_secret = settings.RAZORPAY_KEY_SECRET

    def create_payment_link(self, borrower_id: str, amount: float,
                            purpose: str = "EMI payment") -> dict:
        """Generate a real Razorpay Payment Link (test mode).

        Returns a dict matching the format expected by the rest of the system.
        """
        try:
            # Amount in paise (INR * 100)
            amount_paise = int(amount * 100)
            
            payload = {
                "amount": amount_paise,
                "currency": "INR",
                "accept_partial": False,
                "description": f"LendCo EMI Payment - {borrower_id}",
                "customer": {
                    "name": borrower_id,
                    "email": f"{borrower_id.lower()}@lendco.example",
                    "contact": "+919999999999" # Test mobile number
                },
                "notify": {
                    "sms": False,
                    "email": False
                },
                "reminder_enable": False,
                "notes": {
                    "borrower_id": borrower_id,
                    "purpose": purpose
                },
                "callback_url": "https://lendco.example/payment-success",
                "callback_method": "get"
            }
            
            r = httpx.post(
                "https://api.razorpay.com/v1/payment_links",
                json=payload,
                auth=(self.key_id, self.key_secret),
                timeout=8.0
            )
            r.raise_for_status()
            data = r.json()
            
            return {
                "link_id": data["id"],
                "borrower_id": borrower_id,
                "amount": amount,
                "purpose": purpose,
                "url": data["short_url"],  # Razorpay checkout short URL (e.g. https://rzp.io/i/...)
                "status": "active",
                "valid_for_hours": 24,
                "_source": "razorpay",
            }
        except Exception as e:
            print(f"[Razorpay] payment link error: {e}")
            # Return a mock-format response so the agent doesn't break
            return {
                "link_id": "PL-RAZORPAY-ERR",
                "borrower_id": borrower_id,
                "amount": amount,
                "purpose": purpose,
                "url": f"https://pay.lendco.example/fallback/{borrower_id.lower()}",
                "status": "active",
                "valid_for_hours": 24,
                "_source": "razorpay_error",
                "error": str(e),
            }
