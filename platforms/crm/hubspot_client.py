"""HubSpot CRM API v3 client.

Interfaces with a real HubSpot developer private app using HTTP.
Provides contact properties and maps contact search and details to LendCo format.
"""
from __future__ import annotations

import httpx
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings  # noqa: E402


class HubSpotCRMClient:
    """HubSpot API v3 wrapper client for managing borrower contacts."""

    def __init__(self):
        self.token = settings.HUBSPOT_ACCESS_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.hubapi.com/crm/v3/objects/contacts"

    def _search(self, filter_property: str, value: str) -> dict | None:
        """Helper to search contacts by property value."""
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": filter_property,
                            "operator": "EQ",
                            "value": value
                        }
                    ]
                }
            ],
            "properties": [
                "firstname", "lastname", "email", "phone", "borrower_id", 
                "loan_id", "loan_amount", "interest_rate", "emi_amount", 
                "delinquency_status", "dpd_days"
            ]
        }
        r = httpx.post(f"{self.base_url}/search", json=payload, headers=self.headers, timeout=8.0)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            return data["results"][0]
        return None

    def get_borrower(self, borrower_id: str) -> dict | None:
        contact = self._search("borrower_id", borrower_id)
        if not contact:
            return None
        props = contact["properties"]
        return {
            "borrower_id": props.get("borrower_id"),
            "name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
            "email": props.get("email"),
            "phone": props.get("phone"),
            "loan_id": props.get("loan_id"),
            "loan_amount": float(props.get("loan_amount")) if props.get("loan_amount") else None,
            "interest_rate": float(props.get("interest_rate")) if props.get("interest_rate") else None,
            "emi_amount": float(props.get("emi_amount")) if props.get("emi_amount") else None,
            "delinquency_status": props.get("delinquency_status"),
            "dpd_days": int(props.get("dpd_days")) if props.get("dpd_days") else 0,
            "hubspot_contact_id": contact["id"],
            "_source": "hubspot"
        }

    def find_by_phone(self, phone: str) -> dict | None:
        contact = self._search("phone", phone)
        if not contact and not phone.startswith("+91"):
            contact = self._search("phone", f"+91{phone}")
        if not contact:
            return None
        
        props = contact["properties"]
        return {
            "borrower_id": props.get("borrower_id"),
            "name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
            "email": props.get("email"),
            "phone": props.get("phone"),
            "loan_id": props.get("loan_id"),
            "loan_amount": float(props.get("loan_amount")) if props.get("loan_amount") else None,
            "interest_rate": float(props.get("interest_rate")) if props.get("interest_rate") else None,
            "emi_amount": float(props.get("emi_amount")) if props.get("emi_amount") else None,
            "delinquency_status": props.get("delinquency_status"),
            "dpd_days": int(props.get("dpd_days")) if props.get("dpd_days") else 0,
            "hubspot_contact_id": contact["id"],
            "_source": "hubspot"
        }

    def get_kyc(self, borrower_id: str) -> dict | None:
        payload = {
            "filterGroups": [{"filters": [{"propertyName": "borrower_id", "operator": "EQ", "value": borrower_id}]}],
            "properties": ["firstname", "lastname", "email", "phone", "address", "city", "state", "zipcode"]
        }
        r = httpx.post(f"{self.base_url}/search", json=payload, headers=self.headers, timeout=8.0)
        r.raise_for_status()
        res = r.json().get("results")
        if not res:
            return None
        props = res[0]["properties"]
        addr_parts = [props.get("address"), props.get("city"), props.get("state"), props.get("zipcode")]
        full_address = ", ".join([p for p in addr_parts if p]) or "Address not specified in HubSpot"
        return {
            "borrower_id": borrower_id,
            "name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
            "email": props.get("email"),
            "phone": props.get("phone"),
            "address": full_address,
            "_source": "hubspot"
        }

    def update(self, borrower_id: str, field: str, value: str, note: str | None = None) -> dict:
        contact = self._search("borrower_id", borrower_id)
        if not contact:
            return {"updated": False, "error": f"Borrower {borrower_id} not found in HubSpot"}
        
        contact_id = contact["id"]
        hubspot_field = field
        if field == "phone_number":
            hubspot_field = "phone"
        
        payload = {
            "properties": {
                hubspot_field: value
            }
        }
        
        r = httpx.patch(f"{self.base_url}/{contact_id}", json=payload, headers=self.headers, timeout=8.0)
        r.raise_for_status()
        
        if note:
            try:
                note_payload = {
                    "properties": {
                        "hs_note_body": note,
                        "hs_timestamp": int(time.time() * 1000)
                    },
                    "associations": [
                        {
                            "to": {"id": contact_id},
                            "types": [
                                {
                                    "associationCategory": "HUBSPOT_DEFINED",
                                    "associationTypeId": 202
                                }
                            ]
                        }
                    ]
                }
                httpx.post(
                    "https://api.hubapi.com/crm/v3/objects/notes",
                    json=note_payload,
                    headers=self.headers,
                    timeout=8.0
                )
            except Exception as note_err:
                print(f"[HubSpot] Note creation failed: {note_err}")
                
        return {"updated": True, "borrower_id": borrower_id, "field": field, "value": value}
