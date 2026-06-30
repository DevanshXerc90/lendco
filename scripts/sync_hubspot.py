"""Sync synthetic borrower data into HubSpot CRM.

Verifies that custom properties exist (or creates them if missing) and pushes the 
first 20 test borrowers to HubSpot CRM using private app access token.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import httpx

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


def main() -> None:
    token = settings.HUBSPOT_ACCESS_TOKEN
    if not token:
        print("ERROR: HUBSPOT_ACCESS_TOKEN is not set.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Define custom properties to create in HubSpot
    custom_props = [
        {
            "name": "borrower_id",
            "label": "Borrower ID",
            "type": "string",
            "fieldType": "text",
            "groupName": "contactinformation"
        },
        {
            "name": "loan_id",
            "label": "Loan ID",
            "type": "string",
            "fieldType": "text",
            "groupName": "contactinformation"
        },
        {
            "name": "loan_amount",
            "label": "Loan Amount",
            "type": "number",
            "fieldType": "number",
            "groupName": "contactinformation"
        },
        {
            "name": "interest_rate",
            "label": "Interest Rate",
            "type": "number",
            "fieldType": "number",
            "groupName": "contactinformation"
        },
        {
            "name": "emi_amount",
            "label": "EMI Amount",
            "type": "number",
            "fieldType": "number",
            "groupName": "contactinformation"
        },
        {
            "name": "delinquency_status",
            "label": "Delinquency Status",
            "type": "string",
            "fieldType": "text",
            "groupName": "contactinformation"
        },
        {
            "name": "dpd_days",
            "label": "DPD Days",
            "type": "number",
            "fieldType": "number",
            "groupName": "contactinformation"
        }
    ]

    # Fetch existing properties
    print("[HubSpot Sync] Fetching existing properties...")
    r = httpx.get("https://api.hubapi.com/crm/v3/properties/contacts", headers=headers, timeout=10.0)
    r.raise_for_status()
    existing_names = {p["name"] for p in r.json()["results"]}

    # Create missing properties
    for prop in custom_props:
        if prop["name"] not in existing_names:
            print(f"[HubSpot Sync] Creating property '{prop['name']}'...")
            res = httpx.post("https://api.hubapi.com/crm/v3/properties/contacts", json=prop, headers=headers, timeout=10.0)
            if res.status_code == 201:
                print(f"[HubSpot Sync] Created property '{prop['name']}' successfully.")
            else:
                print(f"[HubSpot Sync] Error creating '{prop['name']}': {res.text}")
        else:
            print(f"[HubSpot Sync] Property '{prop['name']}' already exists.")

    # Load borrowers
    borrowers_file = settings.DATA_DIR / "borrowers.json"
    if not borrowers_file.exists():
        print(f"[HubSpot Sync] Error: {borrowers_file} does not exist. Run: python -m data.generate first.")
        sys.exit(1)

    borrowers = json.loads(borrowers_file.read_text(encoding="utf-8"))
    # Sync the first 20 borrowers (enough for testing/demos)
    sync_count = min(20, len(borrowers))
    print(f"[HubSpot Sync] Syncing first {sync_count} borrowers to HubSpot...")

    for i in range(sync_count):
        b = borrowers[i]
        email = b["email"]
        name_parts = b["name"].split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Search by email
        search_payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": email
                        }
                    ]
                }
            ]
        }
        search_res = httpx.post("https://api.hubapi.com/crm/v3/objects/contacts/search", json=search_payload, headers=headers, timeout=10.0)
        search_res.raise_for_status()
        results = search_res.json().get("results")

        # Prep properties
        properties = {
            "firstname": first_name,
            "lastname": last_name,
            "phone": b["phone"],
            "borrower_id": b["borrower_id"],
            "loan_id": b["loan_id"],
            "loan_amount": str(b["loan_amount"]),
            "interest_rate": str(b["interest_rate"]),
            "emi_amount": str(b["emi_amount"]),
            "delinquency_status": b["delinquency_status"],
            "dpd_days": str(b["dpd_days"])
        }

        if results:
            contact_id = results[0]["id"]
            print(f"[HubSpot Sync] Updating contact {email} (ID: {contact_id})...")
            update_res = httpx.patch(f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}", json={"properties": properties}, headers=headers, timeout=10.0)
            update_res.raise_for_status()
        else:
            print(f"[HubSpot Sync] Creating new contact {email}...")
            create_res = httpx.post("https://api.hubapi.com/crm/v3/objects/contacts", json={"properties": {**properties, "email": email}}, headers=headers, timeout=10.0)
            create_res.raise_for_status()

    print("[HubSpot Sync] Sync completed successfully!")


if __name__ == "__main__":
    main()
