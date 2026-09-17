"""
data_loader.py — Demo fixture loader.

Loads all data files from data/ and assembles ClaimState dicts for the pipeline.

Each JSON file is loaded once and held in an lru_cache so repeated calls inside
a single process (e.g. running all evals) pay only one disk read per file.

Policy loading priority for load_policy():
  1. JSON file (machine-readable, preferred) — used by policy_rules.py
  2. Markdown fallback — raw policy text for context

All paths are resolved relative to the project root via PROJECT_ROOT so the
module works regardless of the working directory at import time.
"""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Any

from core.claim_state import ClaimState


PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POLICIES_DIR = DATA_DIR / "policies"

# Maps policy_id to the preferred filename; key order = lookup priority.
_POLICY_FILES: dict[str, list[str]] = {
    "AUTO-STD-001": ["standard_auto_policy.json", "auto_basic_policy.md"],
    "AUTO-COMP-001": ["auto_premium_policy.md"],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _load_json_file(path: pathlib.Path) -> Any:
    """Read and parse a JSON file; cache the result for the lifetime of the process."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _index_by(records: list[dict], key: str) -> dict[str, dict]:
    """Build a {key_value: record} lookup dict from a list of dicts."""
    return {r[key]: r for r in records}


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_claim(claim_id: str) -> dict:
    """Return the claim record for claim_id from data/claims.json.

    Raises:
        FileNotFoundError: claims.json is missing.
        KeyError: claim_id not found in the file.
    """
    data = _load_json_file(DATA_DIR / "claims.json")
    index = _index_by(data["claims"], "claim_id")
    if claim_id not in index:
        raise KeyError(f"Claim '{claim_id}' not found in claims.json")
    return index[claim_id]


def load_customer(customer_id: str) -> dict:
    """Return the customer record for customer_id from data/customers.json.

    Raises:
        FileNotFoundError: customers.json is missing.
        KeyError: customer_id not found.
    """
    data = _load_json_file(DATA_DIR / "customers.json")
    index = _index_by(data["customers"], "customer_id")
    if customer_id not in index:
        raise KeyError(f"Customer '{customer_id}' not found in customers.json")
    return index[customer_id]


def load_claim_history(customer_id: str) -> list[dict]:
    """Return all historical claim records for customer_id from data/claim_history.json.

    Returns an empty list when the customer has no prior claims — absence of
    history is valid and should not raise.

    Raises:
        FileNotFoundError: claim_history.json is missing.
    """
    data = _load_json_file(DATA_DIR / "claim_history.json")
    return [h for h in data["history"] if h["customer_id"] == customer_id]


def load_repair_estimate(repair_estimate_id: str) -> dict:
    """Return the repair estimate record for repair_estimate_id from data/repair_estimates.json.

    repair_estimate_id is the 'REP-YYYY-NNN' key, not the claim_id.

    Raises:
        FileNotFoundError: repair_estimates.json is missing.
        KeyError: repair_estimate_id not found.
    """
    data = _load_json_file(DATA_DIR / "repair_estimates.json")
    index = _index_by(data["estimates"], "repair_estimate_id")
    if repair_estimate_id not in index:
        raise KeyError(
            f"Repair estimate '{repair_estimate_id}' not found in repair_estimates.json"
        )
    return index[repair_estimate_id]


def load_image_observation(filename: str) -> dict:
    """Return the image observation record for the given image filename.

    filename is the bare name, e.g. '0005.jpg'. Returns the pre-generated
    vision observations from data/image_observations.json.

    Raises:
        FileNotFoundError: image_observations.json is missing.
        KeyError: no observation entry for filename.
    """
    data = _load_json_file(DATA_DIR / "image_observations.json")
    index = _index_by(data["observations"], "filename")
    if filename not in index:
        raise KeyError(
            f"No image observation found for '{filename}' in image_observations.json"
        )
    return index[filename]


def load_policy(policy_id: str) -> dict:
    """Load the policy document for policy_id and return a normalised dict.

    Tries each candidate filename in order (JSON first, then markdown).
    Returns:
        {
            "policy_id":   str,
            "format":      "json" | "markdown",
            "source_path": str,          # absolute path to the file
            "content":     dict | str,   # parsed dict for JSON; raw text for markdown
        }

    Raises:
        FileNotFoundError: no policy document exists for policy_id.
    """
    candidates = _POLICY_FILES.get(policy_id, [])
    if not candidates:
        raise FileNotFoundError(
            f"No policy file mapping defined for policy_id '{policy_id}'"
        )

    for filename in candidates:
        path = POLICIES_DIR / filename
        if not path.exists():
            continue

        if path.suffix == ".json":
            return {
                "policy_id": policy_id,
                "format": "json",
                "source_path": str(path),
                "content": _load_json_file(path),
            }

        return {
            "policy_id": policy_id,
            "format": "markdown",
            "source_path": str(path),
            "content": path.read_text(encoding="utf-8"),
        }

    raise FileNotFoundError(
        f"Policy document for '{policy_id}' not found. "
        f"Looked for: {candidates} in {POLICIES_DIR}"
    )


def load_expected_outcome(claim_id: str) -> dict:
    """Return the expected outcome record for claim_id from data/expected_outcomes.json.

    Used by eval tests to assert pipeline results against ground truth.

    Raises:
        FileNotFoundError: expected_outcomes.json is missing.
        KeyError: claim_id not found.
    """
    data = _load_json_file(DATA_DIR / "expected_outcomes.json")
    index = _index_by(data["outcomes"], "claim_id")
    if claim_id not in index:
        raise KeyError(
            f"Expected outcome for '{claim_id}' not found in expected_outcomes.json"
        )
    return index[claim_id]


# ── build_initial_state ────────────────────────────────────────────────────────

def build_initial_state(claim_id: str) -> ClaimState:
    """Assemble a fully initialised ClaimState dict for the given claim_id.

    Loads and cross-links: claim → customer → claim history → repair estimate
    → policy document. All agent output fields are set to None. All routing
    flags are set to their zero/False defaults. The returned dict is ready
    to pass directly to graph.invoke().

    Raises:
        KeyError: claim_id not found, or any linked record is missing.
        FileNotFoundError: a required data file is absent.
    """
    claim = load_claim(claim_id)
    customer = load_customer(claim["customer_id"])
    history_records = load_claim_history(claim["customer_id"])
    estimate = load_repair_estimate(claim["repair_estimate_id"])
    policy_doc = load_policy(claim["policy_id"])

    # Resolve the full image path. None for missing-evidence claims.
    image_filename: str | None = claim.get("damage_image")
    image_path: str | None = (
        str(DATA_DIR / "images" / image_filename) if image_filename else None
    )

    # Build ClaimInput-compatible dict.
    claim_input = {
        "claim_id": claim["claim_id"],
        "policy_id": claim["policy_id"],
        "customer_id": claim["customer_id"],
        "claim_type": claim["claim_type"],
        "incident_date": claim["incident_date"],
        "incident_description": claim["customer_statement"],
        "damage_image_path": image_path,
        "claim_amount": claim["claimed_amount"],
        "policy_document_path": policy_doc["source_path"],
        "submitted_at": claim.get("reported_date"),
    }

    # Build CustomerProfile-compatible dict.
    # email / phone / address are not in the fixture files; left as empty strings.
    customer_profile = {
        "customer_id": customer["customer_id"],
        "name": customer["customer_name"],
        "email": "",
        "phone": "",
        "policy_id": customer["policy_id"],
        "policy_start_date": customer["policy_start_date"],
        "address": "",
    }

    # Build ClaimHistory-compatible dict.
    prior_claims = [
        {
            "claim_id": h["history_id"],
            "date": h["incident_date"],
            "claim_type": h["claim_type"],
            "amount": h["repair_estimate"],
            "outcome": h["outcome"],
        }
        for h in history_records
    ]
    claim_history = {
        "customer_id": claim["customer_id"],
        "prior_claims": prior_claims,
        "total_claims_count": len(prior_claims),
        # customers.json carries the 24-month count directly; avoids re-computing dates here.
        "claims_in_last_24_months": customer.get("previous_claims_last_24_months", 0),
    }

    # Build RepairEstimate-compatible dict.
    # damaged_parts from the estimate become line items; individual part costs are
    # not itemised in the fixture, so cost is 0.0 (the total is in estimated_amount).
    repair_estimate = {
        "claim_id": claim["claim_id"],
        "estimated_amount": estimate["total_estimated_cost"],
        "workshop_name": estimate.get("repair_shop", ""),
        "estimate_date": claim.get("reported_date", claim["incident_date"]),
        "line_items": [
            {"description": part, "cost": 0.0}
            for part in estimate.get("damaged_parts", [])
        ],
        "estimate_plausibility": estimate.get("estimate_plausibility", "unclear"),
    }

    return {
        # ── PIPELINE INPUTS ───────────────────────────────────────────────────
        "claim_input": claim_input,
        "customer_profile": customer_profile,
        "claim_history": claim_history,
        "repair_estimate": repair_estimate,
        "image_observation": None,

        # ── AGENT OUTPUTS ─────────────────────────────────────────────────────
        "damage_evidence_assessment": None,
        "policy_decision": None,
        "risk_assessment": None,
        "payout_recommendation": None,

        # ── HUMAN REVIEW ──────────────────────────────────────────────────────
        "human_approval": None,

        # ── ROUTING FLAGS ─────────────────────────────────────────────────────
        "intake_complete": False,
        "image_claim_mismatch": False,
        "policy_valid": False,
        "coverage_decision": "",
        "risk_score": 0.0,
        "risk_category": "",
        "requires_human_approval": False,
        "escalation_reason": "",
        "escalation_flags": [],

        # ── AUDIT AND FINAL OUTPUT ────────────────────────────────────────────
        "audit_events": [],
        "final_report": None,
    }
