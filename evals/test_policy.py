"""
test_policy.py — Policy coverage decision and invalid policy handling tests.

Covers: E-06, E-07, AC-04, AC-07.

Strategy: instantiate PolicyAgent directly with crafted ClaimState dicts.
Uses the real standard_auto_policy.json where possible; writes modified copies
to tmp_path for expired-policy and broken-JSON scenarios.
"""

import json
import pathlib

import pytest

from agents.policy_agent import PolicyAgent


_POLICY_DIR = pathlib.Path(__file__).parent.parent / "data" / "policies"
_STANDARD_JSON = _POLICY_DIR / "standard_auto_policy.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def standard_policy_path() -> str:
    """Return the path to the real standard_auto_policy.json."""
    assert _STANDARD_JSON.exists(), "standard_auto_policy.json not found"
    return str(_STANDARD_JSON)


@pytest.fixture
def expired_policy_path(tmp_path: pathlib.Path) -> str:
    """Write a copy of the standard policy with end_date in the past."""
    data = json.loads(_STANDARD_JSON.read_text(encoding="utf-8"))
    data["end_date"] = "2024-12-31"  # expired before any 2026 incident
    target = tmp_path / "expired_policy.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return str(target)


@pytest.fixture
def broken_policy_path(tmp_path: pathlib.Path) -> str:
    """Write a JSON file missing required PolicyRules fields."""
    data = {"policy_id": "BROKEN", "policy_name": "Broken Policy"}  # no coverages, dates, etc.
    target = tmp_path / "broken_policy.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return str(target)


def _base_state(policy_path: str, claim_type: str = "collision", **claim_overrides) -> dict:
    """Return a minimal ClaimState dict for PolicyAgent tests."""
    claim = {
        "claim_id": "CLM-TEST-001",
        "policy_id": "AUTO-STD-001",
        "customer_id": "CUST-9001",
        "claim_type": claim_type,
        "incident_date": "2026-03-15",
        "incident_description": "Test incident",
        "damage_image_path": "data/images/test.jpg",
        "claim_amount": 1500.0,
        "policy_document_path": policy_path,
        "submitted_at": "2026-03-20",
    }
    claim.update(claim_overrides)
    return {
        "claim_input": claim,
        "audit_events": [],
        "image_observation": None,
        "image_claim_mismatch": False,
        "customer_profile": None,
        "claim_history": None,
        "repair_estimate": None,
        "damage_evidence_assessment": None,
        "policy_decision": None,
        "risk_assessment": None,
        "payout_recommendation": None,
        "human_approval": None,
        "intake_complete": True,
        "policy_valid": False,
        "coverage_decision": "",
        "risk_score": 0.0,
        "risk_category": "",
        "requires_human_approval": False,
        "escalation_reason": "",
        "escalation_flags": [],
        "final_report": None,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_missing_policy_file_sets_invalid(tmp_path):
    """E-06: non-existent policy path → policy_valid=False, coverage=excluded."""
    path = str(tmp_path / "nonexistent.json")
    state = _base_state(path)
    result = PolicyAgent().run(state)

    assert result["policy_valid"] is False
    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert pd["confidence"] == 0.0
    assert any("could not be loaded" in w or "not found" in w.lower() for w in pd["warnings"])


def test_excluded_damage_type_returns_excluded(standard_policy_path):
    """E-07: glass is excluded under AUTO-STD-001 Clause 2.4."""
    state = _base_state(standard_policy_path, claim_type="glass")
    result = PolicyAgent().run(state)

    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert "Clause 2.4" in pd["clause_cited"]
    assert pd["coverage_ratio"] == 0.0


def test_covered_damage_type_returns_covered(standard_policy_path):
    """Collision is covered under AUTO-STD-001 Clause 1.1 at 80% ratio."""
    state = _base_state(standard_policy_path, claim_type="collision")
    result = PolicyAgent().run(state)

    assert result["policy_valid"] is True
    assert result["coverage_decision"] == "covered"
    pd = result["policy_decision"]
    assert pd["coverage_ratio"] == 0.80
    assert pd["deductible"] == 500.0
    assert "Clause 1.1" in pd["clause_cited"]


def test_vandalism_excluded_under_standard_policy(standard_policy_path):
    """Vandalism is excluded under AUTO-STD-001 Clause 2.3."""
    state = _base_state(standard_policy_path, claim_type="vandalism")
    result = PolicyAgent().run(state)

    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert "Clause 2.3" in pd["clause_cited"]


def test_expired_policy_sets_invalid(expired_policy_path):
    """AC-07: incident_date after policy end_date → policy_valid=False."""
    state = _base_state(expired_policy_path, claim_type="collision")
    result = PolicyAgent().run(state)

    assert result["policy_valid"] is False
    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert any("expir" in w.lower() or "after" in w.lower() for w in pd["warnings"])


def test_pre_policy_incident_sets_invalid(standard_policy_path):
    """Incident date before policy effective_date (2025-01-01) → policy_valid=False."""
    state = _base_state(
        standard_policy_path,
        claim_type="collision",
        incident_date="2024-12-01",
    )
    result = PolicyAgent().run(state)

    assert result["policy_valid"] is False
    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert any("before" in w.lower() or "not yet active" in w.lower() for w in pd["warnings"])


def test_clause_cited_comes_from_policy_document(standard_policy_path):
    """AC-04: clause_cited for a covered collision matches the clause in the loaded document."""
    import json
    doc = json.loads(pathlib.Path(standard_policy_path).read_text(encoding="utf-8"))
    expected_clause = doc["coverages"]["collision"]["clause"]

    state = _base_state(standard_policy_path, claim_type="collision")
    result = PolicyAgent().run(state)

    pd = result["policy_decision"]
    assert pd["clause_cited"] == expected_clause


def test_policy_missing_required_fields_sets_invalid(broken_policy_path):
    """ValidationError from missing fields → policy_valid=False, confidence=0.0."""
    state = _base_state(broken_policy_path)
    result = PolicyAgent().run(state)

    assert result["policy_valid"] is False
    assert result["coverage_decision"] == "excluded"
    pd = result["policy_decision"]
    assert pd["confidence"] == 0.0


def test_glass_on_standard_policy_uses_correct_exclusion_clause(standard_policy_path):
    """Glass damage on AUTO-STD-001 returns excluded with the Clause 2.4 exclusion."""
    state = _base_state(standard_policy_path, claim_type="glass")
    result = PolicyAgent().run(state)

    pd = result["policy_decision"]
    # Exclusion clause must be sourced from the policy document, not invented
    assert pd["policy_id"] == "AUTO-STD-001"
    assert pd["coverage_decision"] == "excluded"
    assert "2.4" in pd["clause_cited"]
    assert pd["deductible"] == 0.0
    assert pd["coverage_ratio"] == 0.0


def test_coverage_cap_present_for_covered_collision(standard_policy_path):
    """Covered collision claims must carry a non-zero coverage_cap from the document."""
    state = _base_state(standard_policy_path, claim_type="collision")
    result = PolicyAgent().run(state)

    pd = result["policy_decision"]
    assert pd["coverage_cap"] == 15000.0
