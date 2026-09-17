"""
test_routes.py — Routing edge tests.

Covers: E-04, E-05, E-09, E-10, E-11, E-12 and all escalation conditions.

Strategy: tests call routing.router_node() and route_decision() directly with
crafted state dicts to avoid spinning up the full LangGraph pipeline.
Config thresholds (from risk_thresholds.yaml):
  - risk.human_review_threshold: 70
  - risk.confidence_floor:       0.75
  - payout.auto_approve_limit:   5000
"""

import pytest
from datetime import datetime, timezone

from core.routing import (
    router_node,
    route_decision,
    _check_image_mismatch,
    _check_low_confidence,
    _check_high_risk_score,
    _check_invalid_policy,
    _check_excluded_coverage,
    _check_payout_limit,
    _check_critical_risk,
)
from core.claim_state import HumanApprovalDecision


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    """Minimal passing state — all conditions clear, routing goes to auto_approve."""
    state = {
        "claim_input": {
            "claim_id": "CLM-TEST-001",
            "damage_image_path": "data/images/test.jpg",
        },
        "intake_complete": True,
        "image_claim_mismatch": False,
        "policy_valid": True,
        "coverage_decision": "covered",
        "risk_score": 0.0,
        "risk_category": "low",
        "requires_human_approval": False,
        "escalation_reason": "",
        "escalation_flags": [],
        "repair_estimate": {
            "estimate_plausibility": "reasonable",
            "estimated_amount": 1000.0,
        },
        "damage_evidence_assessment": {"confidence": 0.90},
        "policy_decision": {"confidence": 0.90},
        "risk_assessment": {"confidence": 0.90},
        "payout_recommendation": {"confidence": 0.90, "payout_amount": 500.0},
        "audit_events": [],
        "human_approval": None,
    }
    state.update(overrides)
    return state


def _run_router(state: dict) -> dict:
    """Run router_node and merge result into state for route_decision."""
    result = router_node(state)
    return {**state, **result}


# ── Condition check unit tests ────────────────────────────────────────────────

def test_image_mismatch_triggers_escalation():
    state = _base_state(image_claim_mismatch=True)
    assert _check_image_mismatch(state) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True
    assert any("mismatch" in flag.lower() for flag in updated["escalation_flags"])


def test_low_confidence_triggers_escalation():
    # policy confidence 0.74 is below the 0.75 floor in risk_thresholds.yaml
    state = _base_state(policy_decision={"confidence": 0.74})
    assert _check_low_confidence(state, 0.75) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True


def test_risk_score_at_threshold_triggers_escalation():
    """E-09: risk_score = 70 (at boundary) → escalate."""
    state = _base_state(risk_score=70.0)
    assert _check_high_risk_score(state, 70.0) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True


def test_risk_score_below_threshold_does_not_escalate():
    """E-10: risk_score = 69 (one below boundary) → no risk escalation."""
    state = _base_state(risk_score=69.0)
    assert _check_high_risk_score(state, 70.0) is False
    updated = _run_router(state)
    # risk score alone doesn't trigger; all other conditions are clear
    assert updated["requires_human_approval"] is False


def test_invalid_policy_triggers_escalation():
    state = _base_state(policy_valid=False)
    assert _check_invalid_policy(state) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True


def test_excluded_coverage_routes_to_rejected():
    """Excluded coverage is NOT a human-review trigger — it routes to 'rejected'."""
    state = _base_state(coverage_decision="excluded")
    assert _check_excluded_coverage(state) is True
    updated = _run_router(state)
    # excluded alone does not set requires_human_approval
    assert updated["requires_human_approval"] is False
    assert route_decision(updated) == "rejected"


def test_payout_above_limit_triggers_escalation():
    """E-04: payout = 5001 → human review."""
    state = _base_state(payout_recommendation={"confidence": 0.90, "payout_amount": 5001.0})
    assert _check_payout_limit(state, 5000.0) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True


def test_payout_below_limit_auto_approves():
    """E-05: payout = 4999, all clear → auto-approve."""
    state = _base_state(payout_recommendation={"confidence": 0.90, "payout_amount": 4999.0})
    assert _check_payout_limit(state, 5000.0) is False
    updated = _run_router(state)
    assert updated["requires_human_approval"] is False
    assert route_decision(updated) == "auto_approve"


def test_payout_exactly_at_limit_auto_approves():
    """Limit is exclusive: > 5000 escalates, == 5000 does not."""
    state = _base_state(payout_recommendation={"confidence": 0.90, "payout_amount": 5000.0})
    assert _check_payout_limit(state, 5000.0) is False
    updated = _run_router(state)
    assert route_decision(updated) == "auto_approve"


def test_critical_risk_category_triggers_escalation():
    state = _base_state(risk_category="critical")
    assert _check_critical_risk(state) is True
    updated = _run_router(state)
    assert updated["requires_human_approval"] is True


def test_all_conditions_clear_routes_to_auto_approve():
    state = _base_state()
    updated = _run_router(state)
    assert updated["requires_human_approval"] is False
    assert route_decision(updated) == "auto_approve"


def test_multiple_conditions_all_appear_in_escalation_flags():
    state = _base_state(image_claim_mismatch=True, policy_valid=False)
    updated = _run_router(state)
    flags = updated["escalation_flags"]
    # Both image mismatch and invalid policy should appear
    assert any("mismatch" in f.lower() or "image" in f.lower() for f in flags)
    assert any("policy" in f.lower() or "invalid" in f.lower() or "expired" in f.lower() for f in flags)
    assert len(flags) >= 2


# ── Audit trail checks ────────────────────────────────────────────────────────

def test_agent_output_has_required_base_fields():
    """AC-05: all agent output models declare confidence, evidence, reasoning_summary, warnings."""
    from core.claim_state import (
        DamageEvidenceAssessment,
        PolicyDecision,
        RiskAssessment,
        PayoutRecommendation,
    )
    for model_cls in [DamageEvidenceAssessment, PolicyDecision, RiskAssessment, PayoutRecommendation]:
        fields = model_cls.model_fields
        assert "confidence" in fields, f"{model_cls.__name__} missing 'confidence'"
        assert "evidence" in fields, f"{model_cls.__name__} missing 'evidence'"
        assert "reasoning_summary" in fields, f"{model_cls.__name__} missing 'reasoning_summary'"
        assert "warnings" in fields, f"{model_cls.__name__} missing 'warnings'"


def test_audit_trail_entries_have_required_fields():
    """AC-06: audit entries written by router_node have agent, timestamp, routing_decision."""
    state = _base_state()
    result = router_node(state)
    assert len(result["audit_events"]) == 1
    entry = result["audit_events"][0]
    assert "agent" in entry
    assert "timestamp" in entry
    assert "routing_decision" in entry
    assert entry["agent"] == "router"


def test_adjuster_approval_recorded_in_audit_trail():
    """E-11: HumanApprovalDecision with approved=True validates and carries adjuster note."""
    decision = HumanApprovalDecision(
        required=True,
        escalation_reason="Risk score above threshold",
        approved=True,
        adjuster_note="Claim verified with supporting documents",
        decided_at=datetime.now(timezone.utc),
    )
    data = decision.model_dump()
    assert data["approved"] is True
    assert data["adjuster_note"] is not None
    assert len(data["adjuster_note"]) >= 10


def test_adjuster_rejection_recorded_in_audit_trail():
    """E-12: HumanApprovalDecision with approved=False validates and carries adjuster note."""
    decision = HumanApprovalDecision(
        required=True,
        escalation_reason="Image/claim mismatch detected",
        approved=False,
        adjuster_note="Damage inconsistent with reported incident",
        decided_at=datetime.now(timezone.utc),
    )
    data = decision.model_dump()
    assert data["approved"] is False
    assert len(data["adjuster_note"]) >= 10


def test_adjuster_note_too_short_raises_validation_error():
    """HumanApprovalDecision requires adjuster_note >= 10 chars when a decision is set."""
    with pytest.raises(Exception):
        HumanApprovalDecision(
            required=True,
            approved=True,
            adjuster_note="Too short",  # 9 chars
        )
