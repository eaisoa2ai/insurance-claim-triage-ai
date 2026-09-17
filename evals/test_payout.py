"""
test_payout.py — Payout calculation and auto-approve gating tests.

Covers: E-04, E-05, E-13, AC-08.

Config thresholds (from risk_thresholds.yaml):
  - payout.auto_approve_limit: 5000   (payout > 5000 → not auto-eligible)
  - risk.confidence_floor: 0.75       (policy confidence < 0.75 → not auto-eligible)
  - risk.human_review_threshold: 70   (risk_score >= 70 → not auto-eligible)

Strategy: call PayoutAgent().run() directly with crafted state dicts.
The payout formula is: max(0, min(repair × ratio − deductible, cap)).
"""

import pytest

from agents.payout_agent import PayoutAgent


# ── State builder ─────────────────────────────────────────────────────────────

def _build_state(
    repair_estimate: float,
    coverage_ratio: float = 0.80,
    deductible: float = 500.0,
    coverage_cap: float = 15_000.0,
    risk_score: float = 20.0,
    risk_category: str = "low",
    policy_confidence: float = 0.95,
    image_claim_mismatch: bool = False,
    coverage_decision: str = "covered",
) -> dict:
    """Return a minimal ClaimState for PayoutAgent tests with sensible defaults."""
    return {
        "claim_input": {"claim_id": "CLM-TEST-001"},
        "policy_decision": {
            "coverage_decision": coverage_decision,
            "coverage_ratio": coverage_ratio,
            "deductible": deductible,
            "coverage_cap": coverage_cap,
            "confidence": policy_confidence,
        },
        "damage_evidence_assessment": {
            "image_claim_mismatch": image_claim_mismatch,
            "severity": "moderate",
            "consistency_score": 0.80,
        },
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_category": risk_category,
        },
        "repair_estimate": {
            "estimated_amount": repair_estimate,
        },
        "requires_human_approval": False,
        "audit_events": [],
    }


# ── Calculation correctness ───────────────────────────────────────────────────

def test_payout_formula_two_decimal_places():
    """AC-08: payout = repair × ratio − deductible, rounded to 2dp."""
    # 1000 × 0.80 − 500 = 300.00
    state = _build_state(repair_estimate=1000.0)
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert pr["payout_amount"] == pytest.approx(300.00, abs=0.01)


def test_payout_standard_formula_clm001_equivalent():
    """CLM-2026-001: 910 × 0.80 − 500 = 228.00."""
    state = _build_state(repair_estimate=910.0)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(228.00, abs=0.01)


def test_payout_clamped_to_zero_when_deductible_exceeds_covered_amount():
    """When gross payout < deductible, net payout is clamped to 0."""
    # 500 × 0.80 = 400 < deductible 500 → clamped to 0
    state = _build_state(repair_estimate=500.0, deductible=500.0)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(0.0)


def test_payout_clamped_to_coverage_cap():
    """When net payout exceeds cap, payout is clamped to coverage_cap."""
    # 20000 × 0.80 − 500 = 15500; cap = 15000 → clamped to 15000
    state = _build_state(repair_estimate=20_000.0, coverage_cap=15_000.0)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(15_000.0, abs=0.01)


def test_zero_deductible_glass_coverage():
    """Zero deductible: payout = repair × ratio exactly."""
    # 660 × 1.00 − 0 = 660.00
    state = _build_state(
        repair_estimate=660.0,
        coverage_ratio=1.0,
        deductible=0.0,
        coverage_cap=1_500.0,
    )
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(660.0, abs=0.01)


def test_full_coverage_ratio():
    """coverage_ratio=1.0: payout = repair − deductible."""
    state = _build_state(repair_estimate=1000.0, coverage_ratio=1.0, deductible=200.0)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(800.0, abs=0.01)


# ── Auto-approve gating ───────────────────────────────────────────────────────

def test_payout_above_limit_not_auto_eligible():
    """E-04: payout > 5000 → auto_approve_eligible=False."""
    # 7000 × 0.80 − 500 = 5100 > 5000
    state = _build_state(repair_estimate=7_000.0)
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert pr["payout_amount"] > 5_000.0
    assert pr["auto_approve_eligible"] is False


def test_payout_below_limit_auto_eligible():
    """E-05: payout < 5000, all clear → auto_approve_eligible=True."""
    # 6250 × 0.80 − 500 = 4500 < 5000
    state = _build_state(repair_estimate=6_250.0)
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert pr["payout_amount"] < 5_000.0
    assert pr["auto_approve_eligible"] is True


def test_payout_exactly_at_limit_auto_eligible():
    """Limit is exclusive: payout = 5000 does NOT trigger escalation."""
    # 6875 × 0.80 − 500 = 5000 exactly
    state = _build_state(repair_estimate=6_875.0)
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert pr["payout_amount"] == pytest.approx(5_000.0, abs=0.01)
    assert pr["auto_approve_eligible"] is True


def test_image_mismatch_blocks_auto_approve():
    """image_claim_mismatch=True → auto_approve_eligible=False regardless of payout."""
    state = _build_state(repair_estimate=1_000.0, image_claim_mismatch=True)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["auto_approve_eligible"] is False


def test_high_risk_score_blocks_auto_approve():
    """risk_score >= 70 → auto_approve_eligible=False."""
    state = _build_state(repair_estimate=1_000.0, risk_score=70.0)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["auto_approve_eligible"] is False


def test_low_policy_confidence_blocks_auto_approve():
    """PolicyAgent confidence < 0.75 → auto_approve_eligible=False."""
    state = _build_state(repair_estimate=1_000.0, policy_confidence=0.70)
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["auto_approve_eligible"] is False


def test_excluded_coverage_yields_zero_payout():
    """Excluded coverage → payout = 0 regardless of repair estimate."""
    state = _build_state(
        repair_estimate=5_000.0,
        coverage_decision="excluded",
        coverage_ratio=0.0,
        deductible=0.0,
    )
    result = PayoutAgent().run(state)
    assert result["payout_recommendation"]["payout_amount"] == pytest.approx(0.0)


def test_severe_damage_within_limit_auto_eligible():
    """E-13: severe damage, consistent story, payout within limit → auto_approve_eligible=True."""
    # 5625 × 0.80 − 500 = 4000 < 5000; risk_score=40 < 70; mismatch=False
    state = _build_state(
        repair_estimate=5_625.0,
        risk_score=40.0,
        risk_category="medium",
        image_claim_mismatch=False,
        policy_confidence=0.95,
    )
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert pr["payout_amount"] == pytest.approx(4_000.0, abs=0.01)
    assert pr["auto_approve_eligible"] is True


def test_payout_recommendation_has_required_base_fields():
    """Every PayoutRecommendation has confidence, evidence, reasoning_summary, warnings."""
    state = _build_state(repair_estimate=1_000.0)
    result = PayoutAgent().run(state)
    pr = result["payout_recommendation"]
    assert "confidence" in pr
    assert "evidence" in pr
    assert "reasoning_summary" in pr
    assert "warnings" in pr
    assert isinstance(pr["evidence"], list)
    assert isinstance(pr["warnings"], list)
