"""
test_risk.py — Risk scoring tests with boundary values for every factor weight.

Covers: E-03, E-09, E-10.

Config thresholds (from risk_thresholds.yaml):
  - human_review_threshold: 70   (score >= 70 → human review + category "high" or "critical")
  - factors.consistency_weight: 35
  - factors.prior_claims_weight: 25  (threshold: 2 claims in 24 months)
  - factors.late_reporting_weight: 15  (threshold: > 30 days)
  - factors.estimate_vs_severity_weight: 15  (uses severity_estimate_ranges.minor.max: 3000)
  - factors.policy_tenure_weight: 10  (threshold: < 6 months)

Strategy: call RiskAgent().run() with crafted state dicts.  Assert categories and
require_human_approval flags rather than exact score values to avoid hardcoding weights.
"""

from datetime import date, timedelta

import pytest

from agents.risk_agent import RiskAgent


# ── State builder ─────────────────────────────────────────────────────────────

def _build_state(
    consistency_score: float = 1.0,
    prior_claims_count: int = 0,
    days_since_incident: int = 5,
    repair_estimate: float = 1000.0,
    severity: str = "minor",
    policy_effective_date: str = "2024-01-01",
    incident_date: str = "2026-06-01",
    coverage_decision: str = "covered",
) -> dict:
    """Return a minimal ClaimState for RiskAgent tests.

    submitted_at is derived as incident_date + days_since_incident so the
    late-reporting factor can be tested without manual date arithmetic.
    """
    incident = date.fromisoformat(incident_date)
    submitted = incident + timedelta(days=days_since_incident)
    return {
        "claim_input": {
            "claim_id": "CLM-TEST-001",
            "incident_date": incident_date,
            "submitted_at": submitted.isoformat(),
        },
        "damage_evidence_assessment": {
            "consistency_score": consistency_score,
            "severity": severity,
        },
        "claim_history": {
            "claims_in_last_24_months": prior_claims_count,
        },
        "repair_estimate": {
            "estimated_amount": repair_estimate,
        },
        "customer_profile": {
            "policy_start_date": policy_effective_date,
        },
        "coverage_decision": coverage_decision,
        "escalation_flags": [],
        "escalation_reason": "",
        "requires_human_approval": False,
        "audit_events": [],
    }


# ── Category boundary tests ───────────────────────────────────────────────────

def test_clean_claim_scores_low():
    """Perfect consistency, no prior claims, on-time, small estimate → low risk."""
    state = _build_state(consistency_score=1.0)
    result = RiskAgent().run(state)
    assert result["risk_assessment"]["risk_category"] == "low"
    assert result["requires_human_approval"] is False


def test_score_40_is_low_category():
    """Consistency=0 (c1=35), no other factors → score≈35, category=low (threshold is 41)."""
    state = _build_state(consistency_score=0.0)
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_category"] == "low"
    assert ra["risk_score"] < 41


def test_score_41_is_medium_category():
    """Consistency=0 (c1=35) + 1 prior claim → score>40, category=medium."""
    state = _build_state(consistency_score=0.0, prior_claims_count=1)
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_category"] == "medium"
    assert 41 <= ra["risk_score"] < 70


def test_score_70_is_high_category():
    """E-09 boundary: consistency=0 + 2 prior claims + 31 late days → score≥70, category=high."""
    state = _build_state(
        consistency_score=0.0,
        prior_claims_count=2,
        days_since_incident=31,
    )
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_category"] == "high"
    assert ra["risk_score"] >= 70
    assert result["requires_human_approval"] is True


def test_score_69_is_medium_category():
    """E-10 boundary: consistency=0 + 2 prior claims, on-time → score≈60, category=medium."""
    state = _build_state(consistency_score=0.0, prior_claims_count=2)
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_category"] == "medium"
    assert ra["risk_score"] < 70
    assert result["requires_human_approval"] is False


def test_score_85_is_critical_category():
    """Consistency=0 + 2 claims + 31 days late + over-ceiling estimate → score≥85, critical."""
    state = _build_state(
        consistency_score=0.0,
        prior_claims_count=2,
        days_since_incident=31,
        repair_estimate=5000.0,  # above minor ceiling 3000 → factor fires
        severity="minor",
    )
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_category"] == "critical"
    assert ra["risk_score"] >= 85
    assert result["requires_human_approval"] is True


# ── Factor tests ──────────────────────────────────────────────────────────────

def test_zero_consistency_maximises_contribution():
    """consistency_score=0.0 gives maximum c1 contribution (35 points)."""
    state = _build_state(consistency_score=0.0)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c1 = next(f for f in factors if "consistency" in f["factor"])
    assert c1["triggered"] is True
    assert c1["contribution"] == pytest.approx(35.0)


def test_full_consistency_contributes_nothing():
    """consistency_score=1.0 → inverted = 0 → c1 contribution = 0."""
    state = _build_state(consistency_score=1.0)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c1 = next(f for f in factors if "consistency" in f["factor"])
    assert c1["triggered"] is False
    assert c1["contribution"] == pytest.approx(0.0)


def test_prior_claims_at_threshold_fires():
    """prior_claims_count=2 (at threshold) → full prior_claims weight fires."""
    state = _build_state(prior_claims_count=2)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c2 = next(f for f in factors if "prior_claims" in f["factor"])
    assert c2["triggered"] is True
    assert c2["contribution"] == pytest.approx(25.0)


def test_prior_claims_below_threshold_partial():
    """prior_claims_count=1 (below threshold 2) → partial contribution, not full."""
    state = _build_state(prior_claims_count=1)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c2 = next(f for f in factors if "prior_claims" in f["factor"])
    assert c2["triggered"] is True
    assert 0 < c2["contribution"] < 25.0


def test_zero_prior_claims_silent():
    """0 prior claims → c2 contribution = 0."""
    state = _build_state(prior_claims_count=0)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c2 = next(f for f in factors if "prior_claims" in f["factor"])
    assert c2["triggered"] is False
    assert c2["contribution"] == 0.0


def test_late_reporting_boundary_at_30_does_not_fire():
    """days_since_incident=30 is NOT late (threshold is > 30, not >= 30)."""
    state = _build_state(days_since_incident=30)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c3 = next(f for f in factors if "late_reporting" in f["factor"])
    assert c3["triggered"] is False
    assert c3["contribution"] == 0.0


def test_late_reporting_at_31_fires():
    """days_since_incident=31 (one over) → late reporting factor fires."""
    state = _build_state(days_since_incident=31)
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c3 = next(f for f in factors if "late_reporting" in f["factor"])
    assert c3["triggered"] is True
    assert c3["contribution"] == pytest.approx(15.0)


def test_policy_tenure_under_6_months_fires():
    """Policy started 3 months before incident → new-policy flag fires."""
    state = _build_state(
        policy_effective_date="2026-03-01",
        incident_date="2026-06-01",
    )
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    # Factor key is "new_policy_claim" (label: "Policy tenure at time of incident")
    c5 = next(f for f in factors if "new_policy" in f["factor"])
    assert c5["triggered"] is True
    assert c5["contribution"] == pytest.approx(10.0)


def test_policy_tenure_over_6_months_silent():
    """Policy started >6 months before incident → tenure factor silent."""
    state = _build_state(
        policy_effective_date="2024-01-01",
        incident_date="2026-06-01",
    )
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c5 = next(f for f in factors if "new_policy" in f["factor"])
    assert c5["triggered"] is False
    assert c5["contribution"] == 0.0


def test_estimate_above_severity_ceiling_fires():
    """Repair estimate > minor ceiling (3000) → estimate-vs-severity factor fires."""
    state = _build_state(repair_estimate=4000.0, severity="minor")
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c4 = next(f for f in factors if "estimate" in f["factor"])
    assert c4["triggered"] is True
    assert c4["contribution"] == pytest.approx(15.0)


def test_estimate_within_range_silent():
    """Repair estimate well below ceiling → estimate-vs-severity factor silent."""
    state = _build_state(repair_estimate=1000.0, severity="minor")
    result = RiskAgent().run(state)
    factors = result["risk_assessment"]["contributing_factors"]
    c4 = next(f for f in factors if "estimate" in f["factor"])
    assert c4["triggered"] is False
    assert c4["contribution"] == 0.0


# ── E-03 scenario ─────────────────────────────────────────────────────────────

def test_e03_critical_claim():
    """E-03: total loss, 3 prior claims, 45 days late → score ≥ 85, category=critical."""
    state = _build_state(
        consistency_score=0.0,       # c1=35
        prior_claims_count=3,        # c2=25 (above threshold of 2)
        days_since_incident=45,      # c3=15 (> 30 days)
        repair_estimate=60_000.0,    # c4=15 (above severe ceiling 50000)
        severity="severe",
    )
    result = RiskAgent().run(state)
    ra = result["risk_assessment"]
    assert ra["risk_score"] >= 85
    assert ra["risk_category"] == "critical"
    assert result["requires_human_approval"] is True
    # All five factors should have fired
    triggered = [f for f in ra["contributing_factors"] if f["triggered"]]
    assert len(triggered) >= 4
