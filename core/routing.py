"""
routing.py — Router node and escalation condition checks.

Evaluates conditions in priority order after PayoutAgent. Each condition is an
isolated check function so test_routes.py can exercise them directly.

Priority order (first match sets escalation_reason):
  1. intake_complete is False         — required fields missing     → human_review
  2. damage_image_path is None        — no photographic evidence    → human_review
  3. image_claim_mismatch is True     — image/claim inconsistency   → human_review
  4. estimate_plausibility suspicious — suspicious repair estimate   → human_review
  5. coverage_decision == "excluded"  — claim type not covered      → rejected (no payout)
  6. risk_score >= threshold          — high risk score             → human_review
  7. any agent confidence < floor     — low-confidence output       → human_review
  8. payout_amount > limit            — large recommended payout    → human_review
  9. policy_valid is False            — invalid or expired policy   → human_review
 10. risk_category == "critical"      — critical risk category      → human_review

Condition 5 routes to "rejected" (final report, zero payout, no adjuster needed).
All other triggered conditions route to "human_review".
No condition fired → "auto_approve".

All thresholds are read from config/risk_thresholds.yaml; module-level defaults
are used when the file is absent or the key is missing.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import yaml

from core.audit import write_audit_entry
from core.claim_state import ClaimState


_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "risk_thresholds.yaml"

_DEFAULT_RISK_THRESHOLD   = 70.0   # risk_score is 0–100
_DEFAULT_CONFIDENCE_FLOOR = 0.65   # confidence is 0.0–1.0
_DEFAULT_PAYOUT_LIMIT     = 2_500.0


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _risk_threshold() -> float:
    return float(
        _load_config().get("risk", {}).get("human_review_threshold", _DEFAULT_RISK_THRESHOLD)
    )


def _confidence_floor() -> float:
    return float(
        _load_config().get("risk", {}).get("confidence_floor", _DEFAULT_CONFIDENCE_FLOOR)
    )


def _payout_limit() -> float:
    return float(
        _load_config().get("payout", {}).get("auto_approve_limit", _DEFAULT_PAYOUT_LIMIT)
    )


# ── Condition check functions ──────────────────────────────────────────────────
# Each returns True when the condition is triggered (escalation needed).
# Functions that compare against a configurable threshold take the threshold as
# an argument so tests can pass arbitrary values without touching the YAML file.

def _check_missing_fields(state: ClaimState) -> bool:
    """Condition 1: intake_complete is False — required fields missing or invalid."""
    return not state.get("intake_complete", False)


def _check_missing_image(state: ClaimState) -> bool:
    """Condition 2: damage_image_path is None — no photographic evidence submitted."""
    claim = state.get("claim_input") or {}
    return claim.get("damage_image_path") is None


def _check_image_mismatch(state: ClaimState) -> bool:
    """Condition 3: image_claim_mismatch flagged by DamageEvidenceAgent."""
    return bool(state.get("image_claim_mismatch", False))


def _check_suspicious_estimate(state: ClaimState) -> bool:
    """Condition 4: repair estimate flagged as suspicious by the workshop."""
    estimate = state.get("repair_estimate") or {}
    return estimate.get("estimate_plausibility") == "suspicious"


def _check_excluded_coverage(state: ClaimState) -> bool:
    """Condition 5: coverage_decision is 'excluded' — claim type not covered by policy."""
    return state.get("coverage_decision", "") == "excluded"


def _check_high_risk_score(state: ClaimState, threshold: float) -> bool:
    """Condition 6: risk_score >= human_review_threshold (0–100 scale)."""
    return float(state.get("risk_score", 0.0)) >= threshold


def _check_low_confidence(state: ClaimState, confidence_floor: float) -> bool:
    """Condition 7: any downstream agent output has confidence below the floor.

    Checks damage_evidence, policy, risk, and payout outputs. Intake confidence
    is not stored as a separate field — the intake_complete flag covers it.
    """
    agent_outputs = [
        state.get("damage_evidence_assessment") or {},
        state.get("policy_decision") or {},
        state.get("risk_assessment") or {},
        state.get("payout_recommendation") or {},
    ]
    for output in agent_outputs:
        conf = output.get("confidence")
        if conf is not None and float(conf) < confidence_floor:
            return True
    return False


def _check_payout_limit(state: ClaimState, auto_approve_limit: float) -> bool:
    """Condition 8: payout_amount exceeds the auto-approve limit."""
    payout = state.get("payout_recommendation") or {}
    return float(payout.get("payout_amount", 0.0)) > auto_approve_limit


def _check_invalid_policy(state: ClaimState) -> bool:
    """Condition 9: policy_valid is False (expired or not-yet-active policy)."""
    return not state.get("policy_valid", True)


def _check_critical_risk(state: ClaimState) -> bool:
    """Condition 10: risk_category == 'critical'."""
    return state.get("risk_category", "") == "critical"


# ── Router node ────────────────────────────────────────────────────────────────

def router_node(state: ClaimState) -> dict:
    """Evaluate all escalation conditions and write routing flags to state.

    Returns a partial ClaimState dict with:
      - requires_human_approval  — True when any human-review condition fires
      - escalation_reason        — first triggered condition label
      - escalation_flags         — all triggered condition labels (accumulated)
      - human_approval           — HumanApprovalDecision dict (when escalating)
      - audit_events             — one routing entry appended

    Condition 5 (excluded coverage) is tracked in escalation_flags but does NOT
    set requires_human_approval — route_decision routes it to "rejected" instead.
    """
    threshold = _risk_threshold()
    floor     = _confidence_floor()
    limit     = _payout_limit()

    # Ordered list of (triggered: bool, label: str) for human_review conditions.
    # Condition 5 (excluded) is evaluated separately below.
    human_review_checks: list[tuple[bool, str]] = [
        (
            _check_missing_fields(state),
            "Required claim fields missing or failed validation",
        ),
        (
            _check_missing_image(state),
            "No damage photograph submitted",
        ),
        (
            _check_image_mismatch(state),
            "Image/claim consistency score below mismatch threshold",
        ),
        (
            _check_suspicious_estimate(state),
            "Repair estimate flagged as suspicious",
        ),
        (
            _check_high_risk_score(state, threshold),
            f"Risk score >= {threshold:.0f} (human-review threshold)",
        ),
        (
            _check_low_confidence(state, floor),
            f"Agent confidence below {floor:.2f} floor",
        ),
        (
            _check_payout_limit(state, limit),
            f"Recommended payout exceeds auto-approve limit of €{limit:,.0f}",
        ),
        (
            _check_invalid_policy(state),
            "Policy is invalid, expired, or not yet active",
        ),
        (
            _check_critical_risk(state),
            "Risk category is critical",
        ),
    ]

    # Accumulate escalation state, preserving flags set by earlier nodes (e.g. RiskAgent).
    escalation_flags: list[str] = list(state.get("escalation_flags") or [])
    escalation_reason: str = str(state.get("escalation_reason") or "")
    requires_human = False

    for triggered, label in human_review_checks:
        if triggered:
            requires_human = True
            if label not in escalation_flags:
                escalation_flags.append(label)
            if not escalation_reason:
                escalation_reason = label

    # Condition 5 — excluded coverage is a fast-path rejection (not human review).
    if _check_excluded_coverage(state):
        excluded_label = "Coverage type excluded by policy — no payout issued"
        if excluded_label not in escalation_flags:
            escalation_flags.append(excluded_label)
        if not escalation_reason:
            escalation_reason = excluded_label

    routing_decision = "human_review" if requires_human else "continue"

    claim_id = (state.get("claim_input") or {}).get("claim_id", "")

    audit_event = write_audit_entry(
        agent="router",
        event="routing",
        routing_decision=routing_decision,
        extra={
            "requires_human_approval": requires_human,
            "coverage_decision": state.get("coverage_decision", ""),
            "escalation_reason": escalation_reason,
            "escalation_flags": escalation_flags,
        },
        claim_id=claim_id,
        input_summary={
            "intake_complete":       state.get("intake_complete"),
            "image_claim_mismatch":  state.get("image_claim_mismatch"),
            "policy_valid":          state.get("policy_valid"),
            "coverage_decision":     state.get("coverage_decision", ""),
            "risk_score":            state.get("risk_score"),
            "risk_category":         state.get("risk_category", ""),
            "payout_amount":         (state.get("payout_recommendation") or {}).get("payout_amount"),
            "thresholds_used": {
                "risk":       threshold,
                "confidence": floor,
                "payout":     limit,
            },
        },
    )

    partial: dict = {
        "requires_human_approval": requires_human,
        "escalation_reason": escalation_reason,
        "escalation_flags": escalation_flags,
        "audit_events": state["audit_events"] + [audit_event],
    }

    # Pre-populate human_approval model when escalating so the UI can read it
    # immediately without waiting for human_review_node.
    if requires_human:
        partial["human_approval"] = {
            "required": True,
            "escalation_reason": escalation_reason,
            "escalation_flags": escalation_flags,
            "approved": None,
            "adjuster_note": None,
            "decided_at": None,
        }

    return partial


def route_decision(state: ClaimState) -> str:
    """Conditional edge function — returns the next node name.

    Returns:
      "human_review"  — one or more human-review conditions fired
      "rejected"      — coverage is excluded and no other condition fired
      "auto_approve"  — all conditions clear, payout within limits
    """
    if state.get("requires_human_approval"):
        return "human_review"
    if state.get("coverage_decision") == "excluded":
        return "rejected"
    return "auto_approve"
