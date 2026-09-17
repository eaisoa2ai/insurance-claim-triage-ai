"""
risk_agent.py: RiskAgent scores claim risk from 0-100 using five transparent weighted factors.

Pipeline position: fourth node (after PolicyAgent). Reads prior-agent outputs
and raw claim inputs from ClaimState. All factor weights and thresholds come
from config/risk_thresholds.yaml; never hardcoded in this file.

Five scoring factors (default weights sum to 100):
    1. consistency_score (inverted)      weight 35: low image/claim consistency raises risk
    2. prior_claims_24m frequency        weight 25: high claim count in rolling window
    3. late incident reporting           weight 15: submission delayed past late_reporting_days
    4. estimate vs severity range        weight 15: repair cost exceeds expected ceiling
    5. policy tenure at incident         weight 10: claim filed within new_policy_months

Risk categories (not configurable, semantics are fixed):
    Low: 0-40   Medium: 41-69   High: 70-84   Critical: 85-100

Reads from ClaimState:
    damage_evidence_assessment: consistency_score, severity
    claim_history: claims_in_last_24_months
    repair_estimate: estimated_amount
    customer_profile: policy_start_date
    claim_input: incident_date, submitted_at

Writes to ClaimState (partial dict):
    risk_assessment: RiskAssessment.model_dump()
    risk_score: float 0-100 (routing flag)
    risk_category: str (routing flag)
    requires_human_approval: True when risk_score >= human_review_threshold
    escalation_reason: first trigger reason (preserved if already set)
    escalation_flags: accumulated list of all trigger reasons
    audit_events: one entry appended
"""

from __future__ import annotations

import pathlib
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import yaml

from core.audit import write_audit_entry
from core.claim_state import ClaimState, RiskAssessment


_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "risk_thresholds.yaml"

# Default factor weights (must sum to 100)
_W_CONSISTENCY  = 35.0
_W_PRIOR_CLAIMS = 25.0
_W_LATE_REPORT  = 15.0
_W_EST_SEVERITY = 15.0
_W_TENURE       = 10.0

# Default scalar thresholds
_DEFAULT_PRIOR_CLAIMS_THRESHOLD = 2
_DEFAULT_LATE_DAYS              = 30
_DEFAULT_NEW_POLICY_MONTHS      = 6
_DEFAULT_HUMAN_REVIEW_THRESHOLD = 70.0

# Default max repair estimate per severity label (used when YAML key is absent)
_DEFAULT_SEVERITY_MAX: dict[str, float] = {
    "minor":    3_000.0,
    "moderate": 10_000.0,
    "severe":   50_000.0,
    "unknown":  50_000.0,
}

# Category thresholds evaluated top-down; first match wins
_CATEGORY_THRESHOLDS: list[tuple[float, str]] = [
    (85.0, "critical"),
    (70.0, "high"),
    (41.0, "medium"),
    (0.0,  "low"),
]


# ── Config helpers ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _risk_cfg() -> dict[str, Any]:
    return _load_config().get("risk", {})


def _factor_cfg() -> dict[str, Any]:
    return _risk_cfg().get("factors", {})


def _severity_max(severity: str) -> float:
    raw = _risk_cfg().get("severity_estimate_ranges", {})
    entry = raw.get(severity) or raw.get("unknown")
    if entry is None:
        return _DEFAULT_SEVERITY_MAX.get(severity, _DEFAULT_SEVERITY_MAX["unknown"])
    return float(entry["max"]) if isinstance(entry, dict) else float(entry)


def _human_review_threshold() -> float:
    return float(_risk_cfg().get("human_review_threshold", _DEFAULT_HUMAN_REVIEW_THRESHOLD))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _categorize(score: float) -> str:
    for threshold, category in _CATEGORY_THRESHOLDS:
        if score >= threshold:
            return category
    return "low"


def _parse_date(value: Any) -> date | None:
    """Coerce date / datetime / ISO-string to a date. Returns None on failure.

    Uses the first 10 characters (YYYY-MM-DD) for plain date strings and full
    ISO-8601 datetime strings alike.  The old approach of value[:len(fmt)] was
    wrong because len("%Y-%m-%d") == 8, not 10, so bare date strings never parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return None


def _days_between(start: date | None, end: date | None) -> int | None:
    if start is None or end is None:
        return None
    return (end - start).days


def _months_between(start: date | None, end: date | None) -> int | None:
    if start is None or end is None:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month)


def _factor_entry(
    factor: str,
    label: str,
    weight: float,
    contribution: float,
    triggered: bool,
) -> dict:
    return {
        "factor": factor,
        "label": label,
        "weight": weight,
        "contribution": round(contribution, 2),
        "triggered": triggered,
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class RiskAgent:

    AGENT_NAME = "RiskAgent"

    def run(self, state: ClaimState) -> dict:
        """Compute weighted risk score; classify category; return partial ClaimState dict."""
        fcfg = _factor_cfg()

        # Load weights and thresholds from config (fall back to module defaults)
        w1 = float(fcfg.get("consistency_weight",        _W_CONSISTENCY))
        w2 = float(fcfg.get("prior_claims_weight",       _W_PRIOR_CLAIMS))
        w3 = float(fcfg.get("late_reporting_weight",     _W_LATE_REPORT))
        w4 = float(fcfg.get("estimate_vs_severity_weight", _W_EST_SEVERITY))
        w5 = float(fcfg.get("policy_tenure_weight",      _W_TENURE))

        prior_threshold = int(fcfg.get("prior_claims_count_threshold", _DEFAULT_PRIOR_CLAIMS_THRESHOLD))
        late_days       = int(fcfg.get("late_reporting_days",          _DEFAULT_LATE_DAYS))
        new_policy_m    = int(fcfg.get("new_policy_months",            _DEFAULT_NEW_POLICY_MONTHS))

        # Extract inputs from state
        dea      = state.get("damage_evidence_assessment") or {}
        history  = state.get("claim_history") or {}
        estimate = state.get("repair_estimate") or {}
        profile  = state.get("customer_profile") or {}
        claim    = state["claim_input"]

        consistency_score = float(dea.get("consistency_score", 0.5))
        severity          = str(dea.get("severity") or "unknown")
        prior_claims_24m  = int(history.get("claims_in_last_24_months") or 0)
        repair_amount     = float(estimate.get("estimated_amount") or 0.0)

        incident_date = _parse_date(claim.get("incident_date"))
        submitted_at  = _parse_date(claim.get("submitted_at"))
        policy_start  = _parse_date(profile.get("policy_start_date"))

        days_since    = _days_between(incident_date, submitted_at)
        policy_months = _months_between(policy_start, incident_date)

        # Score each factor using the transparent rule methods
        c1 = self._score_consistency(consistency_score, w1)
        c2 = self._score_prior_claims(prior_claims_24m, prior_threshold, w2)
        c3 = self._score_late_reporting(
            days_since if days_since is not None else 0,
            late_days,
            w3,
        )
        c4 = self._score_estimate_vs_severity(repair_amount, severity, w4)
        c5 = self._score_policy_tenure(
            str(policy_start)   if policy_start   else "",
            str(incident_date)  if incident_date  else "",
            w5,
        )

        risk_score    = round(min(100.0, max(0.0, c1 + c2 + c3 + c4 + c5)), 2)
        risk_category = _categorize(risk_score)
        threshold     = _human_review_threshold()

        contributing_factors = [
            _factor_entry(
                "consistency_score_inverted",
                "Inverted image/claim consistency score",
                w1, c1, c1 > 0,
            ),
            _factor_entry(
                "prior_claims_frequency",
                "Prior claims count in last 24 months",
                w2, c2, c2 > 0,
            ),
            _factor_entry(
                "late_reporting",
                "Days between incident date and claim submission",
                w3, c3, c3 > 0,
            ),
            _factor_entry(
                "estimate_vs_severity",
                "Repair estimate vs expected ceiling for severity level",
                w4, c4, c4 > 0,
            ),
            _factor_entry(
                "new_policy_claim",
                "Policy tenure at time of incident",
                w5, c5, c5 > 0,
            ),
        ]

        triggered = [f for f in contributing_factors if f["triggered"]]

        warnings: list[str] = []
        if days_since is None:
            warnings.append(
                "Incident or submission date missing -late-reporting factor scored as 0"
            )
        if days_since is not None and days_since < 0:
            warnings.append(
                "Submission date precedes incident date -possible data entry error"
            )
        if policy_months is None:
            warnings.append(
                "Customer profile or policy_start_date missing -"
                "policy-tenure factor scored as 0"
            )

        evidence = [
            f"Risk score: {risk_score:.1f}/100 -> category '{risk_category}'",
            (
                f"Factor 1 - consistency (weight {w1}): "
                f"score {consistency_score:.2f} inverted -> contribution {c1:.1f}"
            ),
            (
                f"Factor 2 - prior claims (weight {w2}): "
                f"{prior_claims_24m} in 24 months, threshold {prior_threshold} -> contribution {c2:.1f}"
            ),
            (
                f"Factor 3 - late reporting (weight {w3}): "
                f"{days_since if days_since is not None else 'N/A'} days, "
                f"threshold {late_days} -> contribution {c3:.1f}"
            ),
            (
                f"Factor 4 - estimate vs severity (weight {w4}): "
                f"${repair_amount:,.0f} for '{severity}' -> contribution {c4:.1f}"
            ),
            (
                f"Factor 5 - policy tenure (weight {w5}): "
                f"{policy_months if policy_months is not None else 'N/A'} months, "
                f"threshold {new_policy_m} -> contribution {c5:.1f}"
            ),
        ]

        # Escalation flags accumulate across agents -preserve any set by earlier nodes
        escalation_flags  = list(state.get("escalation_flags") or [])
        escalation_reason = str(state.get("escalation_reason") or "")
        requires_human    = risk_score >= threshold or risk_category == "critical"

        if requires_human:
            new_reason = (
                f"Risk category '{risk_category}', "
                f"score {risk_score:.1f} (threshold {threshold:.0f})"
            )
            if new_reason not in escalation_flags:
                escalation_flags.append(new_reason)
            if not escalation_reason:
                escalation_reason = new_reason

        routing = "human_review" if requires_human else "continue"

        confidence = _compute_confidence(
            data_complete=(days_since is not None and policy_months is not None),
            n_triggered=len(triggered),
        )

        assessment = RiskAssessment(
            confidence=confidence,
            evidence=evidence,
            reasoning_summary=_reasoning(
                risk_score, risk_category, triggered, requires_human, threshold
            ),
            warnings=warnings,
            risk_score=risk_score,
            risk_category=risk_category,
            contributing_factors=contributing_factors,
        )

        audit_event = write_audit_entry(
            agent=self.AGENT_NAME,
            event="agent_complete",
            output_model=assessment,
            routing_decision=routing,
            claim_id=claim.get("claim_id", ""),
            input_summary={
                "consistency_score":       consistency_score,
                "severity":                severity,
                "prior_claims_24m":        prior_claims_24m,
                "repair_amount":           repair_amount,
                "days_since_incident":     days_since,
                "policy_months_at_incident": policy_months,
                "factor_weights": {
                    "consistency":        w1,
                    "prior_claims":       w2,
                    "late_reporting":     w3,
                    "estimate_severity":  w4,
                    "policy_tenure":      w5,
                },
            },
        )

        return {
            "risk_assessment":         assessment.model_dump(),
            "risk_score":              risk_score,
            "risk_category":           risk_category,
            "requires_human_approval": requires_human,
            "escalation_reason":       escalation_reason,
            "escalation_flags":        escalation_flags,
            "audit_events":            state["audit_events"] + [audit_event],
        }

    def _score_consistency(self, consistency_score: float, weight: float) -> float:
        """Inverted consistency score contribution: (1 - consistency_score) * weight."""
        inverted = 1.0 - max(0.0, min(1.0, consistency_score))
        return round(inverted * weight, 2)

    def _score_prior_claims(
        self, prior_claims_count: int, threshold: int, weight: float
    ) -> float:
        """Full weight if prior_claims_count >= threshold; proportional ramp below."""
        if prior_claims_count <= 0:
            return 0.0
        if prior_claims_count >= threshold:
            return round(weight, 2)
        # Linear ramp: 1 claim of 2 threshold -> 50% weight
        return round((prior_claims_count / threshold) * weight, 2)

    def _score_late_reporting(
        self, days_since_incident: int, late_days: int, weight: float
    ) -> float:
        """Full weight if days_since_incident > late_days; else 0."""
        return round(weight, 2) if days_since_incident > late_days else 0.0

    def _score_estimate_vs_severity(
        self, repair_estimate: float, severity: str, weight: float
    ) -> float:
        """Full weight above severity ceiling; half weight at 80–100% of ceiling."""
        if repair_estimate <= 0:
            return 0.0
        ceiling = _severity_max(severity)
        if repair_estimate > ceiling:
            return round(weight, 2)
        if repair_estimate > ceiling * 0.80:
            return round(weight * 0.5, 2)
        return 0.0

    def _score_policy_tenure(
        self, effective_date: str, incident_date: str, weight: float
    ) -> float:
        """Full weight if policy age at incident was less than new_policy_months; else 0."""
        start = _parse_date(effective_date)
        end   = _parse_date(incident_date)
        if start is None or end is None:
            return 0.0
        new_policy_months = int(
            _factor_cfg().get("new_policy_months", _DEFAULT_NEW_POLICY_MONTHS)
        )
        months = _months_between(start, end)
        return round(weight, 2) if (months is not None and months < new_policy_months) else 0.0


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_confidence(data_complete: bool, n_triggered: int) -> float:
    """Confidence in the risk score: reduced when data is missing."""
    base = 1.0
    if not data_complete:
        base -= 0.15   # one or more date fields missing
    if n_triggered == 0:
        base -= 0.05   # no factors fired -score may be under-estimating
    return round(max(0.0, min(1.0, base)), 2)


def _reasoning(
    risk_score: float,
    risk_category: str,
    triggered: list[dict],
    requires_human: bool,
    threshold: float,
) -> str:
    factor_labels = ", ".join(f["label"] for f in triggered) or "none"
    human_note = (
        f" Human review required (score {risk_score:.1f} >= threshold {threshold:.0f})."
        if requires_human
        else (
            f" Score {risk_score:.1f} is below the {threshold:.0f} threshold; "
            f"auto-processing eligible."
        )
    )
    return (
        f"Risk score {risk_score:.1f}/100 places this claim in the '{risk_category}' "
        f"category. Triggered factors: {factor_labels}.{human_note}"

    )
