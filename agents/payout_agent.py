"""
payout_agent.py — PayoutAgent: compute recommended payout and approval eligibility.

Pipeline position: fifth node (after RiskAgent). Reads policy terms, damage
evidence, risk scores, and the repair estimate to compute a recommended payout.

Reads from ClaimState:
    policy_decision            — deductible, coverage_ratio, coverage_cap, coverage_decision
    damage_evidence_assessment — severity, consistency_score, image_claim_mismatch
    risk_assessment            — risk_score, risk_category
    repair_estimate            — estimated_amount (authoritative dollar figure)

Writes to ClaimState (partial dict):
    payout_recommendation  — PayoutRecommendation.model_dump()
    audit_events           — one entry appended

Payout formula:
    gross_payout  = repair_estimate × coverage_ratio
    net_payout    = max(0, gross_payout − deductible)
    payout_amount = min(net_payout, coverage_cap)

auto_approve_eligible is False when ANY of:
    • payout_amount > auto_approve_limit   (config: payout.auto_approve_limit)
    • image_claim_mismatch is True
    • risk_score >= human_review_threshold (config: risk.human_review_threshold)
    • coverage_decision != "covered"
    • policy agent confidence < confidence_floor (config: risk.confidence_floor)

When auto_approve_eligible is False, payout_amount is computed but not finalized.
The routing node (core/routing.py) makes the final escalation decision; this agent
only sets the recommendation and flags eligibility.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import yaml

from core.audit import write_audit_entry
from core.claim_state import ClaimState, PayoutRecommendation


_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "risk_thresholds.yaml"

_DEFAULT_AUTO_APPROVE_LIMIT     = 5_000.0
_DEFAULT_HUMAN_REVIEW_THRESHOLD = 70.0
_DEFAULT_CONFIDENCE_FLOOR       = 0.75


# ── Config ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _auto_approve_limit() -> float:
    return float(
        _load_config().get("payout", {}).get("auto_approve_limit", _DEFAULT_AUTO_APPROVE_LIMIT)
    )


def _human_review_threshold() -> float:
    return float(
        _load_config().get("risk", {}).get("human_review_threshold", _DEFAULT_HUMAN_REVIEW_THRESHOLD)
    )


def _confidence_floor() -> float:
    return float(
        _load_config().get("risk", {}).get("confidence_floor", _DEFAULT_CONFIDENCE_FLOOR)
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class PayoutAgent:

    AGENT_NAME = "PayoutAgent"

    def run(self, state: ClaimState) -> dict:
        """Compute recommended payout amount and eligibility; return partial ClaimState dict."""
        policy   = state.get("policy_decision")            or {}
        damage   = state.get("damage_evidence_assessment") or {}
        risk     = state.get("risk_assessment")            or {}
        estimate = state.get("repair_estimate")            or {}

        # ── Extract inputs from upstream agents ────────────────────────────────
        coverage_decision = str(policy.get("coverage_decision", "excluded"))
        coverage_ratio    = float(policy.get("coverage_ratio",  0.0))
        deductible        = float(policy.get("deductible",      0.0))
        coverage_cap      = float(policy.get("coverage_cap",    0.0))
        policy_confidence = float(policy.get("confidence",      0.0))

        repair_amount     = float(estimate.get("estimated_amount", 0.0))

        mismatch          = bool(damage.get("image_claim_mismatch", False))
        severity          = str(damage.get("severity",          "unknown"))
        consistency_score = float(damage.get("consistency_score", 0.0))

        risk_score        = float(risk.get("risk_score",   0.0))
        risk_category     = str(risk.get("risk_category", "unknown"))

        # ── Config thresholds ──────────────────────────────────────────────────
        approve_limit    = _auto_approve_limit()
        review_threshold = _human_review_threshold()
        floor            = _confidence_floor()

        # ── Payout formula ─────────────────────────────────────────────────────
        gross_payout  = repair_amount * coverage_ratio if coverage_decision != "excluded" else 0.0
        net_payout    = max(0.0, gross_payout - deductible)
        # PayoutRecommendation.coverage_cap requires gt=0; use net_payout or repair_amount
        # as a fallback when the policy document did not supply a cap (e.g. excluded claims).
        effective_cap = coverage_cap if coverage_cap > 0 else max(net_payout, repair_amount, 1.0)
        payout_amount = round(min(net_payout, effective_cap), 2)

        # ── Eligibility — collect all triggered escalation reasons ─────────────
        triggers: list[str] = []

        if coverage_decision != "covered":
            triggers.append(f"coverage_decision is '{coverage_decision}', not 'covered'")
        if mismatch:
            triggers.append("image/claim mismatch detected by DamageEvidenceAgent")
        if risk_score >= review_threshold:
            triggers.append(
                f"risk_score {risk_score:.1f} >= human-review threshold {review_threshold:.1f}"
            )
        if payout_amount > approve_limit:
            triggers.append(
                f"payout €{payout_amount:,.2f} exceeds auto-approve limit"
                f" €{approve_limit:,.2f}"
            )
        if policy_confidence < floor:
            triggers.append(
                f"PolicyAgent confidence {policy_confidence:.2f} < confidence floor {floor:.2f}"
            )

        auto_approve_eligible = len(triggers) == 0
        routing = "continue" if auto_approve_eligible else "human_review"

        # ── Evidence bullets ───────────────────────────────────────────────────
        evidence: list[str] = [
            f"Repair estimate: €{repair_amount:,.2f}",
            f"Coverage: {coverage_decision}"
            f"  |  Ratio: {coverage_ratio:.0%}"
            f"  |  Deductible: €{deductible:,.2f}",
            (f"Coverage cap: €{coverage_cap:,.2f}"
             if coverage_cap > 0 else "Coverage cap: not set (excluded claim)"),
            f"Gross payout: €{gross_payout:,.2f}"
            f"  →  Net after deductible: €{net_payout:,.2f}",
            f"Recommended payout: €{payout_amount:,.2f}",
            f"Auto-approve limit: €{approve_limit:,.2f}",
            f"Risk: {risk_score:.1f} ({risk_category})"
            f"  |  Image mismatch: {mismatch}"
            f"  |  Severity: {severity}",
            f"Consistency score: {consistency_score:.2f}",
        ]

        warnings: list[str] = []
        for trigger in triggers:
            warnings.append(f"Human review required — {trigger}")
        if payout_amount == 0.0 and coverage_decision == "covered":
            warnings.append(
                f"Recommended payout is €0.00 despite covered claim: "
                f"deductible (€{deductible:,.2f}) meets or exceeds net repair cost"
            )
        if coverage_cap == 0.0 and coverage_decision != "excluded":
            warnings.append(
                "Coverage cap is 0.0 — policy document may not have loaded correctly"
            )

        # ── Build recommendation model ─────────────────────────────────────────
        recommendation = PayoutRecommendation(
            confidence=_compute_confidence(
                coverage_decision, consistency_score, risk_score,
                payout_amount, approve_limit, review_threshold, policy_confidence,
            ),
            evidence=evidence,
            reasoning_summary=_reasoning(
                coverage_decision, repair_amount, coverage_ratio, deductible,
                payout_amount, coverage_cap, auto_approve_eligible, triggers,
            ),
            warnings=warnings,
            repair_estimate=repair_amount,
            coverage_ratio=coverage_ratio,
            deductible=deductible,
            payout_amount=payout_amount,
            coverage_cap=effective_cap,
            auto_approve_eligible=auto_approve_eligible,
        )

        claim_id = (state.get("claim_input") or {}).get("claim_id", "")

        audit_event = write_audit_entry(
            agent=self.AGENT_NAME,
            event="agent_complete",
            output_model=recommendation,
            routing_decision=routing,
            claim_id=claim_id,
            input_summary={
                "coverage_decision":  coverage_decision,
                "repair_amount":      repair_amount,
                "coverage_ratio":     coverage_ratio,
                "deductible":         deductible,
                "coverage_cap":       coverage_cap,
                "risk_score":         risk_score,
                "risk_category":      risk_category,
                "image_claim_mismatch": mismatch,
                "policy_confidence":  policy_confidence,
                "auto_approve_limit": approve_limit,
            },
        )

        return {
            "payout_recommendation": recommendation.model_dump(),
            "audit_events":          state["audit_events"] + [audit_event],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_confidence(
    coverage_decision: str,
    consistency_score: float,
    risk_score: float,
    payout_amount: float,
    approve_limit: float,
    review_threshold: float,
    policy_confidence: float,
) -> float:
    """Confidence in the payout recommendation (0.0–1.0).

    Anchored to the upstream policy confidence and reduced by signals of
    uncertainty: low visual consistency, elevated risk, or payout near/over limit.
    """
    if coverage_decision == "excluded":
        return 0.90  # high confidence in zero-payout decision

    base = min(policy_confidence, 0.95)

    # Low consistency score (< 0.60) reduces confidence proportionally
    consistency_penalty = max(0.0, 0.60 - consistency_score) * 0.30

    # Risk score approaching the escalation threshold reduces confidence
    risk_ratio    = risk_score / review_threshold if review_threshold > 0 else 0.0
    risk_penalty  = min(0.15, risk_ratio * 0.15)

    # Payout over the auto-approve limit signals adjuster discretion is needed
    limit_penalty = 0.05 if payout_amount > approve_limit else 0.0

    return round(max(0.0, min(1.0, base - consistency_penalty - risk_penalty - limit_penalty)), 2)


def _reasoning(
    coverage_decision: str,
    repair_amount: float,
    coverage_ratio: float,
    deductible: float,
    payout_amount: float,
    coverage_cap: float,
    auto_approve_eligible: bool,
    triggers: list[str],
) -> str:
    """One-paragraph explanation for the audit log and Gradio UI."""
    if coverage_decision == "excluded":
        return (
            "The claim is excluded from coverage. No payout will be issued. "
            "The claim has been routed for human review to confirm the exclusion decision."
        )

    gross = repair_amount * coverage_ratio
    net   = max(0.0, gross - deductible)
    formula = (
        f"Repair estimate €{repair_amount:,.2f} × {coverage_ratio:.0%} = "
        f"€{gross:,.2f} gross, minus €{deductible:,.2f} deductible = "
        f"€{net:,.2f} net"
    )
    if coverage_cap > 0 and payout_amount < net:
        formula += f", capped at €{coverage_cap:,.2f}"
    formula += f". Recommended payout: €{payout_amount:,.2f}."

    if auto_approve_eligible:
        return (
            f"{formula} All eligibility criteria are satisfied: the payout is within "
            "the auto-approve limit, no image/claim mismatch was detected, and the risk "
            "score is below the escalation threshold. The claim is eligible for "
            "automatic approval."
        )

    trigger_list = "; ".join(triggers)
    return (
        f"{formula} Automatic approval cannot proceed because: {trigger_list}. "
        "The recommended payout amount has been computed but will not be finalized "
        "until a human adjuster reviews and approves the claim."
    )
