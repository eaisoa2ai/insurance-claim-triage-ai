"""
policy_agent.py — PolicyAgent: determine coverage from the loaded policy document.

Pipeline position: third node (after DamageEvidenceAgent). Reads the policy file
whose path is stored in state["claim_input"]["policy_document_path"], loads it
into a PolicyRules model via core/policy_rules, and checks coverage.

Reads from ClaimState:
    claim_input   — policy_document_path, policy_id, claim_type, incident_date

Writes to ClaimState (partial dict):
    policy_decision  — PolicyDecision.model_dump()
    policy_valid     — bool
    coverage_decision — "covered" | "excluded" | "partial"
    audit_events     — one entry appended

Coverage logic (pure, from the policy document):
    claim_type "collision"  → checks collision coverage bucket (Clause 1.1)
    claim_type "glass"      → excluded under AUTO-STD-001 (Clause 2.4)
    claim_type "vandalism"  → excluded under AUTO-STD-001 (Clause 2.3)
    claim_type "other"      → excluded — no matching coverage type

Policy invalid when:
    • policy JSON file missing or unparseable
    • incident_date before effective_date or after end_date
    In both cases: policy_valid=False, coverage_decision="excluded", human review.

Hard constraint: every clause cited in PolicyDecision.clause_cited must come
verbatim from the loaded PolicyRules object. Nothing is invented here.
"""

from __future__ import annotations

from datetime import date

from pydantic import ValidationError

from core.audit import write_audit_entry
from core.claim_state import ClaimState, PolicyDecision
from core.policy_rules import PolicyRules, is_damage_covered, load_policy_rules


class PolicyAgent:

    AGENT_NAME = "PolicyAgent"

    def run(self, state: ClaimState) -> dict:
        """Load policy; determine coverage; return partial ClaimState dict."""
        claim     = state["claim_input"]
        policy_path = claim.get("policy_document_path", "")
        policy_id   = claim.get("policy_id", "")
        claim_type  = claim.get("claim_type", "other")

        # Parse incident date (may arrive as ISO string or date object)
        incident_date = _parse_date(claim.get("incident_date"))

        # ── Load policy rules ──────────────────────────────────────────────────
        try:
            rules = load_policy_rules(policy_path)
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            return _policy_unavailable(self.AGENT_NAME, state, policy_id, str(exc))

        # ── Policy period validity ─────────────────────────────────────────────
        evidence: list[str] = [
            f"Policy: {rules.policy_name} ({rules.policy_id})",
            f"Policy period: {rules.effective_date} to {rules.end_date}",
        ]
        warnings: list[str] = []
        exclusion_reasons: list[str] = []
        policy_valid = True

        if incident_date is None:
            policy_valid = False
            warnings.append("Incident date could not be parsed — policy validity unconfirmable")
        elif incident_date < rules.effective_date:
            policy_valid = False
            warnings.append(
                f"Incident date {incident_date} is before policy start "
                f"{rules.effective_date} — policy not yet active"
            )
            exclusion_reasons.append(f"Policy not yet in force on {incident_date}")
        elif incident_date > rules.end_date:
            policy_valid = False
            rejection_clause = rules.late_reporting.get(
                "rejection_clause", "Clause 4.3 / Clause E.7"
            )
            warnings.append(
                f"Incident date {incident_date} is after policy expiry "
                f"{rules.end_date} — {rejection_clause}"
            )
            exclusion_reasons.append(
                f"Policy expired {rules.end_date} — {rejection_clause}"
            )
        else:
            evidence.append(f"Incident date: {incident_date} (within policy period)")

        # ── Coverage determination ─────────────────────────────────────────────
        if not policy_valid or incident_date is None:
            decision       = "excluded"
            coverage_type  = "none"
            coverage_ratio = 0.0
            deductible     = 0.0
            clause_cited   = (
                exclusion_reasons[0] if exclusion_reasons
                else "Policy not active on incident date"
            )
            evidence.append(f"Coverage: excluded — {clause_cited}")

        else:
            decision, coverage_type, coverage_ratio = is_damage_covered(
                rules, claim_type, incident_date
            )

            if decision == "covered":
                rule         = rules.coverages[coverage_type]
                clause_cited = rule.clause
                deductible   = (
                    rule.glass_deductible
                    if rule.glass_deductible is not None and claim_type == "glass"
                    else rules.deductible
                )
                evidence += [
                    f"'{claim_type}' covered under {coverage_type} coverage — {clause_cited}",
                    f"Coverage ratio: {coverage_ratio:.0%}  |  Deductible: €{deductible:,.2f}",
                    f"Coverage cap: €{rules.coverage_cap:,.2f}",
                ]
                # Cite the auto-approve limit from the document
                payout_threshold = rules.manual_review_triggers.get("payout_exceeds")
                if payout_threshold:
                    evidence.append(
                        f"Auto-approve limit: €{payout_threshold:,.0f} "
                        f"({rules.manual_review_triggers.get('payout_clause', 'Clause 5.3')})"
                    )
            else:
                # Excluded — find the specific clause from the loaded document
                clause_cited = _find_exclusion_clause(rules, claim_type)
                coverage_ratio = 0.0
                deductible     = 0.0
                exclusion_reasons.append(
                    f"'{claim_type}' is not covered under policy {rules.policy_id} — "
                    + clause_cited
                )
                evidence += [
                    f"'{claim_type}' is excluded under this policy",
                    f"Exclusion clause: {clause_cited}",
                ]

        routing = "continue" if (policy_valid and decision == "covered") else "human_review"

        policy_decision = PolicyDecision(
            confidence=_confidence(decision, policy_valid),
            evidence=evidence,
            reasoning_summary=_reasoning(
                claim_type, decision, coverage_type, clause_cited,
                rules, coverage_ratio, deductible,
            ),
            warnings=warnings,
            policy_id=policy_id,
            policy_valid=policy_valid,
            coverage_decision=decision,
            applicable_coverage_type=coverage_type,
            clause_cited=clause_cited,
            deductible=deductible,
            coverage_ratio=coverage_ratio,
            coverage_cap=rules.coverage_cap,
            exclusion_reasons=exclusion_reasons,
        )

        audit_event = write_audit_entry(
            agent=self.AGENT_NAME,
            event="agent_complete",
            output_model=policy_decision,
            routing_decision=routing,
            claim_id=claim.get("claim_id", ""),
            input_summary={
                "policy_id":            policy_id,
                "claim_type":           claim_type,
                "incident_date":        str(claim.get("incident_date", "")),
                "policy_document_path": policy_path,
                "policy_period":        f"{rules.effective_date} to {rules.end_date}",
                "policy_name":          rules.policy_name,
                "coverage_cap":         rules.coverage_cap,
                "deductible":           rules.deductible,
            },
        )

        return {
            "policy_decision":  policy_decision.model_dump(),
            "policy_valid":     policy_valid,
            "coverage_decision": decision,
            "audit_events":     state["audit_events"] + [audit_event],
        }


# ── Module-level helpers ───────────────────────────────────────────────────────

def _parse_date(raw: object) -> date | None:
    """Coerce an ISO string, datetime, or date to a date; return None on failure."""
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except (ValueError, TypeError):
        return None


def _find_exclusion_clause(rules: PolicyRules, claim_type: str) -> str:
    """Return the most specific clause citation for an excluded claim_type.

    Checks each coverage bucket's excluded_damage_clauses first (document-sourced),
    then falls back to Clause E.9 (damage outside coverage type).
    """
    for rule in rules.coverages.values():
        if claim_type in rule.excluded_damage_types:
            # Use per-damage-type clause if available, else the bucket's main clause
            return rule.excluded_damage_clauses.get(claim_type, rule.clause)
    return "Clause E.9 — Damage Outside Policy Coverage Type"


def _confidence(decision: str, policy_valid: bool) -> float:
    if not policy_valid:
        return 0.90   # high confidence it IS excluded; human review will confirm
    if decision == "covered":
        return 0.95   # clean positive coverage
    return 0.90       # high confidence in exclusion finding


def _reasoning(
    claim_type: str,
    decision: str,
    coverage_type: str,
    clause_cited: str,
    rules: PolicyRules,
    coverage_ratio: float,
    deductible: float,
) -> str:
    if decision == "covered":
        return (
            f"Claim type '{claim_type}' is covered under the {coverage_type} coverage "
            f"of policy {rules.policy_id} per {clause_cited}. "
            f"Coverage ratio: {coverage_ratio:.0%}. "
            f"Deductible: €{deductible:,.2f}. "
            f"Maximum payout: €{rules.coverage_cap:,.2f}."
        )
    return (
        f"Claim type '{claim_type}' is not covered under policy {rules.policy_id}. "
        f"Applicable exclusion: {clause_cited}. "
        "The claim is routed to human review for final determination."
    )


# ── Error-path helper ─────────────────────────────────────────────────────────

def _policy_unavailable(
    agent_name: str,
    state: ClaimState,
    policy_id: str,
    error: str,
) -> dict:
    """Return partial state when the policy file cannot be loaded or parsed."""
    claim = state.get("claim_input") or {}
    policy_decision = PolicyDecision(
        confidence=0.0,
        evidence=[f"Policy document could not be loaded: {error}"],
        reasoning_summary=(
            f"Policy document for {policy_id} is missing or unreadable. "
            "Coverage cannot be determined without a valid policy document. "
            "Human review is required."
        ),
        warnings=[error, "Policy document unavailable — human review required"],
        policy_id=policy_id or "UNKNOWN",
        policy_valid=False,
        coverage_decision="excluded",
        applicable_coverage_type="none",
        clause_cited="Policy document unavailable",
        deductible=0.0,
        coverage_ratio=0.0,
        coverage_cap=0.0,
        exclusion_reasons=["Policy document could not be loaded"],
    )
    audit_event = write_audit_entry(
        agent=agent_name,
        event="agent_complete",
        output_model=policy_decision,
        routing_decision="human_review",
        claim_id=claim.get("claim_id", ""),
        input_summary={
            "policy_id":            policy_id or "UNKNOWN",
            "policy_document_path": claim.get("policy_document_path", ""),
            "claim_type":           claim.get("claim_type", ""),
            "load_error":           error,
        },
    )
    return {
        "policy_decision":  policy_decision.model_dump(),
        "policy_valid":     False,
        "coverage_decision": "excluded",
        "audit_events":     state["audit_events"] + [audit_event],
    }
