"""
intake_agent.py — IntakeAgent: validate, normalise, and triage claim inputs.

Pipeline position: first node. Reads state["claim_input"] (raw dict from
data_loader), validates it via ClaimInput, normalises fields, identifies
missing documents, and writes the cleaned input back to state.

Reads from ClaimState:
    claim_input, customer_profile, claim_history, repair_estimate

Writes to ClaimState (partial dict returned to graph node):
    claim_input   — normalised and validated dict
    intake_complete — True when no blocking validation errors
    audit_events  — one AgentAuditEvent appended

Blocking errors (set intake_complete = False, confidence = 0.0):
    • Pydantic validation failure on any required ClaimInput field
    • repair_estimate missing from state

Non-blocking warnings (intake_complete stays True, noted for routing):
    • damage_image_path is None
    • incident_description < 20 words
    • prior_claims_count >= 3 in last 24 months (Clause 5.4)
    • days_since_incident > 30 (Clause 4.1)
    • customer_profile missing from state
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import ValidationError

from core.audit import write_audit_entry
from core.claim_state import ClaimInput, ClaimState


# ── Constants (thresholds from policy Clause 4 / 5) ───────────────────────────

_LATE_REPORTING_DAYS = 30
_MIN_STATEMENT_WORDS = 20
_HIGH_PRIOR_CLAIMS   = 3

_VALID_CLAIM_TYPES = {"collision", "glass", "vandalism", "other"}

# Maps substrings found in free-form claim_type text to canonical values.
_CLAIM_TYPE_MAP: dict[str, str] = {
    "collision":  "collision",
    "rear":       "collision",
    "front":      "collision",
    "side":       "collision",
    "accident":   "collision",
    "crash":      "collision",
    "fender":     "collision",
    "bumper":     "collision",
    "glass":      "glass",
    "windshield": "glass",
    "window":     "glass",
    "windscreen": "glass",
    "crack":      "glass",
    "chip":       "glass",
    "vandalism":  "vandalism",
    "vandal":     "vandalism",
    "theft":      "vandalism",
    "scratch":    "vandalism",
    "keyed":      "vandalism",
}


class IntakeAgent:

    AGENT_NAME = "IntakeAgent"

    def run(self, state: ClaimState) -> dict:
        """Validate and normalise claim inputs; return partial ClaimState dict."""
        completeness_flags: list[str] = []
        warnings: list[str] = []
        evidence: list[str] = []
        blocking = False

        # ── 1. Normalise claim_type before Pydantic validation ────────────────
        raw = dict(state["claim_input"])
        raw["claim_type"] = _normalize_claim_type(raw.get("claim_type", ""))

        # ── 2. Validate via ClaimInput ─────────────────────────────────────────
        claim: ClaimInput | None = None
        try:
            claim = ClaimInput.model_validate(raw)
        except ValidationError as exc:
            for err in exc.errors():
                field = ".".join(str(f) for f in err["loc"])
                completeness_flags.append(field)
                warnings.append(f"Validation failed — {field}: {err['msg']}")
            blocking = True

        # ── 3. Required state dependencies ────────────────────────────────────
        if state.get("repair_estimate") is None:
            completeness_flags.append("repair_estimate")
            warnings.append("Repair estimate is missing from the claim submission")
            blocking = True
        else:
            est = state["repair_estimate"].get("estimated_amount", 0)
            evidence.append(f"Repair estimate: €{est:,.2f}")

        if state.get("customer_profile") is None:
            completeness_flags.append("customer_profile")
            warnings.append(
                "Customer profile could not be loaded; policy tenure cannot be verified"
            )
            # Not blocking — RiskAgent can still run with partial data

        # ── 4. Build evidence bullets from validated fields ───────────────────
        if claim is not None:
            evidence += [
                f"claim_id: {claim.claim_id}",
                f"policy_id: {claim.policy_id}",
                f"customer_id: {claim.customer_id}",
                f"claim_type: {claim.claim_type}",
                f"incident_date: {claim.incident_date}",
                f"claim_amount: €{claim.claim_amount:,.2f}",
            ]

        # ── 5. Damage evidence presence ────────────────────────────────────────
        if claim is not None and claim.damage_image_path is None:
            completeness_flags.append("damage_image_path")
            warnings.append(
                "No damage photograph submitted (required per Clause 6.1; "
                "claim will be referred to manual review per Clause 5.2)"
            )

        # ── 6. Customer statement length ───────────────────────────────────────
        description = str(raw.get("incident_description") or "").strip()
        word_count = len(description.split()) if description else 0
        if word_count < _MIN_STATEMENT_WORDS:
            warnings.append(
                f"Customer statement too short: {word_count} words "
                f"(minimum recommended: {_MIN_STATEMENT_WORDS})"
            )
        else:
            evidence.append(f"Customer statement: {word_count} words")

        # ── 7. Prior claims frequency ──────────────────────────────────────────
        history = state.get("claim_history") or {}
        prior_count: int = history.get("claims_in_last_24_months", 0)
        if prior_count >= _HIGH_PRIOR_CLAIMS:
            warnings.append(
                f"High prior claim frequency: {prior_count} claims in last 24 months "
                f"(threshold: {_HIGH_PRIOR_CLAIMS} per Clause 5.4)"
            )
        else:
            evidence.append(f"Prior claims in last 24 months: {prior_count}")

        # ── 8. Late reporting ──────────────────────────────────────────────────
        days_since = _days_since_incident(
            raw.get("incident_date"),
            raw.get("submitted_at"),
        )
        if days_since < 0:
            warnings.append("Incident date or report date could not be parsed")
        elif days_since > _LATE_REPORTING_DAYS:
            warnings.append(
                f"Late reporting: {days_since} days since incident "
                f"(deadline: {_LATE_REPORTING_DAYS} days per Clause 4.1)"
            )
        else:
            evidence.append(f"Reported within deadline: {days_since} days after incident")

        # ── 9. Derive confidence and intake_complete ───────────────────────────
        intake_complete = not blocking
        confidence      = _confidence(blocking, completeness_flags, warnings)
        reasoning       = _reasoning(
            claim, prior_count, days_since, completeness_flags, intake_complete
        )

        # ── 10. Audit event ────────────────────────────────────────────────────
        audit_event = write_audit_entry(
            agent=self.AGENT_NAME,
            event="agent_complete",
            output_model={
                "confidence":         confidence,
                "evidence":           evidence,
                "warnings":           warnings,
                "reasoning_summary":  reasoning,
                "intake_complete":    intake_complete,
                "completeness_flags": completeness_flags,
                "days_since_incident": days_since,
                "prior_claims_count":  prior_count,
            },
            routing_decision="continue",
            claim_id=raw.get("claim_id", ""),
            input_summary={
                "claim_id":              raw.get("claim_id", ""),
                "policy_id":             raw.get("policy_id", ""),
                "customer_id":           raw.get("customer_id", ""),
                "claim_type_raw":        state["claim_input"].get("claim_type", ""),
                "claim_type_normalised": raw.get("claim_type", ""),
                "incident_date":         str(raw.get("incident_date", "")),
                "claim_amount":          raw.get("claim_amount"),
                "has_image":             raw.get("damage_image_path") is not None,
                "repair_estimate_present": state.get("repair_estimate") is not None,
                "customer_profile_present": state.get("customer_profile") is not None,
                "prior_claims_24m":      prior_count,
                "days_since_incident":   days_since,
            },
        )

        return {
            "claim_input":    claim.model_dump() if claim else raw,
            "intake_complete": intake_complete,
            "audit_events":   state["audit_events"] + [audit_event],
        }


# ── Module-level helpers ───────────────────────────────────────────────────────

def _normalize_claim_type(raw: str) -> str:
    """Map free-form claim_type text to a canonical value; default 'other'."""
    t = (raw or "").lower().strip()
    if t in _VALID_CLAIM_TYPES:
        return t
    for fragment, canonical in _CLAIM_TYPE_MAP.items():
        if fragment in t:
            return canonical
    return "other"


def _days_since_incident(incident_raw: object, submitted_raw: object) -> int:
    """Return days between incident and report dates; -1 on any parse failure."""
    try:
        if isinstance(incident_raw, date):
            incident = incident_raw
        else:
            incident = date.fromisoformat(str(incident_raw)[:10])

        if submitted_raw:
            reported = date.fromisoformat(str(submitted_raw)[:10])
        else:
            reported = date.today()

        return max(0, (reported - incident).days)
    except (ValueError, TypeError, AttributeError):
        return -1


def _confidence(blocking: bool, completeness_flags: list[str], warnings: list[str]) -> float:
    """Derive agent confidence from validation outcome and warning count."""
    if blocking:
        return 0.0
    score = 1.0
    score -= len(completeness_flags) * 0.15   # missing field is significant
    score -= len(warnings) * 0.04             # soft warnings reduce confidence slightly
    return round(max(0.10, min(1.0, score)), 2)


def _reasoning(
    claim: ClaimInput | None,
    prior_count: int,
    days_since: int,
    completeness_flags: list[str],
    intake_complete: bool,
) -> str:
    """Build a one-paragraph human-readable reasoning summary for audit/UI."""
    if claim is None:
        return (
            "IntakeAgent could not validate the claim form. "
            f"Blocking field errors detected: {', '.join(completeness_flags)}. "
            "The claim cannot proceed until all required fields are provided."
        )

    parts = [
        f"Claim {claim.claim_id} received for a {claim.claim_type} incident "
        f"on {claim.incident_date} under policy {claim.policy_id}.",
    ]
    if days_since >= 0:
        parts.append(f"Reported {days_since} day(s) after the incident.")
    if prior_count:
        parts.append(
            f"Customer has {prior_count} prior claim(s) in the last 24 months."
        )
    if completeness_flags:
        parts.append(
            f"Incomplete or missing: {', '.join(completeness_flags)}."
        )
    parts.append(
        "Intake validation passed — claim forwarded to damage assessment."
        if intake_complete
        else "Intake flagged blocking errors — claim cannot proceed without correction."
    )
    return " ".join(parts)
