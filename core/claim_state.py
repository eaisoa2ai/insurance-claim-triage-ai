"""
claim_state.py — All Pydantic models and the shared LangGraph state.

Model hierarchy:
  AgentOutputBase          — base mixin every agent output inherits
  ├── DamageEvidenceAssessment
  ├── PolicyDecision
  ├── RiskAssessment
  └── PayoutRecommendation

Input models (populated before the pipeline starts):
  ClaimInput               — raw claim form; validates claim_id, policy_id,
                             damage_image_path, claim_amount
  CustomerProfile          — customer identity and policy metadata
  ClaimHistory             — prior claims list with 24-month summary counts
  RepairEstimate           — workshop estimate with itemised line items

Internal model (never stored in ClaimState directly):
  ImageObservation         — raw structured output from the vision model

Supporting models:
  HumanApprovalDecision    — escalation flags + adjuster decision + note
  AgentAuditEvent          — one entry per node written to the audit trail
  FinalReport              — end-of-pipeline decision summary

ClaimState (TypedDict):
  All models are stored as plain dicts via .model_dump() so LangGraph can merge
  partial returns. Reconstruct with Model.model_validate(state["field"]).
  Routing flags are kept as top-level scalars so routing.py can read them
  without deserialising the full model.

Validation rules enforced by Pydantic:
  • claim_id       — must not be blank (ClaimInput)
  • policy_id      — must not be blank (ClaimInput, PolicyDecision)
  • damage_image_path — must not be blank (ClaimInput)
  • claim_amount   — must be > 0 (ClaimInput)
  • estimated_amount — must be > 0 (RepairEstimate)
  • adjuster_note  — min 10 chars when a decision (approved/rejected) is recorded
  • coverage_decision — must be "covered" | "excluded" | "partial"
  • risk_category  — must be "low" | "medium" | "high" | "critical"
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Base mixin ─────────────────────────────────────────────────────────────────

class AgentOutputBase(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str]
    reasoning_summary: str
    warnings: list[str] = Field(default_factory=list)


# ── Input models ───────────────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    """Raw claim form submitted by the customer. All required fields validated here."""

    claim_id: str = Field(..., min_length=1)
    policy_id: str = Field(..., min_length=1)
    customer_id: str = Field(..., min_length=1)
    claim_type: str                         # "collision" | "glass" | "vandalism" | "other"
    incident_date: date
    incident_description: str
    damage_image_path: str | None = None   # None for missing-evidence claims (CLM-019 pattern)
    claim_amount: float = Field(..., gt=0, description="Claimed repair cost; must be > 0")
    policy_document_path: str = Field(..., min_length=1)
    submitted_at: datetime = Field(default_factory=_utcnow)

    @field_validator("claim_id", "policy_id", "customer_id",
                     "policy_document_path", mode="after")
    @classmethod
    def no_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be blank or whitespace-only")
        return v.strip()

    @field_validator("claim_amount", mode="after")
    @classmethod
    def claim_amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"claim_amount must be > 0, got {v}")
        return v


class PriorClaim(BaseModel):
    claim_id: str
    date: date
    claim_type: str
    amount: float = Field(..., ge=0)
    outcome: str                            # "approved" | "rejected" | "pending"


class CustomerProfile(BaseModel):
    customer_id: str = Field(..., min_length=1)
    name: str
    email: str
    phone: str
    policy_id: str = Field(..., min_length=1)
    policy_start_date: date
    address: str


class ClaimHistory(BaseModel):
    customer_id: str
    prior_claims: list[PriorClaim] = Field(default_factory=list)
    total_claims_count: int = Field(default=0, ge=0)
    claims_in_last_24_months: int = Field(default=0, ge=0)


class LineItem(BaseModel):
    description: str
    cost: float = Field(..., ge=0)


class RepairEstimate(BaseModel):
    claim_id: str = Field(..., min_length=1)
    estimated_amount: float = Field(..., gt=0, description="Total repair cost; must be > 0")
    workshop_name: str
    estimate_date: date
    line_items: list[LineItem] = Field(default_factory=list)
    estimate_plausibility: str = "unclear"  # "reasonable" | "high" | "suspicious" | "unclear"

    @field_validator("claim_id", mode="after")
    @classmethod
    def claim_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("repair_estimate.claim_id must not be blank")
        return v.strip()

    @field_validator("estimated_amount", mode="after")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"estimated_amount must be > 0, got {v}")
        return v


# ── Vision model (internal — not stored directly in ClaimState) ────────────────

class ImageObservation(BaseModel):
    """Raw structured output from the gpt-4o vision call.

    Produced by core/vision_client.py and consumed only by DamageEvidenceAgent
    to compute the consistency score. Never forwarded as-is to downstream agents.
    """

    damage_type: str        # collision | scrape | dent | glass | fire |
                            # flood | vandalism | total_loss | none_visible
    severity: str           # none | minor | moderate | severe | total_loss
    affected_components: list[str]
    estimated_repair_scope: str     # cosmetic | panel_repair | structural | total_loss
    confidence: float = Field(..., ge=0.0, le=1.0)
    observations: Annotated[list[str], Field(max_length=5)] = Field(default_factory=list)


# ── Agent output models ────────────────────────────────────────────────────────

class DamageEvidenceAssessment(AgentOutputBase):
    damage_type: str
    severity: str
    affected_components: list[str]
    estimated_repair_scope: str
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    image_claim_mismatch: bool
    vision_observations: list[str] = Field(default_factory=list)

    @field_validator("consistency_score", mode="after")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class PolicyDecision(AgentOutputBase):
    policy_id: str = Field(..., min_length=1)
    policy_valid: bool
    coverage_decision: str          # "covered" | "excluded" | "partial"
    applicable_coverage_type: str   # "collision" | "comprehensive" | "none"
    clause_cited: str               # exact key from the loaded policy document
    deductible: float = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    coverage_cap: float = Field(default=0.0, ge=0)  # 0.0 only when policy failed to load
    exclusion_reasons: list[str] = Field(default_factory=list)

    @field_validator("coverage_decision", mode="after")
    @classmethod
    def valid_coverage_decision(cls, v: str) -> str:
        allowed = {"covered", "excluded", "partial"}
        if v not in allowed:
            raise ValueError(f"coverage_decision must be one of {allowed!r}, got {v!r}")
        return v


class RiskAssessment(AgentOutputBase):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_category: str              # "low" | "medium" | "high" | "critical"
    contributing_factors: list[dict] = Field(default_factory=list)
    # each factor: {"factor": str, "weight": float, "contribution": float, "triggered": bool}

    @field_validator("risk_category", mode="after")
    @classmethod
    def valid_risk_category(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "critical"}
        if v not in allowed:
            raise ValueError(f"risk_category must be one of {allowed!r}, got {v!r}")
        return v


class PayoutRecommendation(AgentOutputBase):
    repair_estimate: float = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    deductible: float = Field(..., ge=0)
    payout_amount: float = Field(..., ge=0)     # clamped to [0, coverage_cap]
    coverage_cap: float = Field(..., gt=0)
    auto_approve_eligible: bool


# ── Human review ───────────────────────────────────────────────────────────────

class HumanApprovalDecision(BaseModel):
    """Escalation flags set by the router node; decision fields filled by the adjuster."""

    required: bool = False
    escalation_reason: str = ""         # first triggered condition (shown in UI)
    escalation_flags: list[str] = Field(default_factory=list)

    approved: bool | None = None        # None until adjuster acts
    adjuster_note: str | None = None
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def note_required_when_decided(self) -> HumanApprovalDecision:
        # AC-11: adjuster must supply a note of at least 10 chars before deciding
        if self.approved is not None:
            note = (self.adjuster_note or "").strip()
            if len(note) < 10:
                raise ValueError(
                    "adjuster_note must be at least 10 characters when approved/rejected"
                )
        return self


# ── Audit trail ────────────────────────────────────────────────────────────────

class AgentAuditEvent(BaseModel):
    """One structured entry in the audit trail. Written by every node."""

    timestamp: datetime = Field(default_factory=_utcnow)
    agent: str                          # agent name or "router" | "auto_approve" | "adjuster"
    event: str                          # "agent_complete" | "routing" | "human_review_requested"
                                        # | "adjuster_decision" | "auto_approved" | "error"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    routing_decision: str = "continue"  # "continue" | "human_review" | "auto_approve"
    output_summary: dict = Field(default_factory=dict)


# ── Final report ───────────────────────────────────────────────────────────────

class FinalReport(BaseModel):
    """End-of-pipeline summary written by auto_approve or human_review nodes."""

    claim_id: str
    decision: str                       # "auto_approved" | "human_approved" | "rejected" | "pending"
    payout_amount: float | None = None
    risk_category: str | None = None
    coverage_decision: str | None = None
    escalation_flags: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)


# ── Shared LangGraph state ─────────────────────────────────────────────────────

class ClaimState(TypedDict):
    """Single mutable object threaded through every LangGraph node.

    All Pydantic models are stored as plain dicts via .model_dump() so LangGraph
    can merge partial returns. Reconstruct with Model.model_validate(state["x"]).
    Routing flags are kept as top-level scalars so routing.py reads them without
    deserialising.
    """

    # ── PIPELINE INPUTS ───────────────────────────────────────────────────────
    claim_input: dict                   # ClaimInput.model_dump()
    customer_profile: dict | None       # CustomerProfile.model_dump()
    claim_history: dict | None          # ClaimHistory.model_dump()
    repair_estimate: dict | None        # RepairEstimate.model_dump()
    image_observation: dict | None      # ImageObservation.model_dump()

    # ── AGENT OUTPUTS ─────────────────────────────────────────────────────────
    damage_evidence_assessment: dict | None     # DamageEvidenceAssessment
    policy_decision: dict | None               # PolicyDecision
    risk_assessment: dict | None               # RiskAssessment
    payout_recommendation: dict | None         # PayoutRecommendation

    # ── HUMAN REVIEW ──────────────────────────────────────────────────────────
    human_approval: dict | None                # HumanApprovalDecision

    # ── ROUTING FLAGS (read directly by routing.py) ───────────────────────────
    intake_complete: bool
    image_claim_mismatch: bool
    policy_valid: bool
    coverage_decision: str              # "covered" | "excluded" | "partial" | ""
    risk_score: float
    risk_category: str                  # "low" | "medium" | "high" | "critical" | ""
    requires_human_approval: bool
    escalation_reason: str
    escalation_flags: list[str]

    # ── AUDIT AND FINAL OUTPUT ────────────────────────────────────────────────
    audit_events: list[dict]            # list of AgentAuditEvent.model_dump()
    final_report: dict | None           # FinalReport.model_dump()
