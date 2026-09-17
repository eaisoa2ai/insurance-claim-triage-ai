"""
damage_evidence_agent.py — DamageEvidenceAgent: assess image/claim consistency.

Pipeline position: second node (after IntakeAgent). Reads the pre-loaded image
observation from state["image_observation"] (loaded here from image_observations.json
via data_loader if the field is None), then compares it against the customer's
claim story and the repair estimate.

Reads from ClaimState:
    claim_input        — damage_image_path, claim_type, incident_description
    image_observation  — pre-loaded observation dict (loaded here if None)
    repair_estimate    — line_items (part names), estimate_plausibility, estimated_amount

Writes to ClaimState (partial dict):
    image_observation          — stored if it had to be loaded here
    damage_evidence_assessment — DamageEvidenceAssessment.model_dump()
    image_claim_mismatch       — True when consistency_score < mismatch_threshold
    audit_events               — one entry appended

Consistency score (0.0–1.0) combines three sub-signals:
    s1 = damage_type_alignment  (0.40) — visible damage types vs canonical claim type
    s2 = parts_overlap          (0.35) — image parts vs estimate parts, weighted by plausibility
    s3 = direction_alignment    (0.25) — front/rear keywords in description vs observed parts

Missing image (damage_image_path is None) → confidence 0.0, mismatch True.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

import yaml

from core.audit import write_audit_entry
from core.claim_state import ClaimState, DamageEvidenceAssessment
from core.data_loader import load_image_observation


_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "risk_thresholds.yaml"
_DEFAULT_MISMATCH_THRESHOLD = 0.60

# ── Damage classification tables ──────────────────────────────────────────────

# Damage labels expected for each canonical claim type (from image_observations.json)
_EXPECTED_DAMAGE: dict[str, set[str]] = {
    "collision": {
        "scrape", "dent", "paint_transfer", "crumple", "structural_deformation",
        "total_loss", "buckled", "bent", "impact", "compression", "rear_impact",
        "collision", "deformed", "deployed_airbags",
    },
    "glass": {
        "shattered_glass", "broken_glass", "glass_chip", "windshield_crack",
        "cracked_glass", "glass_damage",
    },
    "vandalism": {
        "scratch", "keyed", "spray_paint", "scratch_marks", "burn_damage",
        "shattered_glass", "dent", "vandalism",
    },
}

# Damage labels that are strongly inconsistent with a given claim type
_INCONSISTENT_DAMAGE: dict[str, set[str]] = {
    "collision": {"burn_damage", "fire_damage", "spray_paint", "scratch_marks"},
    "glass":     {"burn_damage", "crumple", "structural_deformation", "spray_paint"},
    "vandalism": {"crumple", "structural_deformation", "total_loss", "compression"},
}

# Parts by damage zone — used for directional consistency check
_FRONT_PARTS: set[str] = {
    "front_bumper", "front_quarter_panel", "hood", "front_grille",
    "radiator", "front_windshield", "front_headlight", "engine_bay",
}
_REAR_PARTS: set[str] = {
    "rear_bumper", "rear_quarter_panel", "trunk", "boot",
    "rear_windshield", "taillight", "rear_headlight",
}

_SEVERITY_MAP = {"low": "minor", "medium": "moderate", "high": "severe"}
_PLAUSIBILITY_WEIGHT = {"reasonable": 1.0, "high": 0.9, "unclear": 0.6, "suspicious": 0.25}


# ── Config ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _mismatch_threshold() -> float:
    return float(
        _load_config().get("vision", {}).get("mismatch_threshold", _DEFAULT_MISMATCH_THRESHOLD)
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class DamageEvidenceAgent:

    AGENT_NAME = "DamageEvidenceAgent"

    def run(self, state: ClaimState) -> dict:
        """Assess image/claim consistency; return partial ClaimState dict."""
        claim    = state["claim_input"]
        estimate = state.get("repair_estimate") or {}
        image_path: str | None = claim.get("damage_image_path")

        if not image_path:
            return _no_image_result(self.AGENT_NAME, state)

        # Prefer already-loaded observation; fall back to data_loader
        obs: dict | None = state.get("image_observation")
        if obs is None:
            filename = pathlib.Path(image_path).name
            try:
                obs = load_image_observation(filename)
            except (KeyError, FileNotFoundError) as exc:
                return _observation_missing_result(self.AGENT_NAME, state, str(exc))

        visible_damage: list[str] = obs.get("visible_damage", [])
        obs_parts: list[str]      = obs.get("damaged_parts", [])
        severity_hint: str        = obs.get("severity_hint", "low")
        obs_confidence: float     = float(obs.get("confidence", 0.8))
        short_summary: str        = obs.get("short_visual_summary", "")

        # Extract part names from line_items (stored as {"description": part, "cost": 0})
        est_part_names: list[str] = [
            item["description"] if isinstance(item, dict) else str(item)
            for item in estimate.get("line_items", [])
        ]
        plausibility: str = estimate.get("estimate_plausibility", "unclear")
        claim_type: str   = claim.get("claim_type", "other")
        description: str  = str(claim.get("incident_description") or "").lower()

        # Compute sub-scores; inconsistencies collected as a side-effect
        inconsistencies: list[str] = []
        s1 = _score_damage_type_alignment(visible_damage, claim_type, inconsistencies)
        s2 = _score_parts_overlap(obs_parts, est_part_names, plausibility, inconsistencies)
        s3 = _score_direction_alignment(obs_parts, description, inconsistencies)

        consistency_score = round(max(0.0, min(1.0, 0.40 * s1 + 0.35 * s2 + 0.25 * s3)), 4)
        threshold = _mismatch_threshold()
        mismatch  = consistency_score < threshold
        routing   = "human_review" if mismatch else "continue"

        # Confidence: how certain we are about the assessment in either direction
        confidence = round(obs_confidence * (0.5 + abs(consistency_score - 0.5)), 2)

        severity     = _SEVERITY_MAP.get(severity_hint, "unknown")
        damage_type  = visible_damage[0] if visible_damage else "undetermined"
        repair_scope = _infer_repair_scope(severity_hint, float(estimate.get("estimated_amount", 0)))

        evidence: list[str] = [
            f"Observed damage: {', '.join(visible_damage) or 'none detected'}",
            f"Damaged parts (image): {', '.join(obs_parts) or 'not identified'}",
            f"Damaged parts (estimate): {', '.join(est_part_names) or 'not listed'}",
            f"Severity: {severity_hint} → {severity}",
            f"Consistency score: {consistency_score:.2f} (threshold: {threshold:.2f})",
            f"Estimate plausibility: {plausibility}",
        ]
        if short_summary:
            evidence.append(f"Image summary: {short_summary}")

        warnings: list[str] = []
        if mismatch:
            warnings.append(
                f"Image/claim consistency score {consistency_score:.2f} is below "
                f"the {threshold:.2f} threshold — human review required"
            )
        if plausibility == "suspicious":
            warnings.append(
                "Repair estimate flagged as suspicious: stated parts may not match observed damage"
            )
        warnings.extend(inconsistencies)
        if obs.get("uncertainty"):
            warnings.append(f"Image observation uncertainty: {obs['uncertainty']}")

        assessment = DamageEvidenceAssessment(
            confidence=confidence,
            evidence=evidence,
            reasoning_summary=_reasoning(
                claim_type, consistency_score, threshold, mismatch,
                visible_damage, obs_parts, plausibility,
            ),
            warnings=warnings,
            damage_type=damage_type,
            severity=severity,
            affected_components=obs_parts,
            estimated_repair_scope=repair_scope,
            consistency_score=consistency_score,
            image_claim_mismatch=mismatch,
            vision_observations=visible_damage,
        )

        audit_event = write_audit_entry(
            agent=self.AGENT_NAME,
            event="agent_complete",
            output_model=assessment,
            routing_decision=routing,
            claim_id=claim.get("claim_id", ""),
            input_summary={
                "damage_image_path":     image_path,
                "claim_type":            claim_type,
                "estimated_amount":      float(estimate.get("estimated_amount", 0)),
                "estimate_plausibility": plausibility,
                "obs_visible_damage":    visible_damage,
                "obs_damaged_parts":     obs_parts,
                "severity_hint":         severity_hint,
                "observation_source":    "pre_loaded" if state.get("image_observation") else "data_loader",
            },
        )

        return {
            "image_observation":          obs,
            "damage_evidence_assessment": assessment.model_dump(),
            "image_claim_mismatch":       mismatch,
            "audit_events":               state["audit_events"] + [audit_event],
        }


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_damage_type_alignment(
    visible_damage: list[str],
    claim_type: str,
    inconsistencies: list[str],
) -> float:
    """Score how well observed damage types fit the canonical claim type. 0.0–1.0."""
    if not visible_damage:
        return 0.5  # no image data — neutral

    expected = _EXPECTED_DAMAGE.get(claim_type, set())
    bad      = _INCONSISTENT_DAMAGE.get(claim_type, set())

    if not expected and not bad:
        return 0.7  # "other" claim type — give benefit of the doubt

    n          = len(visible_damage)
    matches    = sum(1 for d in visible_damage if d in expected)
    mismatches = sum(1 for d in visible_damage if d in bad)

    for d in visible_damage:
        if d in bad:
            inconsistencies.append(
                f"Observed '{d}' is inconsistent with a {claim_type} claim"
            )

    if mismatches > 0 and matches == 0:
        return 0.10  # definitive mismatch: all evidence contradicts the claim type
    if mismatches > 0:
        return max(0.0, min(1.0, (matches - mismatches) / n))
    if matches > 0:
        return max(0.5, matches / n)  # floor at 0.5 when positive signals exist
    return 0.40  # not matched but not inconsistent — weak neutral


def _score_parts_overlap(
    obs_parts: list[str],
    est_parts: list[str],
    plausibility: str,
    inconsistencies: list[str],
) -> float:
    """Jaccard similarity between image parts and estimate parts, scaled by plausibility."""
    if not obs_parts or not est_parts:
        return 0.5  # can't score without both lists

    obs_set = {p.lower().replace(" ", "_") for p in obs_parts}
    est_set = {p.lower().replace(" ", "_") for p in est_parts}

    intersection = len(obs_set & est_set)
    union        = len(obs_set | est_set)
    jaccard      = intersection / union if union else 0.0

    weight = _PLAUSIBILITY_WEIGHT.get(plausibility, 0.6)
    score  = round(jaccard * weight, 4)

    if intersection == 0:
        inconsistencies.append(
            f"No parts in common between image {list(obs_parts)} "
            f"and estimate {list(est_parts)}"
        )

    return score


def _score_direction_alignment(
    obs_parts: list[str],
    description: str,
    inconsistencies: list[str],
) -> float:
    """Check if directional keywords in the description align with the observed damage zone."""
    if not obs_parts:
        return 0.7  # no parts to compare

    front_kws = ("front", "frontal", "head-on")
    rear_kws  = ("rear", " back ", "behind", "from behind", "rear-end", "rear end")

    desc_front = any(kw in description for kw in front_kws)
    desc_rear  = any(kw in description for kw in rear_kws)

    if not desc_front and not desc_rear:
        return 0.7  # description has no directional signal — can't penalise

    part_set  = {p.lower() for p in obs_parts}
    has_front = bool(part_set & _FRONT_PARTS)
    has_rear  = bool(part_set & _REAR_PARTS)

    # Front claim but image shows only rear damage
    if desc_front and not desc_rear and has_rear and not has_front:
        inconsistencies.append(
            "Description mentions front-end damage but image shows only rear-zone parts"
        )
        return 0.10

    # Rear claim but image shows only front damage
    if desc_rear and not desc_front and has_front and not has_rear:
        inconsistencies.append(
            "Description mentions rear damage but image shows only front-zone parts"
        )
        return 0.10

    return 1.0  # consistent or overlapping zones


# ── Utilities ─────────────────────────────────────────────────────────────────

def _infer_repair_scope(severity_hint: str, estimated_amount: float) -> str:
    if severity_hint == "high" or estimated_amount > 10_000:
        return "structural"
    if severity_hint == "medium" or estimated_amount > 3_000:
        return "panel_repair"
    return "cosmetic"


def _reasoning(
    claim_type: str,
    consistency_score: float,
    threshold: float,
    mismatch: bool,
    visible_damage: list[str],
    obs_parts: list[str],
    plausibility: str,
) -> str:
    verdict = "does NOT match" if mismatch else "is consistent with"
    return (
        f"Photographic evidence {verdict} the reported {claim_type} claim "
        f"(consistency score: {consistency_score:.2f}, threshold: {threshold:.2f}). "
        f"Image shows {', '.join(visible_damage) or 'no identifiable damage'} "
        f"affecting {', '.join(obs_parts) or 'unidentified parts'}. "
        f"Repair estimate plausibility is rated '{plausibility}'."
    )


# ── Error-path helpers (no image / observation missing) ───────────────────────

def _no_image_result(agent_name: str, state: ClaimState) -> dict:
    """Return partial state when no damage photograph was submitted."""
    claim = state.get("claim_input") or {}
    assessment = DamageEvidenceAssessment(
        confidence=0.0,
        evidence=["No damage photograph was submitted with the claim"],
        reasoning_summary=(
            "No image evidence available. The claim cannot be assessed for visual "
            "consistency without a damage photograph. Manual review is mandatory "
            "per Clause 5.2."
        ),
        warnings=[
            "damage_image_path is None — no photographic evidence submitted",
            "Human review required per Clause 5.2",
        ],
        damage_type="unknown",
        severity="unknown",
        affected_components=[],
        estimated_repair_scope="unknown",
        consistency_score=0.0,
        image_claim_mismatch=True,
        vision_observations=[],
    )
    audit_event = write_audit_entry(
        agent=agent_name,
        event="agent_complete",
        output_model=assessment,
        routing_decision="human_review",
        claim_id=claim.get("claim_id", ""),
        input_summary={
            "damage_image_path":     None,
            "claim_type":            claim.get("claim_type", ""),
            "estimated_amount":      (state.get("repair_estimate") or {}).get("estimated_amount"),
            "observation_source":    "none — image missing",
        },
    )
    return {
        "damage_evidence_assessment": assessment.model_dump(),
        "image_claim_mismatch":       True,
        "audit_events":               state["audit_events"] + [audit_event],
    }


def _observation_missing_result(agent_name: str, state: ClaimState, error: str) -> dict:
    """Return partial state when the image observation file lookup fails."""
    claim = state.get("claim_input") or {}
    assessment = DamageEvidenceAssessment(
        confidence=0.0,
        evidence=[f"Image observation not found: {error}"],
        reasoning_summary=(
            "No pre-generated observation found for the submitted image. "
            "Manual review is required before processing can continue."
        ),
        warnings=[error, "Missing image observation — human review required"],
        damage_type="unknown",
        severity="unknown",
        affected_components=[],
        estimated_repair_scope="unknown",
        consistency_score=0.0,
        image_claim_mismatch=True,
        vision_observations=[],
    )
    audit_event = write_audit_entry(
        agent=agent_name,
        event="agent_complete",
        output_model=assessment,
        routing_decision="human_review",
        claim_id=claim.get("claim_id", ""),
        input_summary={
            "damage_image_path":     claim.get("damage_image_path"),
            "claim_type":            claim.get("claim_type", ""),
            "estimated_amount":      (state.get("repair_estimate") or {}).get("estimated_amount"),
            "observation_source":    "none — lookup failed",
            "lookup_error":          error,
        },
    )
    return {
        "damage_evidence_assessment": assessment.model_dump(),
        "image_claim_mismatch":       True,
        "audit_events":               state["audit_events"] + [audit_event],
    }
