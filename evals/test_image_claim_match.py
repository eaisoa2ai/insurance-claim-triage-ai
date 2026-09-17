"""
test_image_claim_match.py — Vision mismatch detection and error handling tests.

Covers: E-02, E-08, E-14.

Strategy:
  - Inject image_observation directly into state (bypasses file lookup).
  - For E-08: mock load_image_observation to raise KeyError.
  - Unit tests call the private scoring helpers directly.

Consistency score formula:
    score = 0.40 * s1 + 0.35 * s2 + 0.25 * s3
    s1 = damage_type_alignment   (0.0–1.0)
    s2 = parts_overlap           (0.0–1.0, weighted by plausibility)
    s3 = direction_alignment     (0.0–1.0)
    mismatch when score < 0.60 (from vision.mismatch_threshold in risk_thresholds.yaml)
"""

from unittest.mock import patch

import pytest

from agents.damage_evidence_agent import (
    DamageEvidenceAgent,
    _score_damage_type_alignment,
    _score_parts_overlap,
    _score_direction_alignment,
)


# ── State builder ─────────────────────────────────────────────────────────────

def _build_state(
    damage_image_path: str = "data/images/test.jpg",
    customer_statement: str = "significant front-end collision damage",
    repair_estimate: float = 4500.0,
    claim_type: str = "collision",
    image_observation: dict | None = None,
    est_parts: list[str] | None = None,
    plausibility: str = "reasonable",
) -> dict:
    """Return a minimal ClaimState for DamageEvidenceAgent tests."""
    line_items = [{"description": p, "cost": 0.0} for p in (est_parts or [])]
    return {
        "claim_input": {
            "claim_id": "CLM-TEST-001",
            "damage_image_path": damage_image_path,
            "claim_type": claim_type,
            "incident_description": customer_statement,
            "claim_amount": repair_estimate,
        },
        "image_observation": image_observation,
        "repair_estimate": {
            "estimated_amount": repair_estimate,
            "line_items": line_items,
            "estimate_plausibility": plausibility,
        },
        "audit_events": [],
    }


def _obs(
    visible_damage: list[str] | None = None,
    damaged_parts: list[str] | None = None,
    severity_hint: str = "low",
    confidence: float = 0.90,
) -> dict:
    """Build an image_observation dict matching the image_observations.json schema."""
    return {
        "filename": "test.jpg",
        "visible_damage": visible_damage or [],
        "damaged_parts": damaged_parts or [],
        "severity_hint": severity_hint,
        "confidence": confidence,
        "short_visual_summary": "test observation",
    }


# ── E-02: zero-damage image + collision story ─────────────────────────────────

def test_zero_damage_image_high_damage_story_mismatch():
    """E-02: burn_damage visible + no front-end parts + front-end collision story → score < 0.10."""
    state = _build_state(
        customer_statement="front-end collision, significant damage to front bumper and hood",
        claim_type="collision",
        image_observation=_obs(
            visible_damage=["burn_damage"],
            damaged_parts=["rear_bumper"],
        ),
        # Estimate lists front-end parts — zero overlap with rear image
        est_parts=["front_bumper", "hood"],
    )
    result = DamageEvidenceAgent().run(state)
    da = result["damage_evidence_assessment"]

    # burn_damage inconsistent with collision + zero parts overlap + wrong zone → very low score
    assert da["consistency_score"] < 0.10
    assert da["image_claim_mismatch"] is True
    assert result["image_claim_mismatch"] is True


# ── E-08: image observation lookup fails ──────────────────────────────────────

def test_vision_api_error_sets_zero_confidence_and_mismatch():
    """E-08: load_image_observation raises KeyError → confidence=0.0, mismatch=True."""
    state = _build_state(
        image_observation=None,  # force file lookup so the mock is hit
    )
    with patch(
        "agents.damage_evidence_agent.load_image_observation",
        side_effect=KeyError("No observation for 'test.jpg'"),
    ):
        result = DamageEvidenceAgent().run(state)

    da = result["damage_evidence_assessment"]
    assert da["confidence"] == 0.0
    assert da["image_claim_mismatch"] is True
    assert result["image_claim_mismatch"] is True
    assert any("missing" in w.lower() or "not found" in w.lower() for w in da["warnings"])


# ── E-14: moderate mismatch (mixed signals) ───────────────────────────────────

def test_severity_understated_produces_moderate_mismatch():
    """E-14: mixed damage types (dent + burn) on collision → score in 0.10–0.59, mismatch=True."""
    state = _build_state(
        customer_statement="rear-end collision",
        claim_type="collision",
        image_observation=_obs(
            visible_damage=["burn_damage", "dent"],  # mixed: one bad, one good
            damaged_parts=["driver_door", "rear_bumper"],
            severity_hint="medium",
        ),
        est_parts=["driver_door"],  # partial overlap with obs_parts
    )
    result = DamageEvidenceAgent().run(state)
    da = result["damage_evidence_assessment"]

    assert da["image_claim_mismatch"] is True
    assert 0.10 <= da["consistency_score"] < 0.60


# ── Matching cases ────────────────────────────────────────────────────────────

def test_matching_image_and_story_no_mismatch():
    """Consistent damage types, matching parts, correct direction → score ≥ 0.60."""
    state = _build_state(
        customer_statement="rear-end collision at traffic lights",
        claim_type="collision",
        image_observation=_obs(
            visible_damage=["dent", "scrape"],
            damaged_parts=["rear_bumper"],
            severity_hint="low",
        ),
        est_parts=["rear_bumper"],
    )
    result = DamageEvidenceAgent().run(state)
    da = result["damage_evidence_assessment"]

    assert da["image_claim_mismatch"] is False
    assert da["consistency_score"] >= 0.60


def test_no_image_path_returns_mismatch():
    """Missing damage_image_path → image_claim_mismatch=True, confidence=0.0."""
    state = _build_state(damage_image_path=None)
    # Override to remove the image path
    state["claim_input"]["damage_image_path"] = None
    result = DamageEvidenceAgent().run(state)

    da = result["damage_evidence_assessment"]
    assert da["image_claim_mismatch"] is True
    assert da["confidence"] == 0.0
    assert result["image_claim_mismatch"] is True


def test_consistency_above_threshold_is_not_flagged():
    """Score 0.625 (above 0.60 threshold) → image_claim_mismatch=False."""
    # s1=0.5 (no visible damage), s2=0.5 (no est_parts), s3=1.0 (rear desc + rear part)
    # 0.4*0.5 + 0.35*0.5 + 0.25*1.0 = 0.20 + 0.175 + 0.25 = 0.625
    state = _build_state(
        customer_statement="rear-end collision",
        claim_type="collision",
        image_observation=_obs(
            visible_damage=[],         # s1=0.5 neutral
            damaged_parts=["rear_bumper"],
            severity_hint="low",
        ),
        est_parts=[],                  # s2=0.5 neutral (no est_parts)
    )
    result = DamageEvidenceAgent().run(state)
    da = result["damage_evidence_assessment"]

    assert da["consistency_score"] == pytest.approx(0.625, abs=0.01)
    assert da["image_claim_mismatch"] is False


# ── Unit tests for scoring helpers ────────────────────────────────────────────

def test_all_expected_damage_types_returns_high_s1():
    """s1 = 1.0 when all visible damage types match the claim type."""
    inconsistencies: list[str] = []
    score = _score_damage_type_alignment(
        visible_damage=["dent", "scrape", "paint_transfer"],
        claim_type="collision",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(1.0)
    assert inconsistencies == []


def test_all_mismatch_damage_types_returns_low_s1():
    """s1 = 0.10 when all visible damage is inconsistent with the claim type."""
    inconsistencies: list[str] = []
    score = _score_damage_type_alignment(
        visible_damage=["burn_damage"],
        claim_type="collision",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(0.10)
    assert len(inconsistencies) == 1


def test_parts_overlap_full_match_returns_high_s2():
    """s2 = 1.0 when image parts and estimate parts are identical."""
    inconsistencies: list[str] = []
    score = _score_parts_overlap(
        obs_parts=["rear_bumper", "taillight"],
        est_parts=["rear_bumper", "taillight"],
        plausibility="reasonable",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(1.0)
    assert inconsistencies == []


def test_parts_overlap_zero_match_returns_zero_s2():
    """s2 = 0.0 when image parts and estimate parts have no intersection."""
    inconsistencies: list[str] = []
    score = _score_parts_overlap(
        obs_parts=["rear_bumper"],
        est_parts=["front_bumper", "hood"],
        plausibility="reasonable",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(0.0)
    assert len(inconsistencies) == 1  # "no parts in common" added


def test_direction_mismatch_returns_low_s3():
    """s3 = 0.10 when description claims front-end but image shows only rear parts."""
    inconsistencies: list[str] = []
    score = _score_direction_alignment(
        obs_parts=["rear_bumper"],
        description="significant front-end collision damage",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(0.10)
    assert len(inconsistencies) == 1


def test_direction_consistent_returns_high_s3():
    """s3 = 1.0 when description rear + image shows rear parts."""
    inconsistencies: list[str] = []
    score = _score_direction_alignment(
        obs_parts=["rear_bumper", "taillight"],
        description="rear-end collision from behind",
        inconsistencies=inconsistencies,
    )
    assert score == pytest.approx(1.0)
    assert inconsistencies == []
