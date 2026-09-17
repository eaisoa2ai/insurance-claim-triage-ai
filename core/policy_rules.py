"""
policy_rules.py — Policy document loader and helpers.

PolicyRules mirrors the structure of data/policies/standard_auto_policy.json.
Every field read by PolicyAgent must be declared here — no raw dict key access
on unparsed JSON inside the agent.

Hard constraint: No policy rule may be invented or defaulted here. If a required
field is absent from the JSON document, Pydantic raises ValidationError and
PolicyAgent sets policy_valid=False and escalates to human review.
"""

from __future__ import annotations

import json
import pathlib
from datetime import date

from pydantic import BaseModel


# ── Policy document models ─────────────────────────────────────────────────────

class CoverageRule(BaseModel):
    clause: str                               # exact clause name; cited verbatim in PolicyDecision
    coverage_ratio: float
    covered_damage_types: list[str]
    excluded_damage_types: list[str] = []
    excluded_damage_clauses: dict[str, str] = {}  # damage_type → clause reference from document
    glass_deductible: float | None = None


class PolicyRules(BaseModel):
    policy_id: str
    policy_name: str
    effective_date: date
    end_date: date
    deductible: float
    coverage_cap: float
    coverages: dict[str, CoverageRule]        # keys: "collision", "comprehensive", …
    exclusions: list[str]
    exclusion_clauses: dict[str, str] = {}    # exclusion key → clause reference
    manual_review_triggers: dict = {}         # raw trigger config from document
    late_reporting: dict = {}                 # reporting deadline config from document


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_policy_rules(path: str | pathlib.Path) -> PolicyRules:
    """Parse a JSON policy file into a validated PolicyRules model.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: file is not JSON (e.g. markdown — cannot be parsed automatically).
        pydantic.ValidationError: document is missing or has invalid required fields.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"PolicyAgent requires a JSON policy file; got {path.suffix!r}. "
            "Markdown policies cannot be parsed into structured rules — escalate to manual review."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return PolicyRules.model_validate(data)


# ── Coverage decision helper ───────────────────────────────────────────────────

def is_damage_covered(
    rules: PolicyRules,
    claim_type: str,
    incident_date: date,
) -> tuple[str, str, float]:
    """Return (coverage_decision, applicable_coverage_type, coverage_ratio).

    coverage_decision:        "covered" | "excluded"
    applicable_coverage_type: key in rules.coverages (e.g. "collision"), or "none"
    coverage_ratio:           0.0 when excluded

    Logic (two-pass — covered always wins over excluded when both apply):
      Pass 1: if claim_type is in any bucket's covered_damage_types → covered
      Pass 2: if claim_type is in any bucket's excluded_damage_types → excluded
      Fallthrough: not mentioned anywhere → excluded (no coverage by default)

    Does not raise. "excluded" is the safe default.
    """
    # Policy date validity — incident must fall inside the policy period
    if incident_date < rules.effective_date or incident_date > rules.end_date:
        return ("excluded", "none", 0.0)

    # Pass 1: look for a coverage match first (positive check wins)
    for bucket, rule in rules.coverages.items():
        if claim_type in rule.covered_damage_types:
            return ("covered", bucket, rule.coverage_ratio)

    # Pass 2: look for an explicit exclusion
    for bucket, rule in rules.coverages.items():
        if claim_type in rule.excluded_damage_types:
            return ("excluded", bucket, 0.0)

    # Fallthrough: claim_type not mentioned in any coverage bucket
    return ("excluded", "none", 0.0)
