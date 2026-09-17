# BUILD_SPEC.md — ClaimSight: Insurance Claims AI Agent Team

**Source of truth for implementation. Do not implement anything not described here.**
**Last updated:** 2026-06-16
**Status:** Implementation-ready

---

## 1. Project Goal

Build a production-style five-agent LangGraph pipeline that processes auto insurance claims end-to-end. The system takes structured claim inputs, runs them through five sequential agents, and produces a structured decision packet — either auto-approving clean claims or escalating ambiguous, mismatched, or high-risk claims to a human adjuster.

The primary deliverable is a working Gradio UI demo suitable for a 15-minute YouTube walkthrough.

**Stack:**

| Layer | Tool |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Structured data | Pydantic v2 |
| Vision model | OpenAI `gpt-4o` via `openai` SDK |
| UI | Gradio |
| Config | `python-dotenv` + YAML |
| Testing | pytest |
| Package manager | `uv` |

**Run command:** `uv run python main.py`

---

## 2. Minimal Demo Scope

Three scenarios, run live during the video. All other claims serve as the evaluation dataset.

### Scenario A — Clean Claim
- **Image:** `data/images/0005.jpg` (minor rear bumper scrape, silver sedan)
- **Story:** "Someone bumped my car while it was parked in a shopping centre lot. Small dent and scrape on the rear bumper."
- **Estimate:** $850
- **Policy:** Collision coverage, $500 deductible
- **History:** No prior claims
- **Expected result:** Auto-approve. Payout $350. Risk score ≈ 12.
- **Demo beat:** All five agents fire sequentially, audit trail populates, green auto-approve badge appears. Under 30 seconds.

### Scenario B — Image-Story Mismatch
- **Image:** `data/images/F0022.jpg` (black Jeep Grand Cherokee, zero visible damage)
- **Story:** "I was in a front-end collision at an intersection. Significant structural damage to the front bumper and engine bay."
- **Estimate:** $4,500
- **Policy:** Collision coverage, $500 deductible
- **History:** No prior claims
- **Expected result:** Human review. Vision model finds no damage. Consistency score ≈ 0.05. Escalation reason displayed.
- **Demo beat:** DamageEvidenceAgent flags mismatch, routing snaps to human review, Approve/Reject panel appears.

### Scenario C — High-Risk / High-Payout
- **Image:** `data/images/S0016.jpg` (blue Smart car, total front-end destruction)
- **Story:** "My car was completely wrecked in a head-on collision. Filing late due to injuries."
- **Estimate:** $14,500
- **Policy:** Collision coverage, $500 deductible, $15,000 cap
- **History:** 3 prior claims in 18 months; incident reported 45 days after it occurred
- **Expected result:** Human review. Risk score ≈ 88, category critical. Multiple escalation triggers fire.
- **Demo beat:** Multiple red warning badges. Adjuster types a note and clicks Reject. Audit trail records decision.

---

## 3. Agent Roles and Responsibilities

### 3.1 IntakeAgent

**Purpose:** Validate and normalise all claim inputs.

**Reads from ClaimState:** `claim_form`, `customer_statement`, `claim_history`, `damage_image_path`, `repair_estimate`, `policy_document_path`

**Writes to ClaimState:** `intake_result`, `intake_complete`

**Logic:**
- Normalise claim fields (strip whitespace, cast types, fill defaults)
- Extract: `claim_type`, `incident_date`, `damage_description`, `prior_claims_count`, `days_since_incident`
- Flag in `warnings` when:
  - Required fields are missing
  - Customer statement is fewer than 20 words
  - `prior_claims_count >= 3` within 24 months
  - `days_since_incident > 30`
- Set `intake_complete = True` when no blocking errors

**Output model:** `IntakeResult` (see Section 6)

---

### 3.2 DamageEvidenceAgent

**Purpose:** Inspect the damage image with `gpt-4o` and compare findings to the customer story and repair estimate.

**Reads from ClaimState:** `damage_image_path`, `customer_statement`, `repair_estimate`, `intake_result`

**Writes to ClaimState:** `damage_evidence_result`, `image_claim_mismatch`

**Logic:**
- Load and base64-encode the image from `damage_image_path`
- Call `tools/vision.py` → `call_vision_model(b64_image, prompt)` with a structured extraction prompt
- Parse the response into a Pydantic model (never forward raw text downstream)
- Compute `consistency_score` (0.0–1.0):
  - Compare detected damage type vs. stated damage description
  - Compare detected severity vs. repair estimate dollar range
  - Compare affected components vs. customer story
- Set `image_claim_mismatch = True` when `consistency_score < 0.60`
- If the vision API errors or returns no content: `confidence = 0.0`, `image_claim_mismatch = True`, add error to `warnings`

**Output model:** `DamageEvidenceResult` (see Section 6)

**Vision prompt template (stored in `tools/vision.py`):**
```
You are a vehicle damage assessment expert. Inspect this car damage image and return a JSON object with:
- damage_type: one of [collision, scrape, dent, glass, fire, flood, vandalism, total_loss, none_visible]
- severity: one of [none, minor, moderate, severe, total_loss]
- affected_components: list of damaged parts visible (e.g. ["rear_bumper", "trunk_lid"])
- estimated_repair_scope: one of [cosmetic, panel_repair, structural, total_loss]
- confidence: float 0.0-1.0 representing your confidence in this assessment
- observations: list of specific visual evidence strings (max 5)
Return only valid JSON. No explanation text.
```

---

### 3.3 PolicyAgent

**Purpose:** Parse the uploaded policy document and determine coverage.

**Reads from ClaimState:** `policy_document_path`, `intake_result`, `damage_evidence_result`

**Writes to ClaimState:** `policy_result`, `policy_valid`, `coverage_decision`

**Logic:**
- Load policy rules via `tools/policy_loader.py` → returns a structured `PolicyRules` object
- Match `damage_type` from DamageEvidenceResult against coverage rules
- Determine `coverage_decision`: `"covered"` / `"excluded"` / `"partial"`
- Cite the exact clause name or rule key that drove the decision
- Set `policy_valid = False` when:
  - Policy file is missing or unreadable
  - Policy `end_date` is before `incident_date`
  - Required fields are absent from the parsed document
- **Hard constraint:** Must not infer or invent policy rules. Only use facts extracted from the loaded document.

**Output model:** `PolicyResult` (see Section 6)

---

### 3.4 RiskAgent

**Purpose:** Score the claim from 0–100 using structured outputs from all prior agents.

**Reads from ClaimState:** `intake_result`, `damage_evidence_result`, `policy_result`

**Writes to ClaimState:** `risk_result`, `risk_score`, `risk_category`

**Logic — weighted risk factors (weights in `config/risk_thresholds.yaml`):**

| Factor | Signal | Default weight |
|---|---|---|
| Image-claim consistency | `consistency_score` inverted | 35 |
| Prior claims frequency | `prior_claims_count >= 3` in 24 months | 25 |
| Late reporting | `days_since_incident > 30` | 15 |
| Estimate vs. severity | estimate outside expected range for severity | 15 |
| Policy tenure | policy age < 6 months | 10 |

**Risk categories:**
- Low: 0–40
- Medium: 41–69
- High: 70–84
- Critical: 85–100

**Output model:** `RiskResult` (see Section 6)

---

### 3.5 PayoutAgent

**Purpose:** Calculate recommended payout and determine auto-approve eligibility.

**Reads from ClaimState:** `damage_evidence_result`, `policy_result`, `risk_result`, `repair_estimate`

**Writes to ClaimState:** `payout_result`

**Logic:**
- `payout_amount = repair_estimate × coverage_ratio − deductible`
- Clamp to `[0, policy_coverage_cap]`
- Set `auto_approve_eligible = True` only when:
  - `payout_amount <= auto_approve_limit` (default $5,000)
  - `requires_human_approval` will be `False` after routing

**Output model:** `PayoutResult` (see Section 6)

---

## 4. Required Data Files

```
insurance-claims-agent-team/
├── data/
│   ├── images/                        # 20 synthetic claim images (already present)
│   │   ├── 0002.jpg … S0022.jpg
│   └── policies/
│       └── standard_auto_policy.json  # demo policy rule table
├── config/
│   └── risk_thresholds.yaml           # all numeric cutoffs
└── _bmad-output/
    └── planning-artifacts/
        ├── brief.md
        ├── prd.md
        └── synthetic-claims.md        # 20 synthetic claims with expected outcomes
```

### `data/policies/standard_auto_policy.json` — required structure

```json
{
  "policy_id": "AUTO-STD-001",
  "policy_name": "Standard Auto Collision & Comprehensive",
  "effective_date": "2025-01-01",
  "end_date": "2026-12-31",
  "deductible": 500,
  "coverage_cap": 15000,
  "coverages": {
    "collision": {
      "coverage_ratio": 0.80,
      "covered_damage_types": ["collision", "scrape", "dent", "total_loss"],
      "excluded_damage_types": ["fire", "flood"]
    },
    "comprehensive": {
      "coverage_ratio": 1.00,
      "covered_damage_types": ["glass", "vandalism", "fire", "flood"],
      "glass_deductible": 0
    }
  },
  "exclusions": [
    "damage_reported_after_90_days",
    "intentional_damage",
    "commercial_use"
  ]
}
```

### `config/risk_thresholds.yaml` — required structure

```yaml
risk:
  human_review_threshold: 70
  confidence_floor: 0.75
  weights:
    consistency_score: 35
    prior_claims_frequency: 25
    late_reporting: 15
    estimate_vs_severity: 15
    policy_tenure: 10
  prior_claims_window_months: 24
  prior_claims_high_threshold: 3
  late_reporting_days: 30

payout:
  auto_approve_limit: 5000

vision:
  mismatch_threshold: 0.60
```

---

## 5. Vision Evidence Workflow

```
DamageEvidenceAgent
  │
  ├── 1. Load image bytes from damage_image_path
  ├── 2. Base64-encode → b64_data
  ├── 3. Call tools/vision.py → call_vision_model(b64_data, prompt)
  │         │
  │         └── openai.OpenAI().chat.completions.create(
  │               model="gpt-4o",
  │               messages=[{
  │                 "role": "user",
  │                 "content": [
  │                   {"type": "image_url",
  │                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
  │                   {"type": "text", "text": VISION_PROMPT}
  │                 ]
  │               }]
  │             )
  │
  ├── 4. Parse response.choices[0].message.content
  │       → model_validate_json() → VisionFindings (Pydantic)
  │
  ├── 5. Compute consistency_score from:
  │       - VisionFindings.damage_type vs. intake_result["damage_description"]
  │       - VisionFindings.severity vs. repair_estimate expected range
  │       - VisionFindings.affected_components vs. customer_statement keywords
  │
  └── 6. Set image_claim_mismatch = (consistency_score < mismatch_threshold)
```

**Error handling:** Any exception in steps 1–4 sets `confidence = 0.0`, `image_claim_mismatch = True`, and adds the error string to `warnings`. The pipeline continues to routing (which will escalate).

---

## 6. Input and Output Schemas

All models live in `state.py`. Every agent output model inherits the four base fields.

### Base output mixin (all agent outputs include these)

```python
class AgentOutputBase(BaseModel):
    confidence: float           # 0.0–1.0
    evidence: list[str]         # facts that drove the decision (bullet strings)
    reasoning_summary: str      # one paragraph for UI / audit display
    warnings: list[str]         # non-fatal issues; empty list if none
```

---

### VisionFindings (internal, not stored in ClaimState directly)

```python
class VisionFindings(BaseModel):
    damage_type: str            # collision | scrape | dent | glass | fire |
                                # flood | vandalism | total_loss | none_visible
    severity: str               # none | minor | moderate | severe | total_loss
    affected_components: list[str]
    estimated_repair_scope: str # cosmetic | panel_repair | structural | total_loss
    confidence: float
    observations: list[str]     # max 5 specific visual evidence strings
```

---

### IntakeResult

```python
class IntakeResult(AgentOutputBase):
    claim_type: str             # "collision" | "glass" | "vandalism" | "other"
    incident_date: str          # ISO-8601 or "unknown"
    damage_description: str     # normalised from customer statement
    prior_claims_count: int
    days_since_incident: int    # -1 if unknown
    completeness_flags: list[str]  # list of missing or weak fields
```

---

### DamageEvidenceResult

```python
class DamageEvidenceResult(AgentOutputBase):
    damage_type: str
    severity: str
    affected_components: list[str]
    estimated_repair_scope: str
    consistency_score: float    # 0.0–1.0
    image_claim_mismatch: bool
    vision_observations: list[str]
```

---

### PolicyResult

```python
class PolicyResult(AgentOutputBase):
    policy_id: str
    policy_valid: bool
    coverage_decision: str      # "covered" | "excluded" | "partial"
    applicable_coverage_type: str   # "collision" | "comprehensive" | "none"
    clause_cited: str           # exact rule key from policy document
    deductible: float
    coverage_ratio: float       # e.g. 0.80
    coverage_cap: float
    exclusion_reasons: list[str]  # empty if covered
```

---

### RiskResult

```python
class RiskResult(AgentOutputBase):
    risk_score: float           # 0–100
    risk_category: str          # "low" | "medium" | "high" | "critical"
    contributing_factors: list[dict]  # [{"factor": str, "contribution": float}]
```

---

### PayoutResult

```python
class PayoutResult(AgentOutputBase):
    repair_estimate: float
    coverage_ratio: float
    deductible: float
    payout_amount: float        # = repair_estimate × coverage_ratio − deductible, clamped [0, cap]
    coverage_cap: float
    auto_approve_eligible: bool
```

---

## 7. Shared ClaimState Design

Defined as a `TypedDict` in `state.py`. LangGraph merges partial dicts from each node into this state — nodes return only the fields they update.

```python
class ClaimState(TypedDict):
    # ── INPUTS ──────────────────────────────────────────────────────────
    claim_id: str
    claim_form: dict
    customer_statement: str
    damage_image_path: str
    repair_estimate: float
    policy_document_path: str
    claim_history: list[dict]        # [{"date": str, "type": str, "amount": float}]

    # ── AGENT OUTPUTS (stored as .model_dump()) ──────────────────────────
    intake_result: dict | None
    damage_evidence_result: dict | None
    policy_result: dict | None
    risk_result: dict | None
    payout_result: dict | None

    # ── ROUTING FLAGS ────────────────────────────────────────────────────
    intake_complete: bool
    image_claim_mismatch: bool
    policy_valid: bool
    coverage_decision: str           # "covered" | "excluded" | "partial"
    risk_score: float
    risk_category: str               # "low" | "medium" | "high" | "critical"
    requires_human_approval: bool
    escalation_reason: str           # first triggered condition (shown in UI)
    escalation_flags: list[str]      # all triggered conditions

    # ── HUMAN REVIEW ─────────────────────────────────────────────────────
    human_approval_granted: bool | None
    adjuster_note: str | None

    # ── AUDIT ────────────────────────────────────────────────────────────
    audit_trail: list[dict]
```

**Rules:**
- Agents never read `reasoning_summary` from another agent's output dict
- Agents read only typed routing flags and typed fields from prior output dicts
- All agent output models are stored via `.model_dump()` — reconstructed with `Model.model_validate(state["x_result"])` when needed

---

## 8. LangGraph Workflow

### Graph definition (`graph.py`)

```
StateGraph(ClaimState)
  │
  ├── node: "intake"           → intake_node()
  ├── node: "damage_evidence"  → damage_evidence_node()
  ├── node: "policy"           → policy_node()
  ├── node: "risk"             → risk_node()
  ├── node: "payout"           → payout_node()
  ├── node: "router"           → router_node()
  ├── node: "human_review"     → human_review_node()
  └── node: "auto_approve"     → auto_approve_node()

Edges:
  START            → "intake"
  "intake"         → "damage_evidence"
  "damage_evidence"→ "policy"
  "policy"         → "risk"
  "risk"           → "payout"
  "payout"         → "router"
  "router"         → conditional_edge(route_decision)
                       "human_review"  if requires_human_approval
                       "auto_approve"  otherwise
  "human_review"   → END
  "auto_approve"   → END
```

### Node return pattern

Every node returns a **partial dict** — only the fields that node updates:

```python
def intake_node(state: ClaimState) -> dict:
    result = IntakeAgent().run(state)
    return {
        "intake_result": result.model_dump(),
        "intake_complete": True,
        "audit_trail": state["audit_trail"] + [build_audit_entry("IntakeAgent", result)]
    }
```

---

## 9. Routing Conditions

The `router_node` evaluates all conditions after PayoutAgent. Sets `requires_human_approval = True` and records `escalation_reason` (first match) and `escalation_flags` (all matches).

| Priority | Condition | Check |
|---|---|---|
| 1 | Image-claim mismatch | `state["image_claim_mismatch"] is True` |
| 2 | Any agent confidence below floor | any `state["x_result"]["confidence"] < 0.75` |
| 3 | Risk score at or above threshold | `state["risk_score"] >= 70` |
| 4 | Policy invalid or missing | `state["policy_valid"] is False` |
| 5 | Coverage excluded | `state["coverage_decision"] == "excluded"` |
| 6 | Payout exceeds auto-approve limit | `state["payout_result"]["payout_amount"] > 5000` |
| 7 | Risk category critical | `state["risk_category"] == "critical"` |

If none match: `requires_human_approval = False` → route to `auto_approve`.

**Config keys** for all thresholds live in `config/risk_thresholds.yaml` — never hardcode them in agent files.

---

## 10. Human Approval Rules

### HumanReviewNode

1. Appends `{"event": "human_review_requested", "escalation_reason": ..., "escalation_flags": [...]}` to `audit_trail`
2. Returns partial state — Gradio reads `requires_human_approval = True` and renders the review panel
3. Adjuster sees: full claim summary, all agent `reasoning_summary` fields, `escalation_reason`, all `warnings`
4. Adjuster must enter a note (minimum 10 characters) before Approve/Reject is enabled
5. On submit: sets `human_approval_granted`, `adjuster_note`
6. Appends `{"event": "adjuster_decision", "granted": bool, "note": str, "timestamp": str}` to `audit_trail`

### AutoApproveNode

1. Sets `human_approval_granted = True`
2. Appends `{"event": "auto_approved", "timestamp": str}` to `audit_trail`

**Hard rule:** Every path through the graph writes to `audit_trail` before reaching END.

---

## 11. Audit Trail Requirements

### Entry schema (every `write_audit_entry()` call produces this)

```python
{
    "timestamp": "2026-06-16T14:32:01Z",   # ISO-8601 UTC
    "agent": "DamageEvidenceAgent",         # or "router" | "auto_approve" | "adjuster"
    "event": "agent_complete",              # agent_complete | routing | human_review_requested
                                            # adjuster_decision | auto_approved | error
    "confidence": 0.82,                     # from agent output; null for non-agent events
    "evidence": ["...", "..."],             # from agent output; empty list for non-agent events
    "warnings": [],                         # from agent output; empty list for non-agent events
    "routing_decision": "continue",         # continue | human_review | auto_approve
    "output_summary": {}                    # agent output model as dict (full)
}
```

### Storage

- Appended in-memory to `ClaimState["audit_trail"]` throughout the run
- Written to `_bmad-output/audit_logs/{claim_id}.jsonl` at pipeline end (one JSON object per line)
- Displayed in Gradio UI as a collapsible expandable section

### Rules

- Every agent node calls `write_audit_entry()` as its last action before returning
- Router, AutoApprove, and HumanReview nodes also write entries
- No agent may skip the audit write even on error — the catch block writes an error entry

---

## 12. Evaluation Cases

Full synthetic claim dataset: `_bmad-output/planning-artifacts/synthetic-claims.md` (20 claims).

Minimum pytest test cases required:

| ID | Scenario | Expected route | Primary assertion |
|---|---|---|---|
| E-01 | Clean rear-end, $850, no history (CLM-001) | Auto-approve | `requires_human_approval = False` |
| E-02 | Zero-damage image, $4,500 claim (CLM-017) | Human review | `image_claim_mismatch = True`, `consistency_score < 0.10` |
| E-03 | Total loss, 3 prior claims, 45 days late (CLM-020) | Human review | `risk_score >= 85`, `risk_category = "critical"` |
| E-04 | Payout = $5,001 (one above limit) | Human review | `payout_amount > auto_approve_limit` |
| E-05 | Payout = $4,999 (one below limit), all clear | Auto-approve | `requires_human_approval = False` |
| E-06 | Policy file missing | Human review | `policy_valid = False` |
| E-07 | Damage type explicitly excluded by policy | Human review | `coverage_decision = "excluded"` |
| E-08 | Vision API raises exception | Human review | `confidence = 0.0`, error in `warnings` |
| E-09 | Risk score = 70 (at threshold — boundary) | Human review | Escalates |
| E-10 | Risk score = 69 (one below threshold — boundary) | No risk escalation | Does not escalate on risk alone |
| E-11 | Adjuster approves escalated claim | `human_approval_granted = True` | Audit trail has adjuster note + timestamp |
| E-12 | Adjuster rejects escalated claim | `human_approval_granted = False` | Audit trail records rejection |
| E-13 | Severe damage image, matching story, $4,200 (CLM-012) | Auto-approve | Severity alone does not escalate |
| E-14 | Slight mismatch — severity understated (CLM-013) | Human review | `consistency_score` in range 0.40–0.59 |

### Test file locations

```
tests/
├── test_routing.py       # E-04, E-05, E-09, E-10 — edge conditions on all 7 routing flags
├── test_risk.py          # E-03, E-09, E-10 — boundary values for every risk factor weight
├── test_vision.py        # E-02, E-08, E-14 — mismatch detection and API error handling
├── test_policy.py        # E-06, E-07 — coverage decisions and invalid policy handling
└── test_payout.py        # E-04, E-05, E-11, E-12, E-13 — calculation and approval gating
```

---

## 13. Acceptance Criteria

| ID | Criterion | Verified by |
|---|---|---|
| AC-01 | Clean claim auto-approves in under 30 seconds | Manual demo timing |
| AC-02 | `consistency_score < 0.60` always routes to human review | `test_routing.py` |
| AC-03 | `risk_score >= 70` always routes to human review, regardless of payout | `test_routing.py` |
| AC-04 | PolicyAgent never cites a clause absent from the loaded policy document | `test_policy.py` + code review |
| AC-05 | Every agent output contains `confidence`, `evidence`, `reasoning_summary`, `warnings` | `test_routing.py` (schema check) |
| AC-06 | Every decision appears in `audit_trail` with agent name, timestamp, routing decision | `test_routing.py` (audit check) |
| AC-07 | Missing or expired policy sets `policy_valid = False` and escalates | `test_policy.py` |
| AC-08 | Payout = `repair_estimate × coverage_ratio − deductible` to two decimal places | `test_payout.py` |
| AC-09 | Gradio UI displays all 8 system outputs on a single page | Manual UI check |
| AC-10 | All 7 routing conditions are covered by pytest with boundary values | `test_routing.py` |
| AC-11 | Human review requires a written note (min 10 chars) before Approve/Reject activates | Manual UI check |
| AC-12 | `adjuster_note` and decision timestamp appear in `audit_trail` | `test_routing.py` (E-11, E-12) |

---

## 14. Out of Scope

The following will **not** be built. Do not add stubs, placeholders, or config for any of these.

| Item | Reason |
|---|---|
| Batch claim processing | Demo is single-claim per session |
| Customer-facing portal | Internal adjuster tool only |
| Real-time fraud database integration | Complexity beyond demo scope |
| PDF parsing of policy documents | Policy rules are pre-parsed to JSON for demo |
| Non-auto claim types (home, health, life) | Auto only |
| Multi-vehicle claims | Single vehicle per claim |
| Policy database or external policy API | Local file only |
| Email or notification on decision | UI-only output |
| User authentication or session management | No auth for demo |
| Async / parallel agent execution | Sequential pipeline; no safe parallelism in this graph |
| Claim editing or resubmission | Submit-once per session |
| LLM fallback or model switching | `gpt-4o` only for vision |
