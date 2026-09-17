---
title: ClaimSight PRD
status: draft
created: 2026-06-16
updated: 2026-06-16
---

# PRD: ClaimSight — Insurance Claims AI Agent Team

---

## 1. User Stories

**US-01 — Submit a claim**
As an adjuster, I upload a damage image, paste a customer statement, enter a repair estimate, upload the policy PDF, and submit claim history so the pipeline processes everything without me switching tools.

**US-02 — View damage assessment**
As an adjuster, I see what damage `gpt-4o` detected in the image, the severity, affected parts, and a consistency score against the customer story so I understand immediately whether the image supports the claim.

**US-03 — View coverage decision**
As an adjuster, I see which policy clause applies, whether the claim is covered / partial / excluded, and the exact rule that drove the decision so I can explain it to the customer.

**US-04 — View risk score**
As an adjuster, I see a 0–100 risk score with the top 3 contributing factors so I understand why a claim was flagged or cleared.

**US-05 — View payout recommendation**
As an adjuster, I see the recommended payout amount with deductible applied and coverage ratio so I have a number to approve or override.

**US-06 — Approve or reject an escalated claim**
As an adjuster, I see an Approve / Reject button and a text box for my written reasoning whenever the system escalates a claim. My decision and note are written to the audit trail.

**US-07 — Review the audit trail**
As an adjuster, I see a timestamped record of every agent decision, evidence list, confidence score, and routing choice so the claim is fully explainable to a supervisor or regulator.

**US-08 — See confidence warnings**
As an adjuster, I am notified whenever any agent's confidence falls below 0.75 so I know exactly where the system is uncertain and why.

---

## 2. Agent Responsibilities

### IntakeAgent
- Validate and normalize all claim inputs (form fields, customer statement, prior claims)
- Extract: claim type, incident date, reported damage description, prior claims count, policy ID
- Flag: missing required fields, vague or inconsistent statement, high prior-claim frequency (≥3 in 24 months)
- Output model: `IntakeResult`

### DamageEvidenceAgent
- Encode damage image to base64 and call OpenAI `gpt-4o` via `tools/vision.py`
- Extract structured findings: damage type, affected components, severity (minor / moderate / severe / total-loss), visible repair scope
- Compute `consistency_score` (0.0–1.0) comparing vision findings against customer statement and repair estimate dollar range
- Set `image_claim_mismatch = True` when `consistency_score < 0.60`
- Output model: `DamageEvidenceResult`

### PolicyAgent
- Parse uploaded policy PDF using `tools/policy_loader.py` → extract coverage rule table
- Determine: `coverage_decision` = covered / excluded / partial
- Cite the specific clause that applies
- Flag: missing document, expired policy, exclusion clauses triggered
- Must not invent or infer policy rules not present in the document
- Output model: `PolicyResult`

### RiskAgent
- Score 0–100 from weighted factors (weights defined in `config/risk_thresholds.yaml`):
  - `consistency_score` from DamageEvidenceAgent (highest weight)
  - Prior claims frequency (≥3 in 24 months = high signal)
  - Repair estimate vs. expected range for detected damage severity
  - Days since incident (late reporting > 30 days = risk signal)
  - Policy tenure (< 6 months = elevated risk)
- Assign `risk_category`: low (0–40) / medium (41–69) / high (70–84) / critical (85–100)
- Output model: `RiskResult` with top 3 factors in `evidence`

### PayoutAgent
- Calculate: `repair_estimate × coverage_ratio − deductible`
- Apply coverage cap from policy
- Set `auto_approve_eligible = True` when `payout_amount ≤ $5,000` and no escalation flags
- Output model: `PayoutResult`

---

## 3. Shared ClaimState

```python
class ClaimState(TypedDict):
    # --- INPUTS ---
    claim_id: str
    claim_form: dict                    # adjuster-entered fields
    customer_statement: str
    damage_image_path: str
    repair_estimate: float
    policy_document_path: str          # uploaded PDF path
    claim_history: list[dict]          # prior claims list

    # --- AGENT OUTPUTS (stored as .model_dump() dicts) ---
    intake_result: dict | None
    damage_evidence_result: dict | None
    policy_result: dict | None
    risk_result: dict | None
    payout_result: dict | None

    # --- ROUTING FLAGS ---
    intake_complete: bool
    image_claim_mismatch: bool
    policy_valid: bool
    coverage_decision: str             # "covered" | "excluded" | "partial"
    risk_score: float
    risk_category: str                 # "low" | "medium" | "high" | "critical"
    requires_human_approval: bool
    escalation_reason: str             # first triggered condition, for UI display
    human_approval_granted: bool | None
    adjuster_note: str | None          # written by adjuster at review

    # --- AUDIT ---
    audit_trail: list[dict]            # appended by every agent node
```

**Key constraint:** Agents read only typed fields from other agents' output dicts — never `reasoning_summary`.

**Every agent output model includes these four base fields:**
```python
confidence: float          # 0.0–1.0
evidence: list[str]        # facts that drove the decision
reasoning_summary: str     # one-paragraph explanation for UI / audit
warnings: list[str]        # non-fatal issues
```

---

## 4. LangGraph Workflow

```
START
  │
  ▼
[IntakeAgent]
  │  → sets intake_result, intake_complete
  ▼
[DamageEvidenceAgent]
  │  → sets damage_evidence_result, consistency_score, image_claim_mismatch
  ▼
[PolicyAgent]
  │  → sets policy_result, policy_valid, coverage_decision
  ▼
[RiskAgent]
  │  → sets risk_result, risk_score, risk_category
  ▼
[PayoutAgent]
  │  → sets payout_result, payout_amount, auto_approve_eligible
  ▼
[RouterNode]  ← conditional_edge evaluates all escalation flags
  │
  ├── requires_human_approval = True ──► [HumanReviewNode] ──► END
  │                                            (Gradio pauses, adjuster acts)
  └── requires_human_approval = False ─► [AutoApproveNode] ──► END
```

**Graph type:** `StateGraph(ClaimState)`, compiled with `graph.compile()`

**Sequential, not parallel:** PolicyAgent needs `policy_id` from IntakeResult. RiskAgent needs outputs from all three prior agents. No safe parallelism.

**Nodes return partial dicts** that LangGraph merges into ClaimState. Never return the full state.

---

## 5. Routing Logic

The `router` function evaluates all conditions after PayoutAgent. Sets `requires_human_approval = True` and records `escalation_reason` on the first match.

| Priority | Condition | Source | Config key |
|---|---|---|---|
| 1 | Image-claim mismatch | `image_claim_mismatch = True` | `vision.mismatch_threshold` |
| 2 | Agent confidence below floor | any `*.confidence < 0.75` | `risk.confidence_floor` |
| 3 | Risk score at or above threshold | `risk_score ≥ 70` | `risk.human_review_threshold` |
| 4 | Policy invalid or missing | `policy_valid = False` | — |
| 5 | Coverage excluded | `coverage_decision = "excluded"` | — |
| 6 | Payout exceeds auto-approve limit | `payout_amount > 5000` | `payout.auto_approve_limit` |
| 7 | Risk category critical | `risk_category = "critical"` | — |

All conditions are checked together. Priority determines which reason is surfaced in the UI; remaining matched conditions are listed in `warnings`.

---

## 6. Human Approval Rules

**Trigger:** `requires_human_approval = True`

**HumanReviewNode behaviour:**
1. Writes a `human_review_requested` entry to `audit_trail`
2. Surfaces to Gradio: full claim summary, all agent outputs, `escalation_reason`, full `warnings` list
3. Adjuster reads, then clicks **Approve** or **Reject** and enters a written note (required, min 10 chars)
4. Decision + note + timestamp written to `audit_trail`
5. Sets `human_approval_granted = True / False`, `adjuster_note`
6. Pipeline ends

**AutoApproveNode behaviour:**
1. Sets `human_approval_granted = True` automatically
2. Writes `auto_approved` entry to `audit_trail` with all agent summaries
3. Pipeline ends

**Hard rule:** Both paths always call `write_audit_entry()`. There is no path through the graph that skips the audit write.

---

## 7. Failure Modes

| Failure | Agent | Handling |
|---|---|---|
| Vision model API error or timeout | DamageEvidenceAgent | `confidence = 0.0`, `warnings` = error message, escalate |
| Policy PDF unreadable or missing | PolicyAgent | `policy_valid = False`, escalate |
| Policy document ambiguous / incomplete rules | PolicyAgent | `coverage_decision = "partial"`, low confidence, escalate |
| Customer statement too vague to parse | IntakeAgent | Flag in `warnings`, reduce `confidence`, continue |
| Repair estimate far outside expected range for detected severity | DamageEvidenceAgent | Reduce `consistency_score`, flag in `warnings` |
| Risk model produces out-of-range score | RiskAgent | Clamp to 100, `confidence = 0.0`, escalate |
| Claim history missing | IntakeAgent | Continue with `prior_claims_count = 0`, add warning |
| Any unhandled exception in a node | LangGraph catch | `requires_human_approval = True`, write error to `audit_trail` |
| Image is blank, corrupted, or missing | DamageEvidenceAgent | `confidence = 0.0`, `image_claim_mismatch = True`, escalate |

**No silent failures.** Every failure adds a `warnings` entry and triggers escalation.

---

## 8. Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-01 | A clean claim (matching image, covered policy, risk < 70, payout ≤ $5,000) auto-approves in under 30 seconds with a full audit trail |
| AC-02 | Any claim with `consistency_score < 0.60` always routes to human review |
| AC-03 | Any claim with `risk_score ≥ 70` always routes to human review, regardless of payout |
| AC-04 | PolicyAgent never cites a coverage clause not present in the uploaded policy document |
| AC-05 | Every agent output contains `confidence`, `evidence`, `reasoning_summary`, `warnings` |
| AC-06 | Every decision — including auto-approvals — appears in `audit_trail` with agent name, timestamp, and routing decision |
| AC-07 | A missing or expired policy document sets `policy_valid = False` and escalates |
| AC-08 | Payout calculation matches `repair_estimate × coverage_ratio − deductible` to two decimal places |
| AC-09 | Gradio UI displays all 8 system outputs on a single page |
| AC-10 | All routing logic is covered by pytest with boundary values at every threshold |
| AC-11 | Human review requires a written note before Approve / Reject is accepted |
| AC-12 | The `adjuster_note` and decision timestamp appear in the audit trail |

---

## 9. Evaluation Cases

See `synthetic-claims.md` for the full 20-claim dataset with expected outcomes.

Summary of required test scenarios for the eval suite:

| ID | Scenario | Expected route | Key assertion |
|---|---|---|---|
| E-01 | Clean rear-end, low estimate, clean history | Auto-approve | `requires_human_approval = False` |
| E-02 | No visible image damage, customer claims collision | Human review | `image_claim_mismatch = True`, `consistency_score < 0.30` |
| E-03 | Total loss, 3 prior claims, 45-day late report | Human review | `risk_score ≥ 85`, `risk_category = "critical"` |
| E-04 | Payout = $5,001 (one above limit) | Human review | `payout_amount > auto_approve_limit` |
| E-05 | Payout = $4,999 (one below limit), all clear | Auto-approve | `requires_human_approval = False` |
| E-06 | Policy PDF missing | Human review | `policy_valid = False` |
| E-07 | Damage type explicitly excluded by policy | Human review | `coverage_decision = "excluded"` |
| E-08 | Vision model API timeout | Human review | `confidence = 0.0`, error in `warnings` |
| E-09 | Risk score = 70 (at threshold) | Human review | Boundary value escalates |
| E-10 | Risk score = 69 (one below threshold) | Continue | Does not escalate on risk alone |
| E-11 | Adjuster approves escalated claim | `human_approval_granted = True` | Audit trail records note + timestamp |
| E-12 | Adjuster rejects escalated claim | `human_approval_granted = False` | Audit trail records rejection |
| E-13 | Severe damage image, matching story | Auto-approve if under limit | Damage severity alone does not escalate |
| E-14 | Slight mismatch (severity understated) | Human review | `consistency_score` between 0.45–0.59 |

---

## 10. Minimal YouTube Demo Scope

**Target runtime:** 15 minutes — 4 min intro, 9 min live demo (3 scenarios × 3 min each), 2 min wrap-up.

### Scenario A — Clean Claim (CLM-001)
- **Image:** `data/images/0005.jpg` — minor rear bumper scrape, silver car
- **Story:** "Someone bumped my car in a parking lot. There's a small dent and scrape on the rear bumper."
- **Estimate:** $850
- **Policy:** Collision coverage, $500 deductible
- **History:** No prior claims
- **Expected path:** Auto-approve. Payout = $350. Risk score ≈ 15.
- **What the audience sees:** All five agents fire, audit trail populates, green auto-approve badge appears. 30 seconds end-to-end.

### Scenario B — Image-Story Mismatch (CLM-017)
- **Image:** `data/images/F0022.jpg` — black Jeep Grand Cherokee with zero visible damage
- **Story:** "I was in a front-end collision at an intersection. The front bumper has significant structural damage."
- **Estimate:** $4,500
- **Policy:** Collision coverage, $500 deductible
- **History:** No prior claims
- **Expected path:** Human review. Vision model finds no damage. `consistency_score ≈ 0.05`. Escalation reason: image does not support claimed damage.
- **What the audience sees:** DamageEvidenceAgent flags the mismatch, routing snaps to human review, Approve/Reject panel appears with the vision findings displayed.

### Scenario C — High-Risk / High-Payout (CLM-020)
- **Image:** `data/images/S0016.jpg` — blue Smart car, total front-end destruction
- **Story:** "My car was completely wrecked in a collision. I need a full payout."
- **Estimate:** $14,500
- **Policy:** Collision coverage, $500 deductible
- **History:** 3 prior claims in 18 months; incident reported 45 days after it occurred
- **Expected path:** Human review. Risk score ≈ 88. `risk_category = "critical"`. Payout exceeds auto-approve limit. Multiple escalation triggers fire.
- **What the audience sees:** Multiple red warning badges — high risk score, late reporting, claim frequency, high payout. Adjuster reviews, types a note, clicks Reject. Audit trail records the decision.

### What is out of scope for the demo
- Batch claim processing
- Policy database or external lookup
- Customer portal or self-service flow
- Multi-vehicle or non-auto claim types
- Real-time fraud database integration
