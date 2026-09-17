# CLAUDE.md — Insurance Claims AI Agent Team

This file provides guidance to Claude Code when working in this repository.

## Project Purpose

A production-style multi-agent insurance claims processing system built with LangGraph, Pydantic, Gradio, and Python. The goal is a clean, demo-ready system that shows how real-world agentic workflows handle structured data, vision, policy logic, risk scoring, and human-in-the-loop approval.

Claude Code is not the agent framework. Claude Code helps plan, scaffold, code, test, debug, and refactor the system.

## Environment

```bash
uv venv && uv sync          # create env and install deps
uv add <package>            # add a dependency
uv run python main.py       # run the full pipeline
uv run pytest               # run all tests
uv run pytest tests/test_routing.py::test_name  # run a single test
```

Python 3.11+. Use `uv`; fall back to `pip install -r requirements.txt` only if `uv` is unavailable.

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| Structured data | Pydantic v2 |
| Vision model | OpenAI `gpt-4o` (via `openai` SDK) |
| UI | Gradio |
| Config | `python-dotenv` + `.env` |
| Testing | pytest |

## Project Layout

```
insurance-claims-agent-team/
├── main.py                  # entry point: loads .env, builds graph, runs demo
├── graph.py                 # StateGraph wiring — imports nodes, defines edges
├── state.py                 # ClaimState (TypedDict) and all Pydantic I/O models
├── agents/
│   ├── intake.py            # IntakeAgent
│   ├── damage_evidence.py   # DamageEvidenceAgent (vision model)
│   ├── policy.py            # PolicyAgent
│   ├── risk.py              # RiskAgent
│   └── payout.py            # PayoutAgent
├── tools/
│   ├── audit_log.py         # write_audit_entry() — every decision logged here
│   ├── policy_loader.py     # loads policy documents / rule tables
│   └── vision.py            # call_vision_model() wrapper
├── config/
│   └── risk_thresholds.yaml # risk score cutoffs, payout limits, confidence floors
├── data/
│   ├── images/              # sample damage images for demo
│   └── policies/            # sample policy PDFs or JSON rule tables
├── ui/
│   └── app.py               # Gradio interface
└── tests/
    ├── test_routing.py      # graph edge routing logic
    ├── test_risk.py         # risk threshold and score tests
    ├── test_vision.py       # image-claim mismatch detection
    ├── test_policy.py       # coverage decision correctness
    └── test_payout.py       # payout calculation and approval gating
```

## Architecture Rules (non-negotiable)

### 1. LangGraph for orchestration
Use `StateGraph(ClaimState)` in `graph.py`. Nodes are the five agents plus a `human_review` node. Edges are conditional based on flags set in state.

### 2. Pydantic for all structured data
No agent may pass a plain string or dict to another agent. Every input and output is a typed Pydantic model defined in `state.py`. `ClaimState` (TypedDict) holds the full pipeline state; Pydantic models live inside it as fields.

### 3. Vision model for damage evidence
`DamageEvidenceAgent` calls the vision model via `tools/vision.py`. Pass the base64-encoded image and the damage description from the claim form. Extract structured findings (damage type, severity, affected parts, estimated repair scope) as a Pydantic model — never as raw LLM text.

### 4. No unstructured text between agents
If an agent produces a natural-language summary, it must be stored in a `reasoning_summary` field of a Pydantic model, not used as the primary data passed forward. Downstream agents read typed fields, not summaries.

### 5. Every agent output must include these four fields
```python
class AgentOutput(BaseModel):
    confidence: float          # 0.0–1.0
    evidence: list[str]        # bullet list of facts that drove the decision
    reasoning_summary: str     # one-paragraph explanation for audit / UI
    warnings: list[str]        # non-fatal issues (missing docs, anomalies, etc.)
```
Embed this as a base class or mixin for every agent's output model.

### 6. Audit log on every decision
Call `tools/audit_log.write_audit_entry()` at the end of every agent node. Log: agent name, timestamp, input summary, output model (JSON), routing decision taken. Write to a structured JSONL file or append to `ClaimState.audit_trail`.

### 7. Human approval routing
Route to `human_review` node when **any** of these are true:
- Risk score >= threshold defined in `config/risk_thresholds.yaml`
- Confidence of any agent output < floor defined in config
- Image-claim mismatch detected by `DamageEvidenceAgent`
- Required policy document is missing
- Policy is invalid or expired
- Payout exceeds the auto-approve limit in config

The `human_review` node sets `ClaimState.requires_human_approval = True` and halts the graph (returns to Gradio for human input).

### 8. No invented policy rules
`PolicyAgent` must load coverage rules from `data/policies/` (PDFs or JSON tables) or from `config/risk_thresholds.yaml`. It must not hallucinate policy terms. If a policy document is missing or ambiguous, set `warnings` and flag for human review.

### 9. Test coverage required
Write tests before or alongside each agent. Minimum required:
- Routing tests: correct edges taken for each combination of flags
- Risk threshold tests: boundary values (at, above, below each cutoff)
- Vision/claim mismatch tests: matching vs mismatching image + description pairs
- Policy decision tests: covered, excluded, and edge-case scenarios
- Payout tests: calculation correctness and auto-approve vs escalate gating

### 10. Demo simplicity
The full pipeline must run end-to-end in `uv run python main.py` with the sample data in `data/`. The Gradio UI must show all outputs on a single page. Keep each agent file under ~150 lines.

## The Five Agents

### IntakeAgent (`agents/intake.py`)
- **Input**: raw claim form (dict), customer statement (str), claim history (list)
- **Output**: `IntakeResult` — normalized claim fields, completeness flags, extracted claim type, prior claims count
- **Routing trigger**: sets `ClaimState.intake_complete = True`

### DamageEvidenceAgent (`agents/damage_evidence.py`)
- **Input**: damage image (base64 or path), repair estimate, claim description
- **Output**: `DamageEvidenceResult` — detected damage types, severity, affected components, consistency score vs claim description, mismatch flag
- **Vision call**: `tools/vision.py` wraps the `openai` SDK multimodal call; returns a structured dict parsed into Pydantic
- **Routing trigger**: sets `ClaimState.image_claim_mismatch` if consistency score < threshold

### PolicyAgent (`agents/policy.py`)
- **Input**: `IntakeResult`, policy document (loaded by `tools/policy_loader.py`)
- **Output**: `PolicyResult` — coverage decision (covered/excluded/partial), applicable clauses, exclusion reasons
- **Constraint**: only uses facts from the loaded policy document; no hallucinated rules
- **Routing trigger**: sets `ClaimState.policy_valid`, `ClaimState.coverage_decision`

### RiskAgent (`agents/risk.py`)
- **Input**: `IntakeResult`, `DamageEvidenceResult`, `PolicyResult`, claim history
- **Output**: `RiskResult` — risk score (0–100), risk category (low/medium/high/critical), contributing factors
- **Routing trigger**: sets `ClaimState.risk_score`, triggers human review if score >= config threshold

### PayoutAgent (`agents/payout.py`)
- **Input**: `DamageEvidenceResult`, `PolicyResult`, `RiskResult`, repair estimate
- **Output**: `PayoutResult` — recommended payout amount, deductible applied, coverage ratio, auto-approve eligibility
- **Routing trigger**: sets `ClaimState.payout_amount`, triggers human review if amount > config limit

## ClaimState Shape (sketch)

```python
class ClaimState(TypedDict):
    # inputs
    claim_form: dict
    customer_statement: str
    damage_image_path: str
    repair_estimate: float
    policy_document: str
    claim_history: list[dict]

    # agent outputs (Pydantic models stored as dicts via .model_dump())
    intake_result: dict | None
    damage_evidence_result: dict | None
    policy_result: dict | None
    risk_result: dict | None
    payout_result: dict | None

    # routing flags
    intake_complete: bool
    image_claim_mismatch: bool
    policy_valid: bool
    coverage_decision: str          # "covered" | "excluded" | "partial"
    risk_score: float
    requires_human_approval: bool
    human_approval_granted: bool | None

    # audit
    audit_trail: list[dict]
```

## System Inputs and Outputs

**Inputs** (provided via Gradio or `main.py` demo fixtures):
- Claim form (structured fields)
- Customer statement (free text — IntakeAgent normalizes it)
- Damage image (JPEG/PNG from `data/images/`)
- Repair estimate (dollar amount)
- Policy document (PDF or JSON from `data/policies/`)
- Claim history (list of prior claims)

**Outputs** (displayed in Gradio UI and written to audit log):
- Claim summary
- Image damage assessment
- Image-claim consistency result
- Coverage decision
- Risk score
- Payout recommendation
- Human approval status
- Full audit trail (JSONL)

## Config

`config/risk_thresholds.yaml` controls all numeric cutoffs. Never hardcode thresholds in agent files.

```yaml
risk:
  human_review_threshold: 70       # risk score >= this → escalate
  confidence_floor: 0.75           # any agent confidence < this → escalate

payout:
  auto_approve_limit: 5000         # payout > this → human review

vision:
  mismatch_threshold: 0.60         # consistency score < this → flag mismatch
```

## Key Conventions

### `load_dotenv()` before framework imports
`main.py` calls `load_dotenv()` as the first statement before importing LangGraph or `anthropic`. The `anthropic` client reads `ANTHROPIC_API_KEY` at import time on some versions.

### Agent nodes return partial state dicts
LangGraph merges partial dicts into `ClaimState`. Never return the full state from a node — only the fields that agent updates.

### Vision model usage
```python
# tools/vision.py pattern
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}},
            {"type": "text", "text": prompt}
        ]
    }]
)
```
Parse `response.choices[0].message.content` into a Pydantic model using `model_validate_json()`. Never forward raw text.

### Audit log entry shape
```python
{
    "timestamp": "ISO-8601",
    "agent": "DamageEvidenceAgent",
    "input_summary": {...},
    "output": {...},          # agent output model as dict
    "routing_decision": "continue | human_review",
    "warnings": [...]
}
```
