# ClaimSight — Insurance Claims AI Agent Team

A production-style five-agent LangGraph pipeline that processes auto insurance claims end-to-end. Structured claim inputs flow through five sequential agents and produce a decision packet: either auto-approving clean claims or escalating ambiguous, mismatched, or high-risk claims to a human adjuster.

Built to show what an agentic workflow looks like when it has to survive contact with a real decision process: every agent-to-agent handoff is a typed Pydantic model (never a raw string), every decision is written to an audit trail, and seven independent escalation checks gate anything ambiguous into human review instead of auto-approving it.

## Why this project

Most agent demos stop at "the LLM produced a plausible-looking answer." Insurance claims can't work that way — a wrong auto-approval is a real payout. This project explores the patterns that make an LLM pipeline safe to put in front of that kind of decision:

- **Structured handoffs, not prose.** No agent passes free text to the next one; everything downstream reads typed fields (see `core/claim_state.py`).
- **Confidence and evidence on every output.** Each agent reports a `confidence` score and the `evidence` that drove its decision, so low-confidence outputs can be caught mechanically instead of trusted blindly.
- **Human-in-the-loop by design, not by exception.** The router (`core/routing.py`) checks seven independent conditions — risk score, confidence floor, image/claim mismatch, missing or invalid policy, coverage exclusion, payout cap, risk category — and routes to a human reviewer the moment any one of them trips.
- **A vision model as a fact-checker.** `DamageEvidenceAgent` scores how consistent the submitted photo is with the claimant's story, and a low score is itself an escalation trigger — a lightweight fraud-detection signal.

See [`_bmad-output/planning-artifacts/prd.md`](_bmad-output/planning-artifacts/prd.md) for the product brief this was built against.

## Quick Start

```bash
cp .env.example .env          # add OPENAI_API_KEY
uv venv && uv sync
uv run python main.py         # run the demo pipeline (three scenarios)
uv run python ui/app.py       # launch the Gradio UI
uv run pytest evals/          # run all evaluation tests
```

## Architecture

```
START
  └─► IntakeAgent          validate & normalise claim inputs
        └─► DamageEvidenceAgent   inspect image with gpt-4o; compute consistency score
              └─► PolicyAgent     load policy JSON; determine coverage decision
                    └─► RiskAgent     score 0–100 from five weighted factors
                          └─► PayoutAgent  calculate payout; set auto-approve eligibility
                                └─► Router      evaluate 7 escalation conditions
                                      ├─► AutoApproveNode  → END
                                      └─► HumanReviewNode  → END
```

## Project Layout

```
agents/                  one file per agent (business logic)
core/
  claim_state.py         ClaimState TypedDict + all Pydantic I/O models
  graph.py               LangGraph StateGraph wiring
  routing.py             router node + 7 escalation condition checks
  audit.py               write_audit_entry() + flush_audit_log()
  policy_rules.py        PolicyRules model + JSON policy loader
  data_loader.py         demo fixture loaders + build_initial_state()
  vision_client.py       OpenAI gpt-4o multimodal wrapper
data/
  images/                20 synthetic claim photos
  policies/              standard_auto_policy.json
  claims.json            20 synthetic claims
  customers.json         customer records
  claim_history.json     prior claims per customer
  repair_estimates.json  repair estimate per claim
  expected_outcomes.json expected pipeline results (used by evals)
evals/                   pytest test suite
  test_routes.py         routing edge conditions + audit checks
  test_policy.py         coverage decisions + invalid policy handling
  test_risk.py           risk score boundary values per factor
  test_payout.py         payout calculation + auto-approve gating
  test_image_claim_match.py   vision mismatch detection + API error handling
ui/
  app.py                 Gradio single-page interface
scripts/
  create_image_observations.py   generate image_observations.json from raw images
config/
  risk_thresholds.yaml   all numeric cutoffs (never hardcoded in agents)
```

## Demo Scenarios

| Scenario | Claim | Expected result |
|---|---|---|
| A — Clean claim | Minor rear bumper scrape, $850, no history | Auto-approve, payout $350 |
| B — Image mismatch | Zero-damage image, $4,500 front-collision story | Human review (consistency ≈ 0.05) |
| C — High risk | Total loss, 3 prior claims, 45 days late, $14,500 | Human review (risk score ≈ 88) |

## Escalation Conditions (router checks in priority order)

1. Image-claim mismatch
2. Any agent confidence < 0.75
3. Risk score >= 70
4. Policy invalid or missing
5. Coverage excluded
6. Payout > $5,000
7. Risk category = critical

## Config

All numeric thresholds live in `config/risk_thresholds.yaml`. Never hardcode them in agent files.

## Testing

72/72 eval tests passing across routing, policy, risk, payout, and image-claim matching (`evals/eval_summary.json`).

Writing the risk-boundary tests caught a real bug: `RiskAgent` was slicing ISO date strings incorrectly, which silently zeroed out the late-reporting and policy-tenure risk factors — the pipeline ran without errors, but risk scores were quietly wrong. Fixed with `date.fromisoformat(value[:10])`. A good reminder that "no exceptions raised" isn't the same as "correct."

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Structured data | Pydantic v2 |
| Vision model | OpenAI `gpt-4o` |
| UI | Gradio |
| Config | python-dotenv + YAML |
| Testing | pytest |
