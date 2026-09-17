---
title: ClaimSight — Insurance Claims AI Agent Team
status: draft
created: 2026-06-16
updated: 2026-06-16
---

# Product Brief: ClaimSight

## The Problem

Insurance adjusters review claims manually — reading customer statements, inspecting damage photos, cross-checking repair estimates against policy documents, and scoring risk by hand. The process is slow, inconsistent across adjusters, and leaves fraud signals buried in unstructured text and images no one has time to compare carefully.

## The Solution

A five-agent AI pipeline that processes all claim inputs in under 30 seconds and hands the adjuster a structured, evidence-backed decision packet. Low-risk, well-supported claims are auto-approved. Ambiguous, mismatched, high-risk, or high-payout claims are escalated with a clear explanation and an Approve / Reject button.

The system does not replace the adjuster. It eliminates the 80% of claims that are routine so adjusters spend their time on the 20% that actually need human judgment.

## Users

**Primary:** Auto insurance adjusters (internal tool). Adjusters submit claim materials through a Gradio UI and receive a structured output. For escalated claims they approve or reject with a written note.

## Scope

- Auto / vehicle damage claims only
- Single claim per session
- Policy document uploaded by the adjuster at submission time
- No batch processing, no customer-facing portal

## What It Takes In

| Input | Format |
|---|---|
| Customer statement | Free text |
| Damage image | JPEG / PNG |
| Repair estimate | Dollar amount |
| Policy document | PDF (uploaded by adjuster) |
| Claim history | Structured list of prior claims |

## What It Produces

| Output | Description |
|---|---|
| Claim summary | Normalized intake fields + completeness flags |
| Damage assessment | Vision model findings — type, severity, affected parts |
| Consistency result | Image vs. story alignment score + mismatch flag |
| Coverage decision | Covered / excluded / partial + policy clause cited |
| Risk score | 0–100 with top contributing factors |
| Payout recommendation | Calculated amount, deductible applied, coverage ratio |
| Human approval status | Auto-approved or escalated; adjuster decision if escalated |
| Audit trail | Timestamped JSONL of every agent decision |

## The Five Agents

1. **IntakeAgent** — validates and normalizes all claim inputs
2. **DamageEvidenceAgent** — sends image to OpenAI `gpt-4o` vision, compares findings to customer story and estimate
3. **PolicyAgent** — reads uploaded policy PDF, determines coverage
4. **RiskAgent** — scores claim risk from 0–100 using structured agent outputs
5. **PayoutAgent** — calculates recommended payout and flags auto-approve eligibility

## Success Looks Like

A clean claim (matching image, covered policy, low risk, payout under $5,000) auto-approves in under 30 seconds with a full audit trail. An image-story mismatch or high-risk claim surfaces to the adjuster with the exact reason flagged. Every decision is explainable.

## Demo Goal

A 15-minute YouTube demo showing three scenarios: a clean auto-approve, an image-story mismatch escalation, and a high-risk / high-payout escalation. Twenty synthetic claims with real damage images serve as the evaluation dataset.

## Open Questions / Assumptions

- [ASSUMPTION] Policy PDF is parsed to extract a coverage rule table; for demo a pre-parsed JSON is used
- [ASSUMPTION] Auto-approve limit: $5,000
- [ASSUMPTION] Risk escalation threshold: 70 / 100
- [ASSUMPTION] Agent confidence floor: 0.75
- [ASSUMPTION] Standard deductible: $500
- [ASSUMPTION] Coverage ratios: 80% collision, 100% comprehensive
