"""
graph.py — LangGraph StateGraph wiring.

Edge topology:
    START → intake → damage_evidence → policy → risk → payout → router
    router → conditional(route_decision) → human_review | rejected | auto_approve
    human_review → END
    rejected     → END
    auto_approve → END

Node return pattern: every node returns a partial ClaimState dict.
LangGraph merges it into the shared state; never return the full state object.

Terminal nodes (human_review, rejected, auto_approve) write a FinalReport and
flush the JSONL audit log before returning.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.audit import flush_audit_log, write_audit_entry
from core.claim_state import ClaimState, FinalReport, HumanApprovalDecision
from core.routing import route_decision, router_node


# ── Agent nodes ────────────────────────────────────────────────────────────────
# Agent classes are imported inside the function so build_graph() is importable
# without triggering model/config loading at module level.

def intake_node(state: ClaimState) -> dict:
    """Run IntakeAgent; return partial state with claim_input, intake_complete, audit_events."""
    from agents.intake_agent import IntakeAgent
    return IntakeAgent().run(state)


def damage_evidence_node(state: ClaimState) -> dict:
    """Run DamageEvidenceAgent; return partial state with damage_evidence_assessment."""
    from agents.damage_evidence_agent import DamageEvidenceAgent
    return DamageEvidenceAgent().run(state)


def policy_node(state: ClaimState) -> dict:
    """Run PolicyAgent; return partial state with policy_decision, policy_valid, coverage_decision."""
    from agents.policy_agent import PolicyAgent
    return PolicyAgent().run(state)


def risk_node(state: ClaimState) -> dict:
    """Run RiskAgent; return partial state with risk_assessment, risk_score, risk_category."""
    from agents.risk_agent import RiskAgent
    return RiskAgent().run(state)


def payout_node(state: ClaimState) -> dict:
    """Run PayoutAgent; return partial state with payout_recommendation."""
    from agents.payout_agent import PayoutAgent
    return PayoutAgent().run(state)


# ── Terminal nodes ─────────────────────────────────────────────────────────────

def human_review_node(state: ClaimState) -> dict:
    """Halt for adjuster input.

    Writes a FinalReport(decision="pending"), populates human_approval with
    escalation context, and flushes the audit log. The Gradio UI reads
    final_report and human_approval to display the escalation summary.
    """
    claim_id = (state.get("claim_input") or {}).get("claim_id", "unknown")

    report = FinalReport(
        claim_id=claim_id,
        decision="pending",
        payout_amount=None,
        risk_category=state.get("risk_category") or None,
        coverage_decision=state.get("coverage_decision") or None,
        escalation_flags=list(state.get("escalation_flags") or []),
    )

    approval = HumanApprovalDecision(
        required=True,
        escalation_reason=state.get("escalation_reason") or "",
        escalation_flags=list(state.get("escalation_flags") or []),
    )

    audit_event = write_audit_entry(
        agent="human_review",
        event="human_review_requested",
        routing_decision="human_review",
        extra={
            "escalation_reason": state.get("escalation_reason", ""),
            "escalation_flags": list(state.get("escalation_flags") or []),
        },
        claim_id=claim_id,
        input_summary={
            "escalation_reason":  state.get("escalation_reason", ""),
            "escalation_flags":   list(state.get("escalation_flags") or []),
            "risk_score":         state.get("risk_score"),
            "coverage_decision":  state.get("coverage_decision", ""),
            "payout_amount":      (state.get("payout_recommendation") or {}).get("payout_amount"),
        },
    )

    events = state["audit_events"] + [audit_event]
    flush_audit_log(claim_id, events)

    return {
        "human_approval": approval.model_dump(),
        "requires_human_approval": True,
        "final_report": report.model_dump(),
        "audit_events": events,
    }


def rejected_node(state: ClaimState) -> dict:
    """Fast-path rejection for claims excluded by policy.

    Coverage type is not in scope for this policy — no adjuster decision needed.
    Writes a FinalReport(decision="rejected", payout_amount=0.0) and flushes log.
    """
    claim_id = (state.get("claim_input") or {}).get("claim_id", "unknown")

    report = FinalReport(
        claim_id=claim_id,
        decision="rejected",
        payout_amount=0.0,
        risk_category=state.get("risk_category") or None,
        coverage_decision=state.get("coverage_decision") or None,
        escalation_flags=list(state.get("escalation_flags") or []),
    )

    audit_event = write_audit_entry(
        agent="auto_reject",
        event="auto_rejected",
        routing_decision="rejected",
        extra={"reason": "Coverage type excluded by policy — no payout issued"},
        claim_id=claim_id,
        input_summary={
            "coverage_decision":  state.get("coverage_decision", ""),
            "escalation_flags":   list(state.get("escalation_flags") or []),
            "risk_score":         state.get("risk_score"),
        },
    )

    events = state["audit_events"] + [audit_event]
    flush_audit_log(claim_id, events)

    return {
        "final_report": report.model_dump(),
        "audit_events": events,
    }


def auto_approve_node(state: ClaimState) -> dict:
    """Auto-approve a clean claim that cleared all escalation conditions.

    Writes a FinalReport(decision="auto_approved") with the computed payout
    and flushes the audit log.
    """
    claim_id = (state.get("claim_input") or {}).get("claim_id", "unknown")
    payout_rec = state.get("payout_recommendation") or {}
    payout_amount = payout_rec.get("payout_amount", 0.0)

    report = FinalReport(
        claim_id=claim_id,
        decision="auto_approved",
        payout_amount=float(payout_amount),
        risk_category=state.get("risk_category") or None,
        coverage_decision=state.get("coverage_decision") or None,
        escalation_flags=list(state.get("escalation_flags") or []),
    )

    audit_event = write_audit_entry(
        agent="auto_approve",
        event="auto_approved",
        routing_decision="auto_approve",
        claim_id=claim_id,
        input_summary={
            "payout_amount":      float(payout_amount),
            "coverage_decision":  state.get("coverage_decision", ""),
            "risk_score":         state.get("risk_score"),
            "risk_category":      state.get("risk_category", ""),
        },
    )

    events = state["audit_events"] + [audit_event]
    flush_audit_log(claim_id, events)

    return {
        "final_report": report.model_dump(),
        "audit_events": events,
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """Compile and return the LangGraph runnable for the claims pipeline.

    Importable without side effects — no API calls or file I/O happen here.
    """
    graph = StateGraph(ClaimState)

    graph.add_node("intake",          intake_node)
    graph.add_node("damage_evidence", damage_evidence_node)
    graph.add_node("policy",          policy_node)
    graph.add_node("risk",            risk_node)
    graph.add_node("payout",          payout_node)
    graph.add_node("router",          router_node)
    graph.add_node("human_review",    human_review_node)
    graph.add_node("rejected",        rejected_node)
    graph.add_node("auto_approve",    auto_approve_node)

    graph.add_edge(START,            "intake")
    graph.add_edge("intake",         "damage_evidence")
    graph.add_edge("damage_evidence","policy")
    graph.add_edge("policy",         "risk")
    graph.add_edge("risk",           "payout")
    graph.add_edge("payout",         "router")

    graph.add_conditional_edges("router", route_decision, {
        "human_review": "human_review",
        "rejected":     "rejected",
        "auto_approve": "auto_approve",
    })

    graph.add_edge("human_review", END)
    graph.add_edge("rejected",     END)
    graph.add_edge("auto_approve", END)

    return graph.compile()
