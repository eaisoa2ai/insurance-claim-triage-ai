"""
audit.py — Structured audit trail for every pipeline node.

write_audit_entry() builds one structured dict and returns it; the caller
appends it to ClaimState["audit_events"]. Nothing is written to disk here —
the dict stays in-memory until flush_audit_log() is called.

flush_audit_log() drains the in-memory list to a JSONL file at
    _bmad-output/audit_logs/{claim_id}.jsonl
one JSON object per line. Called once at pipeline end by the terminal node
(auto_approve, rejected, or human_review).

Every node — agent nodes, router, and terminal nodes — must call
write_audit_entry() before returning. The error path must write an "error"
event so the audit trail is never broken.

JSONL record shape (all 9 required fields at top level):
    {
        "timestamp":        ISO-8601 UTC string,
        "claim_id":         str,
        "agent":            str,          # agent name or "router" / "auto_approve" etc.
        "event":            str,          # "agent_complete" | "routing" | "error" | …
        "input_summary":    dict,         # key inputs that drove this node
        "output_summary":   dict,         # full agent output model as a dict
        "confidence":       float | null, # extracted from output_summary if present
        "evidence":         list[str],    # extracted from output_summary if present
        "warnings":         list[str],    # extracted from output_summary if present
        "routing_decision": str           # "continue" | "human_review" | "auto_approve" | …
    }
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any


AUDIT_LOG_DIR = pathlib.Path("_bmad-output/audit_logs")


def write_audit_entry(
    agent: str,
    event: str,
    output_model: Any | None = None,
    routing_decision: str = "continue",
    extra: dict | None = None,
    claim_id: str = "",
    input_summary: dict | None = None,
) -> dict:
    """Build and return one structured audit trail entry.

    Parameters
    ----------
    agent:            Name of the agent or node writing this entry.
    event:            Event type: "agent_complete", "routing", "human_review_requested",
                      "auto_approved", "auto_rejected", "adjuster_decision", "error".
    output_model:     Pydantic model (calls .model_dump()) or plain dict.
                      confidence, evidence, and warnings are extracted automatically.
    routing_decision: Routing outcome taken at this node.
    extra:            Arbitrary extra fields merged into output_summary last.
    claim_id:         The pipeline claim ID — included verbatim in every record.
    input_summary:    Key inputs that drove this node's decision (agent-defined).

    Returns a plain dict ready to be appended to ClaimState["audit_events"].
    """
    if output_model is not None:
        if hasattr(output_model, "model_dump"):
            data: dict = output_model.model_dump()
        elif isinstance(output_model, dict):
            data = dict(output_model)
        else:
            data = {"raw": str(output_model)}
    else:
        data = {}

    if extra:
        data = {**data, **extra}

    return {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "claim_id":         claim_id,
        "agent":            agent,
        "event":            event,
        "input_summary":    input_summary or {},
        "output_summary":   data,
        "confidence":       data.get("confidence"),
        "evidence":         data.get("evidence", []),
        "warnings":         data.get("warnings", []),
        "routing_decision": routing_decision,
    }


def flush_audit_log(claim_id: str, audit_events: list[dict]) -> pathlib.Path:
    """Write audit_events to _bmad-output/audit_logs/{claim_id}.jsonl.

    Creates the directory if it does not exist. Returns the path written.
    Uses default=str so dates and Pydantic models always serialise cleanly.
    """
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_LOG_DIR / f"{claim_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in audit_events:
            fh.write(json.dumps(entry, default=str) + "\n")
    return path
