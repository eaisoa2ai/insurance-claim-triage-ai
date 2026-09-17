"""
main.py — Entry point: run the five-agent pipeline end-to-end on the three
demo scenarios and print a decision summary for each.

    uv run python main.py

Runs fully offline against the fixtures in data/ (pre-computed vision
observations, sample policies, synthetic claim history) — no OpenAI API
key required for this demo path.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from core.data_loader import build_initial_state
from core.graph import build_graph

# Three scenarios chosen to exercise each end of the routing logic:
#   - a clean claim that auto-approves
#   - an image/claim mismatch that escalates on a fraud signal
#   - a high-value, high-risk claim that escalates on multiple thresholds
DEMO_CLAIMS = ["CLM-2026-001", "CLM-2026-017", "CLM-2026-020"]


def run_claim(graph, claim_id: str) -> None:
    state = build_initial_state(claim_id)
    result = graph.invoke(state)
    report = result["final_report"]

    print(f"\n=== {claim_id} ===")
    print(f"  Decision:          {report['decision']}")
    print(f"  Coverage:          {result['coverage_decision']}")
    print(f"  Risk score:        {result['risk_score']} ({result['risk_category']})")
    print(f"  Payout:            {report['payout_amount']}")
    print(f"  Human review:      {result['requires_human_approval']}")
    if result["escalation_flags"]:
        print("  Escalation flags:")
        for flag in result["escalation_flags"]:
            print(f"    - {flag}")


def main() -> None:
    graph = build_graph()
    for claim_id in DEMO_CLAIMS:
        run_claim(graph, claim_id)
    print(
        "\nFull audit trail for each claim written to "
        "_bmad-output/audit_logs/<claim_id>.jsonl"
    )


if __name__ == "__main__":
    main()
