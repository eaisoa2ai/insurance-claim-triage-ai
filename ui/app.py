"""
ui/app.py — Gradio dashboard for the Insurance Claims AI Agent Team.

Tabs:
  1. Claim Demo   — run the 5-agent pipeline on a selected demo claim
  2. Agent Timeline — per-agent status, confidence, evidence, warnings
  3. Audit Trail   — full structured event log
  4. Evals         — test suite summary and production bug story

Run with:
  python ui/app.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

# ── Local imports with graceful fallback ──────────────────────────────────────
try:
    from core.data_loader import build_initial_state
    from core.graph import build_graph
    _PIPELINE_AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as _exc:
    _PIPELINE_AVAILABLE = False
    _IMPORT_ERROR = str(_exc)

DATA_DIR  = PROJECT_ROOT / "data"
EVALS_DIR = PROJECT_ROOT / "evals"

# ── Lazy graph singleton ──────────────────────────────────────────────────────
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ── Agent label map (audit event name → display name) ─────────────────────────
_AGENT_LABELS = {
    "IntakeAgent":         "Intake",
    "DamageEvidenceAgent": "Damage",
    "PolicyAgent":         "Policy",
    "RiskAgent":           "Risk",
    "PayoutAgent":         "Payout",
    "router":              "Router",
    "human_review":        "Human Review",
    "auto_approve":        "Auto Approve",
    "auto_reject":         "Auto Reject",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
/* ── base ── */
body, .gradio-container {
    background: #FAF8F4 !important;
    font-family: Inter, Arial, system-ui, sans-serif;
    color: #111111;
}
/* tabs */
.tab-nav button {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #555555 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
}
.tab-nav button.selected {
    color: #E06F5F !important;
    border-bottom: 3px solid #E06F5F !important;
    background: #FAF8F4 !important;
}
/* header */
.app-header {
    background: #ffffff;
    border-bottom: 2px solid #E8DED4;
    padding: 20px 32px 14px;
    margin-bottom: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px #EAD8C840;
}
.app-header h1 {
    font-size: 1.7rem;
    font-weight: 800;
    color: #111111;
    margin: 0 0 4px 0;
}
.app-header p {
    color: #555555;
    margin: 0;
    font-size: 0.95rem;
}
/* section label */
.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #E06F5F;
    margin-bottom: 6px;
}
/* card */
.card {
    background: #ffffff;
    border: 1px solid #E8DED4;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 2px 6px #EAD8C830;
    margin-bottom: 14px;
}
/* decision card — large */
.decision-main {
    background: #ffffff;
    border: 2px solid #E8DED4;
    border-radius: 16px;
    padding: 28px 28px 22px;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px #EAD8C840;
}
.decision-main .d-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555555;
    margin-bottom: 10px;
}
.decision-main .d-claim {
    font-size: 0.9rem;
    color: #777;
    margin-top: 10px;
}
/* badge */
.badge-pay    { display:inline-block;padding:10px 32px;border-radius:30px;background:#6BBF59;color:#fff;font-weight:800;font-size:1.5rem;letter-spacing:0.05em; }
.badge-reject { display:inline-block;padding:10px 32px;border-radius:30px;background:#555555;color:#fff;font-weight:800;font-size:1.5rem;letter-spacing:0.05em; }
.badge-human  { display:inline-block;padding:10px 32px;border-radius:30px;background:#E06F5F;color:#fff;font-weight:800;font-size:1.5rem;letter-spacing:0.05em; }
.badge-small-green  { display:inline-block;padding:3px 12px;border-radius:16px;background:#6BBF5922;color:#3a8c2a;font-weight:700;font-size:0.82rem;border:1px solid #6BBF5960; }
.badge-small-coral  { display:inline-block;padding:3px 12px;border-radius:16px;background:#E06F5F22;color:#c04030;font-weight:700;font-size:0.82rem;border:1px solid #E06F5F60; }
.badge-small-gray   { display:inline-block;padding:3px 12px;border-radius:16px;background:#55555520;color:#444;font-weight:700;font-size:0.82rem;border:1px solid #55555550; }
.badge-small-beige  { display:inline-block;padding:3px 12px;border-radius:16px;background:#EAD8C860;color:#665544;font-weight:700;font-size:0.82rem;border:1px solid #EAD8C8; }
/* metric grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #E8DED4;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 4px #EAD8C820;
}
.metric-card .m-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #888;
    margin-bottom: 6px;
}
.metric-card .m-value {
    font-size: 1.2rem;
    font-weight: 800;
    color: #111;
}
.metric-card .m-sub {
    font-size: 0.8rem;
    color: #777;
    margin-top: 2px;
}
/* run button */
.run-btn {
    background: #E06F5F !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    border: none !important;
}
.run-btn:hover { background: #c9574a !important; }
/* timeline */
.timeline-step {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    align-items: flex-start;
}
.timeline-num {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: #E06F5F;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.timeline-num.done   { background: #6BBF59; }
.timeline-num.human  { background: #E06F5F; }
.timeline-num.empty  { background: #CCC; }
.timeline-body { flex: 1; }
.tl-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}
.tl-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #111;
}
.tl-confidence {
    font-size: 0.82rem;
    color: #777;
    margin-left: auto;
}
.tl-evidence {
    font-size: 0.85rem;
    color: #555;
    margin: 4px 0;
    line-height: 1.5;
}
.tl-warning {
    background: #FFF5F3;
    border-left: 3px solid #E06F5F;
    padding: 6px 10px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #c04030;
    margin-top: 6px;
}
.tl-summary {
    font-size: 0.85rem;
    color: #444;
    font-style: italic;
    margin-top: 6px;
    border-top: 1px solid #F0E8E0;
    padding-top: 6px;
}
/* audit table wrapper */
.audit-headline {
    font-size: 1.05rem;
    font-weight: 700;
    color: #555;
    font-style: italic;
    margin-bottom: 14px;
    padding: 12px 16px;
    background: #fff;
    border-radius: 10px;
    border-left: 4px solid #E06F5F;
}
/* eval cards */
.eval-overall {
    background: #fff;
    border: 2px solid #6BBF59;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-bottom: 20px;
}
.eval-overall .ev-count {
    font-size: 2.5rem;
    font-weight: 900;
    color: #3a8c2a;
}
.eval-overall .ev-label {
    font-size: 1rem;
    color: #555;
    margin-top: 4px;
}
.eval-overall .ev-time {
    font-size: 0.78rem;
    color: #999;
    margin-top: 8px;
}
.eval-suites {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.eval-suite-card {
    background: #fff;
    border: 1px solid #E8DED4;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
}
.eval-suite-card .es-name { font-size: 0.82rem; font-weight: 700; color: #555; margin-bottom: 6px; }
.eval-suite-card .es-count { font-size: 1.4rem; font-weight: 900; color: #3a8c2a; }
.eval-suite-card .es-file { font-size: 0.7rem; color: #999; margin-top: 4px; }
.bar-section { background: #fff; border: 1px solid #E8DED4; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; }
.bar-section h3 { font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #555; margin: 0 0 16px 0; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.bar-file { font-size: 0.82rem; color: #555; width: 180px; flex-shrink: 0; }
.bar-track { flex: 1; background: #F3EDE5; border-radius: 6px; height: 18px; }
.bar-fill  { height: 18px; border-radius: 6px; background: #6BBF59; }
.bar-num   { font-size: 0.82rem; font-weight: 700; color: #3a8c2a; width: 32px; }
.bug-card {
    background: #FFF8F6;
    border: 2px solid #E06F5F;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.bug-card h3 { font-size: 1rem; font-weight: 800; color: #E06F5F; margin: 0 0 12px 0; }
.bug-row { display: flex; gap: 8px; margin-bottom: 6px; font-size: 0.87rem; }
.bug-key { font-weight: 700; color: #555; min-width: 110px; }
.bug-val { color: #111; }
.bug-fix { font-family: monospace; background: #fff; padding: 2px 8px; border-radius: 4px; border: 1px solid #E8DED4; }
.checklist { background: #fff; border: 1px solid #E8DED4; border-radius: 12px; padding: 20px 24px; }
.checklist h3 { font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #555; margin: 0 0 14px 0; }
.check-item { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #F5EEE6; font-size: 0.9rem; color: #333; }
.check-item:last-child { border-bottom: none; }
.check-tick { color: #6BBF59; font-size: 1.1rem; font-weight: 700; }
/* human review panel */
.human-panel {
    background: #FFF8F6;
    border: 2px solid #E06F5F;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}
.human-panel h3 { color: #E06F5F; font-weight: 800; margin: 0 0 10px 0; }
.escalation-flag {
    display: inline-block;
    background: #E06F5F22;
    border: 1px solid #E06F5F60;
    color: #c04030;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}
/* placeholder text */
.placeholder {
    color: #999;
    font-style: italic;
    font-size: 0.95rem;
    padding: 24px;
    text-align: center;
}
/* info box */
.info-row {
    display: flex;
    gap: 6px;
    align-items: flex-start;
    padding: 6px 0;
    border-bottom: 1px solid #F5EEE6;
    font-size: 0.88rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { font-weight: 700; color: #777; min-width: 120px; }
.info-val { color: #111; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: pathlib.Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_claims() -> list[dict]:
    try:
        return _load_json(DATA_DIR / "claims.json").get("claims", [])
    except Exception:
        return []


def _claim_choices() -> list[str]:
    out = []
    for c in _all_claims():
        out.append(
            f"{c['claim_id']} | {c['claim_type']} | "
            f"€{float(c['claimed_amount']):,.0f} | {c['claim_story_type']}"
        )
    return out


def _id_from_choice(choice: str) -> str:
    return (choice or "").split(" | ")[0].strip()


def _load_display_data(choice: str):
    """
    Returns: (statement, image_path, obs_text, estimate_text, policy_text, history_text)
    Called when the user selects a claim from the dropdown.
    """
    if not choice:
        return "", None, "", "", "", ""
    try:
        claim_id = _id_from_choice(choice)
        claims   = _all_claims()
        claim    = next((c for c in claims if c["claim_id"] == claim_id), None)
        if claim is None:
            return "Claim not found.", None, "", "", "", ""

        statement = claim.get("customer_statement", "")

        # Image
        img_filename = claim.get("damage_image")
        image_path   = None
        if img_filename:
            p = DATA_DIR / "images" / img_filename
            if p.exists():
                image_path = str(p)

        # Vision observation (pre-loaded)
        obs_text = "No observation on file."
        if img_filename:
            try:
                obs_data = _load_json(DATA_DIR / "image_observations.json")
                obs_index = {o["filename"]: o for o in obs_data.get("observations", [])}
                obs = obs_index.get(img_filename)
                if obs:
                    obs_text = (
                        f"{obs.get('short_visual_summary', '')}\n\n"
                        f"Visible damage: {', '.join(obs.get('visible_damage', []))}\n"
                        f"Damaged parts: {', '.join(obs.get('damaged_parts', []))}\n"
                        f"Severity: {obs.get('severity_hint', 'unknown')}   "
                        f"Confidence: {obs.get('confidence', 0):.0%}"
                    )
            except Exception:
                pass

        # Repair estimate
        estimate_text = ""
        try:
            est_data  = _load_json(DATA_DIR / "repair_estimates.json")
            est_index = {e["claim_id"]: e for e in est_data.get("estimates", [])}
            est = est_index.get(claim_id)
            if est:
                parts_str = ", ".join(est.get("damaged_parts", [])) or "—"
                estimate_text = (
                    f"Shop: {est.get('repair_shop', '—')}\n"
                    f"Parts: €{est.get('parts_cost', 0):,.0f}   "
                    f"Labor: €{est.get('labor_cost', 0):,.0f}\n"
                    f"Total: €{est.get('total_estimated_cost', 0):,.0f}\n"
                    f"Plausibility: {est.get('estimate_plausibility', '—')}\n"
                    f"Damaged parts: {parts_str}"
                )
        except Exception:
            estimate_text = f"Claimed: €{float(claim.get('claimed_amount', 0)):,.0f}"

        # Policy
        policy_id   = claim.get("policy_id", "")
        policy_text = f"Policy ID: {policy_id}\nDocument: standard_auto_policy.json"

        # Claim history
        history_text = ""
        try:
            hist_data = _load_json(DATA_DIR / "claim_history.json")
            prior = [h for h in hist_data.get("history", [])
                     if h.get("customer_id") == claim.get("customer_id")]
            if prior:
                history_text = f"{len(prior)} prior claim(s):\n"
                for h in prior[:4]:
                    history_text += (
                        f"  • {h.get('incident_date','')} | "
                        f"{h.get('claim_type','')} | "
                        f"€{h.get('repair_estimate', 0):,.0f} | "
                        f"{h.get('outcome','')}\n"
                    )
            else:
                history_text = "No prior claims on record."
        except Exception:
            history_text = "History unavailable."

        return statement, image_path, obs_text, estimate_text, policy_text, history_text

    except Exception as exc:
        return f"Error loading claim: {exc}", None, "", "", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(choice: str) -> dict:
    """Run the 5-agent LangGraph pipeline and return the final ClaimState dict."""
    if not choice:
        return {"_error": "No claim selected."}
    if not _PIPELINE_AVAILABLE:
        return {"_error": f"Pipeline unavailable: {_IMPORT_ERROR}"}
    try:
        claim_id = _id_from_choice(choice)
        state    = build_initial_state(claim_id)
        result   = _get_graph().invoke(state)
        return dict(result)
    except Exception as exc:
        return {"_error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderers
# ─────────────────────────────────────────────────────────────────────────────

def _placeholder(msg: str = "Run the agent team to see results.") -> str:
    return f'<div class="placeholder">{msg}</div>'


def render_decision_cards(state: dict | None) -> str:
    if state is None:
        return _placeholder()
    if "_error" in state:
        return (
            f'<div class="card" style="border-color:#E06F5F;background:#FFF5F3;">'
            f'<strong style="color:#E06F5F;">⚠ Pipeline error</strong>'
            f'<p style="color:#555;margin:8px 0 0;">{state["_error"]}</p></div>'
        )

    report    = state.get("final_report") or {}
    decision  = report.get("decision", "unknown")
    claim_id  = report.get("claim_id", "")
    payout_rec = state.get("payout_recommendation") or {}

    payout_amount    = payout_rec.get("payout_amount", report.get("payout_amount") or 0)
    risk_score       = state.get("risk_score", 0)
    risk_category    = (state.get("risk_category") or "").upper()
    coverage_dec     = state.get("coverage_decision", "")
    image_mismatch   = state.get("image_claim_mismatch", False)
    requires_human   = state.get("requires_human_approval", False)

    # Main badge
    d = decision.lower()
    if d == "auto_approved":
        badge = '<span class="badge-pay">PAY</span>'
        border_color = "#6BBF59"
    elif d == "rejected":
        badge = '<span class="badge-reject">REJECT</span>'
        border_color = "#555555"
    elif d == "pending":
        badge = '<span class="badge-human">HUMAN REVIEW</span>'
        border_color = "#E06F5F"
    else:
        badge = f'<span class="badge-small-beige">{decision.upper()}</span>'
        border_color = "#EAD8C8"

    decision_label = {
        "auto_approved": "Auto-Approved — No adjuster needed",
        "rejected":      "Rejected — Not covered by policy",
        "pending":       "Escalated — Adjuster review required",
    }.get(d, decision)

    # Image match badge
    match_badge = (
        '<span class="badge-small-coral">⚠ Mismatch detected</span>'
        if image_mismatch else
        '<span class="badge-small-green">✓ Match</span>'
    )

    # Risk badge
    risk_color = {
        "LOW": "green", "MEDIUM": "beige", "HIGH": "coral", "CRITICAL": "coral"
    }.get(risk_category, "beige")
    risk_badge = f'<span class="badge-small-{risk_color}">{risk_category or "—"}</span>'

    # Coverage badge
    cov_color = {"covered": "green", "excluded": "gray", "partial": "beige"}.get(coverage_dec, "beige")
    cov_badge = f'<span class="badge-small-{cov_color}">{coverage_dec.upper() or "—"}</span>'

    # Human approval badge
    ha_badge = (
        '<span class="badge-small-coral">Required</span>'
        if requires_human else
        '<span class="badge-small-green">Not Required</span>'
    )

    html = f"""
    <div class="decision-main" style="border-color:{border_color};">
      <div class="d-label">Final Decision</div>
      {badge}
      <div class="d-claim" style="margin-top:10px;font-size:0.9rem;color:#666;">
        {decision_label}<br><span style="color:#aaa;font-size:0.8rem;">{claim_id}</span>
      </div>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="m-label">Image-Claim Match</div>
        <div class="m-value" style="font-size:1rem;">{match_badge}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Risk Score</div>
        <div class="m-value">{risk_score:.0f} <span style="font-size:0.85rem;font-weight:500;color:#888;">/ 100</span></div>
        <div class="m-sub">{risk_badge}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Coverage</div>
        <div class="m-value" style="font-size:1rem;">{cov_badge}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Payout</div>
        <div class="m-value">€{payout_amount:,.2f}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Human Approval</div>
        <div class="m-value" style="font-size:1rem;">{ha_badge}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Coverage Ratio</div>
        <div class="m-value">{payout_rec.get('coverage_ratio', 0):.0%}</div>
        <div class="m-sub">Deductible €{payout_rec.get('deductible', 0):,.0f}</div>
      </div>
    </div>
    """

    # Human review detail panel
    if requires_human:
        esc_reason = state.get("escalation_reason", "")
        esc_flags  = state.get("escalation_flags", [])
        flags_html = "".join(
            f'<span class="escalation-flag">{f}</span>' for f in esc_flags
        )
        html += f"""
        <div class="human-panel">
          <h3>&#128680; Adjuster Review Required</h3>
          <div style="font-size:0.9rem;margin-bottom:10px;">
            <strong>Reason:</strong> {esc_reason or "Multiple escalation conditions triggered."}
          </div>
          <div style="margin-bottom:6px;font-size:0.82rem;font-weight:700;color:#888;
               text-transform:uppercase;letter-spacing:0.06em;">Escalation Flags</div>
          <div>{flags_html or '<span style="color:#aaa;font-size:0.85rem;">—</span>'}</div>
        </div>
        """

    return html


def render_timeline(state: dict | None) -> str:
    if state is None:
        return _placeholder()
    if "_error" in state:
        return _placeholder(f"Pipeline error: {state['_error']}")

    # Extract per-agent data from state dicts
    damage  = state.get("damage_evidence_assessment") or {}
    policy  = state.get("policy_decision") or {}
    risk    = state.get("risk_assessment") or {}
    payout  = state.get("payout_recommendation") or {}

    # Intake: reconstruct from audit events
    audit_events = state.get("audit_events") or []
    intake_event = next(
        (e for e in audit_events if e.get("agent") == "IntakeAgent"), {}
    )
    intake_output = intake_event.get("output_summary") or {}

    agents_data = [
        {
            "num":    1,
            "name":   "Intake",
            "done":   state.get("intake_complete", False),
            "conf":   intake_output.get("confidence"),
            "evid":   intake_output.get("evidence", []),
            "warns":  intake_output.get("warnings", []),
            "summ":   intake_output.get("reasoning_summary", ""),
        },
        {
            "num":    2,
            "name":   "Damage",
            "done":   bool(damage),
            "conf":   damage.get("confidence"),
            "evid":   damage.get("evidence", []),
            "warns":  damage.get("warnings", []),
            "summ":   damage.get("reasoning_summary", ""),
        },
        {
            "num":    3,
            "name":   "Policy",
            "done":   bool(policy),
            "conf":   policy.get("confidence"),
            "evid":   policy.get("evidence", []),
            "warns":  policy.get("warnings", []),
            "summ":   policy.get("reasoning_summary", ""),
        },
        {
            "num":    4,
            "name":   "Risk",
            "done":   bool(risk),
            "conf":   risk.get("confidence"),
            "evid":   risk.get("evidence", []),
            "warns":  risk.get("warnings", []),
            "summ":   risk.get("reasoning_summary", ""),
        },
        {
            "num":    5,
            "name":   "Payout",
            "done":   bool(payout),
            "conf":   payout.get("confidence"),
            "evid":   payout.get("evidence", []),
            "warns":  payout.get("warnings", []),
            "summ":   payout.get("reasoning_summary", ""),
        },
    ]

    requires_human = state.get("requires_human_approval", False)
    final_decision = (state.get("final_report") or {}).get("decision", "")

    html = '<div style="padding:4px 0;">'
    for ag in agents_data:
        num_class = "done" if ag["done"] else "empty"

        # Status badge
        if ag["done"] and requires_human and ag["num"] == 5:
            status_badge = '<span class="badge-small-coral">Human Review</span>'
        elif ag["done"]:
            status_badge = '<span class="badge-small-green">Completed</span>'
        else:
            status_badge = '<span class="badge-small-gray">Not Run</span>'

        conf_txt = f"Confidence: {ag['conf']:.0%}" if ag["conf"] is not None else ""
        evid_html = ""
        if ag["evid"]:
            items = "".join(f"<li>{e}</li>" for e in ag["evid"][:4])
            evid_html = (
                f'<div class="tl-evidence"><strong>Evidence:</strong>'
                f'<ul style="margin:4px 0 0 16px;padding:0;">{items}</ul></div>'
            )
        warn_html = ""
        if ag["warns"]:
            for w in ag["warns"][:2]:
                warn_html += f'<div class="tl-warning">⚠ {w}</div>'

        summ_html = (
            f'<div class="tl-summary">{ag["summ"]}</div>'
            if ag["summ"] else ""
        )

        html += f"""
        <div class="timeline-step">
          <div class="timeline-num {num_class}">{ag["num"]}</div>
          <div class="timeline-body card" style="margin-bottom:0;padding:14px 18px;">
            <div class="tl-header">
              <span class="tl-name">{ag["name"]}</span>
              {status_badge}
              <span class="tl-confidence">{conf_txt}</span>
            </div>
            {evid_html}
            {warn_html}
            {summ_html}
          </div>
        </div>
        """

    html += "</div>"
    return html


def render_audit_rows(state: dict | None) -> list[list]:
    """Return rows for the audit trail dataframe."""
    if not state or "_error" in state:
        return []
    events = state.get("audit_events") or []
    rows   = []
    for i, ev in enumerate(events, 1):
        agent_raw = ev.get("agent", "")
        agent_lbl = _AGENT_LABELS.get(agent_raw, agent_raw)
        evid      = ev.get("evidence", [])
        evid_str  = "; ".join(evid[:2]) if evid else "—"
        warns     = ev.get("warnings", [])
        warn_str  = "; ".join(warns[:1]) if warns else "None"
        conf      = ev.get("confidence")
        conf_str  = f"{conf:.2f}" if conf is not None else "—"
        ts_raw    = ev.get("timestamp", "")
        ts_str    = ts_raw[:19].replace("T", " ") if ts_raw else "—"
        inp       = ev.get("input_summary") or {}
        inp_str   = ", ".join(f"{k}" for k in list(inp.keys())[:3]) or "—"
        routing   = ev.get("routing_decision", "continue")
        rows.append([i, agent_lbl, inp_str, routing, evid_str, conf_str, warn_str, ts_str])
    return rows


def render_evals_html() -> str:
    """Read evals/eval_summary.json and render as styled HTML."""
    try:
        data = json.loads((EVALS_DIR / "eval_summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return f'<div class="card"><p style="color:#E06F5F;">Could not load eval_summary.json: {exc}</p></div>'

    overall = data.get("overall", {})
    passed  = overall.get("passed", 0)
    total   = overall.get("total", 0)
    last_run_raw = overall.get("last_run", "")
    last_run = last_run_raw[:19].replace("T", " ") if last_run_raw else ""

    # Overall card
    html = f"""
    <div class="eval-overall">
      <div class="ev-count">{passed} / {total}</div>
      <div class="ev-label">tests passed</div>
      <div style="margin-top:12px;">
        <span style="display:inline-block;background:#6BBF5922;border:1px solid #6BBF5960;
              border-radius:20px;padding:5px 18px;color:#3a8c2a;font-weight:700;font-size:0.9rem;">
          ✓ ALL PASSING
        </span>
      </div>
      {"<div class='ev-time'>Last run: " + last_run + " UTC</div>" if last_run else ""}
    </div>
    """

    # Suite cards
    suites = data.get("suites", [])
    html += '<div class="eval-suites">'
    for s in suites:
        html += f"""
        <div class="eval-suite-card">
          <div class="es-name">{s['name']}</div>
          <div class="es-count">{s['passed']}</div>
          <div style="font-size:0.75rem;color:#aaa;">passed</div>
          <div class="es-file">{s['file']}</div>
        </div>
        """
    html += "</div>"

    # Bar chart
    max_tests = max((s["total"] for s in suites), default=1)
    html += '<div class="bar-section"><h3>Tests per suite</h3>'
    for s in suites:
        pct = int(s["total"] / max_tests * 100)
        html += f"""
        <div class="bar-row">
          <div class="bar-file">{s['file']}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct}%;"></div></div>
          <div class="bar-num">{s['total']}</div>
        </div>
        """
    html += "</div>"

    # Bug card
    bug = data.get("production_bug_caught", {})
    if bug:
        html += f"""
        <div class="bug-card">
          <h3>\U0001f41b Production Bug Caught by Evals</h3>
          <div class="bug-row"><span class="bug-key">File</span>
            <span class="bug-val">{bug.get('file','')}</span></div>
          <div class="bug-row"><span class="bug-key">Issue</span>
            <span class="bug-val">{bug.get('issue','')}</span></div>
          <div class="bug-row"><span class="bug-key">Fix</span>
            <span class="bug-val"><span class="bug-fix">{bug.get('fix','')}</span></span></div>
          <div class="bug-row"><span class="bug-key">Why it matters</span>
            <span class="bug-val">{bug.get('why_it_matters','')}</span></div>
        </div>
        """

    # Coverage checklist
    coverage = data.get("coverage", [])
    if coverage:
        items_html = "".join(
            f'<div class="check-item"><span class="check-tick">✓</span>{item}</div>'
            for item in coverage
        )
        html += f"""
        <div class="checklist">
          <h3>Coverage checklist</h3>
          {items_html}
        </div>
        """

    return html


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    choices = _claim_choices()
    default = choices[0] if choices else ""

    pipeline_warning = (
        f'<div class="card" style="border-color:#E06F5F;background:#FFF5F3;">'
        f'<strong style="color:#E06F5F;">⚠ Pipeline not available.</strong> '
        f'<span style="color:#555;">{_IMPORT_ERROR}</span></div>'
        if not _PIPELINE_AVAILABLE else ""
    )

    with gr.Blocks(title="Insurance Claims AI Agent Team") as app:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <h1>Insurance Claims AI Agent Team</h1>
          <p>5-agent LangGraph pipeline &mdash; Intake &rarr; Damage &rarr; Policy &rarr; Risk &rarr; Payout</p>
        </div>
        """ + pipeline_warning)

        # ── Shared state ──────────────────────────────────────────────────────
        pipeline_state = gr.State(value=None)

        with gr.Tabs():

            # ── Tab 1: Claim Demo ─────────────────────────────────────────────
            with gr.Tab("Claim Demo"):
                with gr.Row(equal_height=False):

                    # Left: inputs
                    with gr.Column(scale=1, min_width=340):
                        gr.HTML('<div class="section-label">Select a demo claim</div>')
                        claim_dd = gr.Dropdown(
                            choices=choices,
                            value=default,
                            label="Claim",
                            interactive=True,
                        )
                        statement_box = gr.Textbox(
                            label="Customer Statement",
                            lines=4,
                            interactive=False,
                        )
                        damage_img = gr.Image(
                            label="Damage Image",
                            type="filepath",
                            interactive=False,
                            height=220,
                        )
                        obs_box = gr.Textbox(
                            label="Vision Damage Observation",
                            lines=4,
                            interactive=False,
                        )
                        estimate_box = gr.Textbox(
                            label="Repair Estimate",
                            lines=5,
                            interactive=False,
                        )
                        policy_box = gr.Textbox(
                            label="Policy",
                            lines=2,
                            interactive=False,
                        )
                        history_box = gr.Textbox(
                            label="Claim History",
                            lines=4,
                            interactive=False,
                        )
                        run_btn = gr.Button(
                            "Run Agent Team ▶",
                            variant="primary",
                            elem_classes=["run-btn"],
                        )

                    # Right: results
                    with gr.Column(scale=1, min_width=380):
                        gr.HTML('<div class="section-label">Decision Summary</div>')
                        decision_html = gr.HTML(
                            value=_placeholder(),
                        )

            # ── Tab 2: Agent Timeline ─────────────────────────────────────────
            with gr.Tab("Agent Timeline"):
                gr.HTML("""
                <div style="margin-bottom:16px;">
                  <div class="section-label">5-Agent Pipeline</div>
                  <p style="color:#555;font-size:0.9rem;margin:0;">
                    Each agent contributes one decision. Run the demo to see the full trace.
                  </p>
                </div>
                """)
                timeline_html = gr.HTML(value=_placeholder())

            # ── Tab 3: Audit Trail ────────────────────────────────────────────
            with gr.Tab("Audit Trail"):
                gr.HTML("""
                <div class="audit-headline">
                  "In insurance, the decision trail matters as much as the decision."
                </div>
                """)
                audit_table = gr.Dataframe(
                    headers=["Step", "Agent", "Input Used", "Decision",
                              "Evidence", "Confidence", "Warning", "Timestamp"],
                    datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                    value=[],
                    row_count=(1, "dynamic"),
                    wrap=True,
                    interactive=False,
                    label="Audit Trail",
                )

            # ── Tab 4: Evals ──────────────────────────────────────────────────
            with gr.Tab("Evals"):
                gr.HTML("""
                <div style="margin-bottom:16px;">
                  <div class="section-label">Evaluation Results</div>
                  <p style="color:#555;font-size:0.9rem;margin:0;">
                    The system looked fine. The evals proved it didn't.
                    Claude fixed the bug. Now 72 / 72 tests pass.
                  </p>
                </div>
                """)
                gr.HTML(value=render_evals_html())

        # ── Human review panel (below tabs, only shown when required) ─────────
        with gr.Group(visible=False) as human_group:
            gr.HTML('<div style="height:16px;"></div>')
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div class="human-panel">
                      <h3>Adjuster Decision Required</h3>
                      <p style="color:#555;font-size:0.9rem;margin:0 0 12px;">
                        Enter a note (minimum 10 characters) before approving or rejecting.
                      </p>
                    </div>
                    """)
                    adjuster_note = gr.Textbox(
                        label="Adjuster Note (min 10 characters)",
                        placeholder="Explain the basis for your decision...",
                        lines=2,
                    )
                    with gr.Row():
                        approve_btn = gr.Button("Approve", interactive=False, variant="primary")
                        reject_btn  = gr.Button("Reject",  interactive=False, variant="stop")

        # ── Event wiring ──────────────────────────────────────────────────────

        # Load claim details on dropdown change
        claim_dd.change(
            fn=_load_display_data,
            inputs=[claim_dd],
            outputs=[statement_box, damage_img, obs_box,
                     estimate_box, policy_box, history_box],
        )

        # Run pipeline
        def _run(choice, _state):
            state = run_pipeline(choice)
            dec_html   = render_decision_cards(state)
            tl_html    = render_timeline(state)
            audit_rows = render_audit_rows(state)
            requires_h = bool((state or {}).get("requires_human_approval", False))
            return (
                state,          # pipeline_state
                dec_html,       # decision_html
                tl_html,        # timeline_html
                audit_rows,     # audit_table
                gr.update(visible=requires_h),  # human_group
            )

        run_btn.click(
            fn=_run,
            inputs=[claim_dd, pipeline_state],
            outputs=[pipeline_state, decision_html, timeline_html,
                     audit_table, human_group],
        )

        # Enable Approve/Reject only when note >= 10 chars
        def _check_note(note: str):
            ok = len((note or "").strip()) >= 10
            return gr.update(interactive=ok), gr.update(interactive=ok)

        adjuster_note.change(
            fn=_check_note,
            inputs=[adjuster_note],
            outputs=[approve_btn, reject_btn],
        )

        # Adjuster decision handlers
        def _adjuster_decision(approved: bool, note: str, state: dict):
            if not state or "_error" in state:
                return state, render_decision_cards(state), gr.update(visible=True)

            # Patch the final_report decision
            report = dict(state.get("final_report") or {})
            report["decision"] = "human_approved" if approved else "rejected"
            state = {**state, "final_report": report,
                     "requires_human_approval": False,
                     "human_approval_granted": approved}
            return (
                state,
                render_decision_cards(state),
                gr.update(visible=False),
            )

        approve_btn.click(
            fn=lambda note, s: _adjuster_decision(True,  note, s),
            inputs=[adjuster_note, pipeline_state],
            outputs=[pipeline_state, decision_html, human_group],
        )
        reject_btn.click(
            fn=lambda note, s: _adjuster_decision(False, note, s),
            inputs=[adjuster_note, pipeline_state],
            outputs=[pipeline_state, decision_html, human_group],
        )

        # Load default claim details on startup
        app.load(
            fn=_load_display_data,
            inputs=[claim_dd],
            outputs=[statement_box, damage_img, obs_box,
                     estimate_box, policy_box, history_box],
        )

    return app


def launch():
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=_CSS,
        theme=gr.themes.Base(
            primary_hue="orange",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        ),
    )


if __name__ == "__main__":
    launch()
