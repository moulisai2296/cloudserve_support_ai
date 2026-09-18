"""FastAPI web service for CloudServe Solutions support triage.

Exposes REST endpoints for real-time ticket ingestion, health diagnostics,
operational metrics, and administrative kill-switch controls.
Adheres to Capstone Specification and Setup Guide Section 08.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.graph import process_ticket
from src.logging_store import count_decisions, get_db_connection

app = FastAPI(
    title="CloudServe Intelligent Customer Support Triage API",
    description="Automated multi-channel triage, knowledge retrieval, and grounded resolution engine.",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    ticket_id: str
    channel: str = "email"
    subject: str = ""
    body: str
    customer_id: str = "CUST-ANON"
    customer_name: str = "Customer"
    customer_tier: str = "standard"
    customer_region: str = "global"
    language_fluency: str = "fluent"
    use_llm: bool = True


class TicketResponse(BaseModel):
    ticket_id: str
    route: str
    action_taken: str
    intent: str
    confidence: float
    urgency: str
    generated_response: Optional[str] = None
    cited_doc_ids: List[str] = Field(default_factory=list)
    escalation_reason: Optional[str] = None
    escalation_packet: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    decision_id: Optional[str] = None
    policy_flags: List[str] = Field(default_factory=list)


class KillSwitchRequest(BaseModel):
    active: bool


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Returns system status and database connectivity."""
    db_ok = False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "service": "cloudserve-triage-api",
        "database_connected": db_ok,
        "kill_switch_active": os.getenv("KILL_SWITCH_ACTIVE", "false").lower() in ("true", "1", "yes"),
    }


@app.post("/tickets", response_model=TicketResponse, tags=["Triage"])
def triage_ticket(ticket: TicketRequest):
    """
    Submits a customer ticket to the LangGraph pipeline for automated
    retrieval, classification, routing, and resolution.
    """
    try:
        payload = ticket.model_dump()
        use_llm = payload.pop("use_llm", True)
        result = process_ticket(payload, use_llm=use_llm)

        route = result.get("route", "escalate")
        action = "auto_respond" if route == "auto_respond" else "escalate"
        if result.get("guardrail_blocked"):
            action = "block"

        packet = result.get("escalation_packet") or {}
        if route == "auto_respond":
            summary = result.get("auto_respond_summary") or "Auto-responded: Grounded in documentation and verified by safety guardrails."
        else:
            summary = packet.get("handover_notes") or result.get("escalation_reason") or "Escalated for human engineering assistance."

        return TicketResponse(
            ticket_id=result.get("ticket_id", ticket.ticket_id),
            route=route,
            action_taken=action,
            intent=result.get("intent", "unclear_request"),
            confidence=float(result.get("classification_confidence", 0.0)),
            urgency=result.get("urgency", "medium"),
            generated_response=result.get("generated_response"),
            cited_doc_ids=result.get("cited_doc_ids", []),
            escalation_reason=result.get("escalation_reason"),
            escalation_packet=result.get("escalation_packet"),
            summary=summary,
            decision_id=result.get("decision_id"),
            policy_flags=result.get("policy_flags", []),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ticket triage processing failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
def get_operational_metrics():
    """Returns real-time triage statistics from the decision audit store."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) AS total FROM decisions")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT action_taken, COUNT(*) AS count FROM decisions GROUP BY action_taken")
        action_counts = {r["action_taken"]: r["count"] for r in cursor.fetchall()}

        cursor.execute("SELECT intent, COUNT(*) AS count FROM decisions GROUP BY intent ORDER BY count DESC LIMIT 5")
        top_intents = {r["intent"]: r["count"] for r in cursor.fetchall()}

        conn.close()

        escalations = action_counts.get("escalate", 0) + action_counts.get("block", 0)
        auto_responses = action_counts.get("auto_respond", 0)
        escalation_rate = (escalations / total * 100.0) if total > 0 else 0.0

        return {
            "total_decisions_logged": total,
            "auto_responded_count": auto_responses,
            "escalated_count": escalations,
            "escalation_rate_pct": round(escalation_rate, 2),
            "top_intents": top_intents,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load metrics: {str(e)}"
        )


@app.post("/admin/kill-switch", tags=["Administration"])
def set_kill_switch(req: KillSwitchRequest):
    """Administrative override toggle to halt/resume autonomous customer responses."""
    os.environ["KILL_SWITCH_ACTIVE"] = "true" if req.active else "false"
    return {
        "kill_switch_active": req.active,
        "status": "All automated responses halted; 100% human escalation active." if req.active else "Autonomous triage operational."
    }


@app.get("/admin/kill-switch", tags=["Administration"])
def get_kill_switch():
    """Inspects the current status of the emergency kill-switch."""
    active = os.getenv("KILL_SWITCH_ACTIVE", "false").lower() in ("true", "1", "yes")
    return {"kill_switch_active": active}
