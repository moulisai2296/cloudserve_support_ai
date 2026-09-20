"""Unit tests for routing logic and Tier-2 escalation packet assembly (AC-A5 / B-07 / FR-06).

Validates confidence thresholds, policy exclusions, enterprise SLA rules,
emergency kill-switch overrides, and rich escalation packet compilation.
"""

import pytest
from src.route import (
    RouteDecision,
    EscalationReason,
    route_ticket,
    route_node,
    build_escalation_packet,
)


def test_route_auto_respond_success():
    """Valid ticket with high confidence, relevant docs, and no risk flags should auto-respond."""
    res = route_ticket(
        intent="deployment_failure",
        urgency="medium",
        confidence=0.91,
        has_relevant_docs=True,
        customer_tier="business",
        must_not_auto_respond=False,
        ticket_id="TICKET-AUTO-01",
    )
    assert res.route == RouteDecision.AUTO_RESPOND
    assert res.escalation_reason is None
    assert res.escalation_packet is None
    assert "ROUTED_FOR_AUTONOMOUS_RESOLUTION" in res.policy_flags


def test_route_low_confidence_escalation():
    """Confidence below threshold (0.80) must escalate with contextual packet."""
    retrieved = [{
        "doc_id": "DOC-AUTH-004",
        "title": "API key rotation and scope management",
        "score": 0.45,
        "content": "To rotate without downtime, create the replacement key first."
    }]
    res = route_ticket(
        intent="api_key_issue",
        urgency="medium",
        confidence=0.74,
        has_relevant_docs=True,
        customer_tier="standard",
        ticket_id="TICKET-LOW-CONF",
        retrieved_passages=retrieved,
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.LOW_CONFIDENCE.value
    assert res.escalation_packet is not None
    assert res.escalation_packet["ticket_id"] == "TICKET-LOW-CONF"
    assert len(res.escalation_packet["attempted_retrieval"]) == 1
    assert res.escalation_packet["attempted_retrieval"][0]["doc_id"] == "DOC-AUTH-004"
    assert "below threshold" in res.escalation_packet["handover_notes"]


def test_route_no_docs_escalation():
    """Tickets without authoritative docs must escalate rather than hallucinate."""
    res = route_ticket(
        intent="account_access",
        urgency="low",
        confidence=0.88,
        has_relevant_docs=False,
        customer_tier="standard",
        ticket_id="TICKET-NO-DOCS",
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.NO_RELEVANT_DOCS.value
    assert res.escalation_packet is not None


def test_route_high_risk_intents():
    """Security incidents and compliance requests must strictly escalate even with 1.0 confidence."""
    # Security incident
    r_sec = route_ticket(
        intent="security_incident",
        urgency="high",
        confidence=0.99,
        has_relevant_docs=True,
        ticket_id="TICKET-SEC-01",
    )
    assert r_sec.route == RouteDecision.ESCALATE
    assert r_sec.escalation_reason == EscalationReason.HIGH_RISK_INTENT.value

    # Compliance request
    r_comp = route_ticket(
        intent="compliance_request",
        urgency="medium",
        confidence=0.95,
        has_relevant_docs=True,
        ticket_id="TICKET-COMP-01",
    )
    assert r_comp.route == RouteDecision.ESCALATE
    assert r_comp.escalation_reason == EscalationReason.HIGH_RISK_INTENT.value

    # Data residency
    r_res = route_ticket(
        intent="data_residency",
        urgency="medium",
        confidence=0.92,
        has_relevant_docs=True,
        ticket_id="TICKET-RES-01",
    )
    assert r_res.route == RouteDecision.ESCALATE
    assert r_res.escalation_reason == EscalationReason.HIGH_RISK_INTENT.value


def test_route_policy_exclusion_flag():
    """must_not_auto_respond == True must force escalation regardless of confidence."""
    res = route_ticket(
        intent="data_export",
        urgency="low",
        confidence=0.96,
        has_relevant_docs=True,
        must_not_auto_respond=True,
        ticket_id="TICKET-POLICY-FLAG",
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.HIGH_RISK_POLICY.value


def test_route_enterprise_high_urgency():
    """Enterprise customers reporting high urgency issues fast-track to human engineers."""
    res = route_ticket(
        intent="deployment_failure",
        urgency="high",
        confidence=0.93,
        has_relevant_docs=True,
        customer_tier="enterprise",
        ticket_id="TICKET-ENTERPRISE-HIGH",
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.ENTERPRISE_HIGH_URGENCY.value


def test_route_emergency_kill_switch():
    """Emergency kill-switch must override all rules and force 100% human escalation."""
    res = route_ticket(
        intent="deployment_failure",
        urgency="low",
        confidence=0.99,
        has_relevant_docs=True,
        customer_tier="free",
        kill_switch=True,
        ticket_id="TICKET-KILL-SWITCH",
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.KILL_SWITCH_ACTIVE.value
    assert "KILL_SWITCH_OVERRIDE" in res.policy_flags


def test_route_unclear_request():
    """Ambiguous tickets with intent unclear_request must escalate."""
    res = route_ticket(
        intent="unclear_request",
        urgency="medium",
        confidence=0.40,
        has_relevant_docs=False,
        ticket_id="TICKET-UNCLEAR",
    )
    assert res.route == RouteDecision.ESCALATE
    assert res.escalation_reason == EscalationReason.UNCLEAR_REQUEST.value


def test_route_node_langgraph():
    """LangGraph route_node should properly update workflow state with decision and handover packet."""
    state = {
        "ticket_id": "LG-TICKET-01",
        "channel": "chat",
        "intent": "api_key_issue",
        "urgency": "medium",
        "classification_confidence": 0.65,
        "has_relevant_docs": True,
        "retrieved_passages": [
            {"doc_id": "DOC-AUTH-004", "title": "Key Rotation", "score": 0.42, "content": "Steps to rotate..."}
        ],
        "metadata": {
            "customer_id": "CUST-999",
            "customer_name": "Acme Corp Lead",
            "customer_tier": "standard",
            "customer_region": "europe"
        }
    }

    update = route_node(state)
    assert update["route"] == "escalate"
    assert update["escalation_reason"] == "low_confidence"
    assert update["escalation_packet"] is not None
    assert update["escalation_packet"]["customer_context"]["customer_id"] == "CUST-999"
    assert len(update["escalation_packet"]["attempted_retrieval"]) == 1
