"""Routing module for CloudServe Solutions support tickets.

Implements confidence-based, policy-based, and governance-driven routing logic.
Directs tickets to 'auto_respond' or 'escalate', compiling rich Tier-2
escalation handover packets with source references.
Adheres to Capstone Specification AC-A5, B-07, and FR-06.
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

# Intents that must strictly never be automated due to security, legal, or compliance risks
HIGH_RISK_INTENTS = {
    "security_incident",
    "compliance_request",
    "data_residency"
}

# Conservative operational exclusions from the discovery/PRD. These inspect
# customer text, not evaluation labels, and also cover mixed-intent tickets
# whose primary classification may be routine. Regex rules are not exhaustive
# semantic risk detection; ambiguous cases still rely on the confidence gate.
CONTENT_POLICY_RULES = {
    "POLICY_SECURITY_EXPOSURE": re.compile(
        r"\b(?:compromis\w*|breach\w*|account takeover|unauthori[sz]ed access|"
        r"suspicious (?:login|activity)|former employee.{0,60}still.{0,20}access)\b|"
        r"\b(?:exposed|leaked|stolen)\b.{0,60}\b(?:key|keys|secret|secrets|credentials?|tokens?|passwords?)\b|"
        r"\b(?:key|keys|secret|secrets|credentials?|tokens?|passwords?)\b.{0,60}\b(?:exposed|leaked|stolen|public)\b",
        re.IGNORECASE,
    ),
    "POLICY_FINANCIAL_DISPUTE": re.compile(
        r"\b(?:refund\w*|chargebacks?|billing dispute|disput\w*\b.{0,30}\b(?:charge|invoice|payment)s?|"
        r"(?:charge|invoice|payment)s? .{0,30}disput\w*|charged (?:twice|incorrectly)|"
        r"duplicate charge|unauthori[sz]ed (?:charge|payment))\b",
        re.IGNORECASE,
    ),
    "POLICY_ACCOUNT_OR_DATA_DELETION": re.compile(
        r"\b(?:delet\w*|eras\w*|clos\w*)\b.{0,40}\b(?:account|organisation|organization|customer data|personal data|all (?:our|my) data)\b|"
        r"\b(?:account|personal data)\b.{0,30}\b(?:deletion|erasure|closure)\b",
        re.IGNORECASE,
    ),
    "POLICY_LEGAL_OR_COMPLIANCE": re.compile(
        r"\b(?:gdpr|hipaa|soc ?2|compliance|auditor|legal dispute|lawsuit|"
        r"custom contract|contract terms|data residency|data locality)\b|"
        r"\bdata\b.{0,40}\b(?:stay|remain|stored|reside)\b.{0,30}\b(?:eu|europe|region)\b",
        re.IGNORECASE,
    ),
}


def content_policy_flags(ticket_text: str) -> List[str]:
    """Return explainable exclusions derived only from the submitted text."""
    text = " ".join(ticket_text.split())
    return [flag for flag, pattern in CONTENT_POLICY_RULES.items() if pattern.search(text)]


class RouteDecision(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class EscalationReason(str, Enum):
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    HIGH_RISK_INTENT = "high_risk_intent"
    HIGH_RISK_POLICY = "high_risk_policy"
    ENTERPRISE_HIGH_URGENCY = "enterprise_high_urgency"
    UNCLEAR_REQUEST = "unclear_request"
    LOW_CONFIDENCE = "low_confidence"
    NO_RELEVANT_DOCS = "no_relevant_docs"


class EscalationPacket(BaseModel):
    """Contextual handover package assembled for Tier-2 human engineers (FR-06 / AC-A5)."""
    ticket_id: str
    escalation_reason: str
    policy_flags: List[str] = Field(default_factory=list)
    intent: str
    urgency: str
    confidence: float
    customer_context: Dict[str, Any] = Field(default_factory=dict)
    attempted_retrieval: List[Dict[str, Any]] = Field(default_factory=list)
    issue_summary: Optional[str] = None
    handover_notes: str = ""


class RoutingResult(BaseModel):
    route: RouteDecision
    escalation_reason: Optional[str] = None
    policy_flags: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    escalation_packet: Optional[Dict[str, Any]] = None


def build_escalation_packet(
    ticket_id: str,
    escalation_reason: str,
    policy_flags: List[str],
    intent: str,
    urgency: str,
    confidence: float,
    retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    customer_context: Optional[Dict[str, Any]] = None,
    classification_reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """Assembles structured contextual handover payload for Tier-2 human engineers."""
    passages = retrieved_passages or []
    cust_context = customer_context or {}

    retrieval_summaries = []
    for p in passages[:3]:
        retrieval_summaries.append({
            "doc_id": p.get("doc_id", "UNKNOWN"),
            "title": p.get("title", ""),
            "score": p.get("score", 0.0),
            "snippet": (p.get("content", "")[:200] + "...") if len(p.get("content", "")) > 200 else p.get("content", "")
        })

    # Human-readable handover explanation
    if escalation_reason == EscalationReason.KILL_SWITCH_ACTIVE.value:
        notes = "Emergency kill-switch is active. All automated processing halted; routing to Tier-2 team."
    elif escalation_reason == EscalationReason.HIGH_RISK_INTENT.value:
        notes = f"High-risk intent detected ('{intent}'). Policy strictly prohibits automated response."
    elif escalation_reason == EscalationReason.HIGH_RISK_POLICY.value:
        notes = f"Human review required by operational policy: {', '.join(policy_flags)}."
    elif escalation_reason == EscalationReason.ENTERPRISE_HIGH_URGENCY.value:
        notes = "Enterprise customer reporting high-urgency issue. Fast-tracked to senior engineer for SLA compliance."
    elif escalation_reason == EscalationReason.LOW_CONFIDENCE.value:
        notes = f"Classifier confidence ({confidence:.2f}) below threshold ({DEFAULT_CONFIDENCE_THRESHOLD:.2f}). Human verification required."
    elif escalation_reason == EscalationReason.NO_RELEVANT_DOCS.value:
        notes = f"No authoritative documentation found above relevance threshold for intent '{intent}'."
    elif escalation_reason == EscalationReason.UNCLEAR_REQUEST.value:
        notes = "Customer request is ambiguous or lacks necessary technical details to classify intent."
    else:
        notes = f"Escalated due to: {escalation_reason}."

    # Combine policy rule with LLM technical reasoning summary
    if classification_reasoning and classification_reasoning.strip():
        notes = f"{notes} | Issue Analysis: {classification_reasoning.strip()}"

    packet = EscalationPacket(
        ticket_id=ticket_id,
        escalation_reason=escalation_reason,
        policy_flags=policy_flags,
        intent=intent,
        urgency=urgency,
        confidence=confidence,
        customer_context=cust_context,
        attempted_retrieval=retrieval_summaries,
        issue_summary=classification_reasoning.strip() if classification_reasoning else None,
        handover_notes=notes,
    )
    return packet.model_dump()


def route_ticket(
    intent: str,
    urgency: str,
    confidence: float,
    has_relevant_docs: bool,
    customer_tier: str = "standard",
    must_not_auto_respond: bool = False,
    kill_switch: Optional[bool] = None,
    threshold: Optional[float] = None,
    ticket_id: str = "TICKET-UNKNOWN",
    retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    customer_context: Optional[Dict[str, Any]] = None,
    classification_reasoning: Optional[str] = None,
    ticket_text: str = "",
) -> RoutingResult:
    """
    Evaluates ticket attributes against policy rules to determine target route.
    If escalated, automatically compiles a rich contextual escalation packet.
    must_not_auto_respond is an explicit trusted caller override, never a
    dataset label. API and batch graph calls derive exclusions from ticket_text.
    """
    active_threshold = threshold if threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
    is_kill_switch = (
        kill_switch
        if kill_switch is not None
        else os.getenv("KILL_SWITCH_ACTIVE", "false").strip().lower() in ("true", "1", "yes")
    )

    policy_flags: List[str] = []

    def _create_packet(reason: str) -> Dict[str, Any]:
        return build_escalation_packet(
            ticket_id=ticket_id,
            escalation_reason=reason,
            policy_flags=policy_flags,
            intent=intent,
            urgency=urgency,
            confidence=confidence,
            retrieved_passages=retrieved_passages,
            customer_context=customer_context,
            classification_reasoning=classification_reasoning,
        )

    # Rule 1: Emergency Kill-Switch override (100% human escalation)
    if is_kill_switch:
        policy_flags.append("KILL_SWITCH_OVERRIDE")
        packet = _create_packet(EscalationReason.KILL_SWITCH_ACTIVE.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.KILL_SWITCH_ACTIVE.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 2: High-risk security
    if intent in HIGH_RISK_INTENTS:
        policy_flags.append(f"HIGH_RISK_INTENT_{intent.upper()}")
        packet = _create_packet(EscalationReason.HIGH_RISK_INTENT.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.HIGH_RISK_INTENT.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 3: Observable policy exclusions and optional trusted caller override.
    policy_flags.extend(content_policy_flags(ticket_text))
    if intent == "feature_request":
        policy_flags.append("POLICY_FEATURE_REQUEST_REQUIRES_HUMAN")
    if must_not_auto_respond:
        policy_flags.append("TRUSTED_MANUAL_REVIEW_OVERRIDE")
    if policy_flags:
        packet = _create_packet(EscalationReason.HIGH_RISK_POLICY.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.HIGH_RISK_POLICY.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 4: Enterprise customer reporting high urgency (SLA protection)
    if customer_tier.lower() == "enterprise" and urgency.lower() == "high":
        policy_flags.append("ENTERPRISE_HIGH_URGENCY_SLA")
        packet = _create_packet(EscalationReason.ENTERPRISE_HIGH_URGENCY.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.ENTERPRISE_HIGH_URGENCY.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 5: Unclear or ambiguous request
    if intent == "unclear_request":
        policy_flags.append("UNCLEAR_INTENT")
        packet = _create_packet(EscalationReason.UNCLEAR_REQUEST.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.UNCLEAR_REQUEST.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 6: Low classification confidence below calibrated threshold
    if confidence < active_threshold:
        policy_flags.append("CONFIDENCE_BELOW_THRESHOLD")
        packet = _create_packet(EscalationReason.LOW_CONFIDENCE.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.LOW_CONFIDENCE.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # Rule 7: Missing knowledge base ground truth / retrieval failure
    if not has_relevant_docs:
        policy_flags.append("NO_RELEVANT_DOCUMENTATION")
        packet = _create_packet(EscalationReason.NO_RELEVANT_DOCS.value)
        return RoutingResult(
            route=RouteDecision.ESCALATE,
            escalation_reason=EscalationReason.NO_RELEVANT_DOCS.value,
            policy_flags=policy_flags,
            confidence=confidence,
            threshold=active_threshold,
            escalation_packet=packet,
        )

    # All safety criteria satisfied: Safe to auto-respond
    policy_flags.append("ROUTED_FOR_AUTONOMOUS_RESOLUTION")
    return RoutingResult(
        route=RouteDecision.AUTO_RESPOND,
        escalation_reason=None,
        policy_flags=policy_flags,
        confidence=confidence,
        threshold=active_threshold,
        escalation_packet=None,
    )


# ==============================================================================
# LangGraph Node
# ==============================================================================

def route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph Node: Determines whether the ticket is routed to auto-respond or escalate."""
    ticket_id = state.get("ticket_id", "UNKNOWN")
    intent = state.get("intent", "unclear_request")
    urgency = state.get("urgency", "medium")
    confidence = float(state.get("classification_confidence", 0.0))
    classification_reasoning = state.get("classification_reasoning")
    has_relevant_docs = bool(state.get("has_relevant_docs", False))
    retrieved_passages = state.get("retrieved_passages", [])

    metadata = state.get("metadata", {})
    customer_tier = metadata.get("customer_tier") or state.get("customer_tier", "standard")

    kill_switch = state.get("kill_switch")
    threshold = state.get("routing_threshold")

    customer_context = {
        "customer_id": metadata.get("customer_id") or state.get("customer_id", "UNKNOWN"),
        "customer_name": metadata.get("customer_name") or state.get("customer_name", "UNKNOWN"),
        "customer_tier": customer_tier,
        "customer_region": metadata.get("customer_region") or state.get("customer_region", "global"),
        "channel": state.get("channel", "email"),
    }

    decision = route_ticket(
        intent=intent,
        urgency=urgency,
        confidence=confidence,
        has_relevant_docs=has_relevant_docs,
        customer_tier=customer_tier,
        ticket_text=f"{state.get('subject', '')}\n{state.get('body', '')}",
        kill_switch=kill_switch,
        threshold=threshold,
        ticket_id=ticket_id,
        retrieved_passages=retrieved_passages,
        customer_context=customer_context,
        classification_reasoning=classification_reasoning,
    )

    return {
        "route": decision.route.value,
        "escalation_reason": decision.escalation_reason,
        "policy_flags": decision.policy_flags,
        "escalation_packet": decision.escalation_packet,
    }
