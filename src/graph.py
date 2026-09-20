"""LangGraph Orchestration Module for CloudServe Solutions.

Stitches all core support pipeline nodes into a unified, compiled StateGraph:
START -> retrieve -> classify -> route -> (generate -> guardrails -> log) | (log) -> END.
Provides the central execution entry point for both evaluation/harness.py and src/api.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

# ==============================================================================
# Unified Workflow State
# ==============================================================================

class SupportState(TypedDict, total=False):
    """
    Unified LangGraph State tracking a ticket from ingestion through resolution.
    Shared across retrieve, classify, route, generate, guardrails, and logging.
    """
    # Ticket Ingestion & Metadata
    ticket_id: str
    run_id: Optional[str]
    input_index: Optional[int]
    channel: str
    subject: str
    body: str
    clean_text: str
    customer_id: str
    customer_name: str
    customer_tier: str
    customer_region: str
    language_fluency: str
    metadata: Dict[str, Any]

    # Retrieval
    chroma_path: str
    retrieval_top_k: int
    retrieval_threshold: float
    retrieved_passages: List[Dict[str, Any]]
    top_score: float
    has_relevant_docs: bool
    retrieval_error: Optional[str]

    # Classification
    use_llm_classification: bool
    model_name: str
    intent: str
    urgency: str
    classification_confidence: float
    classification_alternatives: List[Dict[str, Any]]
    classification_reasoning: str

    # Routing
    routing_threshold: float
    kill_switch: bool
    route: str  # auto_respond | escalate
    escalation_reason: Optional[str]
    policy_flags: List[str]
    escalation_packet: Optional[Dict[str, Any]]

    # Generation
    use_llm_generation: bool
    generated_response: Optional[str]
    cited_doc_ids: List[str]
    generation_confidence: float
    missing_information: bool
    cannot_answer_reason: Optional[str]
    auto_respond_summary: Optional[str]

    # Guardrails
    guardrail_passed: bool
    guardrail_blocked: bool
    guardrail_block_reason: Optional[str]
    guardrail_results: Dict[str, str]
    grounding_review: Dict[str, Any]

    # Governance & Audit Store
    db_path: str
    decision_id: str
    logged_to_audit_store: bool


# Node implementations
from src.ingest import NormalizedTicket, ingest_ticket
from src.retrieve import retrieve_node
from src.classify import classify_node
from src.route import route_node
from src.generate import generate_node
from src.guardrails import guardrails_node
from src.logging_store import log_decision_node

logger = logging.getLogger(__name__)


# ==============================================================================
# Graph Construction & Conditional Edges
# ==============================================================================

def route_condition(state: SupportState) -> str:
    """Determines branching after the routing node."""
    route = state.get("route", "escalate")
    if route == "auto_respond":
        return "generate"
    # If escalated at route time, skip generation and go straight to decision logging
    return "log"


def guardrails_condition(state: SupportState) -> str:
    """Determines branching after the guardrails node."""
    # Regardless of whether guardrails passed or blocked, proceed to log
    return "log"


def create_support_graph(db_path: Optional[str] = None):
    """
    Compiles and returns the unified CloudServe Support Triage StateGraph.
    """
    workflow = StateGraph(SupportState)

    # 1. Add all nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("route", route_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("log", lambda s: log_decision_node(s, db_path=db_path))

    # 2. Wire sequential edges: START -> retrieve -> classify -> route
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "classify")
    workflow.add_edge("classify", "route")

    # 3. Conditional routing: auto_respond -> generate, escalate -> log
    workflow.add_conditional_edges(
        "route",
        route_condition,
        {
            "generate": "generate",
            "log": "log",
        }
    )

    # 4. From generate -> guardrails -> log -> END
    workflow.add_edge("generate", "guardrails")
    workflow.add_edge("guardrails", "log")
    workflow.add_edge("log", END)

    return workflow.compile()


# Global compiled instance
support_graph = create_support_graph()


# ==============================================================================
# Convenience Invocation API
# ==============================================================================

def process_ticket(
    ticket_input: Dict[str, Any] | str | NormalizedTicket,
    db_path: Optional[str] = None,
    use_llm: bool = True,
    model_name: Optional[str] = None,
    chroma_path: Optional[str] = None,
    run_id: Optional[str] = None,
    input_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function: Normalizes raw input, builds initial state,
    and runs it through the compiled LangGraph pipeline end-to-end.
    """
    if isinstance(ticket_input, NormalizedTicket):
        norm = ticket_input
    elif isinstance(ticket_input, (dict, str)):
        norm = ingest_ticket(ticket_input)
    else:
        raise ValueError(f"Unsupported ticket input type: {type(ticket_input)}")

    initial_state: SupportState = {
        "ticket_id": norm.ticket_id,
        "run_id": run_id,
        "input_index": input_index,
        "channel": norm.channel.value,
        "subject": norm.subject,
        "body": norm.body,
        "clean_text": norm.clean_text,
        "customer_id": norm.customer_id,
        "customer_name": norm.customer_name,
        "customer_tier": norm.customer_tier,
        "customer_region": norm.customer_region,
        "language_fluency": norm.language_fluency,
        # Only operational fields enter the graph. Labels and historical outcomes
        # remain on the evaluator's NormalizedTicket for scoring, never inference.
        "metadata": {
            "ticket_id": norm.ticket_id,
            "channel": norm.channel.value,
            "customer_id": norm.customer_id,
            "customer_name": norm.customer_name,
            "customer_tier": norm.customer_tier,
            "customer_region": norm.customer_region,
            "language_fluency": norm.language_fluency,
        },
        "generated_response": None,
        "cited_doc_ids": [],
        "use_llm_classification": use_llm,
        "use_llm_generation": use_llm,
        "model_name": model_name or "",
        "chroma_path": chroma_path or "",
        "db_path": db_path or "",
    }

    graph = create_support_graph(db_path=db_path) if db_path else support_graph
    final_state = graph.invoke(initial_state)
    return final_state
