"""Unit tests for the unified LangGraph support pipeline (src/graph.py).

Verifies end-to-end execution across retrieval, classification, routing,
generation, guardrails, and decision logging.
"""

import tempfile
from pathlib import Path
import pytest

from src.graph import create_support_graph, process_ticket
from src.logging_store import get_decisions_by_ticket


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_graph_auto_respond_flow(temp_db):
    """End-to-end test for an answerable ticket routing to auto_respond."""
    raw_ticket = {
        "ticket_id": "GRAPH-TEST-01",
        "channel": "chat",
        "subject": "",
        "body": "How do I reset my MFA after changing my phone?",
        "customer_id": "CUST-100",
        "customer_name": "David Miller",
        "customer_tier": "standard",
        "customer_region": "north_america",
    }

    result = process_ticket(
        raw_ticket,
        db_path=temp_db,
        use_llm=False,  # Deterministic test
    )

    assert result["ticket_id"] == "GRAPH-TEST-01"
    assert result["intent"] == "account_access"
    assert result["has_relevant_docs"] is True
    assert result["route"] == "auto_respond"
    assert result["generated_response"] is not None
    assert "DOC-AUTH-002" in result["cited_doc_ids"]
    assert result["guardrail_passed"] is True
    assert result["logged_to_audit_store"] is True

    # Check SQLite audit database
    records = get_decisions_by_ticket("GRAPH-TEST-01", db_path=temp_db)
    assert len(records) == 1
    assert records[0]["action_taken"] == "auto_respond"


def test_graph_escalate_flow(temp_db):
    """End-to-end test for a sensitive security ticket routing to escalate."""
    raw_ticket = {
        "ticket_id": "GRAPH-TEST-02",
        "channel": "email",
        "subject": "Possible account breach",
        "body": "We detected suspicious login activity and believe our credentials were leaked.",
        "customer_id": "CUST-200",
        "customer_name": "Eve Security",
        "customer_tier": "business",
        "customer_region": "europe",
    }

    result = process_ticket(
        raw_ticket,
        db_path=temp_db,
        use_llm=False,
    )

    assert result["ticket_id"] == "GRAPH-TEST-02"
    assert result["intent"] == "security_incident"
    assert result["route"] == "escalate"
    assert result["escalation_reason"] == "high_risk_intent"
    assert result.get("generated_response") is None  # Skipped!
    assert result["escalation_packet"] is not None
    assert result["logged_to_audit_store"] is True

    records = get_decisions_by_ticket("GRAPH-TEST-02", db_path=temp_db)
    assert len(records) == 1
    assert records[0]["action_taken"] == "escalate"
