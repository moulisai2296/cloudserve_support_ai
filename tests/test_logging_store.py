"""Unit tests for decision logging audit store (AC-A8 / B-10).

Validates schema compliance with Governance Framework Section 1,
SQLite persistence, coverage reconciliation, and LangGraph node integration.
"""

import tempfile
from pathlib import Path
import pytest

from src.logging_store import (
    DecisionRecord,
    init_db,
    log_decision,
    get_decision,
    get_decisions_by_ticket,
    count_decisions,
    reconcile_decisions_with_tickets,
    log_decision_node,
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_insert_and_retrieve_decision(temp_db):
    record = DecisionRecord(
        ticket_id="TICKET-LOG-01",
        stage="routing",
        input_summary="Build failure during docker push",
        action_taken="auto_respond",
        reason="Confidence 0.92 exceeds 0.80 threshold and DOC-DEPLOY-001 found.",
        prediction={"value": "deployment_failure", "confidence": 0.92},
        alternatives=[{"value": "configuration_help", "confidence": 0.08}],
        sources_used=[{"doc_id": "DOC-DEPLOY-001", "score": 0.45}],
        threshold_applied=0.80,
    )

    decision_id = log_decision(record, db_path=temp_db)
    assert decision_id.startswith("DEC-")

    retrieved = get_decision(decision_id, db_path=temp_db)
    assert retrieved is not None
    assert retrieved["ticket_id"] == "TICKET-LOG-01"
    assert retrieved["action_taken"] == "auto_respond"
    assert retrieved["prediction"]["value"] == "deployment_failure"
    assert retrieved["sources_used"][0]["doc_id"] == "DOC-DEPLOY-001"


def test_minimum_record_schema_compliance(temp_db):
    """Verifies all mandatory fields from Governance Framework Section 1 exist in record."""
    record = DecisionRecord(
        ticket_id="TICKET-SCHEMA-01",
        action_taken="escalate",
        reason="Low classification confidence",
    )
    decision_id = log_decision(record, db_path=temp_db)
    data = get_decision(decision_id, db_path=temp_db)

    mandatory_fields = [
        "decision_id",
        "timestamp",
        "ticket_id",
        "stage",
        "input_summary",
        "model",
        "prediction",
        "alternatives",
        "sources_used",
        "threshold_applied",
        "action_taken",
        "reason",
        "guardrail_results",
        "prompt_version",
        "requirement_ids",
    ]

    for field in mandatory_fields:
        assert field in data, f"Missing mandatory governance field: {field}"


def test_reconciliation_check(temp_db):
    """Verifies 1:1 coverage reconciliation against processed tickets."""
    ticket_ids = ["T-01", "T-02", "T-03"]
    for tid in ticket_ids:
        log_decision(
            DecisionRecord(
                ticket_id=tid,
                action_taken="auto_respond",
                reason="Pass",
            ),
            db_path=temp_db
        )

    # 1. Exact match
    res = reconcile_decisions_with_tickets(ticket_ids, db_path=temp_db)
    assert res["is_reconciled"] is True
    assert res["processed_ticket_count"] == 3
    assert res["logged_ticket_count"] == 3
    assert len(res["missing_ticket_ids"]) == 0

    # 2. Missing ticket detection
    res_missing = reconcile_decisions_with_tickets(["T-01", "T-02", "T-03", "T-04"], db_path=temp_db)
    assert res_missing["is_reconciled"] is False
    assert res_missing["missing_ticket_ids"] == ["T-04"]


def test_log_decision_node_langgraph(temp_db):
    """Tests LangGraph node integration updating state with decision_id."""
    state = {
        "ticket_id": "LG-TICKET-02",
        "route": "auto_respond",
        "clean_text": "How do I setup MFA?",
        "intent": "account_access",
        "classification_confidence": 0.94,
        "urgency": "medium",
        "retrieved_passages": [
            {"doc_id": "DOC-AUTH-002", "score": 0.48}
        ],
        "routing_threshold": 0.80,
    }

    update = log_decision_node(state, db_path=temp_db)
    assert update["logged_to_audit_store"] is True
    assert update["decision_id"].startswith("DEC-")

    stored = get_decisions_by_ticket("LG-TICKET-02", db_path=temp_db)
    assert len(stored) == 1
    assert stored[0]["action_taken"] == "auto_respond"
    assert stored[0]["prediction"]["value"] == "account_access"
