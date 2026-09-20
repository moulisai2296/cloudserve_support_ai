"""Regression tests for label-free policy routing and API/batch parity."""

import json

import pytest
from fastapi.testclient import TestClient

from evaluation.harness import run_evaluation
from src import api, graph, logging_store
from src.route import route_node, route_ticket
from src.grounding import render_source_excerpt


@pytest.mark.parametrize("text,flag", [
    ("Please refund this charge.", "POLICY_FINANCIAL_DISPUTE"),
    ("I dispute this invoice.", "POLICY_FINANCIAL_DISPUTE"),
    ("We were charged twice.", "POLICY_FINANCIAL_DISPUTE"),
    ("How do I rotate a leaked API key?", "POLICY_SECURITY_EXPOSURE"),
    ("Our credentials were posted in a public repository.", "POLICY_SECURITY_EXPOSURE"),
    ("A former employee appears to still have access.", "POLICY_SECURITY_EXPOSURE"),
    ("Please delete our account.", "POLICY_ACCOUNT_OR_DATA_DELETION"),
    ("Erase my personal data.", "POLICY_ACCOUNT_OR_DATA_DELETION"),
    ("Our auditor needs access records.", "POLICY_LEGAL_OR_COMPLIANCE"),
    ("We need our data to remain within the EU.", "POLICY_LEGAL_OR_COMPLIANCE"),
    ("Can you agree to custom contract terms?", "POLICY_LEGAL_OR_COMPLIANCE"),
    ("Explain pagination. Also, our account was compromised.", "POLICY_SECURITY_EXPOSURE"),
])
def test_risky_content_overrides_routine_classification(text, flag):
    decision = route_ticket(
        intent="api_usage_question", urgency="low", confidence=0.99,
        has_relevant_docs=True, ticket_text=text, kill_switch=False,
    )
    assert decision.route == "escalate"
    assert decision.escalation_reason == "high_risk_policy"
    assert flag in decision.policy_flags
    assert flag in decision.escalation_packet["handover_notes"]


@pytest.mark.parametrize("text", [
    "Where can I download my invoice?",
    "How do I rotate an API key on a schedule?",
    "How do I export account data?",
    "How do I delete a local build cache?",
    "How do I configure a public API endpoint?",
])
def test_routine_guidance_is_not_blanket_excluded(text):
    assert route_ticket(
        intent="api_usage_question", urgency="low", confidence=0.99,
        has_relevant_docs=True, ticket_text=text, kill_switch=False,
    ).route == "auto_respond"


def test_feature_requests_require_review_even_with_retrieval_hit():
    result = route_ticket(
        intent="feature_request", urgency="low", confidence=0.99,
        has_relevant_docs=True, kill_switch=False,
    )
    assert result.route == "escalate"
    assert "POLICY_FEATURE_REQUEST_REQUIRES_HUMAN" in result.policy_flags


def test_route_node_ignores_attached_truth_labels():
    state = {
        "intent": "billing_query", "urgency": "low",
        "classification_confidence": 0.99, "has_relevant_docs": True,
        "body": "Where can I download my invoice?", "kill_switch": False,
    }
    baseline = route_node(state)
    for flag in (True, False):
        assert route_node({**state, "labels": {
            "must_not_auto_respond": flag, "expected_route": "escalate",
        }}) == baseline
    assert baseline["route"] == "auto_respond"


@pytest.mark.parametrize("subject,body,expected_route", [
    ("Invoice copy", "Where can I download my invoice?", "auto_respond"),
    ("Please refund this charge", "See invoice 123.", "escalate"),
    ("API question", "Explain pagination. Our credentials were leaked.", "escalate"),
])
def test_api_and_batch_match_without_truth_access(
    tmp_path, monkeypatch, subject, body, expected_route,
):
    # Keep both model and retrieval outputs fixed to isolate routing behavior.
    # Real graph branching, guardrails, SQLite logging, API and harness execute.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KILL_SWITCH_ACTIVE", "false")
    db_path = str(tmp_path / "decisions.db")
    monkeypatch.setattr(logging_store, "DEFAULT_DB_PATH", db_path)
    observed_states = []
    passages = [{"doc_id": "DOC-BILL-001", "score": 0.9,
                 "content": "Download the invoice from the billing page."}]

    def retrieve(state):
        observed_states.append(dict(state))
        return {"has_relevant_docs": True, "retrieved_passages": passages}

    def classify(state):
        observed_states.append(dict(state))
        return {"intent": "billing_query", "urgency": "low",
                "classification_confidence": 0.99}

    def generate(state):
        observed_states.append(dict(state))
        return {"generated_response": render_source_excerpt(passages),
                "cited_doc_ids": ["DOC-BILL-001"], "generation_confidence": 0.99}

    monkeypatch.setattr(graph, "retrieve_node", retrieve)
    monkeypatch.setattr(graph, "classify_node", classify)
    monkeypatch.setattr(graph, "generate_node", generate)
    monkeypatch.setattr(graph, "support_graph", graph.create_support_graph())

    ticket = {"ticket_id": "PARITY", "channel": "email", "subject": subject,
              "body": body, "customer_id": "C-TEST", "customer_name": "Test"}
    variants = [ticket, {**ticket, "labels": {"must_not_auto_respond": True,
                 "intent": "security_incident", "expected_route": "escalate"}},
                {**ticket, "labels": {"must_not_auto_respond": False,
                 "intent": "billing_query", "expected_route": "auto_respond"},
                 "history": {"first_contact_resolution": True},
                 "metadata": {"labels": {"must_not_auto_respond": True}}}]
    results = [graph.process_ticket(t, use_llm=False) for t in variants]
    assert {r["route"] for r in results} == {expected_route}
    assert all(r["policy_flags"] == results[0]["policy_flags"] for r in results)

    input_path = tmp_path / "tickets.json"
    input_path.write_text(json.dumps(variants), encoding="utf-8")
    metrics = run_evaluation(str(input_path), str(tmp_path / "report.json"),
                             use_llm=False, db_path=db_path)
    saved = json.loads((tmp_path / "latest_ticket_results.json").read_text())
    assert metrics["total_tickets_evaluated"] == 3
    assert {r["execution"]["route_decision"] for r in saved["tickets"]} == {expected_route}
    # Ground truth remains available for scoring despite its absence at runtime.
    assert saved["tickets"][1]["ground_truth_labels"]["intent"] == "security_incident"

    with TestClient(api.app) as client:
        for payload in variants:
            response = client.post("/tickets", json={**payload, "use_llm": False})
            assert response.status_code == 200
            assert response.json()["route"] == expected_route
            assert response.json()["policy_flags"] == results[0]["policy_flags"]

    assert observed_states
    for state in observed_states:
        assert "labels" not in state
        assert "history" not in state
        assert "labels" not in state["metadata"]
