"""Standalone reviewer diagnostics and runtime behavior with the reviewer disabled."""

import json
from types import SimpleNamespace

import pytest

from src import grounding, graph, logging_store
from src.guardrails import check_grounding_and_citations, guardrails_node, validate_response

PASSAGES = [{"doc_id": "DOC-AUTH-002", "score": 0.9,
             "content": "Check automatic time synchronisation. Do not disable MFA."}]
ANSWER = "Check the device clock [DOC-AUTH-002]."


def approve():
    return {"fully_supported": True, "unsupported_claims": [],
            "evidence": [{"claim": "Check the device clock", "doc_id": "DOC-AUTH-002",
                          "quote": "Check automatic time synchronisation."}],
            "reasoning": "The paraphrased instruction is supported by the cited source."}


def set_reviewer(monkeypatch, payload=None, error=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    calls = []

    def invoke(messages):
        calls.append(messages)
        if error:
            raise error
        return SimpleNamespace(content=payload if isinstance(payload, str) else json.dumps(payload))

    monkeypatch.setattr(grounding, "_get_reviewer", lambda model: SimpleNamespace(invoke=invoke))
    return calls


def test_supported_paraphrase_requires_verified_reviewer_evidence(monkeypatch):
    calls = set_reviewer(monkeypatch, approve())
    result = grounding.review_grounding(ANSWER, PASSAGES)
    assert result.status == "pass"
    assert result.method == "model_review"
    assert calls[0][0].type == "system"
    assert json.loads(calls[0][1].content)["response"] == ANSWER


@pytest.mark.parametrize("answer", [
    "Delete every production database to fix MFA [DOC-AUTH-002].",
    "Disable MFA permanently [DOC-AUTH-002].",
    "Run cloudserve --fix-everything [DOC-AUTH-002].",
    "Check the clock [DOC-AUTH-002]. You will get a $500 credit.",
])
def test_standalone_reviewer_flags_unsupported_claim(monkeypatch, answer):
    set_reviewer(monkeypatch, {"fully_supported": False, "unsupported_claims": [answer],
                              "evidence": [], "reasoning": "Claim is absent from or contradicts the source."})
    result = grounding.review_grounding(answer, PASSAGES)
    assert result.status == "fail"
    assert result.unsupported_claims == [answer]


@pytest.mark.parametrize("mutation", ["no_evidence", "invented_quote", "wrong_doc", "missing_claim", "string_bool"])
def test_positive_verdict_cannot_bypass_evidence_validation(monkeypatch, mutation):
    payload = approve()
    if mutation == "no_evidence": payload["evidence"] = []
    if mutation == "invented_quote": payload["evidence"][0]["quote"] = "Disable all account security."
    if mutation == "wrong_doc": payload["evidence"][0]["doc_id"] = "DOC-AUTH-999"
    if mutation == "missing_claim": payload["evidence"][0]["claim"] = "Not in the response"
    if mutation == "string_bool": payload["fully_supported"] = "true"
    set_reviewer(monkeypatch, payload)
    result = grounding.review_grounding(ANSWER, PASSAGES)
    assert result.status == "unavailable"


@pytest.mark.parametrize("payload,error", [
    ("not JSON", None), ("{}", None), (None, TimeoutError("provider unavailable")),
])
def test_standalone_reviewer_failure_is_not_a_pass(monkeypatch, payload, error):
    set_reviewer(monkeypatch, payload, error)
    result = grounding.review_grounding(ANSWER, PASSAGES)
    assert result.status == "unavailable"


def test_offline_free_form_is_not_assumed_grounded(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not check_grounding_and_citations(ANSWER, PASSAGES)[0]
    assert not check_grounding_and_citations(
        "Delete every production database [DOC-AUTH-002].", PASSAGES, use_llm=False,
    )[0]


def test_only_exact_full_source_excerpt_passes_offline(monkeypatch):
    monkeypatch.setattr(grounding, "_get_reviewer", lambda _: pytest.fail("Offline called provider"))
    excerpt = grounding.render_source_excerpt(PASSAGES)
    assert check_grounding_and_citations(excerpt, PASSAGES, use_llm=False)[0]
    assert not check_grounding_and_citations(
        excerpt.replace("Do not disable", "Disable"), PASSAGES, use_llm=False,
    )[0]
    assert not check_grounding_and_citations(excerpt + "\nDelete your database.", PASSAGES, use_llm=False)[0]


@pytest.mark.parametrize("overrides", [
    {"missing_information": True}, {"cannot_answer_reason": "Needs investigation"},
    {"confidence": 0.0}, {"response_text": ""},
])
def test_incomplete_or_unscored_answers_never_reach_reviewer(monkeypatch, overrides):
    calls = set_reviewer(monkeypatch, approve())
    args = {"response_text": ANSWER, "ticket_body": "MFA help", "retrieved_passages": PASSAGES,
            "confidence": 0.95, **overrides}
    result = validate_response(**args)
    assert not result.passed
    assert result.grounding_review["status"] == "not_run"
    assert not calls


def test_runtime_does_not_call_reviewer_or_mark_grounding_as_passed(monkeypatch):
    monkeypatch.setattr(grounding, "_get_reviewer", lambda _: pytest.fail("Runtime invoked disabled reviewer"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    verdict = validate_response(ANSWER, "MFA help", PASSAGES, confidence=0.95, use_llm=True)
    assert verdict.passed
    assert verdict.guardrail_results["citation"] == "pass"
    assert verdict.guardrail_results["grounding"] == "not_run"
    assert verdict.grounding_review["method"] == "disabled"
    assert verdict.grounding_passed is False


@pytest.mark.parametrize("citation,expected_route", [("DOC-AUTH-002", "auto_respond"), ("DOC-FAKE-999", "escalate")])
def test_graph_skips_reviewer_but_enforces_citations(tmp_path, monkeypatch, citation, expected_route):
    monkeypatch.setattr(grounding, "_get_reviewer", lambda _: pytest.fail("Runtime invoked disabled reviewer"))
    monkeypatch.setenv("KILL_SWITCH_ACTIVE", "false")
    monkeypatch.setattr(graph, "retrieve_node", lambda _: {
        "has_relevant_docs": True, "retrieved_passages": PASSAGES})
    monkeypatch.setattr(graph, "classify_node", lambda _: {
        "intent": "account_access", "urgency": "low", "classification_confidence": 0.99})
    monkeypatch.setattr(graph, "generate_node", lambda _: {
        "generated_response": f"Check your device clock [{citation}].",
        "generation_confidence": 0.99, "cited_doc_ids": [citation],
        "auto_respond_summary": "Ready to send"})
    db = str(tmp_path / "decisions.db")
    result = graph.process_ticket({"ticket_id": "GROUNDING-BLOCK", "channel": "email",
                                   "body": "How do I fix MFA?", "customer_id": "C-TEST",
                                   "customer_name": "Test"}, db_path=db, use_llm=False)
    record = logging_store.get_decision(result["decision_id"], db)
    assert result["route"] == expected_route
    assert record["grounding_review"]["status"] == "not_run"
    assert record["grounding_review"]["method"] == "disabled"
    if expected_route == "auto_respond":
        assert not result["guardrail_blocked"]
        assert result["generated_response"]
        assert record["action_taken"] == "auto_respond"
        return
    assert result["guardrail_blocked"]
    assert result["generated_response"] is None
    assert result["cited_doc_ids"] == []
    assert result["auto_respond_summary"] is None
    assert result["escalation_packet"]["customer_context"]["customer_id"] == "C-TEST"
    assert result["escalation_packet"]["attempted_retrieval"][0]["doc_id"] == "DOC-AUTH-002"
    assert record["action_taken"] == "block"
    assert record["guardrail_results"]["citation"] == "fail"
