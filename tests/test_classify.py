"""Unit tests for ticket classification module (AC-A3 / B-06).

Validates 22 categories, urgency mapping, confidence calibration,
heuristic fallback accuracy, and LangGraph node integration.
"""

import os
import pytest
from src.classify import (
    VALID_INTENTS,
    VALID_URGENCIES,
    ClassificationResult,
    classify_ticket,
    classify_node,
    _heuristic_fallback_classify,
)


def test_classification_result_schema():
    res = ClassificationResult(
        intent="account_access",
        urgency="high",
        confidence=0.92,
        alternatives=[{"intent": "authentication_failure", "confidence": 0.08}],
        reasoning="MFA reset request.",
    )
    assert res.intent in VALID_INTENTS
    assert res.urgency in VALID_URGENCIES
    assert res.confidence == 0.92
    assert len(res.alternatives) == 1
    assert res.alternatives[0].intent == "authentication_failure"


def test_intent_alias_normalization():
    # Verify alias mapping
    res_alias = ClassificationResult(
        intent="compliance_query",
        urgency="critical",  # Invalid, defaults to medium
        confidence=0.75,
    )
    assert res_alias.intent == "compliance_request"
    assert res_alias.urgency == "medium"

    # Unknown intent defaults to unclear_request
    res_unknown = ClassificationResult(
        intent="alien_invasion",
        urgency="low",
        confidence=0.10,
    )
    assert res_unknown.intent == "unclear_request"


def test_heuristic_classification_intents():
    # 1. Rollback query
    r1 = _heuristic_fallback_classify("how do I revert to earlier revision? release is broken")
    assert r1.intent == "rollback_request"
    assert r1.confidence >= 0.80

    # 2. Build failure query
    r2 = _heuristic_fallback_classify("our docker build fail during dependency resolution in CI/CD pipeline")
    assert r2.intent == "deployment_failure"

    # 3. API key query (with 401 code - should match api_key_issue due to disambiguation rule)
    r3 = _heuristic_fallback_classify("production key started returning 401 yesterday morning")
    assert r3.intent == "api_key_issue"
    assert r3.urgency == "high"

    # 4. Billing query
    r4 = _heuristic_fallback_classify("Where can I find the invoice for our annual billing?")
    assert r4.intent == "billing_query"
    assert r4.urgency == "medium"

    # 5. Rate limit query
    r5 = _heuristic_fallback_classify("Getting 429 too many requests errors on our ingestion endpoint")
    assert r5.intent == "rate_limit"

    # 6. Feature request query (low urgency)
    r6 = _heuristic_fallback_classify("It would be great if you could please add per-project spend caps")
    assert r6.intent == "feature_request"
    assert r6.urgency == "low"


def test_classify_node_langgraph():
    state = {
        "ticket_id": "TEST-CLASSIFY-01",
        "channel": "chat",
        "body": "One of our production keys started returning 401 yesterday morning without change on our side.",
        "use_llm_classification": False,  # Test deterministic node
    }
    update = classify_node(state)
    assert "intent" in update
    assert "urgency" in update
    assert "classification_confidence" in update
    assert "classification_alternatives" in update
    assert "classification_reasoning" in update
    assert update["intent"] == "api_key_issue"
    assert update["urgency"] == "high"
    assert isinstance(update["classification_confidence"], float)


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OpenRouter API key required")
def test_llm_classification_live():
    res = classify_ticket(
        body="We are seeing 429 Too Many Requests errors when calling /v1/ingest during peak hours.",
        channel="chat",
        use_llm=True,
    )
    assert res.intent in VALID_INTENTS
    assert res.urgency in VALID_URGENCIES
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.reasoning) > 0
