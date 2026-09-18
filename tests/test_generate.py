"""Unit tests for answer generation module (AC-A6 / B-08).

Validates passage formatting, [DOC-ID] citation extraction,
channel tone handling, LangGraph node integration, and live LLM grounding.
"""

import os
import pytest
from src.generate import (
    GenerationResult,
    extract_citations,
    format_retrieved_passages,
    generate_response,
    generate_node,
    _heuristic_fallback_generate,
)


def test_extract_citations():
    text = (
        "Please follow the MFA setup instructions in [DOC-AUTH-002]. "
        "If credentials fail, refer to [DOC-AUTH-001]. "
        "Remember that [DOC-AUTH-002] also explains clock drift."
    )
    citations = extract_citations(text)
    assert citations == ["DOC-AUTH-002", "DOC-AUTH-001"]


def test_format_retrieved_passages():
    passages = [
        {
            "doc_id": "DOC-AUTH-002",
            "title": "MFA Setup and Recovery",
            "content": "Step 1: Check device clock synchronization.",
        },
        {
            "doc_id": "DOC-AUTH-001",
            "title": "Invalid Credentials on Login",
            "content": "Step 1: Confirm whether account is locked.",
        }
    ]
    formatted = format_retrieved_passages(passages)
    assert "--- Document [DOC-AUTH-002]: MFA Setup and Recovery ---" in formatted
    assert "Step 1: Check device clock synchronization." in formatted
    assert "--- Document [DOC-AUTH-001]: Invalid Credentials on Login ---" in formatted


def test_heuristic_fallback_generation():
    passages = [{
        "doc_id": "DOC-AUTH-002",
        "title": "MFA Setup",
        "content": "Check device time drift before attempting code entry."
    }]
    res = _heuristic_fallback_generate(
        customer_name="Alice",
        subject="MFA failing",
        body="My MFA code is rejected.",
        passages=passages,
        channel="email",
    )
    assert "[DOC-AUTH-002]" in res.response_text
    assert "DOC-AUTH-002" in res.cited_doc_ids
    assert res.confidence >= 0.80
    assert not res.missing_information


def test_generate_node_bypass_on_escalate():
    """Tickets routed to escalate must not run automated generation."""
    state = {
        "ticket_id": "T-ESC-01",
        "route": "escalate",
        "retrieved_passages": [{"doc_id": "DOC-001", "content": "..."}],
    }
    update = generate_node(state)
    assert update["generated_response"] is None
    assert update["cited_doc_ids"] == []


def test_generate_node_auto_respond():
    """Tickets routed to auto_respond must generate grounded response."""
    state = {
        "ticket_id": "T-AUTO-01",
        "route": "auto_respond",
        "channel": "chat",
        "customer_name": "Bob",
        "body": "How do I reset my MFA?",
        "retrieved_passages": [{
            "doc_id": "DOC-AUTH-002",
            "title": "MFA Setup and Recovery",
            "content": "Ask user to check automatic time synchronisation is enabled."
        }],
        "use_llm_generation": False,  # Deterministic test
    }
    update = generate_node(state)
    assert update["generated_response"] is not None
    assert "DOC-AUTH-002" in update["cited_doc_ids"]


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OpenRouter API key required")
def test_llm_generation_live():
    """Live end-to-end grounding test via OpenRouter."""
    passages = [{
        "doc_id": "DOC-AUTH-002",
        "title": "Multi-factor authentication setup and recovery",
        "content": (
            "## Resolution\n"
            "1. Ask the user to check that automatic time synchronisation is enabled on the device generating the code. Drift is the most common cause by a wide margin.\n"
            "2. If the device is lost, the user should sign in with one of their recovery codes and then re-enrol a new device from the security page."
        )
    }]

    res = generate_response(
        ticket_id="LIVE-TEST-01",
        body="My authenticator app codes are constantly being rejected as invalid. What should I do?",
        subject="Authenticator code invalid",
        channel="chat",
        customer_name="David",
        retrieved_passages=passages,
        use_llm=True,
    )

    assert len(res.response_text) > 50
    assert "DOC-AUTH-002" in res.cited_doc_ids
    assert "[DOC-AUTH-002]" in res.response_text
    assert res.confidence >= 0.70
