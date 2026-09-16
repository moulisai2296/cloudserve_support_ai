"""
tests/test_retrieve.py — Verification of Retrieval Layer & LangGraph Integration (AC-A4)
"""

import pytest
import os
from src.retrieve import (
    build_knowledge_base_index,
    retrieve_passages,
    retrieve_node,
    create_retrieval_graph,
    SupportState
)

TEST_CHROMA_PATH = "./storage/test_chroma"


@pytest.fixture(scope="session", autouse=True)
def setup_test_index():
    """Builds test Chroma index once for test session."""
    build_knowledge_base_index(chroma_path=TEST_CHROMA_PATH, force_rebuild=True)
    yield
    # Cleanup optional


def test_build_index():
    """Verify all 29 articles are chunked and indexed."""
    count = build_knowledge_base_index(chroma_path=TEST_CHROMA_PATH, force_rebuild=False)
    assert count > 29, f"Expected more than 29 chunks, got {count}"


def test_retrieval_accuracy_known_query():
    """
    Test known query maps to the expected document.
    'invalid credentials on login' should retrieve DOC-AUTH-001 with high score.
    """
    query = "invalid credentials on login even with correct password"
    results = retrieve_passages(query, top_k=3, threshold=0.45, chroma_path=TEST_CHROMA_PATH)
    
    assert len(results) > 0, "Expected at least one passage retrieved"
    top_doc = results[0]
    assert top_doc["doc_id"] == "DOC-AUTH-001", f"Expected DOC-AUTH-001, got {top_doc['doc_id']}"
    assert top_doc["score"] >= 0.50
    assert "Invalid credentials" in top_doc["content"]


def test_retrieval_deployment_query():
    """Query about dependency build failure should retrieve DOC-DEPLOY-003."""
    query = "builds fail during dependency resolution in CI/CD pipeline"
    results = retrieve_passages(query, top_k=3, threshold=0.40, chroma_path=TEST_CHROMA_PATH)
    
    assert len(results) > 0
    retrieved_doc_ids = [p["doc_id"] for p in results]
    assert "DOC-DEPLOY-003" in retrieved_doc_ids


def test_retrieval_threshold_filters_gibberish():
    """
    AC-A4 Requirement:
    'Applies a relevance threshold and returns nothing rather than something irrelevant.'
    """
    gibberish = "asdfghjkl qwertyuiop zxcvbnm completely unrelated gibberish xyz"
    results = retrieve_passages(gibberish, top_k=3, threshold=0.65, chroma_path=TEST_CHROMA_PATH)
    assert len(results) == 0, f"Expected 0 results for gibberish, got {len(results)}"


def test_langgraph_retrieve_node():
    """Verify retrieve_node updates LangGraph state correctly."""
    initial_state: SupportState = {
        "ticket_id": "TEST-001",
        "channel": "chat",
        "body": "How do I reset my MFA token after changing my phone?",
        "clean_text": "How do I reset my MFA token after changing my phone?",
        "chroma_path": TEST_CHROMA_PATH
    }
    
    update = retrieve_node(initial_state)
    assert "retrieved_passages" in update
    assert "top_score" in update
    assert update["has_relevant_docs"] is True
    assert len(update["retrieved_passages"]) > 0


def test_langgraph_compiled_graph():
    """Verify compiled LangGraph StateGraph executes end-to-end."""
    graph = create_retrieval_graph()
    
    state: SupportState = {
        "ticket_id": "TEST-002",
        "channel": "email",
        "subject": "Need to export customer audit logs",
        "body": "Our auditor requires access logs for SOC2 compliance.",
        "clean_text": "Need to export customer audit logs\n\nOur auditor requires access logs for SOC2 compliance.",
        "chroma_path": TEST_CHROMA_PATH
    }
    
    final_state = graph.invoke(state)
    assert final_state["has_relevant_docs"] is True
    assert len(final_state["retrieved_passages"]) > 0
    assert final_state["top_score"] > 0.40
