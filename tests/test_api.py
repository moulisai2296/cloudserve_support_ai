"""Unit tests for FastAPI web service (src/api.py).

Verifies health endpoints, ticket ingestion, metrics reporting, and kill-switch controls.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "cloudserve-triage-api"


def test_triage_ticket_endpoint(client):
    payload = {
        "ticket_id": "API-TEST-01",
        "channel": "chat",
        "subject": "",
        "body": "How do I reset my MFA token?",
        "customer_id": "CUST-API-01",
        "customer_name": "API User",
        "customer_tier": "standard",
        "use_llm": False,  # Deterministic test
    }
    response = client.post("/tickets", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == "API-TEST-01"
    assert data["intent"] == "account_access"
    assert data["route"] in ("auto_respond", "escalate")
    assert "decision_id" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_decisions_logged" in data
    assert "escalation_rate_pct" in data


def test_kill_switch_controls(client):
    # Activate
    r1 = client.post("/admin/kill-switch", json={"active": True})
    assert r1.status_code == 200
    assert r1.json()["kill_switch_active"] is True

    # Check status
    r2 = client.get("/admin/kill-switch")
    assert r2.status_code == 200
    assert r2.json()["kill_switch_active"] is True

    # Deactivate
    r3 = client.post("/admin/kill-switch", json={"active": False})
    assert r3.status_code == 200
    assert r3.json()["kill_switch_active"] is False
