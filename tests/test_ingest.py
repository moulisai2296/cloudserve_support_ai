"""Unit tests for ticket ingestion module (AC-A2 / B-02).

Validates all 4 channels (email, chat, docs_comment, forum), metadata preservation,
validation error handling, and 100% successful parsing over development and validation datasets.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.ingest import ChannelType, NormalizedTicket, ingest_ticket, load_tickets_from_json


def test_ingest_email_ticket():
    raw_email = {
        "ticket_id": "TEST-EMAIL-01",
        "channel": "email",
        "subject": "Need help with VPC peering",
        "body": "Hi Support team,\r\nour VPC peering connection is dropping packets.",
        "received_at": "2026-06-01T10:00:00Z",
        "customer_id": "CUST-9999",
        "customer_name": "Dev Ops Lead",
        "customer_tier": "enterprise",
        "customer_region": "north_america",
        "language_fluency": "fluent",
    }
    ticket = ingest_ticket(raw_email)
    assert ticket.ticket_id == "TEST-EMAIL-01"
    assert ticket.channel == ChannelType.EMAIL
    assert ticket.clean_text == "Need help with VPC peering\nHi Support team,\nour VPC peering connection is dropping packets."
    assert ticket.metadata["customer_tier"] == "enterprise"
    assert ticket.metadata["customer_region"] == "north_america"


def test_ingest_chat_ticket_empty_subject():
    raw_chat = {
        "ticket_id": "TEST-CHAT-01",
        "channel": "chat",
        "subject": "",
        "body": "Hey, how do I reset my MFA?",
        "received_at": "2026-06-01T10:05:00Z",
        "customer_id": "CUST-1001",
        "customer_name": "Alice Smith",
        "customer_tier": "standard",
        "customer_region": "europe",
        "language_fluency": "fluent",
    }
    ticket = ingest_ticket(raw_chat)
    assert ticket.channel == ChannelType.CHAT
    assert ticket.clean_text == "Hey, how do I reset my MFA?"
    assert ticket.metadata["customer_id"] == "CUST-1001"


def test_ingest_docs_comment_ticket():
    raw_docs = {
        "ticket_id": "TEST-DOCS-01",
        "channel": "docs_comment",
        "subject": "Doc feedback",
        "body": "Step 3 in DOC-DEPLOY-001 has a broken link to kubectl config.",
        "received_at": "2026-06-01T10:10:00Z",
        "customer_id": "CUST-2002",
        "customer_name": "Bob Reader",
        "customer_tier": "business",
        "customer_region": "asia_pacific",
        "language_fluency": "non_fluent",
    }
    ticket = ingest_ticket(raw_docs)
    assert ticket.channel == ChannelType.DOCS_COMMENT
    assert ticket.clean_text == "Doc feedback\nStep 3 in DOC-DEPLOY-001 has a broken link to kubectl config."
    assert ticket.language_fluency == "non_fluent"


def test_ingest_forum_ticket():
    raw_forum = {
        "ticket_id": "TEST-FORUM-01",
        "channel": "forum",
        "subject": "Community Question: Best practice for multi-region failover",
        "body": "Has anyone configured active-active PostgreSQL across regions?",
        "received_at": "2026-06-01T10:15:00Z",
        "customer_id": "CUST-3003",
        "customer_name": "Charlie Community",
        "customer_tier": "free",
        "customer_region": "latin_america",
        "language_fluency": "fluent",
    }
    ticket = ingest_ticket(raw_forum)
    assert ticket.channel == ChannelType.FORUM
    assert ticket.clean_text.startswith("Community Question:")
    assert ticket.metadata["customer_region"] == "latin_america"


def test_channel_normalization_and_aliases():
    # Case insensitivity and dash normalization
    t1 = ingest_ticket({
        "ticket_id": "T-1",
        "channel": "EMAIL",
        "body": "Test message",
        "customer_id": "C-1",
        "customer_name": "User 1",
    })
    assert t1.channel == ChannelType.EMAIL

    t2 = ingest_ticket({
        "ticket_id": "T-2",
        "channel": "docs-comment",
        "body": "Documentation comment",
        "customer_id": "C-2",
        "customer_name": "User 2",
    })
    assert t2.channel == ChannelType.DOCS_COMMENT


def test_ingest_from_json_string():
    json_str = """{
        "ticket_id": "T-JSON-01",
        "channel": "chat",
        "body": "Direct JSON payload test",
        "customer_id": "C-99",
        "customer_name": "JSON tester"
    }"""
    ticket = ingest_ticket(json_str)
    assert ticket.ticket_id == "T-JSON-01"
    assert ticket.clean_text == "Direct JSON payload test"


def test_validation_errors():
    # Missing body
    with pytest.raises((ValidationError, ValueError)):
        ingest_ticket({
            "ticket_id": "T-ERR-01",
            "channel": "email",
            "body": "   ",
            "customer_id": "C-1",
            "customer_name": "User",
        })

    # Invalid channel
    with pytest.raises((ValidationError, ValueError)):
        ingest_ticket({
            "ticket_id": "T-ERR-02",
            "channel": "carrier_pigeon",
            "body": "Valid body",
            "customer_id": "C-1",
            "customer_name": "User",
        })


def test_load_development_tickets():
    dev_path = Path(__file__).resolve().parent.parent / "data" / "development_tickets.json"
    if not dev_path.exists():
        pytest.skip("development_tickets.json not found")

    tickets = load_tickets_from_json(dev_path)
    assert len(tickets) > 0

    # Ensure all 4 channels are present in dataset
    channels_found = {t.channel.value for t in tickets}
    expected_channels = {"email", "chat", "docs_comment", "forum"}
    assert expected_channels.issubset(channels_found), f"Missing channels: {expected_channels - channels_found}"

    # Check that metadata and clean_text are populated for all
    for t in tickets:
        assert t.ticket_id
        assert t.body
        assert t.clean_text
        assert "customer_id" in t.metadata
        assert "customer_tier" in t.metadata


def test_load_validation_tickets():
    val_path = Path(__file__).resolve().parent.parent / "data" / "validation_tickets.json"
    if not val_path.exists():
        pytest.skip("validation_tickets.json not found")

    tickets = load_tickets_from_json(val_path)
    assert len(tickets) > 0

    # Ensure all 4 channels are present in validation set
    channels_found = {t.channel.value for t in tickets}
    expected_channels = {"email", "chat", "docs_comment", "forum"}
    assert expected_channels.issubset(channels_found)

    for t in tickets:
        assert t.ticket_id
        assert t.body
        assert t.clean_text
