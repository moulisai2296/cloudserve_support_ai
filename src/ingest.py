"""Ticket ingestion and pre-processing module for CloudServe Solutions.

Handles incoming tickets across 4 channels (email, chat, docs_comment, forum),
validating structure, preserving customer metadata, and normalizing text.
Adheres to Capstone Specification AC-A2 and B-02.
"""

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ChannelType(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    DOCS_COMMENT = "docs_comment"
    FORUM = "forum"


class CustomerTier(str, Enum):
    ENTERPRISE = "enterprise"
    BUSINESS = "business"
    STANDARD = "standard"
    FREE = "free"


class TicketLabels(BaseModel):
    intent: Optional[str] = None
    urgency: Optional[str] = None
    expected_route: Optional[str] = None
    answerable_from_docs: Optional[bool] = None
    expected_doc_ids: List[str] = Field(default_factory=list)
    must_not_auto_respond: Optional[bool] = False


class TicketHistory(BaseModel):
    first_contact_resolution: Optional[bool] = None
    resolution_time_minutes: Optional[int] = None
    csat_rating: Optional[int] = None
    escalated: Optional[bool] = None
    repeat_contact: Optional[bool] = None


class NormalizedTicket(BaseModel):
    ticket_id: str
    channel: ChannelType
    subject: str = ""
    body: str
    received_at: Optional[str] = None
    customer_id: str
    customer_name: str
    customer_tier: str = "standard"
    customer_region: str = "global"
    language_fluency: str = "fluent"
    
    # Optional ground-truth metadata present in dev/val datasets
    labels: Optional[TicketLabels] = None
    history: Optional[TicketHistory] = None

    # Normalized / synthesized fields
    clean_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("channel", mode="before")
    @classmethod
    def normalize_channel(cls, v: Any) -> ChannelType:
        if isinstance(v, ChannelType):
            return v
        if not v:
            raise ValueError("Channel cannot be empty.")
        clean = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        if clean in ("docs", "docs_comments", "documentation_comment"):
            clean = "docs_comment"
        try:
            return ChannelType(clean)
        except ValueError:
            valid = [c.value for c in ChannelType]
            raise ValueError(f"Invalid channel '{v}'. Must be one of: {valid}")

    @field_validator("customer_tier", mode="before")
    @classmethod
    def normalize_tier(cls, v: Any) -> str:
        if v is None:
            return "standard"
        return str(v).strip().lower()

    @field_validator("subject", mode="before")
    @classmethod
    def default_subject(cls, v: Any) -> str:
        if v is None:
            return ""
        # Normalize carriage returns and trailing whitespace
        return str(v).replace("\r\n", "\n").replace("\r", "\n").strip()

    @field_validator("body", mode="before")
    @classmethod
    def validate_body(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Ticket body cannot be null.")
        cleaned = str(v).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            raise ValueError("Ticket body cannot be empty or whitespace-only.")
        return cleaned

    def model_post_init(self, __context: Any) -> None:
        """Derive clean unified text and store metadata dictionary."""
        subject_part = self.subject.strip()
        body_part = self.body.strip()
        
        # Strip redundant Subject: prefix inside body if present
        if subject_part and body_part.startswith(f"Subject: {subject_part}"):
            body_part = body_part[len(f"Subject: {subject_part}"):].strip()

        if subject_part and subject_part != body_part:
            self.clean_text = f"{subject_part}\n{body_part}"
        else:
            self.clean_text = body_part

        if not self.metadata:
            self.metadata = {
                "ticket_id": self.ticket_id,
                "channel": self.channel.value if isinstance(self.channel, ChannelType) else self.channel,
                "customer_id": self.customer_id,
                "customer_tier": self.customer_tier,
                "customer_region": self.customer_region,
                "language_fluency": self.language_fluency,
            }


def ingest_ticket(raw_data: Dict[str, Any] | str) -> NormalizedTicket:
    """Ingest, validate, and normalize a single raw ticket dictionary or JSON string."""
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected dict or JSON string, got {type(raw_data)}")
    return NormalizedTicket(**raw_data)


def load_tickets_from_json(file_path: str | Path) -> List[NormalizedTicket]:
    """Load and normalize an array of tickets from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticket file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of tickets in {path}, got {type(data)}")

    normalized_tickets: List[NormalizedTicket] = []
    for item in data:
        normalized_tickets.append(ingest_ticket(item))

    return normalized_tickets
