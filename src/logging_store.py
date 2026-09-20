"""Decision logging module for CloudServe Solutions support triage.

Persists full governance audit records to SQLite database (./storage/decisions.db)
matching the mandatory Minimum Record Schema from Governance Framework Section 1.
Adheres to Capstone Specification AC-A8, B-10, and FR-08.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv("DATABASE_URL", "./storage/decisions.db")
if DEFAULT_DB_PATH.startswith("sqlite:///"):
    DEFAULT_DB_PATH = DEFAULT_DB_PATH.replace("sqlite:///", "")


class DecisionRecord(BaseModel):
    """Minimum Record Schema conforming to Governance Framework"""
    decision_id: str = Field(default_factory=lambda: f"DEC-{uuid.uuid4().hex[:10].upper()}")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ticket_id: str
    run_id: Optional[str] = None
    input_index: Optional[int] = None
    stage: str = "triage"  # classification | routing | generation | validation | triage
    input_summary: str = ""
    model: Dict[str, str] = Field(default_factory=lambda: {
        "name": os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"),
        "version": "v1.0"
    })
    prediction: Dict[str, Any] = Field(default_factory=dict)  # {"value": intent, "confidence": 0.00}
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    sources_used: List[Dict[str, Any]] = Field(default_factory=list)  # [{"doc_id": "...", "score": 0.00}]
    threshold_applied: float = 0.80
    action_taken: str  # auto_respond | escalate | block
    reason: str
    guardrail_results: Dict[str, str] = Field(default_factory=lambda: {
        "pii": "pass",
        "grounding": "pass",
        "citation": "pass"
    })
    prompt_version: str = "PR-01/PR-02 v1.0"
    requirement_ids: List[str] = Field(default_factory=lambda: ["FR-02", "FR-04", "FR-08"])
    grounding_review: Dict[str, Any] = Field(default_factory=dict)


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a thread-safe connection to SQLite database, initializing directory if needed."""
    path_str = db_path or DEFAULT_DB_PATH
    if path_str != ":memory:":
        db_file = Path(path_str).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_file))
    else:
        conn = sqlite3.connect(":memory:")

    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the decisions audit table schema."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                reason TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                threshold_applied REAL,
                sources_used TEXT,
                guardrail_results TEXT,
                model_name TEXT,
                prompt_version TEXT,
                requirement_ids TEXT,
                full_record TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(ticket_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action_taken);")
        # Additive migration preserves historical decisions, which have no run identity.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)")}
        for name, sql_type in (("run_id", "TEXT"), ("input_index", "INTEGER")):
            if name not in columns:
                conn.execute(f"ALTER TABLE decisions ADD COLUMN {name} {sql_type}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, input_index)")
    conn.close()


def log_decision(record: DecisionRecord | Dict[str, Any], db_path: Optional[str] = None) -> str:
    """
    Inserts a structured audit decision record into the SQLite database.
    Returns the decision_id.
    """
    if isinstance(record, dict):
        parsed = DecisionRecord(**record)
    else:
        parsed = record

    init_db(db_path)
    conn = get_db_connection(db_path)

    intent_val = parsed.prediction.get("value") or parsed.prediction.get("intent")
    confidence_val = parsed.prediction.get("confidence", 0.0)
    sources_json = json.dumps(parsed.sources_used)
    guardrails_json = json.dumps(parsed.guardrail_results)
    reqs_json = json.dumps(parsed.requirement_ids)
    full_json = json.dumps(parsed.model_dump())

    with conn:
        conn.execute("""
            INSERT INTO decisions (
                decision_id,
                ticket_id,
                timestamp,
                stage,
                action_taken,
                reason,
                intent,
                confidence,
                threshold_applied,
                sources_used,
                guardrail_results,
                model_name,
                prompt_version,
                requirement_ids,
                full_record, run_id, input_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed.decision_id,
            parsed.ticket_id,
            parsed.timestamp,
            parsed.stage,
            parsed.action_taken,
            parsed.reason,
            intent_val,
            float(confidence_val) if confidence_val is not None else 0.0,
            parsed.threshold_applied,
            sources_json,
            guardrails_json,
            parsed.model.get("name", ""),
            parsed.prompt_version,
            reqs_json,
            full_json,
            parsed.run_id,
            parsed.input_index,
        ))
    conn.close()
    return parsed.decision_id


def get_decision(decision_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single decision record by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT full_record FROM decisions WHERE decision_id = ?", (decision_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row["full_record"])
    return None


def get_decisions_by_ticket(ticket_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all decision audit records for a given ticket ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT full_record FROM decisions WHERE ticket_id = ? ORDER BY timestamp ASC", (ticket_id,))
    rows = cursor.fetchall()
    conn.close()
    return [json.loads(r["full_record"]) for r in rows]


def count_decisions(db_path: Optional[str] = None) -> int:
    """Returns the total number of logged decisions."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS cnt FROM decisions")
    count = cursor.fetchone()["cnt"]
    conn.close()
    return count


def reconcile_decisions_with_tickets(
    processed_ticket_ids: List[str],
    db_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs coverage check comparing processed tickets against logged decisions.
    Satisfies Governance Framework coverage requirement.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if run_id is not None:
        cursor.execute("SELECT ticket_id, input_index FROM decisions WHERE run_id = ?", (run_id,))
    else:
        cursor.execute("SELECT ticket_id, input_index FROM decisions")
    rows = cursor.fetchall()
    conn.close()
    # A repeated ticket ID is a separate input occurrence, identified by its index.
    expected = Counter(enumerate(processed_ticket_ids, 1)) if run_id is not None else Counter(processed_ticket_ids)
    actual = Counter((r["input_index"], r["ticket_id"]) for r in rows) if run_id is not None else Counter(r["ticket_id"] for r in rows)
    missing = list((expected - actual).elements())
    extras = list((actual - expected).elements())
    matched = sum((expected & actual).values())
    total = len(processed_ticket_ids)
    ticket_ids = lambda entries: [entry[1] if run_id is not None else entry for entry in entries]
    return {
        "run_id": run_id,
        "is_reconciled": not missing and not extras,
        "processed_ticket_count": total,
        "logged_ticket_count": len(rows),
        "matched_record_count": matched,
        "coverage_pct": round(100.0 * matched / total, 2) if total else 0.0,
        "missing_ticket_ids": ticket_ids(missing),
        "extra_ticket_ids": ticket_ids(extras),
        "missing_input_indices": [entry[0] for entry in missing] if run_id is not None else [],
        "extra_input_indices": [entry[0] for entry in extras] if run_id is not None else [],
    }


# ==============================================================================
# LangGraph Node
# ==============================================================================

def log_decision_node(state: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph Node: Automatically compiles and persists the governance audit
    record for the completed ticket triage process.
    """
    ticket_id = state.get("ticket_id", "UNKNOWN")
    route = state.get("route", "escalate")
    escalation_reason = state.get("escalation_reason")

    if route == "auto_respond":
        action = "auto_respond"
        reason = state.get("auto_respond_summary") or "Auto-responded: All policy, confidence, grounding, and safety checks passed."
    elif state.get("guardrail_blocked"):
        action = "block"
        reason = state.get("guardrail_block_reason", "Response blocked by safety guardrails.")
    else:
        action = "escalate"
        packet = state.get("escalation_packet") or {}
        reason = packet.get("handover_notes") or escalation_reason or "Escalated for human engineer handling."

    passages = state.get("retrieved_passages", [])
    sources_used = [
        {"doc_id": p.get("doc_id", ""), "score": p.get("score", 0.0)}
        for p in passages
    ]

    record = DecisionRecord(
        ticket_id=ticket_id,
        run_id=state.get("run_id"),
        input_index=state.get("input_index"),
        stage="triage",
        input_summary=(state.get("clean_text") or state.get("body", ""))[:250],
        prediction={
            "value": state.get("intent", "unclear_request"),
            "confidence": float(state.get("classification_confidence", 0.0)),
            "urgency": state.get("urgency", "medium"),
        },
        alternatives=state.get("classification_alternatives", []),
        sources_used=sources_used,
        threshold_applied=float(state.get("routing_threshold", 0.80)),
        action_taken=action,
        reason=reason,
        guardrail_results=state.get("guardrail_results", {
            "pii": "not_run",
            "grounding": "not_run",
            "citation": "not_run"
        }),
        grounding_review=state.get("grounding_review", {}),
    )

    decision_id = log_decision(record, db_path=db_path)
    return {
        "decision_id": decision_id,
        "logged_to_audit_store": True,
    }
