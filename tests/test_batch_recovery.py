"""Recovery and exact audit reconciliation, using real SQLite and mocked processing."""
import json
import sqlite3

import pytest

from evaluation import harness
from src import logging_store as audit


def ticket(ticket_id="SAME"):
    return {"ticket_id": ticket_id, "channel": "email", "body": "Help with login",
            "customer_id": "C1", "customer_name": "Test"}


def decision(run_id, index, ticket_id="SAME"):
    return audit.DecisionRecord(ticket_id=ticket_id, run_id=run_id, input_index=index,
                                action_taken="escalate", reason="test")


def test_reconciliation_counts_occurrences_and_rejects_stale_duplicate_extra_logs(tmp_path):
    db = str(tmp_path / "audit.db")
    audit.log_decision(decision("old", 1), db)
    assert audit.reconcile_decisions_with_tickets(["SAME"], db, "new")["coverage_pct"] == 0
    audit.log_decision(decision("new", 1), db)
    check = audit.reconcile_decisions_with_tickets(["SAME", "SAME"], db, "new")
    assert check["missing_input_indices"] == [2]
    assert check["coverage_pct"] == 50
    audit.log_decision(decision("new", 2), db)
    assert audit.reconcile_decisions_with_tickets(["SAME", "SAME"], db, "new")["is_reconciled"]
    audit.log_decision(decision("new", 2), db)
    check = audit.reconcile_decisions_with_tickets(["SAME", "SAME"], db, "new")
    assert not check["is_reconciled"] and check["extra_input_indices"] == [2]
    kpis = harness.evaluate_contract_kpis({"reconciliation_check": check})
    # Even 100% coverage must fail when the run contains an extra decision.
    rows = kpis["kpis"]
    assert next(k for k in rows if k["id"] == "KPI-06")["status"] == "FAIL"


def test_wrong_identity_does_not_match(tmp_path):
    db = str(tmp_path / "audit.db")
    audit.log_decision(decision("run", 1, "WRONG"), db)
    check = audit.reconcile_decisions_with_tickets(["SAME"], db, "run")
    assert check["missing_ticket_ids"] == ["SAME"]
    assert check["extra_ticket_ids"] == ["WRONG"]


def test_old_database_migration_preserves_records(tmp_path):
    db = str(tmp_path / "audit.db")
    audit.log_decision(decision(None, None), db)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX idx_decisions_run")
        conn.execute("ALTER TABLE decisions DROP COLUMN run_id")
        conn.execute("ALTER TABLE decisions DROP COLUMN input_index")
    audit.init_db(db)
    assert audit.count_decisions(db) == 1
    audit.log_decision(decision("new", 1), db)
    assert audit.reconcile_decisions_with_tickets(["SAME"], db, "new")["logged_ticket_count"] == 1


def test_batch_continues_after_invalid_and_failed_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "audit.db")
    seen = []

    def process(norm, *, db_path, run_id, input_index, **kwargs):
        seen.append(input_index)
        if norm.ticket_id == "CRASH":
            raise RuntimeError("provider error containing sensitive input")
        state = {"ticket_id": norm.ticket_id, "run_id": run_id,
                 "input_index": input_index, "route": "escalate"}
        return {**state, **audit.log_decision_node(state, db_path)}

    monkeypatch.setattr(harness, "process_ticket", process)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([ticket(), {**ticket("BAD"), "body": ""},
                                  None, ticket("CRASH"), ticket()]), encoding="utf-8")
    canonical = tmp_path / "evaluation/results/latest_report.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("original", encoding="utf-8")
    metrics = harness.run_evaluation(str(source), str(tmp_path / "report.json"),
                                    db_path=db, use_llm=False,
                                    ticket_output_path=str(tmp_path / "details.json"))
    assert seen == [1, 4, 5]
    assert metrics["invalid_input_count"] == 2
    assert metrics["processing_error_count"] == 1
    assert metrics["total_tickets_evaluated"] == 5
    assert metrics["reconciliation_check"]["is_reconciled"]
    assert metrics["reconciliation_check"]["logged_ticket_count"] == 5
    saved = json.loads((tmp_path / "details.json").read_text())
    assert [r["input_index"] for r in saved["tickets"]] == [1, 2, 3, 4, 5]
    assert all(r["execution"]["decision_id"] for r in saved["tickets"])
    assert saved["tickets"][2]["ticket_id"].startswith("INVALID-")
    assert "sensitive input" not in (tmp_path / "details.json").read_text()
    assert canonical.read_text() == "original"
    repeat = harness.run_evaluation(str(source), str(tmp_path / "repeat.json"), db_path=db, use_llm=False)
    assert repeat["run_id"] != metrics["run_id"]
    assert repeat["reconciliation_check"]["logged_ticket_count"] == 5
    assert audit.count_decisions(db) == 10


def test_audit_outage_is_reported_without_fabricating_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("unavailable")
    monkeypatch.setattr(harness, "process_ticket", fail)
    monkeypatch.setattr(harness, "log_decision", fail)
    monkeypatch.setattr(harness, "reconcile_decisions_with_tickets", fail)
    source = tmp_path / "input.json"
    source.write_text(json.dumps([ticket()]), encoding="utf-8")
    metrics = harness.run_evaluation(str(source), str(tmp_path / "report.json"), use_llm=False)
    assert metrics["run_status"] == "audit_incomplete"
    assert metrics["reconciliation_check"]["logged_ticket_count"] is None
    saved = json.loads((tmp_path / "latest_ticket_results.json").read_text())
    assert saved["tickets"][0]["execution"]["logged_to_audit_store"] is False


@pytest.mark.parametrize("content", ["{", "{}", "[]"])
def test_unenumerable_or_empty_batch_is_rejected(tmp_path, content):
    source = tmp_path / "input.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        harness.run_evaluation(str(source))
