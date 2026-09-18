"""Unit tests for the evaluation harness (evaluation/harness.py).

Verifies unattended batch evaluation execution, metrics calculation,
reconciliation against SQLite, and JSON report generation.
"""

import tempfile
from pathlib import Path
import pytest

from evaluation.harness import run_evaluation


@pytest.fixture
def temp_output():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    yield out_path
    try:
        Path(out_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_harness_batch_run(temp_output):
    """Verifies that harness runs cleanly unattended and computes metrics."""
    dev_path = "data/development_tickets.json"
    if not Path(dev_path).exists():
        pytest.skip("development_tickets.json not found")

    metrics = run_evaluation(
        input_path=dev_path,
        output_path=temp_output,
        limit=5,
        use_llm=False,
    )

    assert "tier_one_business_outcomes" in metrics
    assert "tier_two_technical_performance" in metrics
    assert "fairness_audit" in metrics
    assert "reconciliation_check" in metrics
    assert metrics["total_tickets_evaluated"] == 5
    assert metrics["reconciliation_check"]["is_reconciled"] is True

    # Confirm file was created on disk
    assert Path(temp_output).exists()
    assert Path(temp_output).stat().st_size > 500
