import pytest

from evaluation.harness import compute_metrics, evaluate_contract_kpis
from src.ingest import ingest_ticket


def norm(labels=None, **kwargs):
    return ingest_ticket({"ticket_id": "T1", "channel": "email", "body": "Help me sign in",
        "customer_id": "C1", "customer_name": "Test", "labels": labels, **kwargs})


def result(**kwargs):
    return {"route": "auto_respond", "intent": "account_access", "urgency": "low",
            "classification_confidence": .9, "generated_response": "See [DOC-AUTH-001]",
            "retrieved_passages": [{"doc_id": "DOC-AUTH-001"}], **kwargs}


def kpi(metrics, key):
    return next(r for r in metrics["contract_kpis"]["kpis"] if r["id"] == key)


def test_proxies_never_claim_resolution_or_independent_quality():
    metrics = compute_metrics([result()], [norm(history={"first_contact_resolution": True})], [1])
    assert metrics["business"]["automation_rate_pct"] == 100
    assert metrics["business"]["first_contact_resolution_pct"] is None
    assert metrics["technical"]["citation_id_validity_pct"] == 100
    assert metrics["technical"]["citation_accuracy_pct"] is None
    assert metrics["technical"]["hallucination_rate_pct"] is None
    assert [row["id"] for row in metrics["contract_kpis"]["kpis"]] == [f"KPI-{index:02}" for index in range(1, 8)]
    for key in ("KPI-01", "KPI-05", "KPI-07"):
        assert kpi(metrics, key)["status"] == "PASS"


def test_no_labels_no_citations_have_unknown_denominators():
    metrics = compute_metrics([result(generated_response="")], [norm()], [1])
    tech = metrics["technical"]
    assert tech["overall_classification"]["accuracy_pct"] is None
    assert tech["retrieval_hit_rate_pct"] is None
    assert tech["citation_id_validity_pct"] is None
    assert tech["auto_responses_without_citations"] == 1
    assert kpi(metrics, "KPI-04")["status"] == "NOT_MEASURED"


def test_routing_safety_and_urgency_are_scored_independently():
    labels = {"intent": "security_incident", "urgency": "high", "expected_route": "escalate",
              "must_not_auto_respond": True, "answerable_from_docs": False}
    metrics = compute_metrics([result()], [norm(labels)], [1])
    assert metrics["routing"]["false_auto_count"] == 1
    assert metrics["routing"]["false_auto_rate_among_expected_escalations_pct"] == 100
    assert metrics["routing"]["must_not_auto_respond_violation_count"] == 1
    assert metrics["routing"]["not_answerable_from_docs_auto_count"] == 1
    assert metrics["technical"]["high_urgency_recall_pct"] == 0
    assert metrics["routing"]["confusion_matrix"]["rows_true_columns_predicted"] == [[0, 0], [1, 0]]


def test_weighted_scorecard_retains_minority_failure_in_diagnostics():
    inputs = [norm({"intent": "account_access"})] * 99 + [norm({"intent": "security_incident"})]
    metrics = compute_metrics([result()] * 100, inputs, [1] * 100)
    assert metrics["technical"]["overall_classification"]["precision_pct"] > 85
    assert kpi(metrics, "KPI-04")["status"] == "PASS"
    assert metrics["technical"]["overall_classification"]["minimum_class_precision_pct"] == 0


def test_unlabelled_tickets_do_not_dilute_segment_accuracy():
    metrics = compute_metrics([result(), result()], [norm({"intent": "account_access"}), norm()], [1, 2])
    group = metrics["fairness_audit"]["by_customer_tier"]["standard"]
    assert group["ticket_count"] == 2 and group["labelled_intent_count"] == 1
    assert group["intent_accuracy_pct"] == 100
    assert "by_customer_region" in metrics["fairness_audit"]
    assert "by_ticket_length" in metrics["fairness_audit"]


def test_calibration_includes_one_and_excludes_invalid_scores():
    rows = [result(classification_confidence=s) for s in (0, .2, .8, 1, None, float("nan"), 2)]
    metrics = compute_metrics(rows, [norm({"intent": "account_access"})] * len(rows), [1] * len(rows))
    cal = metrics["confidence_calibration"]
    assert cal["sample_count"] == 4
    assert cal["invalid_or_missing_confidence_count"] == 3
    assert cal["bins"][-1]["sample_count"] == 2
    assert cal["maximum_gap_pp"] == 100


def test_failures_are_not_guardrail_blocks_or_correct_default_predictions():
    metrics = compute_metrics([{"route": "escalate", "guardrail_passed": False, "error": "TimeoutError"}],
                              [norm({"intent": "unclear_request", "urgency": "medium"})], [1])
    assert metrics["volume"]["blocked_by_guardrails"] == 0
    assert metrics["technical"]["overall_classification"]["accuracy_pct"] == 0
    assert metrics["technical"]["urgency_accuracy_pct"] == 0


def test_blocked_private_draft_is_not_outbound_leak():
    metrics = compute_metrics([result(route="escalate", guardrail_blocked=True, guardrail_results={"pii": "fail"})], [norm()], [1])
    assert metrics["governance"]["draft_pii_detections"] == 1
    assert metrics["governance"]["outbound_scanner_detections"] == 0
    assert metrics["governance"]["verified_outbound_private_data_occurrences"] is None


def test_citations_count_occurrences_in_text_and_doc_mismatch_is_only_proxy():
    metrics = compute_metrics([result(generated_response="[DOC-AUTH-001] [DOC-AUTH-001] [DOC-FAKE]", cited_doc_ids=[])],
        [norm({"expected_doc_ids": ["DOC-OTHER"]})], [1])
    assert metrics["technical"]["citation_occurrence_count"] == 3
    assert metrics["technical"]["citation_id_validity_pct"] == 66.67
    assert metrics["technical"]["reference_doc_mismatch_rate_pct"] == 100
    assert metrics["technical"]["hallucination_rate_pct"] is None


def test_legacy_hallucination_proxy_counts_a_response_once_over_all_auto_responses():
    rows = [result(generated_response="[DOC-FAKE]")] + [result()] * 3
    inputs = [norm({"expected_doc_ids": ["DOC-OTHER"]})] + [norm()] * 3
    metrics = compute_metrics(rows, inputs, [1] * 4)
    assert metrics["technical"]["hallucination_proxy_flag_count"] == 1
    assert metrics["technical"]["hallucination_proxy_denominator"] == 4
    assert metrics["technical"]["hallucination_proxy_rate_pct"] == 25
    assert metrics["technical"]["hallucination_rate_pct"] is None


def test_no_auto_responses_do_not_claim_perfect_quality():
    metrics = compute_metrics([result(route="escalate")], [norm()], [1])
    assert metrics["technical"]["hallucination_proxy_rate_pct"] is None
    assert metrics["technical"]["citation_id_validity_pct"] is None
    assert kpi(metrics, "KPI-07")["status"] == "NOT_MEASURED"


@pytest.mark.parametrize("seconds,expected", [(15, "PASS"), (299.9999, "PASS"), (300, "FAIL")])
def test_latency_target_matches_requested_legacy_benchmark(seconds, expected):
    metrics = compute_metrics([result()], [norm()], [seconds])
    assert kpi(metrics, "KPI-03")["status"] == expected


def test_p95_uses_documented_nearest_rank_and_keeps_precision():
    metrics = compute_metrics([result()] * 21, [norm()] * 21, list(range(1, 22)))
    assert metrics["technical"]["latency"]["p95_seconds"] == 20


def test_missing_kpis_never_pass_and_nonfinite_values_are_unmeasured():
    score = evaluate_contract_kpis({"technical": {"latency": {"p95_seconds": float("nan")}}})
    assert all(r["status"] == "NOT_MEASURED" for r in score["kpis"])
    assert score["summary"]["overall_status"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("timings", [[], [-1], [float("nan")]])
def test_invalid_measurements_are_rejected(timings):
    with pytest.raises(ValueError):
        compute_metrics([result()], [norm()], timings)
