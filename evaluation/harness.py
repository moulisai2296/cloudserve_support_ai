"""Standalone evaluation harness for CloudServe Solutions support triage.

Executes unattended batch evaluation with measured diagnostics and explicit
missing-evidence markers for business outcomes and independent quality review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import statistics
from sklearn.metrics import classification_report, confusion_matrix
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_ticket, NormalizedTicket
from src.graph import process_ticket
from src.logging_store import DecisionRecord, log_decision, reconcile_decisions_with_tickets

# Suppress external library noise (HuggingFace Hub, HTTP requests, progress bars)
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.simplefilter("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation.harness")

for noisy_logger in [
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "urllib3",
    "chromadb",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
]:
    nl = logging.getLogger(noisy_logger)
    nl.setLevel(logging.ERROR)
    nl.propagate = False
    nl.handlers.clear()

try:
    import huggingface_hub.utils
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass

try:
    import transformers.utils.logging
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass


# Metric calculation and console reporting are kept inline for the standalone harness.
def percent(numerator, denominator):
    return round(100 * numerator / denominator, 2) if denominator else None


def classification(truth, predictions):
    if not truth:
        return {"sample_count": 0, "accuracy_pct": None, "precision_pct": None,
                "recall_pct": None, "macro_precision_pct": None,
                "minimum_class_precision_pct": None, "per_class": {},
                "confusion_matrix": {"labels": [], "rows_true_columns_predicted": []}}
    labels = sorted(set(truth) | set(predictions))
    report = classification_report(truth, predictions, labels=labels,
                                   output_dict=True, zero_division=0)
    classes = {label: {"precision_pct": report[label]["precision"] * 100,
                       "recall_pct": report[label]["recall"] * 100,
                       "f1_score": report[label]["f1-score"],
                       "support": int(report[label]["support"])} for label in labels}
    return {
        "sample_count": len(truth),
        "accuracy_pct": percent(sum(a == b for a, b in zip(truth, predictions)), len(truth)),
        "precision_pct": report["weighted avg"]["precision"] * 100,
        "recall_pct": report["weighted avg"]["recall"] * 100,
        "macro_precision_pct": report["macro avg"]["precision"] * 100,
        "minimum_class_precision_pct": min(classes[label]["precision_pct"] for label in set(truth)),
        "per_class": classes,
        "confusion_matrix": {"labels": labels,
            "rows_true_columns_predicted": confusion_matrix(truth, predictions, labels=labels).tolist()},
    }


def calibration(samples):
    rows = []
    for index in range(5):
        low, high = index / 5, (index + 1) / 5
        group = [(score, correct) for score, correct in samples
                 if low <= score and (score < high or index == 4 and score == 1)]
        if not group:
            continue
        mean = statistics.mean(score for score, _ in group) * 100
        observed = 100 * sum(correct for _, correct in group) / len(group)
        rows.append({"lower_inclusive": low, "upper": high,
                     "upper_inclusive": index == 4, "sample_count": len(group),
                     "mean_confidence_pct": mean, "observed_accuracy_pct": observed,
                     "absolute_gap_pp": abs(mean - observed)})
    return {"sample_count": len(samples), "bins": rows,
            "maximum_gap_pp": max((r["absolute_gap_pp"] for r in rows), default=None),
            "interpretation": "Descriptive only; small bins are uncertain. Missing/invalid scores are excluded and counted separately."}


def compute_metrics(processed_results, raw_tickets, timings, reconciliation=None):
    if not (len(processed_results) == len(raw_tickets) == len(timings)):
        raise ValueError("Results, inputs, and timings must have equal lengths")
    total = len(processed_results)
    if not total:
        raise ValueError("No tickets processed")
    if any(not math.isfinite(t) or t < 0 for t in timings):
        raise ValueError("Timings must be finite nonnegative seconds")
    auto = sum(r.get("route") == "auto_respond" for r in processed_results)
    escalated = sum(r.get("route") == "escalate" for r in processed_results)
    blocked = sum(bool(r.get("guardrail_blocked")) for r in processed_results)
    intent_true, intent_pred, urgency_true, urgency_pred = [], [], [], []
    route_true, route_pred, confidence_samples = [], [], []
    counters = Counter()
    failure_examples = []
    segments = {name: defaultdict(list) for name in
                ("customer_tier", "language_fluency", "customer_region", "ticket_length")}
    review_counts = Counter()
    guardrail_counts = Counter()
    for index, (norm, result) in enumerate(zip(raw_tickets, processed_results), 1):
        checks = result.get("guardrail_results") or {}
        for check, status in checks.items():
            if status == "fail":
                guardrail_counts[check] += 1
        if checks.get("pii") == "fail" and result.get("route") == "auto_respond":
            counters["outbound_scanner_detections"] += 1
        if norm is None:
            continue
        labels = norm.labels
        automatic = result.get("route") == "auto_respond"
        predicted = result.get("intent") or "__missing_prediction__"
        truth = labels.intent if labels else None
        if truth:
            intent_true.append(truth)
            intent_pred.append(predicted)
            score = result.get("classification_confidence")
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) and 0 <= score <= 1:
                confidence_samples.append((score, predicted == truth))
            else:
                counters["invalid_or_missing_confidence"] += 1
        if labels and labels.urgency:
            urgency_true.append(labels.urgency)
            urgency_pred.append(result.get("urgency") or "__missing_prediction__")
        expected_route = labels.expected_route if labels else None
        if expected_route:
            route_true.append(expected_route)
            route_pred.append(result.get("route") or "__missing_prediction__")
            if expected_route == "escalate":
                counters["expected_escalations"] += 1
                counters["false_auto"] += automatic
            if expected_route == "auto_respond" and result.get("route") == "escalate":
                counters["false_escalations"] += 1
        if labels and labels.must_not_auto_respond:
            counters["policy_excluded"] += 1
            counters["policy_excluded_auto"] += automatic
        if labels and labels.answerable_from_docs is False:
            counters["not_answerable"] += 1
            counters["not_answerable_auto"] += automatic
        if len(failure_examples) < 25 and ((truth and truth != predicted) or (expected_route and expected_route != result.get("route"))):
            failure_examples.append({"input_index": index, "ticket_id": norm.ticket_id,
                "expected_intent": truth, "predicted_intent": predicted,
                "expected_route": expected_route, "route": result.get("route"),
                "processing_error": result.get("error")})
        expected_docs = set(labels.expected_doc_ids or []) if labels else set()
        retrieved = {p.get("doc_id") for p in result.get("retrieved_passages", [])}
        if expected_docs:
            counters["retrieval_evaluable"] += 1
            counters["retrieval_hits"] += bool(expected_docs & retrieved)
        # Count citation occurrences in actual emitted text, rather than model metadata.
        citations = re.findall(r"\[(DOC-[A-Za-z0-9_-]+)\]", result.get("generated_response") or "") if automatic else []
        valid = sum(doc in retrieved for doc in citations)
        counters["citations"] += len(citations)
        counters["valid_citation_ids"] += valid
        if automatic:
            counters["hallucination_proxy_flags"] += bool(valid < len(citations) or (expected_docs and not expected_docs.intersection(citations)))
            counters["auto_without_citations"] += not bool(citations)
            review = result.get("grounding_review") or {}
            review_counts[f"{review.get('method', 'not_run')}:{review.get('status', 'not_run')}"] += 1
            if expected_docs:
                counters["reference_doc_evaluable"] += 1
                counters["reference_doc_mismatch"] += not bool(expected_docs & set(citations))
        words = len(f"{norm.subject} {norm.body}".split())
        length = "short_1_50_words" if words <= 50 else "medium_51_150_words" if words <= 150 else "long_over_150_words"
        for dimension, value in (("customer_tier", norm.customer_tier),
                                  ("language_fluency", norm.language_fluency),
                                  ("customer_region", norm.customer_region), ("ticket_length", length)):
            segments[dimension][value.lower()].append({"auto": automatic, "truth": truth,
                "predicted": predicted, "citations": len(citations), "valid_ids": valid})
    intents = classification(intent_true, intent_pred)
    urgencies = classification(urgency_true, urgency_pred)
    fairness = {}
    for dimension, groups in segments.items():
        summary = {}
        for name, rows in groups.items():
            labelled = [r for r in rows if r["truth"]]
            summary[name] = {"ticket_count": len(rows), "labelled_intent_count": len(labelled),
                "automation_rate_pct": percent(sum(r["auto"] for r in rows), len(rows)),
                "intent_accuracy_pct": percent(sum(r["truth"] == r["predicted"] for r in labelled), len(labelled)),
                "citation_occurrence_count": sum(r["citations"] for r in rows),
                "citation_id_validity_pct": percent(sum(r["valid_ids"] for r in rows), sum(r["citations"] for r in rows)),
                "resolution_quality_pct": None}
        accuracy = [r["intent_accuracy_pct"] for r in summary.values() if r["intent_accuracy_pct"] is not None]
        fairness[f"by_{dimension}"] = summary
        fairness[f"{dimension}_intent_accuracy_spread_pp"] = max(accuracy) - min(accuracy) if len(accuracy) > 1 else None
    fairness["resolution_quality_spread_pp"] = None
    fairness["interpretation"] = "Automation, accuracy and ID validity are diagnostic proxies, not resolution quality or proof of bias. Length is not measured complexity; consider case mix and sample sizes."
    recon = reconciliation or {"is_reconciled": False, "processed_ticket_count": total,
        "logged_ticket_count": None, "coverage_pct": None, "verification_status": "not_run"}
    latency = {"sample_count": len(timings), "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "p95_seconds": sorted(timings)[math.ceil(.95 * len(timings)) - 1],
        "percentile_method": "nearest_rank_ceil", "scope": "Per-input processing through audit logging; not customer delivery latency."}
    cal = calibration(confidence_samples)
    cal["invalid_or_missing_confidence_count"] = counters["invalid_or_missing_confidence"]
    metrics = {
        "metrics_schema_version": "2.1", "reporting_profile": "legacy_operational_proxies", "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tickets_evaluated": total,
        "volume": {"tickets_processed": total, "answered_automatically": auto, "escalated": escalated,
                   "blocked_by_guardrails": blocked, "answered_automatically_pct": percent(auto, total),
                   "escalated_pct": percent(escalated, total)},
        "business": {"automation_rate_pct": percent(auto, total), "escalation_rate_pct": percent(escalated, total),
            "first_contact_resolution_pct": None, "mean_time_to_first_reply_seconds": None,
            "median_time_to_first_reply_seconds": None, "satisfaction_proxy_score": None,
            "repeat_contact_reduction_pct": None,
            "evidence_status": "No closure, customer delivery, assessor, or post-deployment outcome evidence collected."},
        "technical": {"overall_classification": intents,
            "classification_precision_and_recall_per_class": intents["per_class"],
            "urgency_classification": urgencies, "urgency_accuracy_pct": urgencies["accuracy_pct"],
            "high_urgency_recall_pct": urgencies["per_class"].get("high", {}).get("recall_pct"),
            "retrieval_hits": counters["retrieval_hits"], "retrieval_evaluable_tickets": counters["retrieval_evaluable"],
            "retrieval_hit_rate_pct": percent(counters["retrieval_hits"], counters["retrieval_evaluable"]),
            "citation_occurrence_count": counters["citations"],
            "citation_id_validity_pct": percent(counters["valid_citation_ids"], counters["citations"]),
            "auto_responses_without_citations": counters["auto_without_citations"],
            "reference_doc_mismatch_count": counters["reference_doc_mismatch"],
            "reference_doc_evaluable_responses": counters["reference_doc_evaluable"],
            "reference_doc_mismatch_rate_pct": percent(counters["reference_doc_mismatch"], counters["reference_doc_evaluable"]),
            "hallucination_proxy_rate_pct": percent(counters["hallucination_proxy_flags"], auto),
            "hallucination_proxy_flag_count": counters["hallucination_proxy_flags"],
            "hallucination_proxy_denominator": auto,
            "citation_accuracy_pct": None, "hallucination_rate_pct": None,
            "quality_evidence_status": "Independent claim/citation review pending; runtime grounding checks are not the required two-assessor review.",
            "runtime_grounding_reviews": dict(review_counts), "latency": latency, "availability_pct": None},
        "routing": {"labelled_ticket_count": len(route_true),
            "agreement_pct": percent(sum(a == b for a, b in zip(route_true, route_pred)), len(route_true)),
            "expected_escalation_count": counters["expected_escalations"],
            "false_auto_count": counters["false_auto"],
            "false_auto_rate_among_expected_escalations_pct": percent(counters["false_auto"], counters["expected_escalations"]),
            "false_escalation_count": counters["false_escalations"],
            "must_not_auto_respond_count": counters["policy_excluded"],
            "must_not_auto_respond_violation_count": counters["policy_excluded_auto"],
            "not_answerable_from_docs_count": counters["not_answerable"],
            "not_answerable_from_docs_auto_count": counters["not_answerable_auto"],
            "confusion_matrix": classification(route_true, route_pred)["confusion_matrix"]},
        "governance": {"decisions_logged": recon.get("logged_ticket_count"),
            "decision_reconciliation": recon, "total_guardrail_blocks": blocked,
            "guardrail_activations_by_type": dict(guardrail_counts),
            "draft_pii_detections": guardrail_counts["pii"],
            "outbound_scanner_detections": counters["outbound_scanner_detections"],
            "verified_outbound_private_data_occurrences": None,
            "privacy_evidence_status": "Scanner findings are not proof of zero leaks; manual sample review pending."},
        "confidence_calibration": cal, "fairness_audit": fairness,
        "failure_examples": failure_examples, "reconciliation_check": recon,
    }
    # Retain group names used by existing consumers, but never misleading numeric aliases.
    metrics["tier_one_business_outcomes"] = dict(metrics["business"])
    metrics["tier_two_technical_performance"] = dict(metrics["technical"])
    metrics["contract_kpis"] = evaluate_contract_kpis(metrics)
    return metrics


# Selected console/report scorecard: the seven KPIs requested by the project owner.
KPI_DEFINITIONS = [('First Contact Resolution (auto-response proxy)', 'business.automation_rate_pct', '>=', 60, '%'), ('Escalation rate', 'business.escalation_rate_pct', '<=', 30, '%'), ('P95 Response Latency', 'technical.latency.p95_seconds', '<', 300, 's'), ('Intent Classification Precision (weighted)', 'technical.overall_classification.precision_pct', '>=', 85, '%'), ('PII & Credential Scanner Detections', 'governance.draft_pii_detections', '==', 0, 'count'), ('Exact audit reconciliation', 'reconciliation_check.coverage_pct', '==', 100, '%'), ('Citation Accuracy (retrieved-ID proxy)', 'technical.citation_id_validity_pct', '>=', 95, '%')]


def evaluate_contract_kpis(metrics):
    kpis = []
    for index, (name, path, operator, target, unit) in enumerate(KPI_DEFINITIONS, 1):
        value = metrics
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        measured = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        passed = None
        if measured:
            passed = {">=": value >= target, "<=": value <= target, "<": value < target, "==": value == target}[operator]
            if index == 6:
                recon = metrics.get("reconciliation_check", {})
                passed = passed and recon.get("is_reconciled", False) and recon.get("processed_ticket_count", 0) > 0
        kpis.append({"id": f"KPI-{index:02}", "name": name, "metric_path": path.split("."),
            "target_operator": operator, "target_value": target, "unit": unit,
            "target_display": f"{operator} {target} {unit}", "achieved_value": value,
            "achieved_display": f"{value:.2f} {unit}" if measured else "Not measured",
            "achieved": bool(passed) if measured else None,
            "status": ("PASS" if passed else "FAIL") if measured else "NOT_MEASURED",
            "source": 'User-selected legacy operational scorecard; proxy definitions and 300-second P95 comparison benchmark.'})
    counts = Counter(k["status"] for k in kpis)
    return {"summary": {"total_kpis": len(kpis), "achieved_kpis": counts["PASS"],
        "missed_kpis": counts["FAIL"], "unmeasured_kpis": counts["NOT_MEASURED"],
        "benchmark_score_pct": percent(counts["PASS"], len(kpis)),
        "overall_status": "NOT_ESTABLISHED" if counts["NOT_MEASURED"] else "TARGET_FAILURES" if counts["FAIL"] else "MEASURED_TARGETS_MET",
        "scope": "Seven legacy operational benchmarks. FCR is automation, citation accuracy is ID validity, and PII is scanner detections. The 300-second P95 target is the legacy comparison threshold, not the framework's 3-second target. This is not confirmed resolution, independent quality verification or full framework compliance."},
        "kpis": kpis}


def print_evaluation_report(metrics):
    """Print the original four-section report with the seven selected KPIs."""
    def number(value, digits=2, suffix=""):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return "Not measured"
        return f"{value:.{digits}f}{suffix}"

    def section(title):
        print("\n" + "-" * 100)
        print(title)
        print("-" * 100)

    volume = metrics.get("volume", {})
    business = metrics.get("business", {})
    technical = metrics.get("technical", {})
    governance = metrics.get("governance", {})
    latency = technical.get("latency", {})
    contract = evaluate_contract_kpis(metrics)
    summary = contract["summary"]
    total = metrics.get("total_tickets_evaluated", volume.get("tickets_processed", 0))
    blocked = volume.get("blocked_by_guardrails", 0)
    blocked_pct = percent(blocked, total)

    print("\n" + "=" * 100)
    print("      CLOUDSERVE INTELLIGENT SUPPORT -- EVALUATION METRICS REPORT")
    print("=" * 100)
    print(f"Evaluated At : {metrics.get('timestamp', 'N/A')}")
    print(f"Tickets Run  : {total} unattended tickets")
    print(f"KPI Status  : {summary['achieved_kpis']}/{summary['total_kpis']} selected benchmarks achieved "
          f"({number(summary['benchmark_score_pct'], 1, '%')}); "
          f"{summary['missed_kpis']} failed; {summary.get('unmeasured_kpis', 0)} not measured "
          f"[{summary['overall_status']}]")

    section(" [1] VOLUME METRICS")
    print(f"  Tickets Processed        : {total}")
    print(f"  Answered Automatically   : {volume.get('answered_automatically', 0)} "
          f"({number(volume.get('answered_automatically_pct'), suffix='%')})")
    print(f"  Escalated                : {volume.get('escalated', 0)} "
          f"({number(volume.get('escalated_pct'), suffix='%')})")
    print(f"  Blocked by Guardrails    : {blocked} ({number(blocked_pct, suffix='%')})")

    section(" [2] BUSINESS OUTCOMES")
    print(f"  First Contact Resolution*: {number(business.get('automation_rate_pct'), suffix='%')}  [Target: >= 60.0%]")
    print(f"  Escalation Rate          : {number(business.get('escalation_rate_pct'), suffix='%')}  [Target: <= 30.0%]")
    print(f"  Mean Response Time       : {number(latency.get('mean_seconds'), 3, 's')}")
    print(f"  Median Response Time     : {number(latency.get('median_seconds'), 3, 's')}")

    section(" [3] TECHNICAL PERFORMANCE")
    print(f"  Retrieval Hit Rate       : {number(technical.get('retrieval_hit_rate_pct'), suffix='%')} "
          f"({technical.get('retrieval_hits', 0)}/{technical.get('retrieval_evaluable_tickets', 0)} evaluable tickets)")
    print(f"  Citation Accuracy*       : {number(technical.get('citation_id_validity_pct'), suffix='%')}  [Target: >= 95.0%]")
    print(f"  Hallucination Rate*      : {number(technical.get('hallucination_proxy_rate_pct'), suffix='%')}  [Target: <= 5.0%]")
    print(f"  Latency (Median)         : {number(latency.get('median_seconds'), 3, 's')}")
    print(f"  Latency (P95)            : {number(latency.get('p95_seconds'), 3, 's')}  [Legacy benchmark: < 300.0s]")

    classes = technical.get("classification_precision_and_recall_per_class", {})
    if classes:
        print("\n  Classification Precision & Recall per Class:")
        print(f"    {'Class Name':<28} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>8} | {'Support':>7}")
        print("    " + "-" * 76)
        supported = []
        for name, stats in classes.items():
            if name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            print(f"    {name:<28} | {number(stats.get('precision_pct'), suffix='%'):>9} | "
                  f"{number(stats.get('recall_pct'), suffix='%'):>9} | "
                  f"{number(stats.get('f1_score'), 3):>8} | {stats.get('support', 0):>7}")
            supported.append(stats)
        overall = technical.get("overall_classification", {})
        support = sum(row.get("support", 0) for row in supported)
        weighted_f1 = sum(row.get("f1_score", 0) * row.get("support", 0) for row in supported) / support if support else None
        print("    " + "-" * 76)
        print(f"    {'Weighted Average':<28} | {number(overall.get('precision_pct'), suffix='%'):>9} | "
              f"{number(overall.get('recall_pct'), suffix='%'):>9} | {number(weighted_f1, 3):>8} | {support:>7}")

    section(" [4] GOVERNANCE & RISK")
    recon = governance.get("decision_reconciliation", metrics.get("reconciliation_check", {}))
    logged = governance.get("decisions_logged")
    print(f"  Decisions Logged in Audit: {logged if logged is not None else 'Not verified'} "
          f"(Reconciled: {recon.get('is_reconciled', False)}, Coverage: {number(recon.get('coverage_pct'), suffix='%')})")
    print(f"  Private Data Detections  : {governance.get('draft_pii_detections', governance.get('private_data_detections', 0))} draft scanner detections")
    activations = governance.get("guardrail_activations_by_type", {})
    grounding_blocks = sum(
        activations.get(key, 0) for key in ("grounding", "citation")
    )  # These counters describe checks; one draft can trigger multiple checks.
    print("  Guardrail Activations by Type:")
    for label, count in (
        ("PII / Secrets Leakage", activations.get("pii", activations.get("pii_leakage", 0))),
        ("Citation / Grounding Failure", activations.get("grounding_or_citation", grounding_blocks)),
        ("Prompt Injection Jailbreak", activations.get("instruction_integrity", activations.get("prompt_injection", 0))),
        ("Unauthorized Commitments", activations.get("commitments", activations.get("unauthorized_commitments", 0))),
    ):
        print(f"    - {label:<32}: {count}")
    print(f"  Total Guardrail Blocks   : {governance.get('total_guardrail_blocks', blocked)}")

    section(" SELECTED KPI BENCHMARK SCORECARD")
    rows = contract["kpis"]
    name_width = max(35, max((len(row["name"]) for row in rows), default=0))
    print(f"{'KPI ID':<7} | {'Metric Name':<{name_width}} | {'Target':<15} | {'Actual':<15} | Status")
    print("-" * (name_width + 65))
    for row in rows:
        print(f"{row['id']:<7} | {row['name']:<{name_width}} | {row['target_display']:<15} | "
              f"{row['achieved_display']:<15} | [{row['status']}]")

    fairness = metrics.get("fairness_audit", {})
    if fairness.get("by_customer_tier") or fairness.get("by_language_fluency"):
        print("\nFAIRNESS AUDIT -- SEGMENTED BREAKDOWN:")
        for key, title in (("by_customer_tier", "Customer Tiers"), ("by_language_fluency", "Language Fluency")):
            if not fairness.get(key):
                continue
            print(f"  {title}:")
            for name, stats in fairness[key].items():
                automation = stats.get("automation_rate_pct", stats.get("fcr_rate_pct"))
                accuracy = stats.get("intent_accuracy_pct", stats.get("intent_precision_pct"))
                print(f"    - {name.capitalize():<12}: N={stats.get('ticket_count', 0):<3} | "
                      f"Automation: {number(automation, 1, '%'):>6} | Intent Accuracy: {number(accuracy, 1, '%'):>6}")
    print("\n* FCR = auto-response proxy; citation accuracy = retrieved-ID validity;")
    print("  hallucination = citation/reference-document mismatch heuristic. Times measure processing.")
    print("  These operational benchmarks do not establish customer resolution or full framework compliance.")
    print("=" * 100 + "\n")


def run_evaluation(
    input_path: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    use_llm: bool = True,
    db_path: Optional[str] = None,
    ticket_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs unattended batch evaluation over a dataset and produces metrics report."""
    if output_path:
        destination = Path(output_path).resolve()
        if destination.is_dir() or destination.suffix.lower() != ".json":
            destination = destination / "latest_report.json"
        output_path = str(destination)
    logger.info(f"Loading evaluation tickets from: {input_path}")
    source_bytes = Path(input_path).read_bytes()
    inputs = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Evaluation input must be a nonempty JSON array")

    if limit and limit > 0:
        inputs = inputs[:limit]

    run_id = f"RUN-{uuid.uuid4().hex}"
    started_at = datetime.now(timezone.utc).isoformat()
    raw_tickets: List[Optional[NormalizedTicket]] = []
    logger.info(f"Starting evaluation {run_id} on {len(inputs)} inputs...")

    results: List[Dict[str, Any]] = []
    timings: List[float] = []

    for idx, item in enumerate(inputs, 1):
        ticket = None
        t0 = time.perf_counter()
        try:
            ticket = ingest_ticket(item)
            logger.info("Ticket %s/%s: Processing ticket %s", idx, len(inputs), ticket.ticket_id)
            final_state = process_ticket(
                ticket,
                db_path=db_path,
                use_llm=use_llm,
                run_id=run_id,
                input_index=idx,
            )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            results.append(final_state)
            if idx % 10 == 0 or idx == len(inputs):
                logger.info(f"Progress: [{idx}/{len(inputs)}] tickets processed (last: {elapsed:.2f}s).")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            supplied_id = item.get("ticket_id") if isinstance(item, dict) else None
            ticket_id = ticket.ticket_id if ticket else (supplied_id if isinstance(supplied_id, str) and supplied_id.strip() else f"INVALID-{run_id}-{idx}")
            reason = "invalid_input" if ticket is None else "processing_exception"
            error_type = type(e).__name__
            logger.error("Input %s: %s (%s)", idx, reason, error_type)
            failure = {
                "ticket_id": ticket_id,
                "run_id": run_id,
                "input_index": idx,
                "route": "escalate",
                "escalation_reason": reason,
                "error": error_type,
                "guardrail_passed": False,
                "logged_to_audit_store": False,
                "escalation_packet": {"ticket_id": ticket_id, "run_id": run_id, "input_index": idx,
                    "handover_notes": f"Human review required: {reason} ({error_type}). Locate the original input using the run and input index."},
            }
            try:
                failure["decision_id"] = log_decision(DecisionRecord(
                    ticket_id=ticket_id, run_id=run_id, input_index=idx,
                    stage="ingestion" if ticket is None else "triage",
                    action_taken="escalate", reason=f"{reason}: {error_type}",
                    guardrail_results={"pii": "not_run", "grounding": "not_run", "citation": "not_run"},
                ), db_path=db_path)
                failure["logged_to_audit_store"] = True
            except Exception as audit_error:
                failure["audit_error"] = type(audit_error).__name__
                logger.error("Could not persist failure for input %s (%s)", idx, type(audit_error).__name__)
            results.append(failure)
        raw_tickets.append(ticket)

    # Coverage reconciliation check against SQLite audit store
    ticket_ids = [r["ticket_id"] for r in results]
    try:
        reconciliation = reconcile_decisions_with_tickets(ticket_ids, db_path=db_path, run_id=run_id)
    except Exception as audit_error:
        reconciliation = {"run_id": run_id, "is_reconciled": False,
            "processed_ticket_count": len(inputs), "logged_ticket_count": None,
            "coverage_pct": 0.0, "audit_error": type(audit_error).__name__,
            "missing_ticket_ids": [], "extra_ticket_ids": [], "verification_status": "unavailable"}
    logger.info(f"Decision Audit Reconciliation: {reconciliation['is_reconciled']} ({reconciliation['logged_ticket_count']}/{reconciliation['processed_ticket_count']} logged)")

    # Build detailed per-ticket execution records
    ticket_records: List[Dict[str, Any]] = []
    for norm, res, elapsed in zip(raw_tickets, results, timings):
        labels = norm.labels if norm else None
        labels_dict = None
        if labels:
            labels_dict = {
                "intent": labels.intent,
                "urgency": labels.urgency,
                "expected_route": labels.expected_route,
                "expected_doc_ids": labels.expected_doc_ids or [],
                "must_not_auto_respond": labels.must_not_auto_respond,
                "answerable_from_docs": labels.answerable_from_docs,
            }

        retrieved_passages = res.get("retrieved_passages", [])
        retrieved_doc_ids = [p.get("doc_id") for p in retrieved_passages if p.get("doc_id")]
        retrieved_docs_with_scores = [
            {"doc_id": p.get("doc_id"), "score": p.get("score"), "title": p.get("title", "")}
            for p in retrieved_passages if p.get("doc_id")
        ]
        cited_doc_ids = re.findall(r"\[(DOC-[A-Za-z0-9_-]+)\]", res.get("generated_response") or "")
        valid_citations = [c for c in cited_doc_ids if c in retrieved_doc_ids]
        expected_docs_set = set(labels.expected_doc_ids) if (labels and labels.expected_doc_ids) else set()

        reference_doc_mismatch = None
        route_decision = res.get("route")
        if route_decision == "auto_respond" and res.get("generated_response"):
            if expected_docs_set:
                reference_doc_mismatch = not bool(set(cited_doc_ids) & expected_docs_set)

        rec = {
            "ticket_id": res["ticket_id"],
            "run_id": run_id,
            "input_index": len(ticket_records) + 1,
            "input_valid": norm is not None,
            "customer_id": norm.customer_id if norm else None,
            "customer_tier": norm.customer_tier if norm else None,
            "customer_region": norm.customer_region if norm else None,
            "language_fluency": norm.language_fluency if norm else None,
            "channel": norm.channel.value if norm else None,
            "subject": norm.subject if norm else None,
            "body": norm.body if norm else None,
            "clean_text": res.get("clean_text") or (f"{norm.subject} {norm.body}".strip() if norm else None),
            "ground_truth_labels": labels_dict,
            "execution": {
                "latency_seconds": round(elapsed, 3),
                "predicted_intent": res.get("intent"),
                "classification_confidence": res.get("classification_confidence"),
                "intent_match": (res.get("intent") == labels.intent) if (labels and labels.intent) else None,
                "predicted_urgency": res.get("urgency"),
                "urgency_reason": res.get("classification_reasoning") or res.get("urgency_reason"),
                "urgency_match": (res.get("urgency") == labels.urgency) if (labels and labels.urgency) else None,
                "route_decision": res.get("route"),
                "route_match": (res.get("route") == labels.expected_route) if (labels and labels.expected_route) else None,
                "routing_rule_matched": (", ".join(res.get("policy_flags", [])) if res.get("policy_flags") else None) or res.get("routing_rule_matched"),
                "escalation_reason": res.get("escalation_reason"),
                "escalation_packet": res.get("escalation_packet"),
                "retrieved_passages_count": len(retrieved_passages),
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_docs_with_scores": retrieved_docs_with_scores,
                "cited_doc_ids": cited_doc_ids,
                "valid_citations": valid_citations,
                "citation_ids_valid": (len(valid_citations) == len(cited_doc_ids)) if cited_doc_ids else None,
                "citations_accurate": None,
                "is_hallucinated": None,
                "reference_doc_mismatch": reference_doc_mismatch,
                "generated_response": res.get("generated_response"),
                "guardrail_passed": res.get("guardrail_passed"),
                "guardrail_results": res.get("guardrail_results", {}),
                "guardrail_block_reason": res.get("guardrail_block_reason"),
                "grounding_review": res.get("grounding_review", {}),
                "final_response_text": res.get("final_response_text") or res.get("generated_response"),
                "logged_to_audit_store": res.get("logged_to_audit_store", False),
                "decision_id": res.get("decision_id"),
                "audit_error": res.get("audit_error"),
                "error": res.get("error"),
            }
        }
        ticket_records.append(rec)

    # Compute metrics
    metrics = compute_metrics(results, raw_tickets, timings, reconciliation=reconciliation)
    metrics["reconciliation_check"] = reconciliation
    metrics["run_id"] = run_id
    metrics["dataset"] = str(Path(input_path).resolve())
    metrics["dataset_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    metrics["started_at"] = started_at
    metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
    metrics["execution_mode"] = "llm_enabled" if use_llm else "offline"
    metrics["configuration"] = {key: os.getenv(key, default) for key, default in {
        "MODEL_NAME": "meta-llama/llama-3.1-8b-instruct", "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CONFIDENCE_THRESHOLD": "0.80", "RETRIEVAL_SIMILARITY_THRESHOLD": "0.40",
        "RETRIEVAL_TOP_K": "3", "GUARDRAIL_CONFIDENCE_THRESHOLD": "0.70",
        "KILL_SWITCH_ACTIVE": "false"}.items()}
    metrics["source_sha256"] = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ("src/*.py", "evaluation/*.py", "prompts/build/*.txt")
        for path in sorted(PROJECT_ROOT.glob(pattern))}
    metrics["run_status"] = "completed" if reconciliation["is_reconciled"] else "audit_incomplete"
    metrics["invalid_input_count"] = sum(t is None for t in raw_tickets)
    metrics["processing_error_count"] = sum(t is not None and bool(r.get("error")) for t, r in zip(raw_tickets, results))
    metrics["contract_kpis"] = evaluate_contract_kpis(metrics)

    # Save output report FIRST
    # Save output report JSON
    if output_path:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation report successfully saved to: {out_file}")

    # Save per-ticket execution results JSON (File 2)
    ticket_payload = {
        "metrics_schema_version": metrics["metrics_schema_version"],
        "run_id": run_id,
        "dataset_sha256": metrics["dataset_sha256"],
        "execution_mode": metrics["execution_mode"],
        "run_status": metrics["run_status"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": input_path,
        "total_tickets": len(ticket_records),
        "auto_responded_count": sum(1 for r in ticket_records if r["execution"]["route_decision"] == "auto_respond"),
        "escalated_count": sum(1 for r in ticket_records if r["execution"]["route_decision"] == "escalate"),
        "tickets": ticket_records,
    }

    if ticket_output_path:
        canonical_tickets_json = Path(ticket_output_path).resolve()
    elif output_path:
        out_dir = Path(output_path).resolve().parent
        canonical_tickets_json = out_dir / "latest_ticket_results.json"
    else:
        canonical_tickets_json = Path("evaluation/results/latest_ticket_results.json").resolve()

    canonical_tickets_json.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_tickets_json, "w", encoding="utf-8") as f:
        json.dump(ticket_payload, f, indent=2)
    logger.info(f"Saved detailed per-ticket results to: {canonical_tickets_json}")

    # Print summary report safely
    try:
        print_evaluation_report(metrics)
    except Exception as e:
        logger.warning(f"Could not print console report: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CloudServe Support Evaluation Harness (AC-A9)")
    parser.add_argument(
        "--input",
        type=str,
        default="data/validation_tickets.json",
        help="Path to evaluation dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="Output directory (writes latest_report.json and latest_ticket_results.json), or an explicit metrics .json path"
    )
    parser.add_argument(
        "--ticket-output",
        type=str,
        default=None,
        help="Path to save detailed per-ticket execution results JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tickets to evaluate (useful for smoke tests)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run using heuristic classifier and deterministic generator (no external API)"
    )

    args = parser.parse_args()

    metrics = run_evaluation(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        use_llm=not args.offline,
        ticket_output_path=args.ticket_output,
    )
    if not metrics["reconciliation_check"]["is_reconciled"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
