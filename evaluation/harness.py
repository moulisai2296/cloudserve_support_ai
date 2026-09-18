"""Standalone evaluation harness for CloudServe Solutions support triage.

Executes unattended batch evaluation across ticket datasets, computing all
Business Outcomes and Technical Performance metrics,
including segmented fairness audits across customer tiers and language fluency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from src.ingest import load_tickets_from_json, NormalizedTicket
from src.graph import process_ticket
from src.logging_store import count_decisions, reconcile_decisions_with_tickets

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


def compute_metrics(
    processed_results: List[Dict[str, Any]],
    raw_tickets: List[NormalizedTicket],
    timings: List[float],
    reconciliation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes all evaluation metrics structured into the 4 mandatory Build Specification groups:
      1. Volume: Tickets processed, answered automatically, escalated, blocked by guardrails.
      2. Business: First contact resolution, mean and median response time, escalation rate.
      3. Technical: Classification precision and recall per class, retrieval hit rate, latency at median and 95th percentile.
      4. Governance: Decisions logged, guardrail activations by type, any private data detections.
    """
    total = len(processed_results)
    if total == 0:
        return {"error": "No tickets processed"}

    # =========================================================================
    # Group 1: Volume Calculations
    # =========================================================================
    auto_responded_count = sum(1 for r in processed_results if r.get("route") == "auto_respond")
    escalated_count = sum(1 for r in processed_results if r.get("route") == "escalate")
    fcr_rate = (auto_responded_count / total) * 100.0
    escalation_rate = (escalated_count / total) * 100.0

    # Guardrail blocks & activations by type
    blocked_count = sum(1 for r in processed_results if r.get("guardrail_blocked", False) or not r.get("guardrail_passed", True))
    guardrail_breakdown = {
        "pii_leakage": 0,
        "grounding_or_citation": 0,
        "prompt_injection": 0,
        "unauthorized_commitments": 0,
    }
    pii_violations = 0

    for r in processed_results:
        is_blocked = r.get("guardrail_blocked", False) or not r.get("guardrail_passed", True)
        reason = (r.get("guardrail_block_reason") or "").lower()
        gr_dict = r.get("guardrail_results") or {}

        if "pii" in reason or gr_dict.get("pii") == "fail":
            guardrail_breakdown["pii_leakage"] += 1
            pii_violations += 1
        if "citation" in reason or "grounding" in reason or gr_dict.get("citation") == "fail" or gr_dict.get("grounding") == "fail":
            guardrail_breakdown["grounding_or_citation"] += 1
        if "injection" in reason or "instruction" in reason or gr_dict.get("instruction_integrity") == "fail":
            guardrail_breakdown["prompt_injection"] += 1
        if "commitment" in reason or "refund" in reason or gr_dict.get("commitments") == "fail":
            guardrail_breakdown["unauthorized_commitments"] += 1

    # =========================================================================
    # Group 2: Business & Latency Calculations
    # =========================================================================
    ordered_timings = sorted(timings)
    mean_latency = statistics.mean(timings) if timings else 0.0
    median_latency = statistics.median(timings) if timings else 0.0
    p95_idx = int(0.95 * len(ordered_timings)) - 1 if len(ordered_timings) >= 20 else len(ordered_timings) - 1
    p95_latency = ordered_timings[max(0, p95_idx)] if ordered_timings else 0.0

    # =========================================================================
    # Group 3: Technical Performance Calculations
    # =========================================================================
    true_intents = []
    pred_intents = []
    true_urgencies = []
    pred_urgencies = []

    valid_citations = 0
    total_citations_made = 0
    hallucinated_responses = 0

    retrieval_hits = 0
    retrieval_evaluable = 0

    tier_segments: Dict[str, Dict[str, Any]] = {}
    fluency_segments: Dict[str, Dict[str, Any]] = {}

    for norm, res in zip(raw_tickets, processed_results):
        labels = norm.labels or None
        t_tier = norm.customer_tier.lower()
        t_fluency = norm.language_fluency.lower()

        tier_segments.setdefault(t_tier, {"total": 0, "auto": 0, "correct_intent": 0})
        fluency_segments.setdefault(t_fluency, {"total": 0, "auto": 0, "correct_intent": 0})

        tier_segments[t_tier]["total"] += 1
        fluency_segments[t_fluency]["total"] += 1

        if res.get("route") == "auto_respond":
            tier_segments[t_tier]["auto"] += 1
            fluency_segments[t_fluency]["auto"] += 1

        pred_intent = res.get("intent", "unclear_request")
        pred_urgency = res.get("urgency", "medium")
        route_decision = res.get("route", "escalate")

        if labels:
            if labels.intent:
                true_intents.append(labels.intent)
                pred_intents.append(pred_intent)
                if pred_intent == labels.intent:
                    tier_segments[t_tier]["correct_intent"] += 1
                    fluency_segments[t_fluency]["correct_intent"] += 1

            if labels.urgency:
                true_urgencies.append(labels.urgency)
                pred_urgencies.append(pred_urgency)


            # Retrieval hit rate: check if expected docs were retrieved
            expected_docs = labels.expected_doc_ids or []
            if expected_docs:
                retrieval_evaluable += 1
                retrieved_ids = [p.get("doc_id") for p in res.get("retrieved_passages", [])]
                if any(doc in retrieved_ids for doc in expected_docs):
                    retrieval_hits += 1

        # Citation Accuracy & Hallucination checks
        cited = res.get("cited_doc_ids", [])
        retrieved_docs = {p.get("doc_id") for p in res.get("retrieved_passages", [])}
        expected_docs_set = set(labels.expected_doc_ids) if (labels and labels.expected_doc_ids) else set()

        if route_decision == "auto_respond" and res.get("generated_response"):
            total_citations_made += len(cited)
            valid_for_this = sum(1 for c in cited if c in retrieved_docs)
            valid_citations += valid_for_this

            if any(c not in retrieved_docs for c in cited) or (expected_docs_set and not (set(cited) & expected_docs_set)):
                hallucinated_responses += 1

    # Intent Precision & Recall per class and overall
    per_class_classification: Dict[str, Any] = {}
    if true_intents and pred_intents:
        intent_accuracy = accuracy_score(true_intents, pred_intents) * 100.0
        intent_precision = precision_score(true_intents, pred_intents, average="weighted", zero_division=0) * 100.0
        intent_recall = recall_score(true_intents, pred_intents, average="weighted", zero_division=0) * 100.0

        clf_dict = classification_report(true_intents, pred_intents, output_dict=True, zero_division=0)
        for k, v in clf_dict.items():
            if isinstance(v, dict):
                per_class_classification[k] = {
                    "precision_pct": round(v["precision"] * 100.0, 2),
                    "recall_pct": round(v["recall"] * 100.0, 2),
                    "f1_score": round(v["f1-score"], 3),
                    "support": int(v["support"]),
                }
    else:
        intent_accuracy = intent_precision = intent_recall = 0.0

    urgency_accuracy = (accuracy_score(true_urgencies, pred_urgencies) * 100.0) if (true_urgencies and pred_urgencies) else 0.0

    citation_accuracy_pct = (valid_citations / total_citations_made * 100.0) if total_citations_made > 0 else 100.0
    hallucination_rate_pct = (hallucinated_responses / auto_responded_count * 100.0) if auto_responded_count > 0 else 0.0
    retrieval_hit_rate_pct = (retrieval_hits / retrieval_evaluable * 100.0) if retrieval_evaluable > 0 else 100.0

    # Fairness Segmentation Metrics
    fairness_tier_summary = {}
    for tier, data in tier_segments.items():
        cnt = data["total"]
        fairness_tier_summary[tier] = {
            "ticket_count": cnt,
            "fcr_rate_pct": round((data["auto"] / cnt * 100.0) if cnt > 0 else 0.0, 2),
            "intent_precision_pct": round((data["correct_intent"] / cnt * 100.0) if cnt > 0 else 0.0, 2),
        }

    fairness_fluency_summary = {}
    for fluency, data in fluency_segments.items():
        cnt = data["total"]
        fairness_fluency_summary[fluency] = {
            "ticket_count": cnt,
            "fcr_rate_pct": round((data["auto"] / cnt * 100.0) if cnt > 0 else 0.0, 2),
            "intent_precision_pct": round((data["correct_intent"] / cnt * 100.0) if cnt > 0 else 0.0, 2),
        }

    # Coverage reconciliation default
    reconciliation_active = reconciliation or {
        "is_reconciled": True,
        "processed_ticket_count": total,
        "logged_ticket_count": total,
        "coverage_pct": 100.0,
    }

    # =========================================================================
    # Construct 4 Explicit Build Specification Groups (Table 7)
    # =========================================================================
    volume_group = {
        "tickets_processed": total,
        "answered_automatically": auto_responded_count,
        "escalated": escalated_count,
        "blocked_by_guardrails": blocked_count,
        "answered_automatically_pct": round(fcr_rate, 2),
        "escalated_pct": round(escalation_rate, 2),
        "blocked_by_guardrails_pct": round((blocked_count / total * 100.0) if total > 0 else 0.0, 2),
    }

    business_group = {
        "first_contact_resolution_pct": round(fcr_rate, 2),
        "mean_response_time_seconds": round(mean_latency, 3),
        "median_response_time_seconds": round(median_latency, 3),
        "escalation_rate_pct": round(escalation_rate, 2),
    }

    technical_group = {
        "classification_precision_and_recall_per_class": per_class_classification,
        "overall_classification": {
            "precision_pct": round(intent_precision, 2),
            "recall_pct": round(intent_recall, 2),
            "accuracy_pct": round(intent_accuracy, 2),
        },
        "retrieval_hit_rate_pct": round(retrieval_hit_rate_pct, 2),
        "retrieval_hits": retrieval_hits,
        "retrieval_evaluable_tickets": retrieval_evaluable,
        "latency": {
            "mean_seconds": round(mean_latency, 3),
            "median_seconds": round(median_latency, 3),
            "p95_seconds": round(p95_latency, 3),
        },
        "urgency_accuracy_pct": round(urgency_accuracy, 2),
        "citation_accuracy_pct": round(citation_accuracy_pct, 2),
        "hallucination_rate_pct": round(hallucination_rate_pct, 2),
    }

    governance_group = {
        "decisions_logged": reconciliation_active.get("logged_ticket_count", total),
        "decision_reconciliation": reconciliation_active,
        "guardrail_activations_by_type": guardrail_breakdown,
        "total_guardrail_blocks": blocked_count,
        "private_data_detections": pii_violations,
    }

    raw_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tickets_evaluated": total,

        # Build Specification Table 7 Groups:
        "volume": volume_group,
        "business": business_group,
        "technical": technical_group,
        "governance": governance_group,

        # Backwards-compatible fields for existing scorecards & automated tests:
        "tier_one_business_outcomes": {
            "first_contact_resolution_rate_pct": round(fcr_rate, 2),
            "escalation_rate_pct": round(escalation_rate, 2),
            "auto_responded_count": auto_responded_count,
            "escalated_count": escalated_count,
            "latency": {
                "mean_seconds": round(mean_latency, 3),
                "median_seconds": round(median_latency, 3),
                "p95_seconds": round(p95_latency, 3),
            }
        },
        "tier_two_technical_performance": {
            "intent_classification_precision_pct": round(intent_precision, 2),
            "intent_classification_recall_pct": round(intent_recall, 2),
            "intent_classification_accuracy_pct": round(intent_accuracy, 2),
            "urgency_accuracy_pct": round(urgency_accuracy, 2),
            "citation_accuracy_pct": round(citation_accuracy_pct, 2),
            "hallucination_rate_pct": round(hallucination_rate_pct, 2),
            "pii_leakage_violations": pii_violations,
        },
        "fairness_audit": {
            "by_customer_tier": fairness_tier_summary,
            "by_language_fluency": fairness_fluency_summary,
        },
        "reconciliation_check": reconciliation_active,
    }

    raw_metrics["contract_kpis"] = evaluate_contract_kpis(raw_metrics)
    return raw_metrics


# ==============================================================================
# Contract KPI Specifications (SLA Benchmarks)
# ==============================================================================

CONTRACT_KPIS: List[Dict[str, Any]] = [
    {
        "id": "KPI-01",
        "name": "First Contact Resolution (FCR)",
        "category": "Tier-1 Business Outcomes",
        "target_operator": ">=",
        "target_value": 60.0,
        "target_display": ">= 60.0%",
        "metric_path": ["tier_one_business_outcomes", "first_contact_resolution_rate_pct"],
        "unit": "%",
        "policy_rule": "Auto-responds only when classifier confidence >= 0.80 and authoritative docs are grounded.",
        "description": "Rate of inbound customer inquiries successfully resolved by autonomous AI without requiring Tier-2 human escalation."
    },
    {
        "id": "KPI-02",
        "name": "Escalation Rate",
        "category": "Tier-1 Business Outcomes",
        "target_operator": "<=",
        "target_value": 30.0,
        "target_display": "<= 30.0%",
        "metric_path": ["tier_one_business_outcomes", "escalation_rate_pct"],
        "unit": "%",
        "policy_rule": "Deterministic policy gate safely escalates low-confidence (<0.80), high-risk intents, and policy exclusions.",
        "description": "Rate of tickets routed to human support teams; must remain controlled to avoid agent burnout while prioritizing safety."
    },
    {
        "id": "KPI-03",
        "name": "P95 Response Latency",
        "category": "Tier-1 Business Outcomes",
        "target_operator": "<",
        "target_value": 300.0,
        "target_display": "< 300.0s (5m)",
        "metric_path": ["tier_one_business_outcomes", "latency", "p95_seconds"],
        "unit": "s",
        "policy_rule": "Sub-second vector retrieval + streaming LLM generation pipeline.",
        "description": "95th percentile turnaround time from initial ticket ingestion to customer delivery or escalation dispatch."
    },
    {
        "id": "KPI-04",
        "name": "Intent Classification Precision",
        "category": "Tier-2 Technical Performance",
        "target_operator": ">=",
        "target_value": 85.0,
        "target_display": ">= 85.0%",
        "metric_path": ["tier_two_technical_performance", "intent_classification_precision_pct"],
        "unit": "%",
        "policy_rule": "Few-shot calibrated prompt classifier over 22 ground-truth intents with confidence scoring.",
        "description": "Weighted precision score comparing predicted support intent against verified ground-truth labels."
    },
    {
        "id": "KPI-05",
        "name": "PII & Credential Leakage Violations",
        "category": "Governance & Safety Guardrails",
        "target_operator": "==",
        "target_value": 0,
        "target_display": "0 violations",
        "metric_path": ["tier_two_technical_performance", "pii_leakage_violations"],
        "unit": "count",
        "policy_rule": "Strict safety guardrail: Block and escalate. Never redact and send.",
        "description": "Zero tolerance for leaking API keys, passwords, bearer tokens, or sensitive customer identifiable info."
    },
    {
        "id": "KPI-06",
        "name": "Audit Decision Reconciliation Coverage",
        "category": "Governance & Safety Guardrails",
        "target_operator": "==",
        "target_value": 100.0,
        "target_display": "100.0% (0 gaps)",
        "metric_path": ["reconciliation_check", "coverage_pct"],
        "unit": "%",
        "policy_rule": "100% of triage decisions must be permanently recorded in SQLite store conforming to the 15-field schema.",
        "description": "1:1 coverage reconciliation ensuring zero ghost or unlogged decisions across all 4 customer channels."
    },
]


def evaluate_contract_kpis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all contract KPIs comparing target SLAs against actual achieved metrics.
    Adds clear achieved booleans, status badges ('PASS' / 'FAIL'), variance deltas,
    and status colors ('green' / 'red').
    """
    evaluated_kpis: List[Dict[str, Any]] = []
    achieved_count = 0

    # Ensure reconciliation_check has coverage_pct based on processed tickets matched
    recon = metrics.get("reconciliation_check") or {}
    processed = recon.get("processed_ticket_count", metrics.get("total_tickets_evaluated", 0))
    missing_count = len(recon.get("missing_ticket_ids", []))
    matched = max(0, processed - missing_count)
    recon["coverage_pct"] = round((matched / processed * 100.0) if processed > 0 else 100.0, 2)
    metrics["reconciliation_check"] = recon

    for spec in CONTRACT_KPIS:
        curr: Any = metrics
        for k in spec["metric_path"]:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                curr = None
                break

        val: float = float(curr) if curr is not None else 0.0
        op = spec["target_operator"]
        target = float(spec["target_value"])

        if op == ">=":
            achieved = bool(val >= target)
        elif op == "<=":
            achieved = bool(val <= target)
        elif op == "<":
            achieved = bool(val < target)
        elif op == "==":
            achieved = bool(round(val, 2) == round(target, 2))
        else:
            achieved = False

        if achieved:
            achieved_count += 1

        variance = round(val - target, 2)
        unit = spec["unit"]
        if unit == "%":
            achieved_display = f"{val:.2f}%"
            variance_display = f"{variance:+.2f}%"
        elif unit == "s":
            achieved_display = f"{val:.2f}s"
            variance_display = f"{variance:+.2f}s"
        else:
            achieved_display = f"{int(val)} violations" if "violation" in spec["name"].lower() else str(int(val))
            variance_display = f"{int(variance):+d}"

        evaluated_kpis.append({
            "id": spec["id"],
            "name": spec["name"],
            "category": spec["category"],
            "target_operator": op,
            "target_value": target,
            "target_display": spec["target_display"],
            "achieved_value": val,
            "achieved_display": achieved_display,
            "variance": variance,
            "variance_display": variance_display,
            "achieved": achieved,
            "status": "PASS" if achieved else "FAIL",
            "status_color": "green" if achieved else "red",
            "policy_rule": spec["policy_rule"],
            "description": spec["description"],
        })

    total_kpis = len(CONTRACT_KPIS)
    missed_count = total_kpis - achieved_count
    compliance_score = round(achieved_count / total_kpis * 100.0, 1)

    return {
        "summary": {
            "total_kpis": total_kpis,
            "achieved_kpis": achieved_count,
            "missed_kpis": missed_count,
            "compliance_score_pct": compliance_score,
            "overall_status": "COMPLIANT" if missed_count == 0 else "NON_COMPLIANT",
            "overall_status_display": f"{compliance_score}% Contract Compliance ({achieved_count}/{total_kpis} KPIs Achieved)",
            "overall_color": "green" if missed_count == 0 else "red",
        },
        "kpis": evaluated_kpis,
    }


def print_evaluation_report(metrics: Dict[str, Any]) -> None:
    """Prints a clean ASCII scorecard summarizing the 4 Build Spec groups and contract SLAs."""
    contract_data = metrics.get("contract_kpis") or evaluate_contract_kpis(metrics)
    summary = contract_data["summary"]
    kpis = contract_data["kpis"]

    vol = metrics.get("volume", {})
    biz = metrics.get("business", {})
    tech = metrics.get("technical", {})
    gov = metrics.get("governance", {})

    print("\n" + "=" * 85)
    print("      CLOUDSERVE INTELLIGENT SUPPORT -- EVALUATION METRICS REPORT      ")
    print("=" * 85)
    print(f"Evaluated At : {metrics.get('timestamp', 'N/A')}")
    print(f"Tickets Run  : {metrics.get('total_tickets_evaluated', 0)} unattended tickets")
    print(f"Compliance   : {summary['achieved_kpis']}/{summary['total_kpis']} Contract KPIs Achieved ({summary['compliance_score_pct']}%) [{summary['overall_status']}]")

    # Group 1: Volume
    print("\n" + "-" * 85)
    print(" [1] VOLUME METRICS")
    print("-" * 85)
    print(f"  Tickets Processed        : {vol.get('tickets_processed', metrics.get('total_tickets_evaluated', 0))}")
    print(f"  Answered Automatically   : {vol.get('answered_automatically', 0)} ({vol.get('answered_automatically_pct', 0.0):.2f}%)")
    print(f"  Escalated                : {vol.get('escalated', 0)} ({vol.get('escalated_pct', 0.0):.2f}%)")
    print(f"  Blocked by Guardrails    : {vol.get('blocked_by_guardrails', 0)} ({vol.get('blocked_by_guardrails_pct', 0.0):.2f}%)")

    # Group 2: Business Outcomes
    print("\n" + "-" * 85)
    print(" [2] BUSINESS OUTCOMES")
    print("-" * 85)
    print(f"  First Contact Resolution : {biz.get('first_contact_resolution_pct', 0.0):.2f}%  [Contract Target: >= 60.0%]")
    print(f"  Escalation Rate          : {biz.get('escalation_rate_pct', 0.0):.2f}%  [Contract Target: <= 30.0%]")
    print(f"  Mean Response Time       : {biz.get('mean_response_time_seconds', 0.0):.3f}s")
    print(f"  Median Response Time     : {biz.get('median_response_time_seconds', 0.0):.3f}s")

    # Group 3: Technical Performance
    print("\n" + "-" * 85)
    print(" [3] TECHNICAL PERFORMANCE")
    print("-" * 85)
    evaluable = tech.get('retrieval_evaluable_tickets', 0)
    hits = tech.get('retrieval_hits', 0)
    print(f"  Retrieval Hit Rate       : {tech.get('retrieval_hit_rate_pct', 100.0):.2f}% ({hits}/{evaluable} evaluable tickets)")
    lat = tech.get('latency', {})
    print(f"  Latency (Median)         : {lat.get('median_seconds', 0.0):.3f}s")
    print(f"  Latency (P95)            : {lat.get('p95_seconds', 0.0):.3f}s  [Contract Target: < 300.0s]")

    per_class = tech.get("classification_precision_and_recall_per_class", {})
    if per_class:
        print("\n  Classification Precision & Recall per Class:")
        print(f"    {'Class Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<9} | {'Support'}")
        print("    " + "-" * 73)
        for cname, cstats in per_class.items():
            if cname in ["accuracy", "macro avg", "weighted avg"]:
                continue
            print(f"    {cname:<28} | {cstats['precision_pct']:>8.2f}% | {cstats['recall_pct']:>8.2f}% | {cstats['f1_score']:>8.3f} | {cstats['support']:>6}")
        print("    " + "-" * 73)
        if "weighted avg" in per_class:
            w = per_class["weighted avg"]
            print(f"    {'Weighted Average':<28} | {w['precision_pct']:>8.2f}% | {w['recall_pct']:>8.2f}% | {w['f1_score']:>8.3f} | {w['support']:>6}")

    # Group 4: Governance & Risk
    print("\n" + "-" * 85)
    print(" [4] GOVERNANCE & RISK")
    print("-" * 85)
    recon = gov.get("decision_reconciliation", metrics.get("reconciliation_check", {}))
    print(f"  Decisions Logged in Audit: {gov.get('decisions_logged', 0)} (Reconciled: {recon.get('is_reconciled', True)}, Coverage: {recon.get('coverage_pct', 100.0):.1f}%)")
    print(f"  Private Data Detections  : {gov.get('private_data_detections', 0)} violations  [Contract Target: 0 violations]")
    gr_act = gov.get("guardrail_activations_by_type", {})
    print("  Guardrail Activations by Type:")
    print(f"    - PII / Secrets Leakage           : {gr_act.get('pii_leakage', 0)}")
    print(f"    - Citation / Grounding Failure    : {gr_act.get('grounding_or_citation', 0)}")
    print(f"    - Prompt Injection Jailbreak      : {gr_act.get('prompt_injection', 0)}")
    print(f"    - Unauthorized Commitments        : {gr_act.get('unauthorized_commitments', 0)}")
    print(f"  Total Guardrail Blocks              : {gov.get('total_guardrail_blocks', 0)}")

    # Contract KPI SLA Benchmark
    print("\n" + "-" * 85)
    print(" CONTRACT SLA BENCHMARK SCORECARD")
    print("-" * 85)
    print(f"{'KPI ID':<7} | {'Metric Name':<35} | {'Target':<14} | {'Actual':<11} | {'Status'}")
    print("-" * 85)
    for kpi in kpis:
        status_bracket = f"[{kpi['status']}]"
        t_disp = str(kpi['target_display']).replace('≥', '>=').replace('≤', '<=')
        a_disp = str(kpi['achieved_display'])
        print(f"{kpi['id']:<7} | {kpi['name']:<35} | {t_disp:<14} | {a_disp:<11} | {status_bracket}")

    # Segmented fairness summary
    fairness = metrics.get("fairness_audit", {})
    if "by_customer_tier" in fairness:
        print("\nFAIRNESS AUDIT -- SEGMENTED BREAKDOWN:")
        print("  Customer Tiers:")
        for tier, stats in fairness["by_customer_tier"].items():
            print(f"    - {tier.capitalize():<12} : N={stats['ticket_count']:<3} | FCR: {stats['fcr_rate_pct']:>5.1f}% | Intent Prec: {stats['intent_precision_pct']:>5.1f}%")

    if "by_language_fluency" in fairness:
        print("  Language Fluency:")
        for flu, stats in fairness["by_language_fluency"].items():
            print(f"    - {flu.capitalize():<12} : N={stats['ticket_count']:<3} | FCR: {stats['fcr_rate_pct']:>5.1f}% | Intent Prec: {stats['intent_precision_pct']:>5.1f}%")

    print("=" * 85 + "\n")


def run_evaluation(
    input_path: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    use_llm: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs unattended batch evaluation over a dataset and produces metrics report."""
    logger.info(f"Loading evaluation tickets from: {input_path}")
    raw_tickets = load_tickets_from_json(input_path)

    if limit and limit > 0:
        raw_tickets = raw_tickets[:limit]

    logger.info(f"Starting evaluation on {len(raw_tickets)} tickets...")

    results: List[Dict[str, Any]] = []
    timings: List[float] = []

    for idx, ticket in enumerate(raw_tickets, 1):
        logger.info(f"Ticket {idx}/{len(raw_tickets)}: Processing ticket {ticket.ticket_id}")
        t0 = time.perf_counter()
        try:
            final_state = process_ticket(
                ticket,
                db_path=db_path,
                use_llm=use_llm,
            )
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            results.append(final_state)
            if idx % 10 == 0 or idx == len(raw_tickets):
                logger.info(f"Progress: [{idx}/{len(raw_tickets)}] tickets processed (last: {elapsed:.2f}s).")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            logger.error(f"Error processing ticket {ticket.ticket_id}: {e}")
            results.append({
                "ticket_id": ticket.ticket_id,
                "route": "escalate",
                "escalation_reason": "evaluation_exception",
                "error": str(e),
            })

    # Coverage reconciliation check against SQLite audit store
    ticket_ids = [t.ticket_id for t in raw_tickets]
    reconciliation = reconcile_decisions_with_tickets(ticket_ids, db_path=db_path)
    logger.info(f"Decision Audit Reconciliation: {reconciliation['is_reconciled']} ({reconciliation['logged_ticket_count']}/{reconciliation['processed_ticket_count']} logged)")

    # Build detailed per-ticket execution records
    ticket_records: List[Dict[str, Any]] = []
    for norm, res, elapsed in zip(raw_tickets, results, timings):
        labels = norm.labels
        labels_dict = None
        if labels:
            labels_dict = {
                "intent": labels.intent,
                "urgency": labels.urgency,
                "expected_route": labels.expected_route,
                "expected_doc_ids": labels.expected_doc_ids or [],
            }

        retrieved_doc_ids = [p.get("doc_id") for p in res.get("retrieved_passages", []) if p.get("doc_id")]
        cited_doc_ids = res.get("cited_doc_ids", [])
        valid_citations = [c for c in cited_doc_ids if c in retrieved_doc_ids]

        rec = {
            "ticket_id": norm.ticket_id,
            "customer_id": norm.customer_id,
            "customer_tier": norm.customer_tier,
            "language_fluency": norm.language_fluency,
            "channel": str(norm.channel.value) if hasattr(norm.channel, "value") else str(norm.channel),
            "subject": norm.subject,
            "body": norm.body,
            "clean_text": res.get("clean_text") or (f"{norm.subject} {norm.body}".strip()),
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
                "retrieved_passages_count": len(res.get("retrieved_passages", [])),
                "retrieved_doc_ids": retrieved_doc_ids,
                "cited_doc_ids": cited_doc_ids,
                "valid_citations": valid_citations,
                "citations_accurate": (len(valid_citations) == len(cited_doc_ids)) if cited_doc_ids else True,
                "generated_response": res.get("generated_response"),
                "guardrail_passed": res.get("guardrail_passed", True),
                "guardrail_block_reason": res.get("guardrail_block_reason"),
                "final_response_text": res.get("final_response_text") or res.get("generated_response"),
                "logged_to_audit_store": res.get("logged_to_audit_store", True),
                "error": res.get("error"),
            }
        }
        ticket_records.append(rec)

    # Compute metrics
    metrics = compute_metrics(results, raw_tickets, timings, reconciliation=reconciliation)
    metrics["reconciliation_check"] = reconciliation
    metrics["contract_kpis"] = evaluate_contract_kpis(metrics)

    # Save output report FIRST
    # Save output report JSON
    if output_path:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation report successfully saved to: {out_file}")

        # Always maintain canonical evaluation/results/latest_report.json
        canonical_json = Path("evaluation/results/latest_report.json").resolve()
        if out_file != canonical_json:
            canonical_json.parent.mkdir(parents=True, exist_ok=True)
            with open(canonical_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Updated canonical report: {canonical_json}")

    # Save per-ticket execution results JSON (File 2)
    ticket_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": input_path,
        "total_tickets": len(ticket_records),
        "auto_responded_count": sum(1 for r in ticket_records if r["execution"]["route_decision"] == "auto_respond"),
        "escalated_count": sum(1 for r in ticket_records if r["execution"]["route_decision"] == "escalate"),
        "tickets": ticket_records,
    }

    if output_path:
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
        default="evaluation/results/latest_report.json",
        help="Path to save evaluation metrics report JSON (defaults to evaluation/results/latest_report.json)"
    )
    parser.add_argument(
        "--ticket-output",
        type=str,
        default="evaluation/results/latest_ticket_results.json",
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

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        use_llm=not args.offline,
    )


if __name__ == "__main__":
    main()
