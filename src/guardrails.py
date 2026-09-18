"""Safety guardrails and pre-release validation module for CloudServe Solutions.

Enforces the 5 mandatory Governance Framework Section 4 checks:
1. Private data (PII & secrets leak detection)
2. Grounding & citation validity (strict verification against retrieved passages)
3. Instruction integrity (prompt injection & jailbreak detection)
4. Tone & scope (unauthorized refund/timeline commitments)
5. Confidence floor & completeness

Golden Rule: "Block and escalate. Never redact and send."
Adheres to Capstone Specification AC-A7, B-09, and FR-07.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_GUARDRAIL_CONFIDENCE_THRESHOLD = float(os.getenv("GUARDRAIL_CONFIDENCE_THRESHOLD", "0.70"))

# ==============================================================================
# Detection Patterns
# ==============================================================================

# PII & Secrets Patterns
API_KEY_REGEX = re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,}|key-[a-zA-Z0-9_\-]{20,}|Bearer\s+[a-zA-Z0-9_\-\.]{25,})\b", re.IGNORECASE)
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PASSWORD_EXPOSURE_REGEX = re.compile(r"(?:password|passwd|secret)\s*[:=]\s*['\"][^\s'\"]+['\"]", re.IGNORECASE)

# Prompt Injection & Jailbreak Patterns
PROMPT_INJECTION_REGEX = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|"
    r"you\s+are\s+now\s+in\s+developer\s+mode|"
    r"system\s+prompt\s*:|"
    r"DAN\s+mode|"
    r"disregard\s+all\s+rules)",
    re.IGNORECASE,
)

# Unauthorized Financial or Timeline Commitments
UNAUTHORIZED_COMMITMENTS_REGEX = re.compile(
    r"(\bwe\s+(will|hereby)\s+refund\b|"
    r"\bcredit\s+your\s+account\s+with\s+\$|"
    r"\bguarantee\s+(a\s+)?(full\s+)?refund\b|"
    r"\bwe\s+promise\s+this\s+feature\s+will\s+be\s+released\s+by\b|"
    r"\bguarantee\s+resolution\s+within\s+\d+\s+minutes\b)",
    re.IGNORECASE,
)

CITATION_REGEX = re.compile(r"\[(DOC-[A-Z0-9]+-\d+)\]")


class GuardrailVerdict(BaseModel):
    """Detailed audit verdict across all pre-release safety checks."""
    passed: bool
    pii_passed: bool = True
    grounding_passed: bool = True
    instruction_integrity_passed: bool = True
    commitments_passed: bool = True
    confidence_floor_passed: bool = True
    violations: List[str] = Field(default_factory=list)
    block_reason: Optional[str] = None
    guardrail_results: Dict[str, str] = Field(default_factory=lambda: {
        "pii": "pass",
        "grounding": "pass",
        "citation": "pass"
    })


def check_pii_and_secrets(text: str) -> tuple[bool, List[str]]:
    """Detects API keys, bearer tokens, credit cards, SSNs, and exposed passwords."""
    violations = []
    if API_KEY_REGEX.search(text):
        violations.append("Leaked API key or Bearer token detected in response")
    if CREDIT_CARD_REGEX.search(text):
        violations.append("Potential credit card number detected in response")
    if SSN_REGEX.search(text):
        violations.append("Social Security Number format detected in response")
    if PASSWORD_EXPOSURE_REGEX.search(text):
        violations.append("Hardcoded password or secret credential exposed in response")

    return len(violations) == 0, violations


def check_instruction_integrity(ticket_body: str, response_text: str = "") -> tuple[bool, List[str]]:
    """Detects jailbreak and prompt injection attempts in input ticket or response."""
    violations = []
    if PROMPT_INJECTION_REGEX.search(ticket_body):
        violations.append("Adversarial prompt injection attempt detected in customer ticket")
    if response_text and PROMPT_INJECTION_REGEX.search(response_text):
        violations.append("Model echoed or confirmed instruction override prompt")

    return len(violations) == 0, violations


def check_unauthorized_commitments(response_text: str) -> tuple[bool, List[str]]:
    """Detects unauthorized refund promises, financial credits, or roadmap guarantees."""
    violations = []
    if UNAUTHORIZED_COMMITMENTS_REGEX.search(response_text):
        violations.append("Unauthorized financial refund, credit, or delivery timeline commitment detected")

    return len(violations) == 0, violations


def check_grounding_and_citations(
    response_text: str,
    retrieved_passages: List[Dict[str, Any]]
) -> tuple[bool, List[str]]:
    """
    Verifies that every [DOC-ID] cited in response_text actually exists
    within the retrieved documentation passages.
    """
    violations = []
    valid_doc_ids = {p.get("doc_id") for p in retrieved_passages if p.get("doc_id")}

    cited_ids = CITATION_REGEX.findall(response_text)
    if not cited_ids:
        violations.append("Response contains no verifiable [DOC-ID] citations")

    for cited in cited_ids:
        if cited not in valid_doc_ids:
            violations.append(f"Invented citation detected: [{cited}] was not in retrieved context")

    return len(violations) == 0, violations


def validate_response(
    response_text: str,
    ticket_body: str,
    retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.90,
    confidence_threshold: Optional[float] = None,
) -> GuardrailVerdict:
    """
    Executes all 5 pre-release safety guardrails.
    Returns structured GuardrailVerdict. If any check fails, passed = False.
    """
    active_threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else DEFAULT_GUARDRAIL_CONFIDENCE_THRESHOLD
    )
    passages = retrieved_passages or []
    all_violations: List[str] = []

    # 1. PII Check
    pii_ok, pii_errs = check_pii_and_secrets(response_text)
    if not pii_ok:
        all_violations.extend(pii_errs)

    # 2. Instruction Integrity Check
    inj_ok, inj_errs = check_instruction_integrity(ticket_body, response_text)
    if not inj_ok:
        all_violations.extend(inj_errs)

    # 3. Tone & Scope Commitments Check
    comm_ok, comm_errs = check_unauthorized_commitments(response_text)
    if not comm_ok:
        all_violations.extend(comm_errs)

    # 4. Grounding & Citation Check
    ground_ok, ground_errs = check_grounding_and_citations(response_text, passages)
    if not ground_ok:
        all_violations.extend(ground_errs)

    # 5. Confidence Floor
    conf_ok = confidence >= active_threshold
    if not conf_ok:
        all_violations.append(f"Response confidence ({confidence:.2f}) below floor ({active_threshold:.2f})")

    overall_passed = (len(all_violations) == 0)
    block_reason = "; ".join(all_violations) if not overall_passed else None

    guardrail_results = {
        "pii": "pass" if pii_ok else "fail",
        "grounding": "pass" if ground_ok else "fail",
        "citation": "pass" if ground_ok else "fail",
        "instruction_integrity": "pass" if inj_ok else "fail",
        "commitments": "pass" if comm_ok else "fail",
        "confidence_floor": "pass" if conf_ok else "fail",
    }

    return GuardrailVerdict(
        passed=overall_passed,
        pii_passed=pii_ok,
        grounding_passed=ground_ok,
        instruction_integrity_passed=inj_ok,
        commitments_passed=comm_ok,
        confidence_floor_passed=conf_ok,
        violations=all_violations,
        block_reason=block_reason,
        guardrail_results=guardrail_results,
    )


# ==============================================================================
# LangGraph Node
# ==============================================================================

def guardrails_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node: Evaluates drafted response against safety guardrails.
    Enforces 'Block and escalate. Never redact and send.'
    """
    route = state.get("route", "escalate")
    # If already escalated, guardrails pass through
    if route == "escalate":
        return {
            "guardrail_passed": True,
            "guardrail_blocked": False,
            "guardrail_results": {
                "pii": "pass",
                "grounding": "pass",
                "citation": "pass"
            },
        }

    response_text = state.get("generated_response") or ""
    ticket_body = state.get("clean_text") or state.get("body", "")
    passages = state.get("retrieved_passages", [])
    confidence = float(state.get("generation_confidence", 0.90))

    verdict = validate_response(
        response_text=response_text,
        ticket_body=ticket_body,
        retrieved_passages=passages,
        confidence=confidence,
    )

    if not verdict.passed:
        logger.warning(f"Guardrail tripped on ticket {state.get('ticket_id')}: {verdict.block_reason}")
        # Rule: Block and escalate. Suppress drafted response.
        escalation_packet = state.get("escalation_packet") or {}
        escalation_packet["guardrail_block_reason"] = verdict.block_reason
        escalation_packet["violations"] = verdict.violations

        return {
            "route": "escalate",
            "escalation_reason": "guardrail_blocked",
            "guardrail_passed": False,
            "guardrail_blocked": True,
            "guardrail_block_reason": verdict.block_reason,
            "guardrail_results": verdict.guardrail_results,
            "generated_response": None,  # Suppress response
            "escalation_packet": escalation_packet,
        }

    return {
        "guardrail_passed": True,
        "guardrail_blocked": False,
        "guardrail_block_reason": None,
        "guardrail_results": verdict.guardrail_results,
    }
