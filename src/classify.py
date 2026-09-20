"""Classification module for CloudServe Solutions support tickets.

Implements multi-class intent classification (22 categories) and urgency detection
(high, medium, low) with calibrated numeric confidence and candidate alternatives.
Adheres to Capstone Specification AC-A3 and B-06.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

# The 22 canonical intents defined in the CloudServe ground truth dataset
VALID_INTENTS = (
    "account_access",
    "api_key_issue",
    "api_usage_question",
    "authentication_failure",
    "billing_query",
    "compliance_request",
    "configuration_help",
    "data_export",
    "data_residency",
    "database_issue",
    "deployment_failure",
    "feature_request",
    "integration_help",
    "onboarding",
    "performance_degradation",
    "quota_or_overage",
    "rate_limit",
    "rollback_request",
    "security_incident",
    "sso_configuration",
    "unclear_request",
    "webhook_issue",
)

VALID_URGENCIES = ("high", "medium", "low")

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "build" / "classify_v1.txt"


class ClassificationAlternative(BaseModel):
    intent: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    intent: str
    urgency: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alternatives: List[ClassificationAlternative] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        clean = v.strip().lower()
        # Normalization mapping for occasional LLM alias synonyms
        aliases = {
            "compliance_query": "compliance_request",
            "billing_inquiry": "billing_query",
            "storage_limit": "quota_or_overage",
            "service_outage": "performance_degradation",
            "network_timeout": "performance_degradation",
            "telemetry": "configuration_help",
            "custom_domain": "configuration_help",
            "cli_tooling": "configuration_help",
            "permission_error": "account_access",
        }
        clean = aliases.get(clean, clean)
        if clean not in VALID_INTENTS:
            return "unclear_request"
        return clean

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in VALID_URGENCIES:
            return "medium"
        return clean


def _load_prompt_template() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"Classification prompt file not found at {DEFAULT_PROMPT_PATH}")


def _get_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    
    selected_model = model_name or os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
    
    return ChatOpenAI(
        model=selected_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=300,
    )


def _heuristic_fallback_classify(text: str) -> ClassificationResult:
    """Deterministic heuristic fallback classifier when LLM is offline or unconfigured."""
    lower = text.lower()

    # Rule-based intent detection (ordered to respect boundary disambiguation)
    if any(k in lower for k in ["api key", "production key", "rotate key", "key compromised", "invalid api key"]):
        intent = "api_key_issue"
    elif any(k in lower for k in ["revert", "roll back", "rollback", "previous revision", "earlier build"]):
        intent = "rollback_request"
    elif any(k in lower for k in ["build fail", "ci/cd", "pipeline", "docker", "dependency resolution"]):
        intent = "deployment_failure"
    elif any(k in lower for k in ["mfa", "2fa", "two-factor", "password reset", "locked account", "login fail"]):
        intent = "account_access"
    elif any(k in lower for k in ["sso", "saml", "okta", "azure ad", "single sign-on"]):
        intent = "sso_configuration"
    elif any(k in lower for k in ["401", "unauthorized", "expired token", "jwt", "token verification"]):
        intent = "authentication_failure"
    elif any(k in lower for k in ["rate limit", "429", "too many requests", "throttle"]):
        intent = "rate_limit"
    elif any(k in lower for k in ["webhook", "delivery failed", "signature verification"]):
        intent = "webhook_issue"
    elif any(k in lower for k in ["invoice", "charge", "refund", "credit card", "billing", "payment"]):
        intent = "billing_query"
    elif any(k in lower for k in ["gdpr", "soc2", "hipaa", "audit", "compliance", "retention period"]):
        intent = "compliance_request"
    elif any(k in lower for k in ["residency", "data locality", "eu region", "store data in"]):
        intent = "data_residency"
    elif any(k in lower for k in ["export data", "backup download", "database dump"]):
        intent = "data_export"
    elif any(k in lower for k in ["slow query", "database connection", "postgres", "pool exhausted"]):
        intent = "database_issue"
    elif any(k in lower for k in ["latency", "cpu spike", "slow response", "degraded", "outage", "downtime"]):
        intent = "performance_degradation"
    elif any(k in lower for k in ["quota", "overage", "storage full", "disk limit"]):
        intent = "quota_or_overage"
    elif any(k in lower for k in ["security breach", "vulnerability", "leaked", "compromised credentials"]):
        intent = "security_incident"
    elif any(k in lower for k in ["feature request", "would be great if", "spend cap", "please add"]):
        intent = "feature_request"
    elif any(k in lower for k in ["onboarding", "walkthrough", "new account", "getting started"]):
        intent = "onboarding"
    elif any(k in lower for k in ["github integration", "slack bot", "datadog", "integration"]):
        intent = "integration_help"
    elif any(k in lower for k in ["how do i use", "parameter format", "api endpoint", "sdk"]):
        intent = "api_usage_question"
    elif any(k in lower for k in ["env var", "config", "setting", "environment variable"]):
        intent = "configuration_help"
    else:
        intent = "unclear_request"

    # Rule-based urgency detection
    if any(k in lower for k in ["outage", "down", "critical", "security incident", "blocked", "production key", "urgent", "emergency"]):
        urgency = "high"
    elif intent in ("feature_request", "onboarding") or any(k in lower for k in ["feature request", "would be useful", "would be great", "general question", "curious", "when you have time", "please add"]):
        urgency = "low"
    else:
        urgency = "medium"

    return ClassificationResult(
        intent=intent,
        urgency=urgency,
        confidence=0.85 if intent != "unclear_request" else 0.40,
        alternatives=[ClassificationAlternative(intent="unclear_request", confidence=0.15)],
        reasoning="Heuristic keyword fallback matching.",
    )


def classify_ticket(
    body: str,
    subject: str = "",
    channel: str = "email",
    customer_tier: str = "standard",
    language_fluency: str = "fluent",
    use_llm: bool = True,
    model_name: Optional[str] = None,
) -> ClassificationResult:
    """Classifies a support ticket into primary intent, urgency, and calibrated confidence."""
    full_text = f"{subject}\n{body}".strip() if subject and subject != body else body.strip()

    if not use_llm or not os.getenv("OPENROUTER_API_KEY"):
        return _heuristic_fallback_classify(full_text)

    try:
        template = _load_prompt_template()
        formatted_prompt = template.format(
            channel=channel,
            customer_tier=customer_tier,
            language_fluency=language_fluency,
            subject=subject or "(None)",
            body=body,
        )

        llm = _get_llm(model_name=model_name)
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        content = response.content.strip()

        # Extract JSON if wrapped in markdown fences
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return ClassificationResult(**parsed)

        return _heuristic_fallback_classify(full_text)

    except Exception as e:
        logger.warning(f"LLM classification failed ({e}), falling back to heuristic engine.")
        return _heuristic_fallback_classify(full_text)


# ==============================================================================
# LangGraph Node
# ==============================================================================

def classify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph Node: Executes intent & urgency classification for incoming ticket state."""
    body = state.get("body", "")
    subject = state.get("subject", "")
    channel = state.get("channel", "email")
    metadata = state.get("metadata", {})
    customer_tier = metadata.get("customer_tier") or state.get("customer_tier", "standard")
    language_fluency = metadata.get("language_fluency") or state.get("language_fluency", "fluent")

    use_llm = state.get("use_llm_classification", True)
    model_name = state.get("model_name")

    result = classify_ticket(
        body=body,
        subject=subject,
        channel=channel,
        customer_tier=customer_tier,
        language_fluency=language_fluency,
        use_llm=use_llm,
        model_name=model_name,
    )

    return {
        "intent": result.intent,
        "urgency": result.urgency,
        "classification_confidence": result.confidence,
        "classification_alternatives": [alt.model_dump() for alt in result.alternatives],
        "classification_reasoning": result.reasoning,
    }
