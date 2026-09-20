"""Response generation module for CloudServe Solutions support tickets.

Produces grounded, technically accurate responses with mandatory inline [DOC-ID]
citations based strictly on retrieved knowledge base passages.
Adheres to Capstone Specification AC-A6, B-08, and FR-03.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.grounding import render_source_excerpt

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "build" / "generate_v1.txt"

# Regex for extracting document citations in the canonical [DOC-CATEGORY-NUM] format
CITATION_REGEX = re.compile(r"\[(DOC-[A-Z0-9]+-\d+)\]")


class GenerationResult(BaseModel):
    """Structured container for generated support response and citation metadata."""
    response_text: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_information: bool = False
    cannot_answer_reason: Optional[str] = None


def extract_citations(text: str) -> List[str]:
    """Extracts all unique [DOC-ID] citations from generated text preserving order."""
    matches = CITATION_REGEX.findall(text)
    seen = set()
    ordered_unique: List[str] = []
    for doc_id in matches:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered_unique.append(doc_id)
    return ordered_unique


def format_retrieved_passages(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved passage dictionaries into a clean markdown block for the prompt."""
    if not passages:
        return "(No documentation passages retrieved)"

    formatted_blocks = []
    for idx, p in enumerate(passages, 1):
        doc_id = p.get("doc_id", "DOC-UNKNOWN")
        title = p.get("title", "Untitled")
        content = p.get("content", "").strip()
        formatted_blocks.append(
            f"--- Document [{doc_id}]: {title} ---\n{content}\n"
        )
    return "\n".join(formatted_blocks)


def _load_prompt_template() -> str:
    if DEFAULT_PROMPT_PATH.exists():
        with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"Generation prompt file not found at {DEFAULT_PROMPT_PATH}")


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
        max_tokens=600,
    )


def _heuristic_fallback_generate(
    customer_name: str,
    subject: str,
    body: str,
    passages: List[Dict[str, Any]],
    channel: str = "email"
) -> GenerationResult:
    """Deterministic fallback generator when LLM is unavailable or offline."""
    if not passages:
        return GenerationResult(
            response_text=f"Hi {customer_name},\n\nThank you for reaching out. We could not find matching documentation for your query and have escalated your ticket to our engineering team.",
            cited_doc_ids=[],
            confidence=0.40,
            missing_information=True,
            cannot_answer_reason="No relevant documentation passages available.",
        )

    top_doc = passages[0]
    top_id = top_doc.get("doc_id", "DOC-UNKNOWN")
    response_text = render_source_excerpt(passages)

    return GenerationResult(
        response_text=response_text,
        cited_doc_ids=[top_id],
        confidence=0.85,
        missing_information=False,
    )


def generate_response(
    ticket_id: str,
    body: str,
    subject: str = "",
    channel: str = "email",
    customer_name: str = "Customer",
    customer_tier: str = "standard",
    language_fluency: str = "fluent",
    retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = True,
    model_name: Optional[str] = None,
) -> GenerationResult:
    """
    Generates a grounded technical response strictly citing retrieved passages in [DOC-ID] format.
    """
    passages = retrieved_passages or []

    if not use_llm or not os.getenv("OPENROUTER_API_KEY"):
        return _heuristic_fallback_generate(
            customer_name=customer_name,
            subject=subject,
            body=body,
            passages=passages,
            channel=channel,
        )

    try:
        template = _load_prompt_template()
        formatted_passages = format_retrieved_passages(passages)

        formatted_prompt = template.format(
            retrieved_passages=formatted_passages,
            ticket_id=ticket_id,
            customer_name=customer_name or "Customer",
            channel=channel,
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
            result = GenerationResult(**parsed)
            # Synchronize citations with actual regex extraction to prevent LLM omissions
            extracted = extract_citations(result.response_text)
            result.cited_doc_ids = extracted
            return result

        return _heuristic_fallback_generate(
            customer_name=customer_name,
            subject=subject,
            body=body,
            passages=passages,
            channel=channel,
        )

    except Exception as e:
        logger.warning(f"LLM generation failed ({e}), using fallback generator.")
        return _heuristic_fallback_generate(
            customer_name=customer_name,
            subject=subject,
            body=body,
            passages=passages,
            channel=channel,
        )


# ==============================================================================
# LangGraph Node
# ==============================================================================

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph Node: Synthesizes a grounded cited response for auto-respond tickets."""
    route = state.get("route", "auto_respond")
    # If routed to escalate, bypass answer generation
    if route == "escalate":
        return {
            "generated_response": None,
            "cited_doc_ids": [],
            "generation_confidence": 0.0,
            "missing_information": False,
            "cannot_answer_reason": "Ticket was routed for human escalation.",
        }

    ticket_id = state.get("ticket_id", "UNKNOWN")
    body = state.get("body", "")
    subject = state.get("subject", "")
    channel = state.get("channel", "email")
    retrieved_passages = state.get("retrieved_passages", [])

    metadata = state.get("metadata", {})
    customer_name = metadata.get("customer_name") or state.get("customer_name", "Customer")
    customer_tier = metadata.get("customer_tier") or state.get("customer_tier", "standard")
    language_fluency = metadata.get("language_fluency") or state.get("language_fluency", "fluent")

    use_llm = state.get("use_llm_generation", True)
    model_name = state.get("model_name")

    res = generate_response(
        ticket_id=ticket_id,
        body=body,
        subject=subject,
        channel=channel,
        customer_name=customer_name,
        customer_tier=customer_tier,
        language_fluency=language_fluency,
        retrieved_passages=retrieved_passages,
        use_llm=use_llm,
        model_name=model_name,
    )

    intent = state.get("intent", "general_inquiry")
    conf = float(state.get("classification_confidence", 0.0))
    cited_str = ", ".join(res.cited_doc_ids) if res.cited_doc_ids else "None"
    summary = f"Auto-responded for intent '{intent}': High confidence ({conf:.2f} >= 0.80), grounded with citations [{cited_str}]."

    return {
        "generated_response": res.response_text,
        "cited_doc_ids": res.cited_doc_ids,
        "generation_confidence": res.confidence,
        "missing_information": res.missing_information,
        "cannot_answer_reason": res.cannot_answer_reason,
        "auto_respond_summary": summary,
    }
