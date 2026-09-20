"""Pre-release claim support review, separate from citation-ID validation.

Only an exact, locally rendered source quotation bypasses model review. An
unavailable or malformed reviewer never authorizes release of free-form text.
Model review is an additional control, not proof of factual correctness.
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictBool

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts/build/grounding_v1.txt"
PROMPT_VERSION = "grounding_v1"


def render_source_excerpt(passages: list[dict[str, Any]]) -> str:
    """Quote the entire first retrieved passage without asserting resolution."""
    if not passages or not str(passages[0].get("content", "")).strip():
        return ""
    passage = passages[0]
    quoted = "\n".join("> " + line for line in passage["content"].strip().splitlines())
    return (
        "AI-drafted documentation excerpt. Please verify its relevance to your issue.\n\n"
        f"{quoted}\n\nSource: [{passage.get('doc_id', '')}]"
    )


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claim: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ReviewerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fully_supported: StrictBool
    unsupported_claims: list[str]
    evidence: list[Evidence]
    reasoning: str = Field(min_length=1)


class GroundingReview(BaseModel):
    status: Literal["pass", "fail", "unavailable", "not_run"]
    method: str
    reason: str
    unsupported_claims: list[str] = Field(default_factory=list)
    evidence: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = None
    prompt_version: str = PROMPT_VERSION


def _get_reviewer(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name, api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1", temperature=0.0,
        max_tokens=1600, timeout=15, max_retries=0,
    )


def review_grounding(
    response_text: str,
    passages: list[dict[str, Any]],
    *,
    use_llm: bool = True,
    model_name: str | None = None,
) -> GroundingReview:
    expected = render_source_excerpt(passages)
    if expected and response_text == expected:
        return GroundingReview(
            status="pass", method="exact_source_excerpt",
            reason="Output exactly matches the locally rendered full retrieved passage.",
            prompt_version="source_excerpt_v1",
        )
    selected_model = model_name or os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
    if not use_llm or not os.getenv("OPENROUTER_API_KEY"):
        return GroundingReview(
            status="unavailable", method="model_review", model=selected_model,
            reason="Free-form answer requires claim review; reviewer is offline or unconfigured.",
        )
    try:
        payload = json.dumps({"response": response_text, "passages": passages}, ensure_ascii=False)
        result = _get_reviewer(selected_model).invoke([
            SystemMessage(content=PROMPT_PATH.read_text(encoding="utf-8")),
            HumanMessage(content=payload),
        ])
        review = ReviewerResponse.model_validate_json(result.content)
        if not review.fully_supported or review.unsupported_claims:
            return GroundingReview(
                status="fail", method="model_review", model=selected_model,
                reason=review.reasoning, unsupported_claims=review.unsupported_claims,
                evidence=[e.model_dump() for e in review.evidence],
            )
        # Independently verify the reviewer's evidence references. Do not accept
        # invented quotations, or evidence attributed to the wrong document.
        import re
        cited = set(re.findall(r"\[(DOC-[A-Z0-9]+-\d+)\]", response_text))
        covered = set()
        if not review.evidence:
            raise ValueError("Reviewer returned no supporting evidence")
        for item in review.evidence:
            if not item.claim.strip() or item.claim not in response_text:
                raise ValueError("Reviewer claim does not occur in the response")
            if item.doc_id not in cited or not item.quote.strip() or not any(
                p.get("doc_id") == item.doc_id and item.quote in p.get("content", "")
                for p in passages
            ):
                raise ValueError("Reviewer evidence does not resolve to the cited passage")
            covered.add(item.doc_id)
        if covered != cited:
            raise ValueError("Reviewer did not verify every cited document")
        return GroundingReview(
            status="pass", method="model_review", model=selected_model,
            reason=review.reasoning, evidence=[e.model_dump() for e in review.evidence],
        )
    except Exception as exc:
        # Avoid persisting provider exceptions that may include request content.
        return GroundingReview(
            status="unavailable", method="model_review", model=selected_model,
            reason=f"Claim review could not be verified ({type(exc).__name__}); human review required.",
        )
