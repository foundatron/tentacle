"""Stage 2: Deep analysis and maturity scoring using Sonnet."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import anthropic

from tentacle.llm.client import BudgetExceededError, LLMClient
from tentacle.llm.prompts import ANALYZE_SYSTEM, ANALYZE_USER
from tentacle.models import Analysis, Article

logger = logging.getLogger(__name__)


class _AnalysisToolOutput(TypedDict, total=False):
    """Typed representation of the create_analysis tool output."""

    key_insights: list[str]
    applicable_scopes: list[str]
    suggested_type: str
    suggested_title: str
    suggested_body: str
    maturity_score: int
    maturity_reasoning: str
    confidence_score: float


ANALYSIS_TOOL: anthropic.types.ToolParam = {
    "name": "create_analysis",
    "description": (
        "Record the structured analysis of a research article for OctopusGarden, "
        "including key insights, a draft GitHub issue, maturity score, and confidence."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key_insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key insights from the article relevant to OctopusGarden.",
            },
            "applicable_scopes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "OctopusGarden scopes this applies to (e.g. attractor, llm).",
            },
            "suggested_type": {
                "type": "string",
                "description": "Conventional commit type: feat, fix, perf, or refactor.",
            },
            "suggested_title": {
                "type": "string",
                "description": "Issue title in conventional commits format.",
            },
            "suggested_body": {
                "type": "string",
                "description": "Full GitHub issue body in markdown using the provided template.",
            },
            "maturity_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "Implementation maturity score 1-5. "
                    "Score 1 if the current approach is adequate. "
                    "Score 4-5 only if the proposal addresses a real gap "
                    "and benefits outweigh complexity."
                ),
            },
            "maturity_reasoning": {
                "type": "string",
                "description": "Explanation of the maturity rating.",
            },
            "confidence_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in this analysis (0.0-1.0).",
            },
        },
        "required": [
            "key_insights",
            "applicable_scopes",
            "suggested_type",
            "suggested_title",
            "suggested_body",
            "maturity_score",
            "maturity_reasoning",
            "confidence_score",
        ],
    },
}


def analyze_article(
    client: LLMClient,
    article: Article,
    context: str,
    *,
    model: str,
    relevance_score: float,
    relevance_reasoning: str,
) -> Analysis | None:
    """Perform deep analysis of an article. Returns Analysis or None on failure."""
    authors_str = ", ".join(article.authors) if article.authors else "Unknown"
    published_str = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
    content = article.full_text or article.abstract or "(no content available)"

    user_msg = ANALYZE_USER.format(
        title=article.title,
        authors=authors_str,
        url=article.url,
        published=published_str,
        content=content,
        context=context,
    )

    # Use prompt caching on system prompt (repeated across analyses)
    system_blocks: list[dict[str, object]] = [
        {
            "type": "text",
            "text": ANALYZE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    try:
        raw = client.complete_with_tools(
            model=model,
            system=system_blocks,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": user_msg}],
            tools=[ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "create_analysis"},
            max_tokens=4096,
            temperature=0.0,
        )
    except BudgetExceededError:
        raise
    except Exception:
        logger.warning("Failed to analyze article '%s'", article.title[:60])
        return None

    data: _AnalysisToolOutput = raw  # type: ignore[assignment]

    # Get the last usage record for token counts
    last_record = client.costs.records[-1] if client.costs.records else None

    return Analysis(
        article_id=article.id,
        relevance_score=relevance_score,
        relevance_reasoning=relevance_reasoning,
        key_insights=data.get("key_insights"),
        applicable_scopes=data.get("applicable_scopes"),
        suggested_type=data.get("suggested_type"),
        suggested_title=data.get("suggested_title"),
        suggested_body=data.get("suggested_body"),
        maturity_score=int(data.get("maturity_score", 1)),
        maturity_reasoning=data.get("maturity_reasoning"),
        confidence_score=float(data["confidence_score"]) if "confidence_score" in data else None,
        model_used=model,
        input_tokens=last_record.input_tokens if last_record else None,
        output_tokens=last_record.output_tokens if last_record else None,
        cost_usd=last_record.cost_usd if last_record else None,
        analyzed_at=datetime.now(UTC),
    )
