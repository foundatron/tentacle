"""Stage 2: Deep analysis and maturity scoring using Sonnet."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from tentacle.llm.client import LLMClient
from tentacle.llm.prompts import ANALYZE_SYSTEM, ANALYZE_USER
from tentacle.models import Analysis, Article

logger = logging.getLogger(__name__)


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

    response = client.complete(
        model=model,
        system=system_blocks,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=4096,
        temperature=0.0,
    )

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse analysis response for '%s'", article.title[:60])
        return None

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
        model_used=model,
        input_tokens=last_record.input_tokens if last_record else None,
        output_tokens=last_record.output_tokens if last_record else None,
        cost_usd=last_record.cost_usd if last_record else None,
        analyzed_at=datetime.now(UTC),
    )
