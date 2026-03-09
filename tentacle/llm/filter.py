"""Stage 1: Cheap relevance filtering using Haiku."""

from __future__ import annotations

import json
import logging

from tentacle.llm.client import LLMClient
from tentacle.llm.prompts import FILTER_SYSTEM, FILTER_USER
from tentacle.models import Article

logger = logging.getLogger(__name__)


def filter_article(
    client: LLMClient,
    article: Article,
    *,
    model: str,
    threshold: float = 0.3,
) -> tuple[float, str]:
    """Score an article's relevance. Returns (score, reasoning).

    Uses the cheap model (Haiku) for cost efficiency.
    """
    abstract = article.abstract or "(no abstract available)"
    user_msg = FILTER_USER.format(title=article.title, abstract=abstract)

    response = client.complete(
        model=model,
        system=FILTER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=256,
        temperature=0.0,
    )

    try:
        data = json.loads(response)
        relevance = float(data["relevance"])
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse filter response: %s", response[:200])
        return 0.0, "parse error"

    logger.info(
        "Filter: %.2f for '%s' (%s)",
        relevance,
        article.title[:60],
        "PASS" if relevance >= threshold else "SKIP",
    )

    return relevance, reasoning
