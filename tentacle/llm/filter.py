"""Stage 1: Cheap relevance filtering using Haiku."""

from __future__ import annotations

import json
import logging

from tentacle.llm.client import LLMClient
from tentacle.llm.prompts import FILTER_BATCH_SYSTEM, FILTER_BATCH_USER, FILTER_SYSTEM, FILTER_USER
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


def filter_batch(
    client: LLMClient,
    articles: list[Article],
    *,
    model: str,
    threshold: float = 0.3,
    batch_size: int = 10,
) -> list[tuple[float, str]]:
    """Score a list of articles for relevance in batches. Returns (score, reasoning) per article.

    Sends each batch in a single LLM call. Falls back to individual filter_article() calls
    on JSON parse failure or missing/invalid entries.
    """
    if not articles:
        return []

    results: list[tuple[float, str]] = [(0.0, "not processed")] * len(articles)

    for batch_start in range(0, len(articles), batch_size):
        batch = articles[batch_start : batch_start + batch_size]
        batch_len = len(batch)

        # Build numbered article list (1-based)
        parts = []
        for i, article in enumerate(batch, start=1):
            abstract = article.abstract or "(no abstract available)"
            parts.append(f"[{i}] Title: {article.title}\nAbstract: {abstract}")
        articles_text = "\n\n".join(parts)

        user_msg = FILTER_BATCH_USER.format(articles=articles_text)

        response = client.complete(
            model=model,
            system=FILTER_BATCH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=batch_size * 200,
            temperature=0.0,
        )

        parsed: dict[int, tuple[float, str]] = {}
        parse_failed = False

        try:
            data = json.loads(response)
            if not isinstance(data, list):
                raise ValueError("expected JSON array")
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                try:
                    idx = int(entry["index"])
                    relevance = float(entry["relevance"])
                    reasoning = str(entry.get("reasoning", ""))
                except (KeyError, ValueError, TypeError):
                    continue
                # Validate 1-based index is in range
                if idx < 1 or idx > batch_len:
                    logger.warning(
                        "Batch filter: index %d out of range (batch size %d)", idx, batch_len
                    )
                    continue
                if idx not in parsed:  # first entry wins on duplicate
                    parsed[idx] = (max(0.0, min(1.0, relevance)), reasoning)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Batch filter: JSON parse failure (%s), falling back to individual calls", exc
            )
            parse_failed = True

        if parse_failed:
            # Full-batch fallback
            for i, article in enumerate(batch):
                results[batch_start + i] = filter_article(
                    client, article, model=model, threshold=threshold
                )
        else:
            # Partial fallback: use parsed results where available, fallback for missing
            batch_scored = 0
            for i, article in enumerate(batch):
                one_based = i + 1
                if one_based in parsed:
                    score, reasoning = parsed[one_based]
                    results[batch_start + i] = (score, reasoning)
                    batch_scored += 1
                    logger.info(
                        "Filter (batch): %.2f for '%s' (%s)",
                        score,
                        article.title[:60],
                        "PASS" if score >= threshold else "SKIP",
                    )
                else:
                    logger.warning(
                        "Batch filter: missing result for index %d ('%s'), falling back",
                        one_based,
                        article.title[:60],
                    )
                    results[batch_start + i] = filter_article(
                        client, article, model=model, threshold=threshold
                    )
            logger.info("Batch filter: scored %d/%d articles", batch_scored, batch_len)

    unprocessed = [i for i, r in enumerate(results) if r == (0.0, "not processed")]
    if unprocessed:
        raise RuntimeError(
            f"filter_batch: {len(unprocessed)} article(s) were not processed: indices {unprocessed}"
        )

    return results
