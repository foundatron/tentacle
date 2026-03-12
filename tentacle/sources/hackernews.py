"""Hacker News Algolia search API source adapter."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Literal

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter, fetch_with_backoff

logger = logging.getLogger(__name__)

_HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"


def _utcnow() -> datetime:
    """Return current UTC time. Extracted for testability."""
    return datetime.now(UTC)


class HackerNewsAdapter(SourceAdapter):
    """Fetch stories from the Hacker News Algolia search API."""

    def __init__(
        self,
        min_points: int = 10,
        days_back: int | None = None,
        story_type: Literal["story", "show_hn", "ask_hn"] = "story",
    ) -> None:
        self._min_points = min_points
        self._days_back = days_back
        self._story_type = story_type

    @property
    def name(self) -> str:
        return "hn"

    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        articles: list[Article] = []
        per_query = max(1, max_results // max(len(queries), 1))

        for query in queries:
            try:
                results = self._search(query, per_query)
                articles.extend(results)
            except Exception:
                logger.exception("HN query failed: %s", query)

        return articles[:max_results]

    def _search(self, query: str, limit: int) -> list[Article]:
        numeric_filters = f"points>={self._min_points}"
        if self._days_back is not None:
            cutoff = _utcnow() - timedelta(days=self._days_back)
            # Algolia accepts comma-separated AND filters in a single numericFilters value.
            numeric_filters += f",created_at_i>{int(cutoff.timestamp())}"

        params = urllib.parse.urlencode(
            {
                "query": query,
                "tags": self._story_type,
                "hitsPerPage": min(limit, 50),
                "numericFilters": numeric_filters,
            }
        )
        url = f"{_HN_SEARCH_API}?{params}"

        req = urllib.request.Request(url, headers={"User-Agent": "tentacle/0.1"})
        data = json.loads(fetch_with_backoff(req, source_name="HN"))

        articles: list[Article] = []
        for hit in data.get("hits", []):
            article = self._parse_hit(hit)
            if article:
                articles.append(article)

        logger.info("HN: %d results for '%s'", len(articles), query)
        return articles

    def _parse_hit(self, hit: dict[str, object]) -> Article | None:
        title = hit.get("title")
        if not isinstance(title, str) or not title:
            return None

        object_id = hit.get("objectID")
        discussion_url = (
            f"https://news.ycombinator.com/item?id={object_id}"
            if isinstance(object_id, str)
            else None
        )

        # Prefer the actual URL, fall back to HN discussion
        story_url = hit.get("url")
        if not isinstance(story_url, str) or not story_url:
            if discussion_url is None:
                return None
            story_url = discussion_url

        # Author
        author = hit.get("author")
        authors = [author] if isinstance(author, str) and author else None

        # Published date
        published_at = None
        created_at_str = hit.get("created_at")
        if isinstance(created_at_str, str):
            try:
                published_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        source_id = str(object_id) if isinstance(object_id, str) else None

        # Points and comments
        points_raw = hit.get("points")
        points = points_raw if isinstance(points_raw, int) else 0
        num_comments_raw = hit.get("num_comments")
        num_comments = num_comments_raw if isinstance(num_comments_raw, int) else 0

        metadata: dict[str, str | int | float | bool | None] = {
            "points": points,
            "num_comments": num_comments,
        }
        if discussion_url is not None:
            metadata["discussion_url"] = discussion_url

        return Article(
            id=fingerprint(story_url),
            source="hn",
            source_id=source_id,
            title=title,
            url=story_url,
            discovered_at=_utcnow(),
            authors=authors,
            published_at=published_at,
            access_status="unknown",
            metadata=metadata,
        )
