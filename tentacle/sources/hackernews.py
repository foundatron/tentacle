"""Hacker News Algolia search API source adapter."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"


class HackerNewsAdapter(SourceAdapter):
    """Fetch stories from the Hacker News Algolia search API."""

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
        params = urllib.parse.urlencode(
            {
                "query": query,
                "tags": "story",
                "hitsPerPage": min(limit, 50),
            }
        )
        url = f"{_HN_SEARCH_API}?{params}"

        req = urllib.request.Request(url, headers={"User-Agent": "tentacle/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

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

        # Prefer the actual URL, fall back to HN discussion
        story_url = hit.get("url")
        if not isinstance(story_url, str) or not story_url:
            object_id = hit.get("objectID")
            if not isinstance(object_id, str):
                return None
            story_url = f"https://news.ycombinator.com/item?id={object_id}"

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

        object_id = hit.get("objectID")
        source_id = str(object_id) if isinstance(object_id, str) else None

        return Article(
            id=fingerprint(story_url),
            source="hn",
            source_id=source_id,
            title=title,
            url=story_url,
            discovered_at=datetime.now(UTC),
            authors=authors,
            published_at=published_at,
            access_status="unknown",
        )
