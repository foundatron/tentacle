"""arXiv Atom API source adapter."""

from __future__ import annotations

import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import RetriesExhaustedError, SourceAdapter, fetch_with_backoff

logger = logging.getLogger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
PAGE_SIZE = 100


class ArxivAdapter(SourceAdapter):
    """Fetch papers from the arXiv Atom API."""

    def __init__(
        self,
        days_back: int | None = None,
        sort_order: str = "descending",
    ) -> None:
        self._days_back = days_back
        self._sort_order = sort_order

    @property
    def name(self) -> str:
        return "arxiv"

    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        articles: list[Article] = []
        per_query = max(1, max_results // max(len(queries), 1))

        for query in queries:
            try:
                results = self._search(query, per_query)
                articles.extend(results)
            except RetriesExhaustedError as exc:
                if exc.status_code == 429:
                    logger.warning("arXiv: rate-limited, skipping remaining queries")
                    break
                logger.exception("arXiv query failed: %s", query)
            except Exception:
                logger.exception("arXiv query failed: %s", query)

        return articles[:max_results]

    @staticmethod
    def _fetch_url(url: str) -> bytes:
        """Fetch URL with exponential backoff on retryable errors."""
        return fetch_with_backoff(url, source_name="arXiv")

    def _search(self, query: str, max_results: int) -> list[Article]:
        search_query = f"all:{query}"
        if self._days_back is not None:
            now = datetime.now(UTC)
            start_date = now - timedelta(days=self._days_back)
            start_str = start_date.strftime("%Y%m%d") + "0000"
            end_str = now.strftime("%Y%m%d") + "2359"
            search_query += f" AND submittedDate:[{start_str} TO {end_str}]"

        articles: list[Article] = []
        start = 0

        while len(articles) < max_results:
            page_size = min(PAGE_SIZE, max_results - len(articles))
            params = urllib.parse.urlencode(
                {
                    "search_query": search_query,
                    "start": start,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": self._sort_order,
                }
            )
            url = f"{_ARXIV_API}?{params}"

            if start > 0:
                time.sleep(1)

            data = self._fetch_url(url)
            root = ET.fromstring(data)  # noqa: S314
            entries = root.findall(f"{_ATOM_NS}entry")

            for entry in entries:
                try:
                    article = self._parse_entry(entry)
                    if article:
                        articles.append(article)
                except Exception:
                    logger.warning("arXiv: skipping malformed entry", exc_info=True)

            start += page_size

            if len(entries) < page_size:
                break

        logger.info("arXiv: %d results (cap %d) for '%s'", len(articles), max_results, query)
        return articles

    def _parse_entry(self, entry: ET.Element) -> Article | None:
        title_el = entry.find(f"{_ATOM_NS}title")
        if title_el is None or not title_el.text:
            return None

        # Get canonical URL (abs link)
        url = ""
        pdf_url = None
        for link in entry.findall(f"{_ATOM_NS}link"):
            if link.get("type") == "text/html":
                url = link.get("href", "")
            elif link.get("title") == "pdf":
                pdf_url = link.get("href")
        if not url:
            id_el = entry.find(f"{_ATOM_NS}id")
            url = id_el.text if id_el is not None and id_el.text else ""
        if not url:
            return None

        # Authors
        authors = []
        for author in entry.findall(f"{_ATOM_NS}author"):
            name_el = author.find(f"{_ATOM_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text)

        # Abstract
        summary_el = entry.find(f"{_ATOM_NS}summary")
        abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else None

        # Published date
        published_el = entry.find(f"{_ATOM_NS}published")
        published_at = None
        if published_el is not None and published_el.text:
            try:
                published_at = datetime.fromisoformat(published_el.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Extract arXiv ID from URL
        source_id = url.rsplit("/", 1)[-1] if url else None

        # Categories as tags
        tags = []
        for cat in entry.findall("{http://arxiv.org/schemas/atom}category"):
            term = cat.get("term")
            if term:
                tags.append(term)

        return Article(
            id=fingerprint(url),
            source="arxiv",
            source_id=source_id,
            title=title_el.text.strip().replace("\n", " "),
            url=url,
            discovered_at=datetime.now(UTC),
            authors=authors or None,
            abstract=abstract,
            pdf_url=pdf_url,
            published_at=published_at,
            tags=tags or None,
            access_status="open",
        )
