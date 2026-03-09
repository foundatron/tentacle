"""arXiv Atom API source adapter."""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivAdapter(SourceAdapter):
    """Fetch papers from the arXiv Atom API."""

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
            except Exception:
                logger.exception("arXiv query failed: %s", query)

        return articles[:max_results]

    def _search(self, query: str, max_results: int) -> list[Article]:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{_ARXIV_API}?{params}"

        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()

        root = ET.fromstring(data)  # noqa: S314
        articles: list[Article] = []

        for entry in root.findall(f"{_ATOM_NS}entry"):
            article = self._parse_entry(entry)
            if article:
                articles.append(article)

        logger.info("arXiv: %d results for '%s'", len(articles), query)
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
