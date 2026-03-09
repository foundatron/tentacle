"""Semantic Scholar API source adapter."""

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

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarAdapter(SourceAdapter):
    """Fetch papers from the Semantic Scholar API."""

    @property
    def name(self) -> str:
        return "semantic_scholar"

    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        articles: list[Article] = []
        per_query = max(1, max_results // max(len(queries), 1))

        for query in queries:
            try:
                results = self._search(query, per_query)
                articles.extend(results)
            except Exception:
                logger.exception("Semantic Scholar query failed: %s", query)

        return articles[:max_results]

    def _search(self, query: str, limit: int) -> list[Article]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": min(limit, 100),
                "fields": "title,authors,abstract,url,externalIds,publicationDate,openAccessPdf",
            }
        )
        url = f"{_S2_API}?{params}"

        req = urllib.request.Request(url, headers={"User-Agent": "tentacle/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        articles: list[Article] = []
        for paper in data.get("data", []):
            article = self._parse_paper(paper)
            if article:
                articles.append(article)

        logger.info("Semantic Scholar: %d results for '%s'", len(articles), query)
        return articles

    def _parse_paper(self, paper: dict[str, object]) -> Article | None:
        title = paper.get("title")
        if not isinstance(title, str) or not title:
            return None

        paper_url = paper.get("url")
        if not isinstance(paper_url, str) or not paper_url:
            return None

        # Authors
        authors_data = paper.get("authors")
        authors: list[str] | None = None
        if isinstance(authors_data, list):
            authors = [
                a["name"]
                for a in authors_data
                if isinstance(a, dict) and isinstance(a.get("name"), str)
            ]

        abstract = paper.get("abstract")
        abstract_str = abstract if isinstance(abstract, str) else None

        # PDF URL
        pdf_url = None
        oa_pdf = paper.get("openAccessPdf")
        if isinstance(oa_pdf, dict):
            pdf_url_val = oa_pdf.get("url")
            if isinstance(pdf_url_val, str):
                pdf_url = pdf_url_val

        # Published date
        published_at = None
        pub_date = paper.get("publicationDate")
        if isinstance(pub_date, str):
            try:
                published_at = datetime.fromisoformat(pub_date).replace(tzinfo=UTC)
            except ValueError:
                pass

        # External IDs for source_id
        ext_ids = paper.get("externalIds")
        source_id = None
        if isinstance(ext_ids, dict):
            source_id = ext_ids.get("CorpusId")
            if source_id is not None:
                source_id = str(source_id)

        return Article(
            id=fingerprint(paper_url),
            source="semantic_scholar",
            source_id=source_id,
            title=title,
            url=paper_url,
            discovered_at=datetime.now(UTC),
            authors=authors,
            abstract=abstract_str,
            pdf_url=pdf_url,
            published_at=published_at,
            access_status="open" if pdf_url else "unknown",
        )
