"""Semantic Scholar API source adapter."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarAdapter(SourceAdapter):
    """Fetch papers from the Semantic Scholar API."""

    def __init__(
        self,
        api_key: str = "",
        min_citations: int = 0,
        days_back: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._min_citations = min_citations
        self._days_back = days_back

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

    def _fetch_with_retry(self, req: urllib.request.Request) -> bytes | None:
        """Fetch URL with retry on 429. Non-retryable errors are logged and return None."""
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()  # type: ignore[no-any-return]
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    logger.error("Semantic Scholar HTTP error %d: %s", e.code, e)
                    return None
                # 429 rate-limit handling
                retry_after = 1
                raw = e.headers.get("Retry-After")
                if raw is not None:
                    try:
                        retry_after = int(raw)
                    except ValueError:
                        logger.debug(
                            "Retry-After header is not an integer (%r), defaulting to 1s", raw
                        )
                if attempt < max_retries:
                    logger.warning(
                        "Semantic Scholar rate-limited (429), retrying in %ds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(retry_after)
                else:
                    logger.error("Semantic Scholar rate-limited (429), max retries exhausted")
            except urllib.error.URLError as e:
                logger.error("Semantic Scholar network error: %s", e)
                return None
        return None

    def _search(self, query: str, limit: int) -> list[Article]:
        query_params: dict[str, str | int] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": (
                "title,authors,abstract,url,externalIds,publicationDate,openAccessPdf,citationCount"
            ),
        }

        if self._days_back is not None:
            now = datetime.now(UTC)
            start_date = now - timedelta(days=self._days_back)
            query_params["publicationDateOrYear"] = (
                f"{start_date.strftime('%Y-%m-%d')}:{now.strftime('%Y-%m-%d')}"
            )

        params = urllib.parse.urlencode(query_params)
        url = f"{_S2_API}?{params}"

        headers: dict[str, str] = {"User-Agent": "tentacle/0.1"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        req = urllib.request.Request(url, headers=headers)

        data_bytes = self._fetch_with_retry(req)
        if data_bytes is None:
            return []

        data = json.loads(data_bytes)
        articles: list[Article] = []
        for paper in data.get("data", []):
            citation_count = paper.get("citationCount")
            if citation_count is None:
                citation_count = 0
            if citation_count < self._min_citations:
                continue
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
