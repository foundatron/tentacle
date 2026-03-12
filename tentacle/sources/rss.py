"""Generic RSS/Atom feed source adapter."""

from __future__ import annotations

import html.parser
import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter, fetch_with_backoff

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_SKIP_TAGS = {"script", "style", "nav", "header", "footer"}
_CONTENT_FETCH_WORKERS = 10


class _HTMLToTextParser(html.parser.HTMLParser):
    """Strip HTML tags and extract plain text, skipping non-content elements.

    Uses per-tag depth counters so that mismatched end tags (malformed HTML)
    only close the matching open tag, preventing script/style content from
    leaking into extracted text.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skip_counts: dict[str, int] = {}
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_counts[tag] = self._skip_counts.get(tag, 0) + 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_counts.get(tag, 0) > 0:
            self._skip_counts[tag] -= 1

    def handle_data(self, data: str) -> None:
        if sum(self._skip_counts.values()) == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def _fetch_content(url: str, timeout: int, max_bytes: int) -> str | None:
    """Fetch a URL and return extracted plain text, or None on any failure.

    Uses a direct urlopen (no retry) because content fetches are best-effort;
    this also preserves charset detection from Content-Type and limits the
    socket read to *max_bytes* to avoid buffering huge responses.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tentacle/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type: str = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip().strip("\"'")
            data = resp.read(max_bytes)
        html_text = data.decode(charset, errors="replace")
        parser = _HTMLToTextParser()
        parser.feed(html_text)
        text = parser.get_text()
        return text if text else None
    except Exception:
        logger.warning("Failed to fetch content from %s", url, exc_info=True)
        return None


class RSSAdapter(SourceAdapter):
    """Fetch articles from RSS/Atom feeds.

    The `queries` parameter is interpreted as a list of feed URLs.
    """

    def __init__(
        self,
        extract_content: bool = False,
        content_timeout: int = 30,
        content_max_bytes: int = 1_048_576,
    ) -> None:
        self._extract_content = extract_content
        self._content_timeout = content_timeout
        self._content_max_bytes = content_max_bytes

    @property
    def name(self) -> str:
        return "rss"

    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        articles: list[Article] = []

        for feed_url in queries:
            try:
                results = self._fetch_feed(feed_url)
                articles.extend(results)
            except Exception:
                logger.exception("RSS fetch failed: %s", feed_url)

        return articles[:max_results]

    def _fetch_feed(self, feed_url: str) -> list[Article]:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "tentacle/0.1"})
        data = fetch_with_backoff(req, source_name="RSS")

        root = ET.fromstring(data)  # noqa: S314

        # Detect RSS vs Atom
        if root.tag == "rss" or root.find("channel") is not None:
            articles = self._parse_rss(root)
        elif root.tag == f"{_ATOM_NS}feed":
            articles = self._parse_atom(root)
        else:
            logger.warning("Unknown feed format: %s", root.tag)
            return []

        if self._extract_content:
            articles = self._fetch_contents_parallel(articles)

        return articles

    def _fetch_contents_parallel(self, articles: list[Article]) -> list[Article]:
        """Fetch full-text content for all articles in parallel."""
        urls: list[tuple[int, str]] = [(i, a.url) for i, a in enumerate(articles) if a.url]
        if not urls:
            return articles

        workers = min(_CONTENT_FETCH_WORKERS, len(urls))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fetch_content, url, self._content_timeout, self._content_max_bytes
                ): i
                for i, url in urls
            }
            for future in as_completed(futures):
                idx = futures[future]
                articles[idx].full_text = future.result()

        return articles

    def _parse_rss(self, root: ET.Element) -> list[Article]:
        articles: list[Article] = []
        channel = root.find("channel")
        if channel is None:
            return articles

        for item in channel.findall("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            if title_el is None or not title_el.text or link_el is None or not link_el.text:
                continue

            # Description as abstract
            desc_el = item.find("description")
            abstract = desc_el.text.strip() if desc_el is not None and desc_el.text else None

            # Published date
            pub_date_el = item.find("pubDate")
            published_at = None
            if pub_date_el is not None and pub_date_el.text:
                try:
                    published_at = parsedate_to_datetime(pub_date_el.text)
                except Exception:
                    pass

            # Author
            author_el = item.find("author")
            dc_creator = item.find("{http://purl.org/dc/elements/1.1/}creator")
            author = None
            if author_el is not None and author_el.text:
                author = author_el.text
            elif dc_creator is not None and dc_creator.text:
                author = dc_creator.text

            article = Article(
                id=fingerprint(link_el.text),
                source="rss",
                title=title_el.text.strip(),
                url=link_el.text.strip(),
                discovered_at=datetime.now(UTC),
                authors=[author] if author else None,
                abstract=abstract,
                published_at=published_at,
                access_status="open",
            )

            articles.append(article)

        logger.info("RSS: %d items parsed", len(articles))
        return articles

    def _parse_atom(self, root: ET.Element) -> list[Article]:
        articles: list[Article] = []

        for entry in root.findall(f"{_ATOM_NS}entry"):
            title_el = entry.find(f"{_ATOM_NS}title")
            if title_el is None or not title_el.text:
                continue

            # Get link
            url = ""
            for link in entry.findall(f"{_ATOM_NS}link"):
                rel = link.get("rel", "alternate")
                if rel == "alternate":
                    url = link.get("href", "")
                    break
            if not url:
                # Fall back to first link
                first_link = entry.find(f"{_ATOM_NS}link")
                url = first_link.get("href", "") if first_link is not None else ""
            if not url:
                continue

            # Summary
            summary_el = entry.find(f"{_ATOM_NS}summary")
            abstract = (
                summary_el.text.strip() if summary_el is not None and summary_el.text else None
            )

            # Authors
            authors = []
            for author in entry.findall(f"{_ATOM_NS}author"):
                name_el = author.find(f"{_ATOM_NS}name")
                if name_el is not None and name_el.text:
                    authors.append(name_el.text)

            # Published date
            published_el = entry.find(f"{_ATOM_NS}published")
            updated_el = entry.find(f"{_ATOM_NS}updated")
            published_at = None
            date_el = published_el if published_el is not None else updated_el
            if date_el is not None and date_el.text:
                try:
                    published_at = datetime.fromisoformat(date_el.text.replace("Z", "+00:00"))
                except ValueError:
                    pass

            article = Article(
                id=fingerprint(url),
                source="rss",
                title=title_el.text.strip(),
                url=url,
                discovered_at=datetime.now(UTC),
                authors=authors or None,
                abstract=abstract,
                published_at=published_at,
                access_status="open",
            )

            articles.append(article)

        logger.info("Atom: %d entries parsed", len(articles))
        return articles
