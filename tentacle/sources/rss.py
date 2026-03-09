"""Generic RSS/Atom feed source adapter."""

from __future__ import annotations

import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from tentacle.dedup import fingerprint
from tentacle.models import Article
from tentacle.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class RSSAdapter(SourceAdapter):
    """Fetch articles from RSS/Atom feeds.

    The `queries` parameter is interpreted as a list of feed URLs.
    """

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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        root = ET.fromstring(data)  # noqa: S314

        # Detect RSS vs Atom
        if root.tag == "rss" or root.find("channel") is not None:
            return self._parse_rss(root)
        if root.tag == f"{_ATOM_NS}feed":
            return self._parse_atom(root)

        logger.warning("Unknown feed format: %s", root.tag)
        return []

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

            articles.append(
                Article(
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
            )

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

            articles.append(
                Article(
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
            )

        logger.info("Atom: %d entries parsed", len(articles))
        return articles
