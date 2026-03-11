"""Tests for source adapters using mocked HTTP responses."""

from __future__ import annotations

import unittest
import urllib.error
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, call, patch

from tentacle.sources.arxiv import _PAGE_SIZE, ArxivAdapter
from tentacle.sources.hackernews import HackerNewsAdapter
from tentacle.sources.rss import RSSAdapter
from tentacle.sources.semantic_scholar import SemanticScholarAdapter

ARXIV_RESPONSE = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Autonomous Code Generation with LLMs</title>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <link href="http://arxiv.org/abs/2401.00001v1" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.00001v1" title="pdf"/>
    <summary>We present a novel approach to autonomous code generation.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:category term="cs.SE"/>
    <arxiv:category term="cs.AI"/>
  </entry>
</feed>"""

HN_RESPONSE = b"""\
{
  "hits": [
    {
      "objectID": "12345",
      "title": "Show HN: AI Code Generator",
      "url": "https://example.com/ai-codegen",
      "author": "hacker",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}"""

S2_RESPONSE = b"""\
{
  "data": [
    {
      "title": "LLM-based Software Testing",
      "url": "https://api.semanticscholar.org/paper/abc123",
      "authors": [{"name": "Dr. Test"}],
      "abstract": "A study on using LLMs for testing.",
      "publicationDate": "2024-02-01",
      "externalIds": {"CorpusId": "999"},
      "openAccessPdf": {"url": "https://example.com/paper.pdf"}
    }
  ]
}"""

RSS_RESPONSE = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>AI Blog</title>
    <item>
      <title>New Findings in Code Gen</title>
      <link>https://blog.example.com/codegen</link>
      <description>Exciting new results in autonomous code generation.</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <author>writer@example.com</author>
    </item>
  </channel>
</rss>"""


def _mock_urlopen(data: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_arxiv_feed(count: int) -> bytes:
    """Build a minimal Atom feed with `count` entries."""
    entries = ""
    for i in range(count):
        entries += f"""\
  <entry>
    <title>Paper {i}</title>
    <id>http://arxiv.org/abs/2401.{i:05d}v1</id>
    <link href="http://arxiv.org/abs/2401.{i:05d}v1" type="text/html"/>
    <summary>Abstract {i}.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Author {i}</name></author>
  </entry>
"""
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}</feed>""".encode()


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=MagicMock(), fp=None)  # type: ignore[arg-type]


class TestArxivAdapter(unittest.TestCase):
    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_fetch_parses_atom(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(ARXIV_RESPONSE)
        adapter = ArxivAdapter()
        articles = adapter.fetch(["autonomous code generation"], max_results=10)

        assert len(articles) == 1
        a = articles[0]
        assert a.source == "arxiv"
        assert "Autonomous Code Generation" in a.title
        assert a.authors == ["Alice Smith", "Bob Jones"]
        assert a.pdf_url is not None
        assert a.tags == ["cs.SE", "cs.AI"]

    @patch("tentacle.sources.arxiv.time.sleep")
    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_pagination_multiple_pages(
        self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock
    ) -> None:
        page1 = _make_arxiv_feed(_PAGE_SIZE)  # full page → more to fetch
        page2 = _make_arxiv_feed(3)  # partial page → signals end

        mock_urlopen_fn.side_effect = [
            _mock_urlopen(page1),
            _mock_urlopen(page2),
        ]

        adapter = ArxivAdapter()
        # Request more than one page's worth so pagination is triggered
        articles = adapter.fetch(["query"], max_results=_PAGE_SIZE + 50)

        assert len(articles) == _PAGE_SIZE + 3
        assert mock_urlopen_fn.call_count == 2

        # Verify start params in URLs
        url1 = mock_urlopen_fn.call_args_list[0][0][0]
        url2 = mock_urlopen_fn.call_args_list[1][0][0]
        assert "start=0" in url1
        assert f"start={_PAGE_SIZE}" in url2

        # Sleep called once between pages
        mock_sleep.assert_called_once_with(1)

    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_empty_feed(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_arxiv_feed(0))
        adapter = ArxivAdapter()
        articles = adapter.fetch(["query"], max_results=10)
        assert articles == []

    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_malformed_entry_skipped(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_arxiv_feed(2))
        adapter = ArxivAdapter()

        call_count = 0

        def patched_parse(entry: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("malformed")
            return adapter.__class__._parse_entry(adapter, entry)  # type: ignore[attr-defined]

        with (
            patch.object(adapter, "_parse_entry", side_effect=patched_parse),
            self.assertLogs("tentacle.sources.arxiv", level="WARNING") as log,
        ):
            articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        assert any("skipping malformed" in m for m in log.output)

    @patch("tentacle.sources.arxiv.time.sleep")
    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_retry_on_503(self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock) -> None:
        mock_urlopen_fn.side_effect = [
            _make_http_error(503),
            _make_http_error(503),
            _mock_urlopen(ARXIV_RESPONSE),
        ]

        adapter = ArxivAdapter()
        articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        assert mock_urlopen_fn.call_count == 3
        assert mock_sleep.call_args_list == [call(1), call(2)]

    @patch("tentacle.sources.arxiv.time.sleep")
    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_retry_exhausted_raises(
        self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_urlopen_fn.side_effect = _make_http_error(503)

        adapter = ArxivAdapter()
        with self.assertLogs("tentacle.sources.arxiv", level="ERROR") as log:
            articles = adapter.fetch(["query"], max_results=10)

        assert articles == []
        assert any("arXiv query failed" in m for m in log.output)

    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_date_range_query_param(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_arxiv_feed(0))
        adapter = ArxivAdapter(days_back=7)

        now = datetime.now(UTC)
        start_date = now - timedelta(days=7)
        expected_start = start_date.strftime("%Y%m%d") + "0000"
        expected_end = now.strftime("%Y%m%d") + "2359"

        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0]
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        search_query = params["search_query"][0]

        assert "submittedDate" in search_query
        assert expected_start in search_query
        assert expected_end in search_query

    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_sort_order_passed(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_arxiv_feed(0))
        adapter = ArxivAdapter(sort_order="ascending")
        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0]
        assert "sortOrder=ascending" in url


class TestHackerNewsAdapter(unittest.TestCase):
    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_fetch_parses_hits(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(HN_RESPONSE)
        adapter = HackerNewsAdapter()
        articles = adapter.fetch(["AI code generator"], max_results=10)

        assert len(articles) == 1
        a = articles[0]
        assert a.source == "hn"
        assert a.title == "Show HN: AI Code Generator"
        assert a.url == "https://example.com/ai-codegen"
        assert a.source_id == "12345"


class TestSemanticScholarAdapter(unittest.TestCase):
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_fetch_parses_papers(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(S2_RESPONSE)
        adapter = SemanticScholarAdapter()
        articles = adapter.fetch(["LLM testing"], max_results=10)

        assert len(articles) == 1
        a = articles[0]
        assert a.source == "semantic_scholar"
        assert a.title == "LLM-based Software Testing"
        assert a.authors == ["Dr. Test"]
        assert a.pdf_url == "https://example.com/paper.pdf"
        assert a.access_status == "open"


class TestRSSAdapter(unittest.TestCase):
    @patch("tentacle.sources.rss.urllib.request.urlopen")
    def test_fetch_parses_rss(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(RSS_RESPONSE)
        adapter = RSSAdapter()
        articles = adapter.fetch(["https://blog.example.com/feed"], max_results=10)

        assert len(articles) == 1
        a = articles[0]
        assert a.source == "rss"
        assert a.title == "New Findings in Code Gen"
        assert a.url == "https://blog.example.com/codegen"


if __name__ == "__main__":
    unittest.main()
