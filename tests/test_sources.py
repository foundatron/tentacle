"""Tests for source adapters using mocked HTTP responses."""

from __future__ import annotations

import unittest
import urllib.error
import urllib.parse
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, call, patch

from tentacle.sources.arxiv import PAGE_SIZE, ArxivAdapter
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
      "created_at": "2024-01-15T10:00:00Z",
      "points": 42,
      "num_comments": 7
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
      "openAccessPdf": {"url": "https://example.com/paper.pdf"},
      "citationCount": 5
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
        page1 = _make_arxiv_feed(PAGE_SIZE)  # full page → more to fetch
        page2 = _make_arxiv_feed(3)  # partial page → signals end

        mock_urlopen_fn.side_effect = [
            _mock_urlopen(page1),
            _mock_urlopen(page2),
        ]

        adapter = ArxivAdapter()
        # Request more than one page's worth so pagination is triggered
        articles = adapter.fetch(["query"], max_results=PAGE_SIZE + 50)

        assert len(articles) == PAGE_SIZE + 3
        assert mock_urlopen_fn.call_count == 2

        # Verify start params in URLs
        url1 = mock_urlopen_fn.call_args_list[0][0][0]
        url2 = mock_urlopen_fn.call_args_list[1][0][0]
        assert "start=0" in url1
        assert f"start={PAGE_SIZE}" in url2

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

    @patch("tentacle.sources.arxiv.datetime")
    @patch("tentacle.sources.arxiv.urllib.request.urlopen")
    def test_date_range_query_param(
        self, mock_urlopen_fn: MagicMock, mock_datetime: MagicMock
    ) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(_make_arxiv_feed(0))
        fixed_now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = fixed_now

        adapter = ArxivAdapter(days_back=7)

        start_date = fixed_now - timedelta(days=7)
        expected_start = start_date.strftime("%Y%m%d") + "0000"
        expected_end = fixed_now.strftime("%Y%m%d") + "2359"

        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0]
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
        assert a.metadata is not None
        assert a.metadata["points"] == 42
        assert a.metadata["num_comments"] == 7
        assert a.metadata["discussion_url"] == "https://news.ycombinator.com/item?id=12345"

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_min_points_in_query_params(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"hits": []}')
        adapter = HackerNewsAdapter(min_points=25)
        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0].full_url
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert "numericFilters" in params
        assert "points>=25" in params["numericFilters"][0]

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_days_back_in_query_params(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"hits": []}')
        fixed_now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        expected_ts = int((fixed_now - timedelta(days=7)).timestamp())

        with patch("tentacle.sources.hackernews._utcnow", return_value=fixed_now):
            adapter = HackerNewsAdapter(days_back=7)
            adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0].full_url
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert "numericFilters" in params
        assert f"created_at_i>{expected_ts}" in params["numericFilters"][0]

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_story_type_tag(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"hits": []}')
        adapter = HackerNewsAdapter(story_type="show_hn")
        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0].full_url
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        assert params["tags"] == ["show_hn"]

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"hits": []}')
        adapter = HackerNewsAdapter()
        articles = adapter.fetch(["query"], max_results=10)
        assert articles == []

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_missing_points_defaults_to_zero(self, mock_urlopen_fn: MagicMock) -> None:
        response = b"""\
{
  "hits": [
    {
      "objectID": "99999",
      "title": "Some HN Post",
      "url": "https://example.com/post",
      "author": "user",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}"""
        mock_urlopen_fn.return_value = _mock_urlopen(response)
        adapter = HackerNewsAdapter()
        articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        assert articles[0].metadata is not None
        assert articles[0].metadata["points"] == 0

    @patch("tentacle.sources.hackernews.urllib.request.urlopen")
    def test_missing_url_falls_back_to_discussion(self, mock_urlopen_fn: MagicMock) -> None:
        response = b"""\
{
  "hits": [
    {
      "objectID": "77777",
      "title": "Ask HN: Something",
      "author": "asker",
      "created_at": "2024-01-01T00:00:00Z",
      "points": 5,
      "num_comments": 3
    }
  ]
}"""
        mock_urlopen_fn.return_value = _mock_urlopen(response)
        adapter = HackerNewsAdapter()
        articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        a = articles[0]
        discussion_url = "https://news.ycombinator.com/item?id=77777"
        assert a.url == discussion_url
        assert a.metadata is not None
        assert a.metadata["discussion_url"] == discussion_url


def _make_429_error(retry_after: str | None = "1") -> urllib.error.HTTPError:
    headers = MagicMock()
    headers.get.return_value = retry_after
    return urllib.error.HTTPError(
        url="http://x",
        code=429,
        msg="Too Many Requests",
        hdrs=headers,
        fp=None,  # type: ignore[arg-type]
    )


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

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_api_key_header(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(S2_RESPONSE)
        adapter = SemanticScholarAdapter(api_key="test-key")
        adapter.fetch(["query"], max_results=10)

        req = mock_urlopen_fn.call_args[0][0]
        assert req.get_header("X-api-key") == "test-key"

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_no_api_key_header(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(S2_RESPONSE)
        adapter = SemanticScholarAdapter()
        adapter.fetch(["query"], max_results=10)

        req = mock_urlopen_fn.call_args[0][0]
        assert req.get_header("X-api-key") is None

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_min_citations_filter(self, mock_urlopen_fn: MagicMock) -> None:
        response = b"""\
{
  "data": [
    {
      "title": "High Citations Paper",
      "url": "https://s2.example.com/paper/high",
      "authors": [],
      "abstract": "Abstract.",
      "publicationDate": "2024-01-01",
      "externalIds": {},
      "openAccessPdf": null,
      "citationCount": 20
    },
    {
      "title": "Low Citations Paper",
      "url": "https://s2.example.com/paper/low",
      "authors": [],
      "abstract": "Abstract.",
      "publicationDate": "2024-01-01",
      "externalIds": {},
      "openAccessPdf": null,
      "citationCount": 3
    }
  ]
}"""
        mock_urlopen_fn.return_value = _mock_urlopen(response)
        adapter = SemanticScholarAdapter(min_citations=10)
        articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        assert articles[0].title == "High Citations Paper"

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_min_citations_null_treated_as_zero(self, mock_urlopen_fn: MagicMock) -> None:
        response = b"""\
{
  "data": [
    {
      "title": "No Citation Count Paper",
      "url": "https://s2.example.com/paper/nocite",
      "authors": [],
      "abstract": "Abstract.",
      "publicationDate": "2024-01-01",
      "externalIds": {},
      "openAccessPdf": null,
      "citationCount": null
    }
  ]
}"""
        mock_urlopen_fn.return_value = _mock_urlopen(response)
        adapter = SemanticScholarAdapter(min_citations=1)
        articles = adapter.fetch(["query"], max_results=10)

        assert articles == []

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"data": []}')
        adapter = SemanticScholarAdapter()
        articles = adapter.fetch(["query"], max_results=10)
        assert articles == []

    @patch("tentacle.sources.semantic_scholar.time.sleep")
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock) -> None:
        mock_urlopen_fn.side_effect = [
            _make_429_error("1"),
            _mock_urlopen(S2_RESPONSE),
        ]
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="WARNING"):
            articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        mock_sleep.assert_called_once_with(1)

    @patch("tentacle.sources.semantic_scholar.time.sleep")
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_retry_429_exhausted(self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock) -> None:
        mock_urlopen_fn.side_effect = _make_429_error("1")
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="ERROR"):
            articles = adapter.fetch(["query"], max_results=10)

        assert articles == []
        assert mock_sleep.call_count == 3  # 3 retries before exhaustion

    @patch("tentacle.sources.semantic_scholar.datetime")
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_date_range_filter(self, mock_urlopen_fn: MagicMock, mock_datetime: MagicMock) -> None:
        mock_urlopen_fn.return_value = _mock_urlopen(b'{"data": []}')
        fixed_now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = fixed_now

        adapter = SemanticScholarAdapter(days_back=7)
        adapter.fetch(["query"], max_results=10)

        url = mock_urlopen_fn.call_args[0][0].full_url
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        date_param = params["publicationDateOrYear"][0]

        start_date = fixed_now - timedelta(days=7)
        expected = f"{start_date.strftime('%Y-%m-%d')}:{fixed_now.strftime('%Y-%m-%d')}"
        assert date_param == expected

    @patch("tentacle.sources.semantic_scholar.time.sleep")
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_retry_after_non_integer_falls_back_to_one_second(
        self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Non-integer Retry-After header should fall back to 1s and log at DEBUG."""
        mock_urlopen_fn.side_effect = [
            _make_429_error("Thu, 01 Dec 1994 16:00:00 GMT"),
            _mock_urlopen(S2_RESPONSE),
        ]
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="DEBUG") as log:
            articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        mock_sleep.assert_called_once_with(1)  # fell back to 1s
        assert any("not an integer" in m for m in log.output)

    @patch("tentacle.sources.semantic_scholar.time.sleep")
    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_retry_after_absent_falls_back_to_one_second(
        self, mock_urlopen_fn: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Absent Retry-After header should default to 1s sleep."""
        mock_urlopen_fn.side_effect = [
            _make_429_error(None),
            _mock_urlopen(S2_RESPONSE),
        ]
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="WARNING"):
            articles = adapter.fetch(["query"], max_results=10)

        assert len(articles) == 1
        mock_sleep.assert_called_once_with(1)

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_url_error_returns_empty(self, mock_urlopen_fn: MagicMock) -> None:
        """URLError (e.g. DNS failure, timeout) should log and return no articles."""
        import urllib.error as _ue

        mock_urlopen_fn.side_effect = _ue.URLError("Name or service not known")
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="ERROR") as log:
            articles = adapter.fetch(["query"], max_results=10)

        assert articles == []
        assert any("network error" in m for m in log.output)

    @patch("tentacle.sources.semantic_scholar.urllib.request.urlopen")
    def test_non_retryable_error_skips_query(self, mock_urlopen_fn: MagicMock) -> None:
        mock_urlopen_fn.side_effect = [
            _make_http_error(500),
            _mock_urlopen(S2_RESPONSE),
        ]
        adapter = SemanticScholarAdapter()
        with self.assertLogs("tentacle.sources.semantic_scholar", level="ERROR"):
            articles = adapter.fetch(["query1", "query2"], max_results=20)

        assert len(articles) == 1
        assert mock_urlopen_fn.call_count == 2


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
