"""Tests for source adapters using mocked HTTP responses."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tentacle.sources.arxiv import ArxivAdapter
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
