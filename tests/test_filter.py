"""Tests for the relevance filter stage."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from tentacle.llm.client import CostTracker, LLMClient
from tentacle.llm.filter import filter_article
from tentacle.models import Article


def _make_article() -> Article:
    return Article(
        id="test123",
        source="arxiv",
        title="Autonomous Code Generation with LLMs",
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        abstract="We present a novel approach to autonomous code generation.",
    )


class TestFilter(unittest.TestCase):
    def test_high_relevance_passes(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.85, "reasoning": "Directly applicable"}'
        client.costs = CostTracker()

        score, reasoning = filter_article(
            client, _make_article(), model="claude-haiku-4-5-20251001"
        )
        assert score == 0.85
        assert reasoning == "Directly applicable"

    def test_low_relevance(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.1, "reasoning": "Not relevant"}'
        client.costs = CostTracker()

        score, _reasoning = filter_article(
            client, _make_article(), model="claude-haiku-4-5-20251001"
        )
        assert score == 0.1

    def test_parse_error_returns_zero(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = "not json at all"
        client.costs = CostTracker()

        score, reasoning = filter_article(
            client, _make_article(), model="claude-haiku-4-5-20251001"
        )
        assert score == 0.0
        assert reasoning == "parse error"

    def test_missing_abstract(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.5, "reasoning": "Partial info"}'
        client.costs = CostTracker()

        article = _make_article()
        article.abstract = None
        score, _ = filter_article(client, article, model="claude-haiku-4-5-20251001")
        assert score == 0.5
        # Verify prompt includes "(no abstract available)"
        call_args = client.complete.call_args
        messages = call_args.kwargs["messages"]
        assert "(no abstract available)" in messages[0]["content"]


if __name__ == "__main__":
    unittest.main()
