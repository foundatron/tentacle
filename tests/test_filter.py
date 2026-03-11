"""Tests for the relevance filter stage."""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from tentacle.llm.client import CostTracker, LLMClient
from tentacle.llm.filter import filter_article, filter_batch
from tentacle.models import Article


def _make_article(n: int = 0) -> Article:
    return Article(
        id=f"test{n:03d}",
        source="arxiv",
        title=f"Article {n}: Autonomous Code Generation with LLMs",
        url=f"https://arxiv.org/abs/2401.{n:05d}",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        abstract=f"Abstract {n}: We present a novel approach to autonomous code generation.",
    )


def _make_articles(count: int) -> list[Article]:
    return [_make_article(i) for i in range(count)]


class TestFilter(unittest.TestCase):
    def test_high_relevance_passes(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.85, "reasoning": "Directly applicable"}'
        client.costs = CostTracker()

        score, reasoning = filter_article(client, _make_article(), model="claude-haiku-4-5")
        assert score == 0.85
        assert reasoning == "Directly applicable"

    def test_low_relevance(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.1, "reasoning": "Not relevant"}'
        client.costs = CostTracker()

        score, _reasoning = filter_article(client, _make_article(), model="claude-haiku-4-5")
        assert score == 0.1

    def test_parse_error_returns_zero(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = "not json at all"
        client.costs = CostTracker()

        score, reasoning = filter_article(client, _make_article(), model="claude-haiku-4-5")
        assert score == 0.0
        assert reasoning == "parse error"

    def test_missing_abstract(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = '{"relevance": 0.5, "reasoning": "Partial info"}'
        client.costs = CostTracker()

        article = _make_article()
        article.abstract = None
        score, _ = filter_article(client, article, model="claude-haiku-4-5")
        assert score == 0.5
        # Verify prompt includes "(no abstract available)"
        call_args = client.complete.call_args
        messages = call_args.kwargs["messages"]
        assert "(no abstract available)" in messages[0]["content"]


class TestFilterBatch(unittest.TestCase):
    def test_batch_all_scored(self) -> None:
        """3 articles, valid JSON array with all 3 scores. Single LLM call."""
        articles = _make_articles(3)
        batch_response = json.dumps(
            [
                {"index": 1, "relevance": 0.9, "reasoning": "Very relevant"},
                {"index": 2, "relevance": 0.5, "reasoning": "Somewhat relevant"},
                {"index": 3, "relevance": 0.1, "reasoning": "Not relevant"},
            ]
        )
        client = MagicMock(spec=LLMClient)
        client.complete.return_value = batch_response

        results = filter_batch(client, articles, model="claude-haiku-4-5")

        assert len(results) == 3
        assert results[0] == (0.9, "Very relevant")
        assert results[1] == (0.5, "Somewhat relevant")
        assert results[2] == (0.1, "Not relevant")
        assert client.complete.call_count == 1

    def test_batch_json_parse_failure_falls_back(self) -> None:
        """LLM returns garbage. Falls back to 3 individual calls (4 total)."""
        articles = _make_articles(3)
        individual_response = '{"relevance": 0.7, "reasoning": "fallback"}'
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [
            "not json garbage",
            individual_response,
            individual_response,
            individual_response,
        ]

        results = filter_batch(client, articles, model="claude-haiku-4-5")

        assert client.complete.call_count == 4  # 1 batch + 3 individual
        assert len(results) == 3
        for score, reasoning in results:
            assert score == 0.7
            assert reasoning == "fallback"

    def test_batch_partial_failure(self) -> None:
        """Valid JSON array with only 2 of 3 entries. 2 parsed, 1 individual fallback."""
        articles = _make_articles(3)
        partial_response = json.dumps(
            [
                {"index": 1, "relevance": 0.8, "reasoning": "Good"},
                {"index": 3, "relevance": 0.2, "reasoning": "Weak"},
            ]
        )
        individual_response = '{"relevance": 0.6, "reasoning": "fallback"}'
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [partial_response, individual_response]

        results = filter_batch(client, articles, model="claude-haiku-4-5")

        assert client.complete.call_count == 2  # 1 batch + 1 individual fallback
        assert results[0] == (0.8, "Good")
        assert results[1] == (0.6, "fallback")  # index 2 was missing
        assert results[2] == (0.2, "Weak")

    def test_batch_empty_list(self) -> None:
        """Empty input returns empty results, no LLM call."""
        client = MagicMock(spec=LLMClient)

        results = filter_batch(client, [], model="claude-haiku-4-5")

        assert results == []
        client.complete.assert_not_called()

    def test_batch_respects_batch_size(self) -> None:
        """15 articles with batch_size=10 results in 2 batch LLM calls."""
        articles = _make_articles(15)
        batch_10 = json.dumps(
            [{"index": i, "relevance": 0.5, "reasoning": "ok"} for i in range(1, 11)]
        )
        batch_5 = json.dumps(
            [{"index": i, "relevance": 0.5, "reasoning": "ok"} for i in range(1, 6)]
        )
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [batch_10, batch_5]

        results = filter_batch(client, articles, model="claude-haiku-4-5", batch_size=10)

        assert client.complete.call_count == 2
        assert len(results) == 15

    def test_batch_out_of_range_index(self) -> None:
        """LLM returns index beyond batch size. That entry is discarded; falls back individually."""
        articles = _make_articles(2)
        bad_response = json.dumps(
            [
                {"index": 1, "relevance": 0.9, "reasoning": "Good"},
                {"index": 5, "relevance": 0.8, "reasoning": "Out of range"},  # batch size is 2
            ]
        )
        individual_response = '{"relevance": 0.4, "reasoning": "fallback"}'
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [bad_response, individual_response]

        results = filter_batch(client, articles, model="claude-haiku-4-5")

        assert len(results) == 2
        assert results[0] == (0.9, "Good")
        assert results[1] == (0.4, "fallback")  # index 5 discarded, fallback for article[1]
        assert client.complete.call_count == 2

    def test_batch_duplicate_index(self) -> None:
        """Duplicate index entries; first wins, missing indices fall back."""
        articles = _make_articles(3)
        dup_response = json.dumps(
            [
                {"index": 1, "relevance": 0.9, "reasoning": "First"},
                {"index": 1, "relevance": 0.1, "reasoning": "Duplicate - ignored"},
                {"index": 3, "relevance": 0.7, "reasoning": "Third"},
                # index 2 missing
            ]
        )
        individual_response = '{"relevance": 0.5, "reasoning": "fallback"}'
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = [dup_response, individual_response]

        results = filter_batch(client, articles, model="claude-haiku-4-5")

        assert len(results) == 3
        assert results[0] == (0.9, "First")  # first duplicate wins
        assert results[1] == (0.5, "fallback")  # index 2 missing, fell back
        assert results[2] == (0.7, "Third")
        assert client.complete.call_count == 2


if __name__ == "__main__":
    unittest.main()
