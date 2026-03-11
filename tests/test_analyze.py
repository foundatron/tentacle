"""Tests for the deep analysis stage."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock

from tentacle.llm.analyze import ANALYSIS_TOOL, analyze_article
from tentacle.llm.client import CostTracker, LLMClient, UsageRecord
from tentacle.models import Article


def _make_article() -> Article:
    return Article(
        id="test123",
        source="arxiv",
        title="Novel Attractor Convergence Method",
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        authors=["Alice Smith"],
        abstract="A method for improving convergence in iterative optimization.",
        published_at=datetime(2024, 12, 1, tzinfo=UTC),
    )


_ANALYSIS_DATA: dict[str, object] = {
    "key_insights": ["Adaptive temperature scheduling improves convergence"],
    "applicable_scopes": ["attractor"],
    "suggested_type": "feat",
    "suggested_title": "feat(attractor): implement adaptive temperature scheduling",
    "suggested_body": (
        "## Problem Statement\nConvergence can stall.\n\n"
        "## Proposed Change\nAdd temperature scheduling."
    ),
    "maturity_score": 4,
    "maturity_reasoning": "Clear implementation path",
    "confidence_score": 0.9,
}


class TestAnalyze(unittest.TestCase):
    def test_successful_analysis(self) -> None:
        cost_tracker = CostTracker()
        cost_tracker.add(
            UsageRecord(
                model="claude-sonnet-4-6",
                input_tokens=500,
                output_tokens=300,
                cost_usd=0.01,
            )
        )

        client = MagicMock(spec=LLMClient)
        client.complete_with_tools.return_value = _ANALYSIS_DATA
        client.costs = cost_tracker

        result = analyze_article(
            client,
            _make_article(),
            "OctopusGarden context here",
            model="claude-sonnet-4-6",
            relevance_score=0.85,
            relevance_reasoning="Relevant",
        )

        assert result is not None
        assert result.maturity_score == 4
        assert result.suggested_type == "feat"
        assert result.applicable_scopes == ["attractor"]
        assert result.relevance_score == 0.85
        assert result.confidence_score == 0.9

    def test_tool_use_error_returns_none(self) -> None:
        client = MagicMock(spec=LLMClient)
        client.complete_with_tools.side_effect = ValueError("No tool_use block in response")
        client.costs = CostTracker()

        result = analyze_article(
            client,
            _make_article(),
            "context",
            model="claude-sonnet-4-6",
            relevance_score=0.5,
            relevance_reasoning="test",
        )
        assert result is None

    def test_tool_use_schema_matches_analysis_fields(self) -> None:
        """ANALYSIS_TOOL schema must cover the expected Analysis output fields."""
        expected_fields = {
            "key_insights",
            "applicable_scopes",
            "suggested_type",
            "suggested_title",
            "suggested_body",
            "maturity_score",
            "maturity_reasoning",
            "confidence_score",
        }
        schema_props = set(ANALYSIS_TOOL["input_schema"]["properties"].keys())  # type: ignore[index]
        assert expected_fields == schema_props

    def test_missing_confidence_score_returns_none_for_field(self) -> None:
        """analyze_article handles a response missing the optional confidence_score."""
        data_without_confidence = {
            k: v for k, v in _ANALYSIS_DATA.items() if k != "confidence_score"
        }
        cost_tracker = CostTracker()
        cost_tracker.add(
            UsageRecord(
                model="claude-sonnet-4-6",
                input_tokens=500,
                output_tokens=300,
                cost_usd=0.01,
            )
        )

        client = MagicMock(spec=LLMClient)
        client.complete_with_tools.return_value = data_without_confidence
        client.costs = cost_tracker

        result = analyze_article(
            client,
            _make_article(),
            "context",
            model="claude-sonnet-4-6",
            relevance_score=0.8,
            relevance_reasoning="test",
        )

        assert result is not None
        assert result.confidence_score is None

    def test_uses_prompt_caching(self) -> None:
        cost_tracker = CostTracker()
        cost_tracker.add(
            UsageRecord(
                model="claude-sonnet-4-6",
                input_tokens=100,
                output_tokens=100,
                cost_usd=0.001,
            )
        )

        client = MagicMock(spec=LLMClient)
        client.complete_with_tools.return_value = _ANALYSIS_DATA
        client.costs = cost_tracker

        analyze_article(
            client,
            _make_article(),
            "context",
            model="claude-sonnet-4-6",
            relevance_score=0.5,
            relevance_reasoning="test",
        )

        call_args = client.complete_with_tools.call_args
        system = call_args.kwargs["system"]
        # Should be a list with cache_control
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}


if __name__ == "__main__":
    unittest.main()
