"""Tests for GitHub issue creation."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from tentacle.issues import _format_body, create_issue
from tentacle.models import Analysis, Article


def _make_article() -> Article:
    return Article(
        id="test123",
        source="arxiv",
        title="Test Paper",
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        authors=["Alice"],
        published_at=datetime(2024, 12, 1, tzinfo=UTC),
    )


def _make_analysis() -> Analysis:
    return Analysis(
        id=1,
        article_id="test123",
        relevance_score=0.85,
        maturity_score=4,
        model_used="claude-sonnet-4-5-20250514",
        analyzed_at=datetime(2025, 1, 1, tzinfo=UTC),
        suggested_type="feat",
        suggested_title="feat(attractor): add thing",
        suggested_body="## Problem Statement\nTest body\n\n## Proposed Change\nDo things.",
    )


class TestIssueCreation(unittest.TestCase):
    def test_dry_run_returns_none(self) -> None:
        result = create_issue(
            _make_article(),
            _make_analysis(),
            repo="foundatron/octopusgarden",
            label="tentacle",
            dry_run=True,
        )
        assert result is None

    def test_missing_title_returns_none(self) -> None:
        analysis = _make_analysis()
        analysis.suggested_title = None
        result = create_issue(
            _make_article(),
            analysis,
            repo="foundatron/octopusgarden",
            label="tentacle",
        )
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_successful_creation(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/foundatron/octopusgarden/issues/42\n",
        )

        result = create_issue(
            _make_article(),
            _make_analysis(),
            repo="foundatron/octopusgarden",
            label="tentacle",
        )

        assert result is not None
        assert result.github_number == 42
        assert result.maturity_score == 4

    @patch("tentacle.issues.subprocess.run")
    def test_gh_failure_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        result = create_issue(
            _make_article(),
            _make_analysis(),
            repo="foundatron/octopusgarden",
            label="tentacle",
        )
        assert result is None

    def test_format_body_adds_source(self) -> None:
        body = _format_body(_make_article(), _make_analysis())
        assert "## Source" in body
        assert "Test Paper" in body
        assert "tentacle" in body.lower()

    def test_format_body_preserves_existing_source(self) -> None:
        analysis = _make_analysis()
        analysis.suggested_body = "## Problem\nBody\n\n## Source\nAlready here"
        body = _format_body(_make_article(), analysis)
        # Should not duplicate Source section
        assert body.count("## Source") == 1


if __name__ == "__main__":
    unittest.main()
