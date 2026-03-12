"""Tests for GitHub issue creation."""

from __future__ import annotations

import json
import subprocess
import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from tentacle.issues import (
    _format_body,
    _sanitize_search_query,
    _title_similarity,
    check_duplicate,
    close_issue,
    comment_on_issue,
    create_issue,
)
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
        model_used="claude-sonnet-4-6",
        analyzed_at=datetime(2025, 1, 1, tzinfo=UTC),
        suggested_type="feat",
        suggested_title="feat(attractor): add thing",
        suggested_body="## Problem Statement\nTest body\n\n## Proposed Change\nDo things.",
    )


class TestTitleSimilarity(unittest.TestCase):
    def test_identical(self) -> None:
        assert _title_similarity("add streaming support", "add streaming support") == 1.0

    def test_disjoint(self) -> None:
        assert _title_similarity("add streaming", "remove caching") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _title_similarity("add streaming support", "add caching support")
        # intersection: {add, support} = 2; union: {add, streaming, support, caching} = 4
        assert sim == 0.5

    def test_case_insensitive(self) -> None:
        assert _title_similarity("Add Streaming", "add streaming") == 1.0

    def test_strips_cc_prefix(self) -> None:
        # Different type prefixes but identical bodies — should be 1.0 after stripping
        assert _title_similarity("feat(llm): add streaming", "fix(llm): add streaming") == 1.0

    def test_empty_after_strip(self) -> None:
        # Both empty after prefix stripping — should return 0.0 (not division-by-zero)
        assert _title_similarity("feat:", "fix:") == 0.0


class TestSanitizeSearchQuery(unittest.TestCase):
    def test_plain_title_unchanged(self) -> None:
        assert _sanitize_search_query("add streaming support") == "add streaming support"

    def test_strips_quotes(self) -> None:
        result = _sanitize_search_query('feat: "add" streaming')
        assert '"' not in result

    def test_strips_github_operators(self) -> None:
        result = _sanitize_search_query("add NOT remove OR cache AND stream")
        assert "NOT" not in result
        assert "OR" not in result
        assert "AND" not in result

    def test_strips_special_chars(self) -> None:
        result = _sanitize_search_query("foo:bar (baz) <qux>")
        assert ":" not in result
        assert "(" not in result
        assert "<" not in result

    def test_collapses_whitespace(self) -> None:
        result = _sanitize_search_query("add   streaming")
        assert "  " not in result


class TestCheckDuplicate(unittest.TestCase):
    @patch("tentacle.issues.subprocess.run")
    def test_finds_match(self, mock_run: MagicMock) -> None:
        issues = [
            {
                "title": "feat(llm): add streaming support",
                "number": 7,
                "url": "https://github.com/org/repo/issues/7",
            }
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(issues))

        result = check_duplicate("feat(api): add streaming support", repo="org/repo")
        assert result == "https://github.com/org/repo/issues/7"

    @patch("tentacle.issues.subprocess.run")
    def test_no_match(self, mock_run: MagicMock) -> None:
        issues = [
            {
                "title": "fix(db): remove stale connections",
                "number": 3,
                "url": "https://github.com/org/repo/issues/3",
            }
        ]
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(issues))

        result = check_duplicate("feat(llm): add streaming support", repo="org/repo")
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_gh_failure_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="authentication required", stdout="")

        result = check_duplicate("feat: add thing", repo="org/repo")
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_timeout_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=30)

        result = check_duplicate("feat: add thing", repo="org/repo")
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_gh_not_installed_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError

        result = check_duplicate("feat: add thing", repo="org/repo")
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_invalid_json_returns_none(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="not valid json{{")

        result = check_duplicate("feat: add thing", repo="org/repo")
        assert result is None


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

    @patch("tentacle.issues.check_duplicate", return_value=None)
    @patch("tentacle.issues.subprocess.run")
    def test_successful_creation(self, mock_run: MagicMock, _mock_dup: MagicMock) -> None:
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

        args = mock_run.call_args[0][0]
        # Should have both the primary label and the maturity label
        label_indices = [i for i, a in enumerate(args) if a == "--label"]
        assert len(label_indices) == 2
        assert args[label_indices[0] + 1] == "tentacle"
        assert args[label_indices[1] + 1] == "m4"

    @patch("tentacle.issues.check_duplicate", return_value=None)
    @patch("tentacle.issues.subprocess.run")
    def test_maturity_label_in_args(self, mock_run: MagicMock, _mock_dup: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/foundatron/octopusgarden/issues/10\n",
        )

        analysis = _make_analysis()
        analysis.maturity_score = 3

        create_issue(
            _make_article(),
            analysis,
            repo="foundatron/octopusgarden",
            label="tentacle",
        )

        args = mock_run.call_args[0][0]
        label_indices = [i for i, a in enumerate(args) if a == "--label"]
        assert args[label_indices[1] + 1] == "m3"

    @patch("tentacle.issues.check_duplicate", return_value=None)
    @patch("tentacle.issues.subprocess.run")
    def test_gh_failure_returns_none(self, mock_run: MagicMock, _mock_dup: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        result = create_issue(
            _make_article(),
            _make_analysis(),
            repo="foundatron/octopusgarden",
            label="tentacle",
        )
        assert result is None

    @patch("tentacle.issues.subprocess.run")
    def test_skips_duplicate(self, mock_run: MagicMock) -> None:
        dup_url = "https://github.com/foundatron/octopusgarden/issues/99"
        with patch("tentacle.issues.check_duplicate", return_value=dup_url):
            result = create_issue(
                _make_article(),
                _make_analysis(),
                repo="foundatron/octopusgarden",
                label="tentacle",
            )
        assert result is None
        mock_run.assert_not_called()

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


class TestCommentOnIssue(unittest.TestCase):
    @patch("tentacle.issues.subprocess.run")
    def test_comment_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        result = comment_on_issue(7, "Great finding!", repo="org/repo")

        assert result is True
        args = mock_run.call_args[0][0]
        assert args[0] == "gh"
        assert "comment" in args
        assert str(7) in args
        assert "--body" in args
        assert "Great finding!" in args
        assert "--repo" in args
        assert "org/repo" in args

    @patch("tentacle.issues.subprocess.run")
    def test_comment_dry_run(self, mock_run: MagicMock) -> None:
        result = comment_on_issue(7, "Test", repo="org/repo", dry_run=True)
        assert result is True
        mock_run.assert_not_called()

    @patch("tentacle.issues.subprocess.run")
    def test_comment_gh_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="auth error", stdout="")

        result = comment_on_issue(7, "Test", repo="org/repo")

        assert result is False


class TestCloseIssue(unittest.TestCase):
    @patch("tentacle.issues.subprocess.run")
    def test_close_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        result = close_issue(42, "Closing now.", repo="org/repo")

        assert result is True
        args = mock_run.call_args[0][0]
        assert args[0] == "gh"
        assert "close" in args
        assert str(42) in args
        assert "--comment" in args
        assert "Closing now." in args
        assert "--repo" in args
        assert "org/repo" in args

    @patch("tentacle.issues.subprocess.run")
    def test_close_dry_run(self, mock_run: MagicMock) -> None:
        result = close_issue(42, "Closing.", repo="org/repo", dry_run=True)
        assert result is True
        mock_run.assert_not_called()

    @patch("tentacle.issues.subprocess.run")
    def test_close_gh_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="not found", stdout="")

        result = close_issue(42, "Closing.", repo="org/repo")

        assert result is False


if __name__ == "__main__":
    unittest.main()
