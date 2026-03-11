"""Tests for ticket maturity decay logic."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from tentacle.db import Store
from tentacle.decay import apply_decay
from tentacle.llm.client import BudgetExceededError
from tentacle.models import Analysis, Article, Issue


def _setup_store_with_issue(
    created_days_ago: int,
    current_maturity: int = 4,
    last_decay_days_ago: int | None = None,
) -> tuple[Store, int]:
    """Create a store with one article, analysis, and issue. Returns (store, issue_id)."""
    store = Store(":memory:")
    now = datetime.now(UTC)

    article = Article(
        id="test123",
        source="arxiv",
        title="Test",
        url="https://example.com",
        discovered_at=now,
    )
    store.insert_article(article)

    analysis = Analysis(
        article_id="test123",
        relevance_score=0.8,
        maturity_score=4,
        model_used="test",
        analyzed_at=now,
        suggested_body="## Problem\nTest body.",
    )
    analysis_id = store.insert_analysis(analysis)

    created_at = now - timedelta(days=created_days_ago)
    last_decay_at = None
    if last_decay_days_ago is not None:
        last_decay_at = now - timedelta(days=last_decay_days_ago)

    issue = Issue(
        article_id="test123",
        analysis_id=analysis_id,
        github_number=42,
        github_url="https://github.com/test/42",
        title="feat: test",
        created_at=created_at,
        maturity_score=4,
        current_maturity=current_maturity,
        last_decay_at=last_decay_at,
    )
    issue_id = store.insert_issue(issue)
    return store, issue_id


class TestDecay(unittest.TestCase):
    def test_no_decay_within_grace_period(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=15)
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 0
        store.close()

    def test_decay_after_grace_period(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100)
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 1

        issues = store.get_open_issues()
        assert issues[0].current_maturity == 3
        store.close()

    def test_no_decay_before_interval(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, last_decay_days_ago=30)
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 0
        store.close()

    def test_decay_after_interval(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=200, last_decay_days_ago=70)
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 1
        store.close()

    def test_no_decay_at_minimum(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=200, current_maturity=1)
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 0
        store.close()

    def test_multiple_issues(self) -> None:
        store = Store(":memory:")
        now = datetime.now(UTC)

        for i in range(3):
            aid = f"art{i}"
            store.insert_article(
                Article(
                    id=aid,
                    source="arxiv",
                    title=f"Test {i}",
                    url=f"https://example.com/{i}",
                    discovered_at=now,
                )
            )
            analysis_id = store.insert_analysis(
                Analysis(
                    article_id=aid,
                    relevance_score=0.8,
                    maturity_score=4,
                    model_used="test",
                    analyzed_at=now,
                )
            )
            store.insert_issue(
                Issue(
                    article_id=aid,
                    analysis_id=analysis_id,
                    github_number=i,
                    github_url=f"https://github.com/test/{i}",
                    title=f"feat: test {i}",
                    created_at=now - timedelta(days=100),
                    maturity_score=4,
                    current_maturity=4,
                )
            )

        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 3
        store.close()

    # -- LLM recheck tests --

    @patch("tentacle.decay.comment_on_issue")
    def test_threshold_4_to_3_llm_halt(self, mock_comment: MagicMock) -> None:
        store, issue_id = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"action": "halt", "reasoning": "still relevant", "comment": "Staying open."}'
        )

        decayed = apply_decay(
            store,
            grace_days=30,
            interval_days=60,
            llm_client=mock_llm,
            repo="org/repo",
        )

        assert decayed == 0
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 4
        mock_comment.assert_called_once_with(42, "Staying open.", repo="org/repo", dry_run=False)
        log = store.get_decay_log_for_issue(issue_id)
        assert len(log) == 1
        assert "llm_recheck:halt" in log[0].reason
        store.close()

    def test_threshold_4_to_3_llm_decay(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '{"action": "decay", "reasoning": "losing relevance"}'

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 3
        store.close()

    @patch("tentacle.decay.close_issue", return_value=True)
    def test_threshold_4_to_3_llm_accelerate(self, mock_close: MagicMock) -> None:
        store, issue_id = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"action": "accelerate", "reasoning": "superseded", "comment": "Closing."}'
        )

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        # Issue should be closed in DB
        issues = store.get_open_issues()
        assert len(issues) == 0
        mock_close.assert_called_once_with(42, "Closing.", repo="org/repo", dry_run=False)
        log = store.get_decay_log_for_issue(issue_id)
        assert len(log) == 1
        assert "llm_recheck:accelerate" in log[0].reason
        assert log[0].new_maturity == 1
        store.close()

    @patch("tentacle.decay.comment_on_issue")
    def test_threshold_2_to_1_llm_halt(self, mock_comment: MagicMock) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=2)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"action": "halt", "reasoning": "still needed", "comment": ""}'
        )

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 0
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 2
        mock_comment.assert_not_called()  # comment is empty
        store.close()

    @patch("tentacle.decay.close_issue", return_value=True)
    def test_threshold_2_to_1_llm_accelerate(self, mock_close: MagicMock) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=2)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '{"action": "accelerate", "reasoning": "no longer relevant", "comment": "Done."}'
        )

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        mock_close.assert_called_once()
        issues = store.get_open_issues()
        assert len(issues) == 0
        store.close()

    def test_non_threshold_no_llm_call(self) -> None:
        # maturity 5 -> 4 is NOT a threshold, LLM should not be called
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=5)
        mock_llm = MagicMock()

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        mock_llm.complete.assert_not_called()
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 4
        store.close()

    def test_no_llm_client_skips_recheck(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        # No llm_client — should behave identically to original mechanical decay
        decayed = apply_decay(store, grace_days=30, interval_days=60)
        assert decayed == 1
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 3
        store.close()

    def test_llm_exception_falls_through(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("network error")

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        # Fail-open: normal -1 decay
        assert decayed == 1
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 3
        store.close()

    def test_llm_malformed_json_falls_through(self) -> None:
        store, _ = _setup_store_with_issue(created_days_ago=100, current_maturity=4)
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '{"action": "bogus_unknown_action"}'

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        issues = store.get_open_issues()
        assert issues[0].current_maturity == 3
        store.close()

    def test_llm_budget_exceeded_disables_recheck(self) -> None:
        """BudgetExceededError on first call disables LLM for subsequent issues."""
        store = Store(":memory:")
        now = datetime.now(UTC)

        for i in range(2):
            aid = f"art{i}"
            store.insert_article(
                Article(
                    id=aid,
                    source="arxiv",
                    title=f"Test {i}",
                    url=f"https://example.com/{i}",
                    discovered_at=now,
                )
            )
            analysis_id = store.insert_analysis(
                Analysis(
                    article_id=aid,
                    relevance_score=0.8,
                    maturity_score=4,
                    model_used="test",
                    analyzed_at=now,
                )
            )
            store.insert_issue(
                Issue(
                    article_id=aid,
                    analysis_id=analysis_id,
                    github_number=i,
                    github_url=f"https://github.com/test/{i}",
                    title=f"feat: test {i}",
                    created_at=now - timedelta(days=100),
                    maturity_score=4,
                    current_maturity=4,
                )
            )

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = BudgetExceededError(1.0, 0.5, "scan")

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        # Both issues should be mechanically decayed
        assert decayed == 2
        for issue in store.get_open_issues():
            assert issue.current_maturity == 3
        # LLM called only once (disabled after BudgetExceededError)
        assert mock_llm.complete.call_count == 1
        store.close()

    @patch("tentacle.decay.close_issue", return_value=True)
    def test_decay_to_1_closes_issue(self, mock_close: MagicMock) -> None:
        store, _issue_id = _setup_store_with_issue(created_days_ago=100, current_maturity=2)
        # No LLM client — mechanical decay to 1
        decayed = apply_decay(store, grace_days=30, interval_days=60, repo="org/repo")

        assert decayed == 1
        mock_close.assert_called_once()
        # Issue status updated to closed in DB
        issues = store.get_open_issues()
        assert len(issues) == 0
        store.close()

    @patch("tentacle.decay.close_issue")
    def test_decay_to_1_no_repo_skips_close(self, mock_close: MagicMock) -> None:
        """When repo='', close_issue must NOT be called (gh would error with --repo '')."""
        store, _issue_id = _setup_store_with_issue(created_days_ago=100, current_maturity=2)
        decayed = apply_decay(store, grace_days=30, interval_days=60, repo="")

        assert decayed == 1
        mock_close.assert_not_called()
        # Issue is decayed in DB but status stays open (no repo to close against)
        issues = store.get_open_issues()
        assert len(issues) == 1
        assert issues[0].current_maturity == 1
        store.close()

    @patch("tentacle.decay.close_issue", return_value=False)
    def test_decay_to_1_close_failure_logs_warning(self, mock_close: MagicMock) -> None:
        """When close_issue returns False, the issue remains open in DB at maturity 1."""
        store, _issue_id = _setup_store_with_issue(created_days_ago=100, current_maturity=2)
        decayed = apply_decay(store, grace_days=30, interval_days=60, repo="org/repo")

        assert decayed == 1
        mock_close.assert_called_once()
        # Status NOT updated — gh close failed
        issues = store.get_open_issues()
        assert len(issues) == 1
        assert issues[0].current_maturity == 1
        store.close()

    def test_analysis_body_none_handled(self) -> None:
        """Analysis with suggested_body=None falls back to placeholder text."""
        store = Store(":memory:")
        now = datetime.now(UTC)

        store.insert_article(
            Article(
                id="art_none",
                source="arxiv",
                title="Test None Body",
                url="https://example.com/none",
                discovered_at=now,
            )
        )
        analysis = Analysis(
            article_id="art_none",
            relevance_score=0.8,
            maturity_score=4,
            model_used="test",
            analyzed_at=now,
            suggested_body=None,
        )
        analysis_id = store.insert_analysis(analysis)
        store.insert_issue(
            Issue(
                article_id="art_none",
                analysis_id=analysis_id,
                github_number=99,
                github_url="https://github.com/test/99",
                title="feat: none body",
                created_at=now - timedelta(days=100),
                maturity_score=4,
                current_maturity=4,
            )
        )

        mock_llm = MagicMock()
        mock_llm.complete.return_value = '{"action": "decay", "reasoning": "ok"}'

        decayed = apply_decay(
            store, grace_days=30, interval_days=60, llm_client=mock_llm, repo="org/repo"
        )

        assert decayed == 1
        # Verify LLM was called with fallback body text
        call_args = mock_llm.complete.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "(no body available)" in user_content
        store.close()


if __name__ == "__main__":
    unittest.main()
