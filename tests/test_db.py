"""Tests for SQLite catalog store."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from tentacle.db import Store
from tentacle.models import Analysis, Article, DecayEntry, Issue


def _make_article(article_id: str = "abc123", title: str = "Test Article") -> Article:
    return Article(
        id=article_id,
        source="arxiv",
        source_id="2401.00001",
        title=title,
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
        authors=["Alice", "Bob"],
        abstract="Test abstract",
    )


def _make_issue(article_id: str = "abc123", analysis_id: int = 1, github_number: int = 42) -> Issue:
    return Issue(
        article_id=article_id,
        analysis_id=analysis_id,
        github_number=github_number,
        github_url=f"https://github.com/foundatron/octopusgarden/issues/{github_number}",
        title="feat: test",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        maturity_score=4,
        current_maturity=4,
    )


def _make_analysis(article_id: str = "abc123") -> Analysis:
    return Analysis(
        article_id=article_id,
        relevance_score=0.85,
        maturity_score=4,
        model_used="claude-haiku-4-5-20251001",
        analyzed_at=datetime(2025, 1, 1, tzinfo=UTC),
        relevance_reasoning="Highly relevant",
        key_insights=["insight1", "insight2"],
        applicable_scopes=["attractor", "llm"],
        suggested_type="feat",
        suggested_title="feat(attractor): add thing",
        suggested_body="## Problem\nTest body",
        maturity_reasoning="Clear path",
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.001,
    )


class TestStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_insert_and_get_article(self) -> None:
        article = _make_article()
        self.store.insert_article(article)
        got = self.store.get_article("abc123")
        assert got is not None
        assert got.title == "Test Article"
        assert got.authors == ["Alice", "Bob"]

    def test_article_exists(self) -> None:
        assert not self.store.article_exists("abc123")
        self.store.insert_article(_make_article())
        assert self.store.article_exists("abc123")

    def test_insert_duplicate_article_ignored(self) -> None:
        self.store.insert_article(_make_article())
        self.store.insert_article(_make_article())  # should not raise
        assert self.store.article_exists("abc123")

    def test_unanalyzed_articles(self) -> None:
        self.store.insert_article(_make_article("a1"))
        self.store.insert_article(_make_article("a2"))
        unanalyzed = self.store.get_unanalyzed_articles()
        assert len(unanalyzed) == 2

        self.store.insert_analysis(_make_analysis("a1"))
        unanalyzed = self.store.get_unanalyzed_articles()
        assert len(unanalyzed) == 1
        assert unanalyzed[0].id == "a2"

    def test_insert_and_get_analysis(self) -> None:
        self.store.insert_article(_make_article())
        analysis = _make_analysis()
        analysis_id = self.store.insert_analysis(analysis)
        assert analysis_id > 0

        got = self.store.get_analysis_for_article("abc123")
        assert got is not None
        assert got.relevance_score == 0.85
        assert got.key_insights == ["insight1", "insight2"]

    def test_issueable_analyses(self) -> None:
        self.store.insert_article(_make_article("a1"))
        self.store.insert_article(_make_article("a2"))

        a1 = _make_analysis("a1")
        a1.maturity_score = 4
        self.store.insert_analysis(a1)

        a2 = _make_analysis("a2")
        a2.maturity_score = 2
        self.store.insert_analysis(a2)

        results = self.store.get_issueable_analyses(3)
        assert len(results) == 1
        assert results[0].article_id == "a1"

    def test_insert_and_get_issue(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())

        issue = _make_issue(analysis_id=analysis_id)
        issue.title = "feat(attractor): add thing"
        self.store.insert_issue(issue)

        issues = self.store.get_open_issues()
        assert len(issues) == 1
        assert issues[0].github_number == 42

    def test_update_issue_maturity(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())
        issue_id = self.store.insert_issue(_make_issue(analysis_id=analysis_id))

        self.store.update_issue_maturity(issue_id, 3)
        issues = self.store.get_open_issues()
        assert issues[0].current_maturity == 3
        assert issues[0].last_decay_at is not None

    def test_scan_run_lifecycle(self) -> None:
        run_id = self.store.start_scan_run("arxiv")
        assert run_id > 0

        self.store.finish_scan_run(
            run_id,
            articles_found=10,
            articles_new=5,
            articles_relevant=2,
            issues_created=1,
            total_cost_usd=0.05,
        )

        runs = self.store.get_recent_scan_runs(1)
        assert len(runs) == 1
        assert runs[0].articles_found == 10
        assert runs[0].status == "complete"

    def test_decay_log(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())
        issue_id = self.store.insert_issue(_make_issue(analysis_id=analysis_id))

        self.store.insert_decay(
            DecayEntry(
                issue_id=issue_id,
                old_maturity=4,
                new_maturity=3,
                reason="time decay",
                decayed_at=datetime(2025, 3, 1, tzinfo=UTC),
            )
        )
        # Just verify no exception

    def test_get_articles_by_source(self) -> None:
        a1 = _make_article("a1")
        a1.source = "arxiv"
        a1.discovered_at = datetime(2025, 1, 2, tzinfo=UTC)

        a2 = _make_article("a2")
        a2.source = "arxiv"
        a2.discovered_at = datetime(2025, 1, 3, tzinfo=UTC)

        a3 = _make_article("a3")
        a3.source = "hn"

        self.store.insert_article(a1)
        self.store.insert_article(a2)
        self.store.insert_article(a3)

        results = self.store.get_articles_by_source("arxiv")
        assert len(results) == 2
        assert all(r.source == "arxiv" for r in results)
        # descending by discovered_at: a2 (Jan 3) before a1 (Jan 2)
        assert results[0].id == "a2"
        assert results[1].id == "a1"

    def test_get_articles_by_source_empty(self) -> None:
        self.store.insert_article(_make_article())
        results = self.store.get_articles_by_source("semantic_scholar")
        assert results == []

    def test_get_stats(self) -> None:
        # insert articles
        self.store.insert_article(_make_article("a1"))
        self.store.insert_article(_make_article("a2"))
        # insert analysis for a1
        analysis_id = self.store.insert_analysis(_make_analysis("a1"))
        # insert open issue
        self.store.insert_issue(
            Issue(
                article_id="a1",
                analysis_id=analysis_id,
                github_number=1,
                github_url="https://github.com/foundatron/octopusgarden/issues/1",
                title="feat: thing",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                maturity_score=4,
                current_maturity=4,
            )
        )
        # insert analysis for a2 and a closed issue
        analysis_id2 = self.store.insert_analysis(_make_analysis("a2"))
        issue_id2 = self.store.insert_issue(
            Issue(
                article_id="a2",
                analysis_id=analysis_id2,
                github_number=2,
                github_url="https://github.com/foundatron/octopusgarden/issues/2",
                title="feat: other",
                created_at=datetime(2025, 1, 2, tzinfo=UTC),
                maturity_score=4,
                current_maturity=4,
            )
        )
        self.store.update_issue_status(issue_id2, "closed")
        # insert scan run
        run_id = self.store.start_scan_run("arxiv")
        self.store.finish_scan_run(run_id)

        stats = self.store.get_stats()
        assert stats["total_articles"] == 2
        assert stats["total_analyses"] == 2
        assert stats["total_issues"] == 2
        assert stats["open_issues"] == 1
        assert stats["total_scan_runs"] == 1
        assert stats["latest_scan_at"] is not None

    def test_get_stats_empty_db(self) -> None:
        stats = self.store.get_stats()
        assert stats["total_articles"] == 0
        assert stats["total_analyses"] == 0
        assert stats["total_issues"] == 0
        assert stats["open_issues"] == 0
        assert stats["total_scan_runs"] == 0
        assert stats["latest_scan_at"] is None

    def test_get_decay_log_for_issue(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())
        issue_id = self.store.insert_issue(_make_issue(analysis_id=analysis_id))
        self.store.insert_decay(
            DecayEntry(
                issue_id=issue_id,
                old_maturity=4,
                new_maturity=3,
                reason="time decay",
                decayed_at=datetime(2025, 3, 2, tzinfo=UTC),
            )
        )
        self.store.insert_decay(
            DecayEntry(
                issue_id=issue_id,
                old_maturity=3,
                new_maturity=2,
                reason="stale",
                decayed_at=datetime(2025, 3, 1, tzinfo=UTC),
            )
        )

        entries = self.store.get_decay_log_for_issue(issue_id)
        assert len(entries) == 2
        # descending by decayed_at: Mar 2 first
        assert entries[0].old_maturity == 4
        assert entries[0].new_maturity == 3
        assert entries[1].old_maturity == 3
        assert entries[1].new_maturity == 2

    def test_get_decay_log_for_issue_empty(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())
        issue_id = self.store.insert_issue(_make_issue(analysis_id=analysis_id))
        entries = self.store.get_decay_log_for_issue(issue_id)
        assert entries == []

    def test_json_roundtrip_empty_lists(self) -> None:
        article = _make_article()
        article.authors = []
        article.tags = []
        self.store.insert_article(article)
        got = self.store.get_article(article.id)
        assert got is not None
        assert got.authors == []
        assert got.tags == []

        analysis = _make_analysis()
        analysis.key_insights = []
        analysis.applicable_scopes = []
        self.store.insert_analysis(analysis)
        got_analysis = self.store.get_analysis_for_article(article.id)
        assert got_analysis is not None
        assert got_analysis.key_insights == []
        assert got_analysis.applicable_scopes == []

    def test_update_issue_status(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())
        issue_id = self.store.insert_issue(_make_issue(analysis_id=analysis_id))
        assert len(self.store.get_open_issues()) == 1
        self.store.update_issue_status(issue_id, "closed")
        assert self.store.get_open_issues() == []

    def test_json_roundtrip_none_lists(self) -> None:
        article = _make_article("none_test")
        article.authors = None
        article.tags = None
        self.store.insert_article(article)
        got = self.store.get_article(article.id)
        assert got is not None
        assert got.authors is None
        assert got.tags is None

        analysis = _make_analysis("none_test")
        analysis.key_insights = None
        analysis.applicable_scopes = None
        self.store.insert_analysis(analysis)
        got_analysis = self.store.get_analysis_for_article("none_test")
        assert got_analysis is not None
        assert got_analysis.key_insights is None
        assert got_analysis.applicable_scopes is None


if __name__ == "__main__":
    unittest.main()
