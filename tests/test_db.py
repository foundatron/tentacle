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

        issue = Issue(
            article_id="abc123",
            analysis_id=analysis_id,
            github_number=42,
            github_url="https://github.com/foundatron/octopusgarden/issues/42",
            title="feat(attractor): add thing",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            maturity_score=4,
            current_maturity=4,
        )
        self.store.insert_issue(issue)

        issues = self.store.get_open_issues()
        assert len(issues) == 1
        assert issues[0].github_number == 42

    def test_update_issue_maturity(self) -> None:
        self.store.insert_article(_make_article())
        analysis_id = self.store.insert_analysis(_make_analysis())

        issue = Issue(
            article_id="abc123",
            analysis_id=analysis_id,
            github_number=42,
            github_url="https://github.com/foundatron/octopusgarden/issues/42",
            title="feat: test",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            maturity_score=4,
            current_maturity=4,
        )
        issue_id = self.store.insert_issue(issue)

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
        issue_id = self.store.insert_issue(
            Issue(
                article_id="abc123",
                analysis_id=analysis_id,
                github_number=42,
                github_url="https://github.com/foundatron/octopusgarden/issues/42",
                title="feat: test",
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                maturity_score=4,
                current_maturity=4,
            )
        )

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


if __name__ == "__main__":
    unittest.main()
