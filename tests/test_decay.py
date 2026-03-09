"""Tests for ticket maturity decay logic."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from tentacle.db import Store
from tentacle.decay import apply_decay
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


if __name__ == "__main__":
    unittest.main()
