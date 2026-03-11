"""Tests for CLI commands."""

from __future__ import annotations

import io
import tempfile
import unittest
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tentacle.cli import cmd_init, cmd_run, cmd_status
from tentacle.config import Config
from tentacle.llm.client import BudgetExceededError, CostTracker, UsageRecord
from tentacle.models import Article


def _make_article(article_id: str = "abc123") -> Article:
    return Article(
        id=article_id,
        source="arxiv",
        source_id="2401.00001",
        title="Test Article",
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


class TestCmdStatus(unittest.TestCase):
    @patch("tentacle.cli._get_store")
    def test_cmd_status_shows_cost_summary(self, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_store.get_open_issues.return_value = []
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 1.2345,
            "scan_count": 5,
            "avg_cost_per_scan": 0.2469,
        }

        config = Config()
        config.monthly_budget = 10.0
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "1.2345" in output
        assert "5" in output
        assert "10.00" in output
        # Remaining = 10.0 - 1.2345 = 8.7655
        assert "8.7655" in output

    @patch("tentacle.cli._get_store")
    def test_cmd_status_no_budget_line_when_zero(self, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_store.get_open_issues.return_value = []
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }

        config = Config()
        config.monthly_budget = 0.0  # no limit
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "Budget:" not in output


class TestCmdRun(unittest.TestCase):
    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.LLMClient")
    @patch("tentacle.cli.fetch_context")
    @patch("tentacle.cli._get_sources")
    @patch("tentacle.cli.filter_batch")
    def test_cmd_run_budget_exceeded_stops_gracefully(
        self,
        mock_filter: MagicMock,
        mock_get_sources: MagicMock,
        mock_fetch_context: MagicMock,
        mock_llm_cls: MagicMock,
        mock_get_store: MagicMock,
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.start_scan_run.side_effect = [1, 2]
        mock_store.article_exists.return_value = False
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_fetch_context.return_value = MagicMock()

        source1_adapter = MagicMock()
        source1_adapter.fetch.return_value = [_make_article("a1")]
        source2_adapter = MagicMock()
        source2_adapter.fetch.return_value = [_make_article("a2")]

        mock_get_sources.return_value = [
            ("arxiv", ["autonomous code"], 10, source1_adapter),
            ("hn", ["llm"], 10, source2_adapter),
        ]

        # filter_batch raises BudgetExceededError on the first call
        mock_filter.side_effect = BudgetExceededError(2.1, 2.0, "scan")

        config = Config()
        config.anthropic_api_key = "fake-key"
        args = Namespace(dry_run=False)

        cmd_run(args, config)

        # filter_batch was called with a list of articles (not a single article)
        mock_filter.assert_called_once()
        call_args = mock_filter.call_args
        articles_arg = (
            call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("articles")
        )
        assert isinstance(articles_arg, list), "filter_batch must receive a list of articles"

        # First source scan run was started and finished with budget_exceeded
        mock_store.start_scan_run.assert_called_once_with("arxiv")
        mock_store.finish_scan_run.assert_called_once_with(1, status="budget_exceeded")

        # Second source adapter was never contacted
        source2_adapter.fetch.assert_not_called()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.LLMClient")
    @patch("tentacle.cli.fetch_context")
    @patch("tentacle.cli._get_sources")
    @patch("tentacle.cli.filter_batch")
    def test_cmd_run_per_source_cost_delta(
        self,
        mock_filter: MagicMock,
        mock_get_sources: MagicMock,
        mock_fetch_context: MagicMock,
        mock_llm_cls: MagicMock,
        mock_get_store: MagicMock,
    ) -> None:
        """finish_scan_run receives per-source cost delta, not cumulative total."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.start_scan_run.side_effect = [10, 20]
        mock_store.article_exists.return_value = False
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }

        mock_fetch_context.return_value = MagicMock()

        # Capture the cost_tracker that cmd_run passes to LLMClient (second positional arg)
        captured: dict[str, CostTracker] = {}

        def make_llm(*args: object, **kwargs: object) -> MagicMock:
            tracker = args[1] if len(args) > 1 else kwargs.get("cost_tracker")
            captured["tracker"] = tracker  # type: ignore[assignment]
            mock_llm = MagicMock()
            mock_llm.costs = tracker
            return mock_llm

        mock_llm_cls.side_effect = make_llm

        source1_adapter = MagicMock()
        source1_adapter.fetch.return_value = [_make_article("a1")]
        source2_adapter = MagicMock()
        source2_adapter.fetch.return_value = [_make_article("a2")]

        mock_get_sources.return_value = [
            ("arxiv", ["q1"], 10, source1_adapter),
            ("hn", ["q2"], 10, source2_adapter),
        ]

        call_count = 0

        def filter_side_effect(
            _client: object, articles: list[Article], **kwargs: object
        ) -> list[tuple[float, str]]:
            nonlocal call_count
            call_count += 1
            # Add cost to the *same* tracker cmd_run uses for cost_before snapshots
            cost = 0.50 if call_count == 1 else 0.30
            captured["tracker"].add(
                UsageRecord(
                    model="claude-haiku-4-5",
                    input_tokens=10,
                    output_tokens=10,
                    cost_usd=cost,
                )
            )
            return [(0.1, "low relevance")] * len(articles)  # below threshold, no analyze

        mock_filter.side_effect = filter_side_effect

        config = Config()
        config.anthropic_api_key = "fake-key"
        config.relevance_threshold = 0.5  # above 0.1, so no articles proceed to analyze
        args = Namespace(dry_run=False)

        cmd_run(args, config)

        # Verify per-source deltas were passed, not cumulative totals
        calls = mock_store.finish_scan_run.call_args_list
        assert len(calls) == 2

        first_cost = calls[0].kwargs["total_cost_usd"]
        second_cost = calls[1].kwargs["total_cost_usd"]

        assert first_cost == pytest.approx(0.50)
        assert second_cost == pytest.approx(0.30)


class TestCmdInit(unittest.TestCase):
    def test_cmd_init_creates_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / ".config" / "tentacle" / "config.toml"
            args = Namespace()
            cmd_init(args, config_path=config_path)
            self.assertTrue(config_path.exists())
            content = config_path.read_text()
            self.assertIn("target_repo", content)

    def test_cmd_init_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / ".config" / "tentacle" / "config.toml"
            config_path.parent.mkdir(parents=True)
            original_content = "original = true\n"
            config_path.write_text(original_content)

            args = Namespace()
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                cmd_init(args, config_path=config_path)
                output = mock_out.getvalue()

            self.assertEqual(config_path.read_text(), original_content)
            self.assertIn("already exists", output)


if __name__ == "__main__":
    unittest.main()
