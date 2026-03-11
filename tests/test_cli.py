"""Tests for CLI commands."""

from __future__ import annotations

import io
import os
import signal as sig_module
import subprocess
import tempfile
import threading
import unittest
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tentacle.cli import cmd_daemon, cmd_health, cmd_init, cmd_run, cmd_status, main
from tentacle.config import Config, ConfigError
from tentacle.llm.client import BudgetExceededError, CostTracker, UsageRecord
from tentacle.models import Article, ScanRun


def _make_article(article_id: str = "abc123") -> Article:
    return Article(
        id=article_id,
        source="arxiv",
        source_id="2401.00001",
        title="Test Article",
        url="https://arxiv.org/abs/2401.00001",
        discovered_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_status_summary(
    total_articles: int = 0,
    total_analyses: int = 0,
    high: int = 0,
    medium: int = 0,
    low: int = 0,
    open_issues: int = 0,
    closed_issues: int = 0,
    monthly_costs: list[dict[str, str | float]] | None = None,
) -> dict[str, object]:
    return {
        "total_articles": total_articles,
        "total_analyses": total_analyses,
        "relevance_tiers": {"high": high, "medium": medium, "low": low},
        "open_issues": open_issues,
        "closed_issues": closed_issues,
        "monthly_costs": monthly_costs or [],
    }


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
        mock_store.get_status_summary.return_value = _make_status_summary()

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
        mock_store.get_status_summary.return_value = _make_status_summary()

        config = Config()
        config.monthly_budget = 0.0  # no limit
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "Budget:" not in output

    @patch("tentacle.cli._get_store")
    def test_cmd_status_shows_article_and_analysis_counts(self, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_store.get_open_issues.return_value = []
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }
        mock_store.get_status_summary.return_value = _make_status_summary(
            total_articles=42, total_analyses=17
        )

        config = Config()
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "42" in output
        assert "17" in output

    @patch("tentacle.cli._get_store")
    def test_cmd_status_shows_relevance_tiers(self, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_store.get_open_issues.return_value = []
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }
        mock_store.get_status_summary.return_value = _make_status_summary(high=5, medium=12, low=3)

        config = Config()
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "high=5" in output
        assert "medium=12" in output
        assert "low=3" in output

    @patch("tentacle.cli._get_store")
    def test_cmd_status_shows_open_closed_issues(self, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_store.get_open_issues.return_value = []
        mock_store.get_monthly_cost.return_value = {
            "total_cost": 0.0,
            "scan_count": 0,
            "avg_cost_per_scan": 0.0,
        }
        mock_store.get_status_summary.return_value = _make_status_summary(
            open_issues=7, closed_issues=23
        )

        config = Config()
        args = Namespace()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            cmd_status(args, config)
            output = mock_out.getvalue()

        assert "open=7" in output
        assert "closed=23" in output


class TestCmdDaemon(unittest.TestCase):
    def test_daemon_argument_parsing(self) -> None:
        """--interval and --dry-run are correctly parsed for the daemon subcommand."""
        captured: dict[str, Any] = {}

        def fake_cmd_daemon(args: Namespace, config: Config) -> None:
            captured["args"] = args

        with (
            patch("sys.argv", ["tentacle", "daemon", "--interval", "120", "--dry-run"]),
            patch("tentacle.cli.load_config") as mock_load_config,
            patch("tentacle.cli.cmd_daemon", side_effect=fake_cmd_daemon),
        ):
            mock_cfg = Config()
            mock_cfg.anthropic_api_key = "fake-key"
            mock_load_config.return_value = mock_cfg
            main()

        assert captured["args"].interval == 120
        assert captured["args"].dry_run is True

    def test_daemon_runs_cycle_then_exits_on_shutdown(self) -> None:
        """Loop executes one full cycle and exits when shutdown_event is set.

        shutdown_event is set inside cmd_run's side effect. cmd_review_backlog still
        runs (the event check happens after both commands), then shutdown_event.wait()
        returns immediately because the event is already set.
        """
        real_event = threading.Event()

        def run_side_effect(args: Any, config: Any) -> None:
            real_event.set()  # trigger shutdown after first cmd_run call

        with (
            patch("tentacle.cli.threading.Event", return_value=real_event),
            patch("tentacle.cli.signal.signal"),
            patch("tentacle.cli.cmd_run", side_effect=run_side_effect) as mock_run,
            patch("tentacle.cli.cmd_review_backlog") as mock_backlog,
        ):
            config = Config()
            config.daemon_interval = 1
            args = Namespace(dry_run=False, interval=None)
            cmd_daemon(args, config)

        # cmd_daemon builds sub_args = Namespace(dry_run=args.dry_run) before the loop
        expected_sub_args = Namespace(dry_run=False, days_back=None)
        mock_run.assert_called_once_with(expected_sub_args, config)
        mock_backlog.assert_called_once_with(expected_sub_args, config)

    def test_daemon_uses_config_interval_by_default(self) -> None:
        """shutdown_event.wait() is called with config.daemon_interval when --interval is absent."""
        mock_event = MagicMock()
        mock_event.wait.return_value = True  # exit after first wait

        with (
            patch("tentacle.cli.threading.Event", return_value=mock_event),
            patch("tentacle.cli.signal.signal"),
            patch("tentacle.cli.cmd_run"),
            patch("tentacle.cli.cmd_review_backlog"),
        ):
            config = Config()
            config.daemon_interval = 7200
            args = Namespace(dry_run=False, interval=None)
            cmd_daemon(args, config)

        mock_event.wait.assert_called_with(7200)

    def test_daemon_interval_flag_overrides_config(self) -> None:
        """--interval overrides config.daemon_interval for the wait call."""
        mock_event = MagicMock()
        mock_event.wait.return_value = True

        with (
            patch("tentacle.cli.threading.Event", return_value=mock_event),
            patch("tentacle.cli.signal.signal"),
            patch("tentacle.cli.cmd_run"),
            patch("tentacle.cli.cmd_review_backlog"),
        ):
            config = Config()
            config.daemon_interval = 7200
            args = Namespace(dry_run=False, interval=120)
            cmd_daemon(args, config)

        mock_event.wait.assert_called_with(120)

    def test_daemon_catches_systemexit_from_cmd_run(self) -> None:
        """SystemExit from cmd_run is caught; loop continues to cmd_review_backlog."""
        call_count = 0

        def run_side_effect(args: Any, config: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise SystemExit(1)

        mock_event = MagicMock()
        # First wait returns False (don't exit), second returns True (exit)
        mock_event.wait.side_effect = [False, True]

        with (
            patch("tentacle.cli.threading.Event", return_value=mock_event),
            patch("tentacle.cli.signal.signal"),
            patch("tentacle.cli.cmd_run", side_effect=run_side_effect),
            patch("tentacle.cli.cmd_review_backlog") as mock_backlog,
        ):
            config = Config()
            config.daemon_interval = 1
            args = Namespace(dry_run=False, interval=None)
            cmd_daemon(args, config)

        # cmd_review_backlog should still have been called despite cmd_run raising SystemExit
        assert mock_backlog.call_count == 2
        assert call_count == 2

    def test_daemon_catches_exception_from_cmd_run(self) -> None:
        """RuntimeError from cmd_run is caught; loop continues to next cycle."""
        call_count = 0

        def run_side_effect(args: Any, config: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("network failure")

        mock_event = MagicMock()
        mock_event.wait.side_effect = [False, True]

        with (
            patch("tentacle.cli.threading.Event", return_value=mock_event),
            patch("tentacle.cli.signal.signal"),
            patch("tentacle.cli.cmd_run", side_effect=run_side_effect),
            patch("tentacle.cli.cmd_review_backlog") as mock_backlog,
        ):
            config = Config()
            config.daemon_interval = 1
            args = Namespace(dry_run=False, interval=None)
            cmd_daemon(args, config)

        assert mock_backlog.call_count == 2
        assert call_count == 2

    def test_daemon_signal_handler_sets_event(self) -> None:
        """The registered signal handler sets the shutdown event when called directly."""
        real_event = threading.Event()
        captured_handlers: dict[int, Any] = {}

        def fake_signal(signum: int, handler: Any) -> Any:
            # Only capture the initial registration (callable handler), not the
            # restoration calls in the finally block (which pass back the old handler,
            # None in this mock context).
            if callable(handler):
                captured_handlers[signum] = handler
            return None

        def stop_after_first(args: Any, config: Any) -> None:
            real_event.set()

        with (
            patch("tentacle.cli.threading.Event", return_value=real_event),
            patch("tentacle.cli.signal.signal", side_effect=fake_signal),
            patch("tentacle.cli.cmd_run", side_effect=stop_after_first),
            patch("tentacle.cli.cmd_review_backlog"),
        ):
            config = Config()
            config.daemon_interval = 1
            args = Namespace(dry_run=False, interval=None)
            cmd_daemon(args, config)

        # Handler should have been registered for SIGINT and SIGTERM
        assert sig_module.SIGINT in captured_handlers
        assert sig_module.SIGTERM in captured_handlers

        # Test the handler directly: clear the event, call the handler, verify it's set
        real_event.clear()
        captured_handlers[sig_module.SIGINT](sig_module.SIGINT, None)
        assert real_event.is_set()


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


class TestCmdHealth(unittest.TestCase):
    def _make_scan_run(self, status: str = "ok", source: str = "arxiv") -> ScanRun:
        return ScanRun(
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
            source=source,
            status=status,
        )

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_all_ok(self, mock_subproc: MagicMock, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = [self._make_scan_run("ok")]
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 0
        output = mock_out.getvalue()
        assert "api_key: ok" in output
        assert "db: ok" in output
        assert "gh_auth: ok" in output

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_missing_api_key(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = ""
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        assert "api_key: FAIL" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_db_inaccessible(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_get_store.side_effect = OSError("permission denied")
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        assert "db: FAIL" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_gh_not_authenticated(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_subproc.return_value = MagicMock(returncode=1)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        assert "gh_auth: FAIL" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_gh_not_installed(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_subproc.side_effect = FileNotFoundError("gh not found")

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        assert "gh_auth: FAIL" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_last_scan_error(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = [self._make_scan_run("error")]
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 0
        assert "warning" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_no_previous_scans(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 0
        assert "no scans yet" in mock_out.getvalue()

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_gh_timeout(self, mock_subproc: MagicMock, mock_get_store: MagicMock) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_store.get_recent_scan_runs.return_value = []
        mock_subproc.side_effect = subprocess.TimeoutExpired(
            cmd=["gh", "auth", "status"], timeout=10
        )

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        output = mock_out.getvalue()
        assert "gh_auth: FAIL" in output
        assert "timed out" in output

    @patch("tentacle.cli._get_store")
    @patch("tentacle.cli.subprocess.run")
    def test_health_db_fail_skips_last_scan(
        self, mock_subproc: MagicMock, mock_get_store: MagicMock
    ) -> None:
        """When DB open fails, last_scan check is skipped rather than retrying the store."""
        mock_get_store.side_effect = OSError("permission denied")
        mock_subproc.return_value = MagicMock(returncode=0)

        config = Config()
        config.anthropic_api_key = "sk-ant-test"
        args = Namespace()

        with (
            patch("sys.stdout", new_callable=io.StringIO) as mock_out,
            self.assertRaises(SystemExit) as ctx,
        ):
            cmd_health(args, config)

        assert ctx.exception.code == 1
        output = mock_out.getvalue()
        assert "db: FAIL" in output
        # Store was opened exactly once (not twice)
        mock_get_store.assert_called_once()
        assert "last_scan: skipped" in output


class TestMainHealthConfigFallback(unittest.TestCase):
    def test_config_error_fallback_uses_env_api_key(self) -> None:
        """ConfigError in main() health path falls back to Config() and warns."""
        with (
            patch("sys.argv", ["tentacle", "health"]),
            patch("tentacle.cli.load_config", side_effect=ConfigError("no config")),
            patch("tentacle.cli.cmd_health") as mock_health,
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}),
        ):
            main()

        called_config = mock_health.call_args[0][1]
        assert called_config.anthropic_api_key == "sk-ant-from-env"

    def test_config_error_fallback_no_env_key(self) -> None:
        """ConfigError fallback with no env var leaves api key empty."""
        env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with (
            patch("sys.argv", ["tentacle", "health"]),
            patch("tentacle.cli.load_config", side_effect=ConfigError("no config")),
            patch("tentacle.cli.cmd_health") as mock_health,
            patch.dict(os.environ, env_without_key, clear=True),
        ):
            main()

        called_config = mock_health.call_args[0][1]
        assert called_config.anthropic_api_key == ""


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
