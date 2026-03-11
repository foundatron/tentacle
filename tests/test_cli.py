"""Tests for CLI commands."""

from __future__ import annotations

import argparse
import io
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from tentacle.cli import cmd_status
from tentacle.config import Config
from tentacle.db import Store
from tentacle.models import ScanRun

_FIXED_NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)


def _make_scan_run(
    source: str = "arxiv",
    started_at: datetime | None = None,
    total_cost_usd: float = 0.50,
    status: str = "complete",
) -> ScanRun:
    return ScanRun(
        source=source,
        started_at=started_at or datetime(2026, 3, 10, tzinfo=UTC),
        finished_at=datetime(2026, 3, 10, 1, tzinfo=UTC),
        articles_found=10,
        articles_new=5,
        articles_relevant=2,
        issues_created=1,
        total_cost_usd=total_cost_usd,
        status=status,
    )


class TestCmdStatus(unittest.TestCase):
    def _run_status(self, store: Store, config: Config) -> str:
        """Run cmd_status with a patched store, capture stdout."""
        args = argparse.Namespace()
        captured = io.StringIO()
        with (
            patch("tentacle.cli._get_store", return_value=store),
            patch("tentacle.cli.datetime") as mock_dt,
            patch("sys.stdout", captured),
        ):
            mock_dt.now.return_value = _FIXED_NOW
            cmd_status(args, config)
        return captured.getvalue()

    def test_cmd_status_shows_cost_summary(self) -> None:
        store = Store(":memory:")

        # Insert two runs in the current month (2026-03)
        run1 = store.start_scan_run("arxiv")
        store.finish_scan_run(run1, total_cost_usd=1.00)

        # Manually set started_at to current month
        store._conn.execute(
            "UPDATE scan_runs SET started_at = ? WHERE id = ?",
            ("2026-03-10T00:00:00+00:00", run1),
        )

        run2 = store.start_scan_run("hn")
        store.finish_scan_run(run2, total_cost_usd=0.50)
        store._conn.execute(
            "UPDATE scan_runs SET started_at = ? WHERE id = ?",
            ("2026-03-11T00:00:00+00:00", run2),
        )
        store._conn.commit()

        config = Config(monthly_budget=10.0)
        output = self._run_status(store, config)
        store.close()

        assert "Monthly cost (2026-03)" in output
        assert "remaining=" in output
        assert "$10.0000" in output

    def test_cmd_status_no_budget_limit(self) -> None:
        """When monthly_budget=0.0, no remaining budget line is shown."""
        store = Store(":memory:")
        config = Config(monthly_budget=0.0)
        output = self._run_status(store, config)
        store.close()

        assert "Monthly cost" in output
        assert "remaining=" not in output


if __name__ == "__main__":
    unittest.main()
