"""Tests for LLMClient budget enforcement."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tentacle.llm.client import BudgetExceededError, LLMClient, UsageRecord


def _make_client(**kwargs: object) -> LLMClient:
    """Create an LLMClient with a mocked Anthropic backend."""
    with patch("tentacle.llm.client.anthropic.Anthropic"):
        return LLMClient("fake-key", **kwargs)  # type: ignore[arg-type]


def _add_cost(client: LLMClient, cost: float) -> None:
    client.costs.add(
        UsageRecord(
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=100,
            cost_usd=cost,
        )
    )


def _mock_api_response(client: LLMClient, text: str = "ok") -> None:
    """Wire a successful API response onto the mocked client."""
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text=text)]
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    resp.usage.cache_creation_input_tokens = 0
    resp.usage.cache_read_input_tokens = 0
    client._client.messages.create.return_value = resp


class TestLLMClientBudget(unittest.TestCase):
    def test_scan_budget_exceeded(self) -> None:
        client = _make_client(scan_budget=1.0)
        _add_cost(client, 1.5)  # already over budget

        with self.assertRaises(BudgetExceededError) as ctx:
            client.complete(
                model="claude-haiku-4-5",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )

        exc = ctx.exception
        assert exc.budget_type == "scan"
        assert exc.limit == 1.0
        assert exc.current_cost == 1.5

    def test_monthly_budget_exceeded(self) -> None:
        get_monthly = MagicMock(return_value=9.5)  # prior spend this month
        client = _make_client(monthly_budget=10.0, get_monthly_cost=get_monthly)
        _add_cost(client, 0.6)  # 9.5 + 0.6 = 10.1 > 10.0

        with self.assertRaises(BudgetExceededError) as ctx:
            client.complete(
                model="claude-haiku-4-5",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )

        exc = ctx.exception
        assert exc.budget_type == "monthly"
        assert exc.limit == 10.0

    def test_budget_check_db_failure_graceful(self) -> None:
        get_monthly = MagicMock(side_effect=RuntimeError("db error"))
        client = _make_client(monthly_budget=10.0, get_monthly_cost=get_monthly)
        _mock_api_response(client)

        # Should NOT raise — DB failure is logged and ignored
        result = client.complete(
            model="claude-haiku-4-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "ok"

    def test_no_budget_enforcement_when_zero(self) -> None:
        client = _make_client(scan_budget=0.0, monthly_budget=0.0)
        _add_cost(client, 9999.0)  # enormous accumulated cost
        _mock_api_response(client)

        # 0.0 means no limit — should proceed without raising
        result = client.complete(
            model="claude-haiku-4-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "ok"

    def test_scan_budget_raises_when_exactly_at_limit(self) -> None:
        # Budget check is >=, so exactly at limit should raise
        client = _make_client(scan_budget=1.0)
        _add_cost(client, 1.0)

        with self.assertRaises(BudgetExceededError):
            client.complete(
                model="claude-haiku-4-5",
                system="sys",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_scan_budget_proceeds_when_under_limit(self) -> None:
        client = _make_client(scan_budget=1.0)
        _add_cost(client, 0.99)
        _mock_api_response(client)

        result = client.complete(
            model="claude-haiku-4-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "ok"

    def test_negative_budget_treated_as_no_limit(self) -> None:
        client = _make_client(scan_budget=-5.0, monthly_budget=-1.0)
        _add_cost(client, 9999.0)
        _mock_api_response(client)

        result = client.complete(
            model="claude-haiku-4-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result == "ok"


if __name__ == "__main__":
    unittest.main()
