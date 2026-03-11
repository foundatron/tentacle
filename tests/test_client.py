"""Tests for LLMClient including complete_with_tools and refactored helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tentacle.llm.client import BudgetExceededError, CostTracker, LLMClient, UsageRecord


def _make_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation
    usage.cache_read_input_tokens = cache_read
    return usage


def _make_text_response(text: str = "hello") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = _make_usage()
    return response


def _make_tool_use_response(input_data: dict[str, object]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_data
    response = MagicMock()
    response.content = [block]
    response.usage = _make_usage(input_tokens=200, output_tokens=100)
    return response


def _make_client(
    scan_budget: float = 0.0,
    monthly_budget: float = 0.0,
    get_monthly_cost: object = None,
    cost_tracker: CostTracker | None = None,
) -> LLMClient:
    with patch("tentacle.llm.client.anthropic.Anthropic"):
        client = LLMClient(
            api_key="test-key",
            cost_tracker=cost_tracker,
            scan_budget=scan_budget,
            monthly_budget=monthly_budget,
            get_monthly_cost=get_monthly_cost,  # type: ignore[arg-type]
        )
    return client


class TestCompleteWithTools(unittest.TestCase):
    def test_complete_with_tools_extracts_tool_input(self) -> None:
        client = _make_client()
        expected = {"maturity_score": 4, "confidence_score": 0.9}
        client._client.messages.create.return_value = _make_tool_use_response(expected)  # type: ignore[attr-defined]

        result = client.complete_with_tools(
            model="claude-sonnet-4-6",
            system="sys",
            messages=[{"role": "user", "content": "analyze this"}],
            tools=[],  # type: ignore[arg-type]
            tool_choice={"type": "tool", "name": "create_analysis"},
        )

        assert result == expected

    def test_complete_with_tools_no_tool_use_block_raises(self) -> None:
        client = _make_client()
        client._client.messages.create.return_value = _make_text_response("oops")  # type: ignore[attr-defined]

        with self.assertRaises(ValueError, msg="No tool_use block in response"):
            client.complete_with_tools(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[{"role": "user", "content": "go"}],
                tools=[],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "create_analysis"},
            )

    def test_complete_with_tools_tracks_usage(self) -> None:
        client = _make_client()
        client._client.messages.create.return_value = _make_tool_use_response({"x": 1})  # type: ignore[attr-defined]

        client.complete_with_tools(
            model="claude-sonnet-4-6",
            system="sys",
            messages=[{"role": "user", "content": "go"}],
            tools=[],  # type: ignore[arg-type]
            tool_choice={"type": "tool", "name": "create_analysis"},
        )

        assert len(client.costs.records) == 1
        assert isinstance(client.costs.records[0], UsageRecord)
        assert client.costs.records[0].input_tokens == 200
        assert client.costs.records[0].output_tokens == 100

    def test_complete_with_tools_scan_budget_check(self) -> None:
        # Pre-call: budget already exhausted
        tracker = CostTracker()
        tracker.add(UsageRecord(model="m", input_tokens=0, output_tokens=0, cost_usd=1.0))
        client = _make_client(scan_budget=0.50, cost_tracker=tracker)

        with self.assertRaises(BudgetExceededError) as ctx:
            client.complete_with_tools(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[],
                tools=[],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "t"},
            )
        assert ctx.exception.budget_type == "scan"

        # Post-call: single call overshoots budget
        client2 = _make_client(scan_budget=0.0001)
        client2._client.messages.create.return_value = _make_tool_use_response({"x": 1})  # type: ignore[attr-defined]

        with self.assertRaises(BudgetExceededError) as ctx2:
            client2.complete_with_tools(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[],
                tools=[],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "t"},
            )
        assert ctx2.exception.budget_type == "scan"

    def test_complete_with_tools_monthly_budget_check(self) -> None:
        get_cost = MagicMock(return_value=9.50)
        client = _make_client(monthly_budget=10.0, get_monthly_cost=get_cost)
        # Add scan costs that push total over monthly limit
        client.costs.add(UsageRecord(model="m", input_tokens=0, output_tokens=0, cost_usd=0.60))

        with self.assertRaises(BudgetExceededError) as ctx:
            client.complete_with_tools(
                model="claude-sonnet-4-6",
                system="sys",
                messages=[],
                tools=[],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "t"},
            )
        assert ctx.exception.budget_type == "monthly"


class TestCompleteAfterRefactor(unittest.TestCase):
    def test_complete_still_works_after_refactor(self) -> None:
        """Smoke test: complete() still returns text after helper extraction."""
        client = _make_client()
        client._client.messages.create.return_value = _make_text_response("result text")  # type: ignore[attr-defined]

        result = client.complete(
            model="claude-sonnet-4-6",
            system="system prompt",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert result == "result text"
        assert len(client.costs.records) == 1


if __name__ == "__main__":
    unittest.main()
