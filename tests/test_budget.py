"""Tests for LLMClient budget enforcement."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tentacle.llm.client import BudgetExceededError, LLMClient


def _make_mock_response(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Build a fake anthropic Messages response."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text="result")]
    response.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return response


def _make_client(**kwargs: object) -> tuple[LLMClient, MagicMock]:
    """Create an LLMClient with a mocked Anthropic SDK."""
    with patch("tentacle.llm.client.anthropic.Anthropic") as mock_cls:
        mock_sdk = MagicMock()
        mock_cls.return_value = mock_sdk
        client = LLMClient("test-key", **kwargs)  # type: ignore[arg-type]
    # Patch the internal client directly so subsequent calls work
    client._client = mock_sdk
    return client, mock_sdk


_COMPLETE_KWARGS = {
    "model": "claude-haiku-4-5-20251001",
    "system": "sys",
    "messages": [{"role": "user", "content": "hello"}],
}


class TestBudgetEnforcement(unittest.TestCase):
    def test_scan_budget_exceeded(self) -> None:
        """Second complete() call raises BudgetExceededError when scan_budget is tiny."""
        client, mock_sdk = _make_client(scan_budget=0.001)
        mock_sdk.messages.create.return_value = _make_mock_response(
            input_tokens=10000, output_tokens=5000
        )

        # First call spends money
        client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]

        # Second call should detect budget exceeded
        with self.assertRaises(BudgetExceededError):
            client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]

    def test_monthly_budget_exceeded(self) -> None:
        """complete() raises BudgetExceededError when monthly cost alone >= monthly_budget."""
        # Return a value already at or over the monthly budget so even zero scan cost triggers it
        monthly_cost_fn = MagicMock(return_value=10.01)
        client, mock_sdk = _make_client(
            monthly_budget=10.0,
            monthly_cost_fn=monthly_cost_fn,
        )
        mock_sdk.messages.create.return_value = _make_mock_response()

        with self.assertRaises(BudgetExceededError):
            client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]

    def test_budget_check_failure_graceful(self) -> None:
        """If monthly_cost_fn raises, the call still proceeds (log warning only)."""

        def bad_fn() -> float:
            raise RuntimeError("DB unavailable")

        client, mock_sdk = _make_client(
            monthly_budget=10.0,
            monthly_cost_fn=bad_fn,
        )
        mock_sdk.messages.create.return_value = _make_mock_response()

        # Should not raise; warning is logged and call proceeds
        result = client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]
        assert result == "result"

    def test_no_budget_when_zero(self) -> None:
        """scan_budget=0.0 and monthly_budget=0.0 disable all checks."""
        monthly_cost_fn = MagicMock(return_value=999.0)
        client, mock_sdk = _make_client(
            scan_budget=0.0,
            monthly_budget=0.0,
            monthly_cost_fn=monthly_cost_fn,
        )
        mock_sdk.messages.create.return_value = _make_mock_response(
            input_tokens=10000, output_tokens=5000
        )

        # Both calls must succeed without raising
        client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]
        client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]

        # monthly_cost_fn should never be called when monthly_budget=0.0
        monthly_cost_fn.assert_not_called()

    def test_no_budget_when_omitted(self) -> None:
        """Default constructor params (backward compat) — no budget checks at all."""
        client, mock_sdk = _make_client()
        mock_sdk.messages.create.return_value = _make_mock_response(
            input_tokens=10000, output_tokens=5000
        )

        client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]
        client.complete(**_COMPLETE_KWARGS)  # type: ignore[arg-type]
        # No exception raised


if __name__ == "__main__":
    unittest.main()
