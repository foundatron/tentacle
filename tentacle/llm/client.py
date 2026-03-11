"""Anthropic SDK wrapper with cost tracking."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

# Pricing per million tokens (as of 2025)
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_M, output_per_M)
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250514": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
}

_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.1


class BudgetExceededError(Exception):
    """Raised when a cost budget limit has been reached.

    The scan budget check fires both before *and* after each API call.  The
    pre-call check prevents calls when the budget is already exhausted; the
    post-call check catches the case where a single expensive call overshoots
    the limit.  Both are best-effort soft ceilings — one call may exceed the
    budget before the post-call guard triggers.
    """

    def __init__(self, current_cost: float, limit: float, budget_type: str) -> None:
        self.current_cost = current_cost
        self.limit = limit
        self.budget_type = budget_type
        super().__init__(
            f"{budget_type} budget exceeded: current=${current_cost:.4f}, limit=${limit:.4f}"
        )


@dataclass
class UsageRecord:
    """Token usage and cost for a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """Tracks cumulative costs across API calls."""

    records: list[UsageRecord] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    input_price, output_price = _PRICING.get(model, (3.0, 15.0))

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    cache_write_cost = (cache_creation_tokens / 1_000_000) * input_price * _CACHE_WRITE_MULTIPLIER
    cache_read_cost = (cache_read_tokens / 1_000_000) * input_price * _CACHE_READ_MULTIPLIER

    return input_cost + output_cost + cache_write_cost + cache_read_cost


class LLMClient:
    """Thin wrapper around the Anthropic SDK with cost tracking."""

    def __init__(
        self,
        api_key: str,
        cost_tracker: CostTracker | None = None,
        *,
        scan_budget: float = 0.0,
        monthly_budget: float = 0.0,
        get_monthly_cost: Callable[[], float] | None = None,
    ) -> None:
        if scan_budget < 0:
            logger.warning("scan_budget=%s is negative; treating as no limit", scan_budget)
        if monthly_budget < 0:
            logger.warning("monthly_budget=%s is negative; treating as no limit", monthly_budget)
        self._client = anthropic.Anthropic(api_key=api_key)
        self.costs = cost_tracker or CostTracker()
        self._scan_budget = scan_budget
        self._monthly_budget = monthly_budget
        self._get_monthly_cost = get_monthly_cost
        # Cached monthly base cost (costs accrued before this scan started).
        # Fetched once on first use to avoid a SQL round-trip per LLM call.
        self._monthly_base_cost: float | None = None

    def _check_budgets(self) -> None:
        """Check scan and monthly budgets before an API call."""
        if self._scan_budget > 0 and self.costs.total_cost >= self._scan_budget:
            raise BudgetExceededError(self.costs.total_cost, self._scan_budget, "scan")

        if self._monthly_budget > 0 and self._get_monthly_cost is not None:
            try:
                if self._monthly_base_cost is None:
                    self._monthly_base_cost = self._get_monthly_cost()
                total_monthly = self._monthly_base_cost + self.costs.total_cost
                if total_monthly >= self._monthly_budget:
                    raise BudgetExceededError(total_monthly, self._monthly_budget, "monthly")
            except BudgetExceededError:
                raise
            except Exception:
                logger.warning(
                    "Failed to check monthly cost; proceeding without monthly budget enforcement"
                )

    def _check_scan_budget_post_call(self) -> None:
        """Post-call scan budget check: catches overshoot from a single expensive call."""
        if self._scan_budget > 0 and self.costs.total_cost >= self._scan_budget:
            raise BudgetExceededError(self.costs.total_cost, self._scan_budget, "scan")

    def _record_usage(self, model: str, response: anthropic.types.Message) -> UsageRecord:
        """Build a UsageRecord from a response, add it to costs, and log it."""
        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        record = UsageRecord(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            cost_usd=_estimate_cost(
                model, usage.input_tokens, usage.output_tokens, cache_creation, cache_read
            ),
        )
        self.costs.add(record)

        logger.info(
            "LLM call: model=%s in=%d out=%d cache_w=%d cache_r=%d cost=$%.4f",
            model,
            usage.input_tokens,
            usage.output_tokens,
            cache_creation,
            cache_read,
            record.cost_usd,
        )
        return record

    def complete(
        self,
        *,
        model: str,
        system: str | list[anthropic.types.TextBlockParam],
        messages: list[anthropic.types.MessageParam],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Send a completion request and return the text response."""
        self._check_budgets()

        response = self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract text
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        self._record_usage(model, response)
        self._check_scan_budget_post_call()

        return text

    def complete_with_tools(
        self,
        *,
        model: str,
        system: str | list[anthropic.types.TextBlockParam],
        messages: list[anthropic.types.MessageParam],
        tools: list[anthropic.types.ToolParam],
        tool_choice: anthropic.types.ToolChoiceToolParam,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Send a completion request forcing a tool call and return the tool input dict."""
        self._check_budgets()

        response = self._client.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._record_usage(model, response)
        self._check_scan_budget_post_call()

        for block in response.content:
            if block.type == "tool_use":
                if not isinstance(block.input, dict):
                    raise TypeError(
                        f"Expected dict from tool_use block.input, got {type(block.input)}"
                    )
                return block.input

        raise ValueError("No tool_use block in response")
