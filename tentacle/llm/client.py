"""Anthropic SDK wrapper with cost tracking."""

from __future__ import annotations

import logging
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

    def __init__(self, api_key: str, cost_tracker: CostTracker | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.costs = cost_tracker or CostTracker()

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

        # Track usage
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

        return text
