"""Base class for source adapters."""

from __future__ import annotations

import abc

from tentacle.models import Article


class SourceAdapter(abc.ABC):
    """Abstract base class for article source adapters."""

    @abc.abstractmethod
    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        """Fetch articles matching queries. Returns up to max_results articles."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Source identifier (e.g., 'arxiv', 'hn')."""
