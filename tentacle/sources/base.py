"""Base class for source adapters."""

from __future__ import annotations

import abc
import logging
import time
import urllib.error
import urllib.request

from tentacle.models import Article

logger = logging.getLogger(__name__)

# Retry defaults: 2, 4, 8, 16, 32 s  (total ~62 s worst-case)
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_INITIAL_DELAY = 2.0

# HTTP status codes worth retrying.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def fetch_with_backoff(
    req: urllib.request.Request | str,
    *,
    timeout: int = 30,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
    source_name: str = "",
) -> bytes:
    """Fetch a URL with exponential backoff on retryable HTTP errors.

    Retries on 429, 500, 502, 503, 504.  Respects the ``Retry-After`` header
    as a *floor* for the computed backoff delay.

    Raises the last :class:`urllib.error.HTTPError` if retries are exhausted,
    or any non-retryable error immediately.
    """
    label = source_name or "source"

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS_CODES:
                raise
            if attempt == max_retries:
                logger.error("%s: HTTP %d, max retries exhausted", label, exc.code)
                raise
            delay = initial_delay * (2**attempt)
            # Respect Retry-After header as a floor.
            raw_retry = exc.headers.get("Retry-After") if exc.headers else None
            if raw_retry is not None:
                try:
                    delay = max(delay, float(raw_retry))
                except ValueError:
                    pass
            logger.warning(
                "%s: HTTP %d, retrying in %gs (attempt %d/%d)",
                label,
                exc.code,
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)

    raise AssertionError("unreachable: loop always raises or returns")


class SourceAdapter(abc.ABC):
    """Abstract base class for article source adapters."""

    @abc.abstractmethod
    def fetch(self, queries: list[str], max_results: int) -> list[Article]:
        """Fetch articles matching queries. Returns up to max_results articles."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Source identifier (e.g., 'arxiv', 'hn')."""
