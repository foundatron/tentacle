"""SHA-256 fingerprinting for article deduplication."""

from __future__ import annotations

import hashlib


def fingerprint(url: str) -> str:
    """Generate a 16-char hex fingerprint from a canonical URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]
