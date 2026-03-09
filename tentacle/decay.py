"""Ticket maturity decay logic."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from tentacle.db import Store
from tentacle.models import DecayEntry

logger = logging.getLogger(__name__)


def apply_decay(
    store: Store,
    *,
    grace_days: int = 30,
    interval_days: int = 60,
) -> int:
    """Apply time-based maturity decay to open issues.

    Returns the number of issues decayed.
    """
    now = datetime.now(UTC)
    open_issues = store.get_open_issues()
    decayed = 0

    for issue in open_issues:
        if issue.id is None:
            continue

        # Check if within grace period
        age_days = (now - issue.created_at).days
        if age_days <= grace_days:
            continue

        # Check if enough time has passed since last decay
        if issue.last_decay_at:
            since_decay = (now - issue.last_decay_at).days
            if since_decay < interval_days:
                continue

        # Already at minimum
        if issue.current_maturity <= 1:
            continue

        new_maturity = issue.current_maturity - 1
        old_maturity = issue.current_maturity

        store.update_issue_maturity(issue.id, new_maturity)
        store.insert_decay(
            DecayEntry(
                issue_id=issue.id,
                old_maturity=old_maturity,
                new_maturity=new_maturity,
                reason=f"time decay: {age_days} days old, interval {interval_days} days",
                decayed_at=now,
            )
        )

        logger.info(
            "Decayed issue #%d: %d -> %d ('%s')",
            issue.github_number,
            old_maturity,
            new_maturity,
            issue.title[:50],
        )
        decayed += 1

    return decayed
