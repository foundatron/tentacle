"""Ticket maturity decay logic."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Literal, NamedTuple

from tentacle.db import Store
from tentacle.issues import close_issue, comment_on_issue
from tentacle.llm.client import BudgetExceededError, LLMClient
from tentacle.llm.prompts import DECAY_CHECK_SYSTEM, DECAY_CHECK_USER
from tentacle.models import DecayEntry, Issue

logger = logging.getLogger(__name__)

# Threshold crossings that trigger LLM recheck: (old, new) maturity pairs.
# These are hard-coded for simplicity; move to Config if they need to be operator-configurable.
_DECAY_THRESHOLDS: frozenset[tuple[int, int]] = frozenset({(4, 3), (2, 1)})


class RecheckResult(NamedTuple):
    """Result of an LLM recheck at a maturity threshold crossing."""

    action: Literal["halt", "decay", "accelerate"]
    reasoning: str
    comment: str


def _llm_recheck(
    store: Store,
    issue: Issue,
    llm_client: LLMClient,
    context: str,
    model: str,
) -> RecheckResult:
    """Ask LLM whether to halt/decay/accelerate at a threshold crossing.

    Raises BudgetExceededError if the budget is exhausted.
    Fails open (returns 'decay') on any other error.
    """
    analysis = store.get_analysis_by_id(issue.analysis_id)
    body = (
        analysis.suggested_body
        if analysis is not None and analysis.suggested_body
        else "(no body available)"
    )

    try:
        raw = llm_client.complete(
            model=model,
            system=DECAY_CHECK_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": DECAY_CHECK_USER.format(
                        title=issue.title,
                        created_at=issue.created_at.strftime("%Y-%m-%d"),
                        current_maturity=issue.current_maturity,
                        body=body,
                        context=context,
                    ),
                }
            ],
            max_tokens=512,
        )
    except BudgetExceededError:
        raise
    except Exception:
        logger.warning(
            "LLM recheck: error calling LLM for issue #%d; defaulting to decay",
            issue.github_number,
            exc_info=True,
        )
        return RecheckResult(action="decay", reasoning="llm error", comment="")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "LLM recheck: failed to parse JSON for issue #%d; defaulting to decay",
            issue.github_number,
        )
        return RecheckResult(action="decay", reasoning="parse error", comment="")

    action = data.get("action", "decay")
    if action not in ("halt", "decay", "accelerate"):
        logger.warning(
            "LLM recheck: unknown action %r for issue #%d; defaulting to decay",
            action,
            issue.github_number,
        )
        action = "decay"
        # The LLM comment is preserved but never posted — "decay" fall-through skips commenting.

    return RecheckResult(
        action=action,
        reasoning=str(data.get("reasoning", "")),
        comment=str(data.get("comment", "")),
    )


def apply_decay(
    store: Store,
    *,
    grace_days: int = 30,
    interval_days: int = 60,
    llm_client: LLMClient | None = None,
    context: str = "",
    repo: str = "",
    model: str = "claude-haiku-4-5",
    dry_run: bool = False,
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

        # LLM recheck at threshold crossings (4→3 or 2→1)
        if llm_client is not None and (old_maturity, new_maturity) in _DECAY_THRESHOLDS:
            try:
                result = _llm_recheck(store, issue, llm_client, context, model)
            except BudgetExceededError:
                logger.warning(
                    "Budget exceeded during LLM recheck for issue #%d; "
                    "disabling LLM for remaining issues",
                    issue.github_number,
                )
                # Intentional: setting llm_client to None disables LLM for all
                # subsequent issues in this loop iteration without raising further.
                llm_client = None
                result = RecheckResult(action="decay", reasoning="budget exceeded", comment="")

            if result.action == "halt":
                store.insert_decay(
                    DecayEntry(
                        issue_id=issue.id,
                        old_maturity=old_maturity,
                        new_maturity=old_maturity,
                        reason=f"llm_recheck:halt — {result.reasoning}",
                        decayed_at=now,
                    )
                )
                if result.comment:
                    comment_on_issue(
                        issue.github_number,
                        result.comment,
                        repo=repo,
                        dry_run=dry_run,
                    )
                logger.info(
                    "Halted decay for issue #%d ('%s'): %s",
                    issue.github_number,
                    issue.title[:50],
                    result.reasoning[:80],
                )
                continue

            if result.action == "accelerate":
                new_maturity = 1
                store.update_issue_maturity(issue.id, new_maturity)
                store.insert_decay(
                    DecayEntry(
                        issue_id=issue.id,
                        old_maturity=old_maturity,
                        new_maturity=1,
                        reason=f"llm_recheck:accelerate — {result.reasoning}",
                        decayed_at=now,
                    )
                )
                if not repo:
                    logger.warning(
                        "repo not configured; cannot close issue #%d (maturity set to 1 in DB)",
                        issue.github_number,
                    )
                else:
                    close_comment = result.comment or "Issue closed (relevance decay accelerated)."
                    closed = close_issue(
                        issue.github_number,
                        close_comment,
                        repo=repo,
                        dry_run=dry_run,
                    )
                    if closed:
                        store.update_issue_status(issue.id, "closed")
                    else:
                        logger.warning(
                            "Failed to close issue #%d on GitHub; "
                            "DB maturity set to 1 but status remains open",
                            issue.github_number,
                        )
                logger.info(
                    "Accelerated decay for issue #%d: %d -> 1 ('%s')",
                    issue.github_number,
                    old_maturity,
                    issue.title[:50],
                )
                decayed += 1
                continue

            # action == "decay": fall through to normal decay

        # Normal decay
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

        if new_maturity <= 1:
            if not repo:
                logger.warning(
                    "repo not configured; cannot close issue #%d (maturity set to 1 in DB)",
                    issue.github_number,
                )
            else:
                closed = close_issue(
                    issue.github_number,
                    "Issue closed (maturity reached minimum).",
                    repo=repo,
                    dry_run=dry_run,
                )
                if closed:
                    store.update_issue_status(issue.id, "closed")
                else:
                    logger.warning(
                        "Failed to close issue #%d on GitHub; "
                        "DB maturity set to 1 but status remains open",
                        issue.github_number,
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
