"""CLI entry point: run, review-backlog, status subcommands."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import types
from datetime import UTC, datetime
from pathlib import Path

from tentacle.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONFIG_TEMPLATE,
    Config,
    ConfigError,
    load_config,
)
from tentacle.context import fetch_context
from tentacle.db import Store
from tentacle.decay import apply_decay
from tentacle.issues import create_issue
from tentacle.llm.analyze import analyze_article
from tentacle.llm.client import BudgetExceededError, CostTracker, LLMClient
from tentacle.llm.filter import filter_batch
from tentacle.sources.arxiv import ArxivAdapter
from tentacle.sources.base import SourceAdapter
from tentacle.sources.hackernews import HackerNewsAdapter
from tentacle.sources.rss import RSSAdapter
from tentacle.sources.semantic_scholar import SemanticScholarAdapter

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def _get_store(config: Config) -> Store:
    db_path = Path(config.db_path).expanduser()
    return Store(str(db_path))


def _get_sources(
    config: Config,
    *,
    days_back_override: int | None = None,
) -> list[tuple[str, list[str], int, SourceAdapter]]:
    """Return enabled sources as (name, queries, max_results, adapter) tuples.

    When *days_back_override* is set it takes precedence over per-source
    config values, letting the CLI do a one-off wider scan.
    """
    sources: list[tuple[str, list[str], int, SourceAdapter]] = []
    if config.arxiv.enabled:
        sources.append(
            (
                "arxiv",
                config.arxiv.queries,
                config.arxiv.max_results,
                ArxivAdapter(
                    days_back=days_back_override or config.arxiv.days_back,
                    sort_order=config.arxiv.sort_order,
                ),
            )
        )
    if config.semantic_scholar.enabled:
        sources.append(
            (
                "semantic_scholar",
                config.semantic_scholar.queries,
                config.semantic_scholar.max_results,
                SemanticScholarAdapter(
                    api_key=config.semantic_scholar.s2_api_key,
                    min_citations=config.semantic_scholar.min_citations,
                    days_back=days_back_override or config.semantic_scholar.days_back,
                ),
            )
        )
    if config.hackernews.enabled:
        sources.append(
            (
                "hn",
                config.hackernews.queries,
                config.hackernews.max_results,
                HackerNewsAdapter(
                    min_points=config.hackernews.min_points,
                    days_back=days_back_override or config.hackernews.days_back,
                    story_type=config.hackernews.story_type,
                ),
            )
        )
    if config.rss.enabled and config.rss.queries:
        sources.append(
            (
                "rss",
                config.rss.queries,
                config.rss.max_results,
                RSSAdapter(extract_content=config.rss.extract_content),
            )
        )
    return sources


def cmd_init(args: argparse.Namespace, config_path: Path | None = None) -> None:
    """Write a default config file to ~/.config/tentacle/config.toml."""
    resolved_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    if resolved_path.exists():
        print(f"Config already exists at {resolved_path}. Aborting.")
        return

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(DEFAULT_CONFIG_TEMPLATE)
    print(f"Config written to {resolved_path}")

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Reminder: set ANTHROPIC_API_KEY in your environment or add it to the config file.")


def cmd_run(args: argparse.Namespace, config: Config) -> None:
    """Run the full scan pipeline."""
    store = _get_store(config)
    cost_tracker = CostTracker()
    now = datetime.now(UTC)

    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = LLMClient(
        config.anthropic_api_key,
        cost_tracker,
        scan_budget=config.scan_budget,
        monthly_budget=config.monthly_budget,
        get_monthly_cost=lambda: store.get_monthly_cost(now.year, now.month)["total_cost"],
    )
    context_result = fetch_context(store=store)
    if context_result.changed_files:
        logger.info("Context files changed since last scan: %s", context_result.changed_files)
    sources = _get_sources(
        config,
        days_back_override=getattr(args, "days_back", None),
    )

    if not sources:
        logger.warning("No sources enabled")
        return

    total_new = 0
    total_relevant = 0
    total_issues = 0

    for source_name, queries, max_results, adapter in sources:
        if not queries:
            logger.info("Skipping %s: no queries configured", source_name)
            continue

        cost_before = cost_tracker.total_cost
        run_id: int | None = None
        if not args.dry_run:
            run_id = store.start_scan_run(source_name)
        logger.info("Scanning %s...", source_name)

        try:
            articles = adapter.fetch(queries, max_results)
            articles_found = len(articles)

            # Dedup: remove duplicates within the batch (same article
            # returned by multiple queries), then filter against the DB.
            seen_ids: set[str] = set()
            unique_articles = []
            for a in articles:
                if a.id not in seen_ids:
                    seen_ids.add(a.id)
                    unique_articles.append(a)
            new_articles = [a for a in unique_articles if not store.article_exists(a.id)]
            if not args.dry_run:
                for a in new_articles:
                    store.insert_article(a)
            else:
                logger.info("DRY RUN: would insert %d new articles", len(new_articles))
            articles_new = len(new_articles)
            total_new += articles_new

            # Filter
            filter_results = filter_batch(
                client,
                new_articles,
                model=config.filter_model,
                threshold=config.relevance_threshold,
            )
            relevant_articles = [
                (article, score, reasoning)
                for article, (score, reasoning) in zip(new_articles, filter_results, strict=True)
                if score >= config.relevance_threshold
            ]

            articles_relevant = len(relevant_articles)
            total_relevant += articles_relevant

            # Analyze
            issues_created = 0
            for article, rel_score, rel_reasoning in relevant_articles:
                analysis = analyze_article(
                    client,
                    article,
                    context_result.context,
                    model=config.analyze_model,
                    relevance_score=rel_score,
                    relevance_reasoning=rel_reasoning,
                )
                if analysis is None:
                    continue

                if not args.dry_run:
                    analysis_id = store.insert_analysis(analysis)
                    analysis.id = analysis_id
                else:
                    logger.info(
                        "DRY RUN: would insert analysis for '%s' (maturity=%d)",
                        article.title[:60],
                        analysis.maturity_score,
                    )

                if (
                    analysis.maturity_score >= config.min_maturity_for_issue
                    and total_issues < config.max_issues_per_cycle
                ):
                    # URL-based dedup: skip without consuming a budget slot so
                    # more net-new issues can be created within the cycle limit.
                    if store.get_issues_by_source_url(article.url):
                        logger.info("Skipping — already has issue for this URL: %s", article.url)
                        continue
                    issue = create_issue(
                        article,
                        analysis,
                        repo=config.target_repo,
                        label=config.issue_label,
                        dry_run=args.dry_run,
                    )
                    if issue:
                        if not args.dry_run:
                            store.insert_issue(issue)
                        issues_created += 1
                        total_issues += 1
                        if total_issues < config.max_issues_per_cycle:
                            delay = config.issue_creation_delay
                            logger.info(
                                "Waiting %ds before next issue creation to avoid rate limits",
                                delay,
                            )
                            time.sleep(delay)

            source_cost = cost_tracker.total_cost - cost_before
            if not args.dry_run and run_id is not None:
                store.finish_scan_run(
                    run_id,
                    articles_found=articles_found,
                    articles_new=articles_new,
                    articles_relevant=articles_relevant,
                    issues_created=issues_created,
                    total_cost_usd=source_cost,
                )
            else:
                logger.info(
                    "DRY RUN: scan %s complete "
                    "(found=%d, new=%d, relevant=%d, issues=%d, cost=$%.4f)",
                    source_name,
                    articles_found,
                    articles_new,
                    articles_relevant,
                    issues_created,
                    source_cost,
                )

        except BudgetExceededError as e:
            logger.warning("Budget limit reached during %s: %s", source_name, e)
            if not args.dry_run and run_id is not None:
                store.finish_scan_run(run_id, status="budget_exceeded")
            break

        except Exception:
            logger.exception("Error scanning %s", source_name)
            if not args.dry_run and run_id is not None:
                store.finish_scan_run(run_id, status="error")

    logger.info(
        "Scan complete: %d new articles, %d relevant, %d issues created, $%.4f total cost",
        total_new,
        total_relevant,
        total_issues,
        cost_tracker.total_cost,
    )
    store.close()


def cmd_review_backlog(args: argparse.Namespace, config: Config) -> None:
    """Review and decay open issues."""
    store = _get_store(config)
    now = datetime.now(UTC)

    llm_client: LLMClient | None = None
    context = ""
    if config.anthropic_api_key:
        cost_tracker = CostTracker()
        # scan_budget is reused here intentionally: scan and backlog review share the
        # same per-operation budget cap. Operators running both in the same window
        # should be aware they draw from the same monthly_budget pool.
        llm_client = LLMClient(
            config.anthropic_api_key,
            cost_tracker,
            scan_budget=config.scan_budget,
            monthly_budget=config.monthly_budget,
            get_monthly_cost=lambda: store.get_monthly_cost(now.year, now.month)["total_cost"],
        )
        context_result = fetch_context(store=store)
        context = context_result.context
    else:
        logger.info("ANTHROPIC_API_KEY not set; running mechanical decay only")

    decayed = apply_decay(
        store,
        grace_days=config.decay_grace_days,
        interval_days=config.decay_interval_days,
        llm_client=llm_client,
        context=context,
        repo=config.target_repo,
        model=config.decay_model,
        dry_run=args.dry_run,
    )

    logger.info("Backlog review: %d issues decayed", decayed)
    store.close()


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    """Show current status."""
    store = _get_store(config)
    now = datetime.now(UTC)

    # Catalog summary
    summary = store.get_status_summary()
    print(f"Articles: {summary['total_articles']}  Analyses: {summary['total_analyses']}")
    tiers = summary["relevance_tiers"]
    print(f"Relevance: high={tiers['high']} medium={tiers['medium']} low={tiers['low']}")
    print(f"Issues: open={summary['open_issues']} closed={summary['closed_issues']}")

    # Monthly costs (last 3 months from summary)
    if summary["monthly_costs"]:
        print("\nMonthly costs (last 3 months):")
        for month_data in summary["monthly_costs"]:
            print(f"  {month_data['month']}  ${month_data['cost']:.4f}")

    # Current month cost + budget
    monthly = store.get_monthly_cost(now.year, now.month)
    print(f"\nMonthly cost ({now.strftime('%Y-%m')}):")
    print(f"  Total:  ${monthly['total_cost']:.4f}")
    print(f"  Scans:  {monthly['scan_count']} (avg ${monthly['avg_cost_per_scan']:.4f}/scan)")
    if config.monthly_budget > 0:
        remaining = config.monthly_budget - monthly["total_cost"]
        print(f"  Budget: ${config.monthly_budget:.2f} (${remaining:.4f} remaining)")

    # Recent scan runs
    runs = store.get_recent_scan_runs(5)
    print("\nRecent scans:")
    for run in runs:
        finished = run.finished_at.strftime("%Y-%m-%d %H:%M") if run.finished_at else "running"
        print(
            f"  {run.source:20s} {finished:16s} "
            f"found={run.articles_found} new={run.articles_new} "
            f"relevant={run.articles_relevant} issues={run.issues_created} "
            f"cost=${run.total_cost_usd:.4f} [{run.status}]"
        )

    # Open issues
    issues = store.get_open_issues()
    print(f"\nOpen issues: {len(issues)}")
    for issue in issues:
        age = (datetime.now(UTC) - issue.created_at).days
        print(
            f"  #{issue.github_number:4d} maturity={issue.current_maturity}/5 "
            f"age={age}d {issue.title[:60]}"
        )

    store.close()


def cmd_daemon(args: argparse.Namespace, config: Config) -> None:
    """Run scan pipeline and backlog review on a recurring interval."""
    shutdown_event = threading.Event()

    def _handle_signal(signum: int, frame: types.FrameType | None) -> None:
        logger.info("Received signal %d; shutting down after current cycle", signum)
        shutdown_event.set()

    old_sigint = signal.signal(signal.SIGINT, _handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    interval = args.interval if args.interval is not None else config.daemon_interval
    logger.info(
        "Starting daemon (interval=%ds). Signals set event; current cycle runs to completion.",
        interval,
    )

    # Build a namespace with only the attributes that cmd_run and cmd_review_backlog
    # actually use, so future flags added to their own subparsers don't cause
    # AttributeError when called from the daemon context.
    sub_args = argparse.Namespace(
        dry_run=args.dry_run,
        days_back=getattr(args, "days_back", None),
    )

    try:
        while True:
            try:
                cmd_run(sub_args, config)
            except SystemExit as e:
                logger.warning("cmd_run exited with code %s; continuing to next cycle", e.code)
            except Exception:
                logger.exception("Unexpected error in cmd_run; continuing to next cycle")

            try:
                cmd_review_backlog(sub_args, config)
            except SystemExit as e:
                logger.warning(
                    "cmd_review_backlog exited with code %s; continuing to next cycle", e.code
                )
            except Exception:
                logger.exception("Unexpected error in cmd_review_backlog; continuing to next cycle")

            if shutdown_event.wait(interval):
                break
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    logger.info("Daemon stopped gracefully")


def cmd_health(args: argparse.Namespace, config: Config) -> None:
    """Check infrastructure readiness: API key, DB, gh auth, last scan status."""
    all_ok = True

    # 1. API key
    if config.anthropic_api_key:
        print("api_key: ok")
    else:
        print("api_key: FAIL (anthropic_api_key is not set)")
        all_ok = False

    # 2. DB writable — open once and reuse for step 4
    store: Store | None = None
    try:
        store = _get_store(config)
        print("db: ok")
    except Exception as exc:
        print(f"db: FAIL ({exc})")
        all_ok = False

    # 3. gh auth
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("gh_auth: ok")
        else:
            print("gh_auth: FAIL (gh auth status returned non-zero)")
            all_ok = False
    except FileNotFoundError:
        print("gh_auth: FAIL (gh not found)")
        all_ok = False
    except subprocess.TimeoutExpired:
        print("gh_auth: FAIL (gh auth status timed out)")
        all_ok = False

    # 4. Last scan status (informational only)
    if store is not None:
        try:
            runs = store.get_recent_scan_runs(1)
            if runs:
                run = runs[0]
                if run.status == "error":
                    print(f"last_scan: warning (last run for '{run.source}' had status=error)")
                else:
                    print(f"last_scan: {run.status} (source={run.source})")
            else:
                print("last_scan: no scans yet")
        except Exception as exc:
            logger.warning("Could not retrieve last scan status: %s", exc)
            print(f"last_scan: could not retrieve ({exc})")
        finally:
            store.close()
    else:
        print("last_scan: skipped (db unavailable)")

    sys.exit(0 if all_ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tentacle",
        description="Research scout service for OctopusGarden",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Write a default config file")

    # run
    run_parser = subparsers.add_parser("run", help="Run the full scan pipeline")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't create GitHub issues or write to the database",
    )
    run_parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Override days_back for all sources (e.g. 90 for initial scan)",
    )

    # review-backlog
    backlog_parser = subparsers.add_parser("review-backlog", help="Review and decay open issues")
    backlog_parser.add_argument(
        "--dry-run", action="store_true", help="Don't modify GitHub issues or write to the database"
    )

    # status
    subparsers.add_parser("status", help="Show current status")

    # health
    subparsers.add_parser("health", help="Check infrastructure readiness")

    # daemon
    daemon_parser = subparsers.add_parser(
        "daemon", help="Run scan pipeline on a recurring schedule"
    )
    daemon_parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between scan cycles (overrides config daemon_interval)",
    )
    daemon_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run through to scan and backlog review",
    )
    daemon_parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Override days_back for all sources",
    )

    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.command == "init":
        cmd_init(args)
    else:
        if args.command == "health":
            try:
                config = load_config(args.config)
            except ConfigError as e:
                logger.warning(
                    "Config file not found or invalid (%s); using defaults — "
                    "DB path may not match your actual setup",
                    e,
                )
                config = Config()
                if api_key := os.environ.get("ANTHROPIC_API_KEY"):
                    config.anthropic_api_key = api_key
        else:
            config = load_config(args.config)
        commands = {
            "run": cmd_run,
            "review-backlog": cmd_review_backlog,
            "status": cmd_status,
            "daemon": cmd_daemon,
            "health": cmd_health,
        }
        commands[args.command](args, config)
