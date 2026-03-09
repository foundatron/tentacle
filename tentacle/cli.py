"""CLI entry point: run, review-backlog, status subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from tentacle.config import Config, load_config
from tentacle.context import fetch_context
from tentacle.db import Store
from tentacle.decay import apply_decay
from tentacle.issues import create_issue
from tentacle.llm.analyze import analyze_article
from tentacle.llm.client import CostTracker, LLMClient
from tentacle.llm.filter import filter_article
from tentacle.sources.arxiv import ArxivAdapter
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


def _get_sources(config: Config) -> list[tuple[str, list[str], int, object]]:
    """Return enabled sources as (name, queries, max_results, adapter) tuples."""
    sources = []
    if config.arxiv.enabled:
        sources.append(("arxiv", config.arxiv.queries, config.arxiv.max_results, ArxivAdapter()))
    if config.semantic_scholar.enabled:
        sources.append(
            (
                "semantic_scholar",
                config.semantic_scholar.queries,
                config.semantic_scholar.max_results,
                SemanticScholarAdapter(),
            )
        )
    if config.hackernews.enabled:
        sources.append(
            (
                "hn",
                config.hackernews.queries,
                config.hackernews.max_results,
                HackerNewsAdapter(),
            )
        )
    if config.rss.enabled and config.rss.queries:
        sources.append(("rss", config.rss.queries, config.rss.max_results, RSSAdapter()))
    return sources


def cmd_run(args: argparse.Namespace, config: Config) -> None:
    """Run the full scan pipeline."""
    store = _get_store(config)
    cost_tracker = CostTracker()

    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = LLMClient(config.anthropic_api_key, cost_tracker)
    context = fetch_context()
    sources = _get_sources(config)

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

        run_id = store.start_scan_run(source_name)
        logger.info("Scanning %s...", source_name)

        try:
            articles = adapter.fetch(queries, max_results)
            articles_found = len(articles)

            # Dedup
            new_articles = [a for a in articles if not store.article_exists(a.id)]
            for a in new_articles:
                store.insert_article(a)
            articles_new = len(new_articles)
            total_new += articles_new

            # Filter
            relevant_articles = []
            for article in new_articles:
                score, reasoning = filter_article(
                    client,
                    article,
                    model=config.filter_model,
                    threshold=config.relevance_threshold,
                )
                if score >= config.relevance_threshold:
                    relevant_articles.append((article, score, reasoning))

            articles_relevant = len(relevant_articles)
            total_relevant += articles_relevant

            # Analyze
            issues_created = 0
            for article, rel_score, rel_reasoning in relevant_articles:
                analysis = analyze_article(
                    client,
                    article,
                    context,
                    model=config.analyze_model,
                    relevance_score=rel_score,
                    relevance_reasoning=rel_reasoning,
                )
                if analysis is None:
                    continue

                analysis_id = store.insert_analysis(analysis)
                analysis.id = analysis_id

                if (
                    analysis.maturity_score >= config.min_maturity_for_issue
                    and total_issues < config.max_issues_per_cycle
                ):
                    issue = create_issue(
                        article,
                        analysis,
                        repo=config.target_repo,
                        label=config.issue_label,
                        dry_run=args.dry_run,
                    )
                    if issue:
                        store.insert_issue(issue)
                        issues_created += 1
                        total_issues += 1

            store.finish_scan_run(
                run_id,
                articles_found=articles_found,
                articles_new=articles_new,
                articles_relevant=articles_relevant,
                issues_created=issues_created,
                total_cost_usd=cost_tracker.total_cost,
            )

        except Exception:
            logger.exception("Error scanning %s", source_name)
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

    decayed = apply_decay(
        store,
        grace_days=config.decay_grace_days,
        interval_days=config.decay_interval_days,
    )

    logger.info("Backlog review: %d issues decayed", decayed)
    store.close()


def cmd_status(args: argparse.Namespace, config: Config) -> None:
    """Show current status."""
    store = _get_store(config)

    # Recent scan runs
    runs = store.get_recent_scan_runs(5)
    print("Recent scans:")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tentacle",
        description="Research scout service for OctopusGarden",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run the full scan pipeline")
    run_parser.add_argument("--dry-run", action="store_true", help="Don't create GitHub issues")

    # review-backlog
    subparsers.add_parser("review-backlog", help="Review and decay open issues")

    # status
    subparsers.add_parser("status", help="Show current status")

    args = parser.parse_args()

    _setup_logging(args.verbose)
    config = load_config(args.config)

    commands = {
        "run": cmd_run,
        "review-backlog": cmd_review_backlog,
        "status": cmd_status,
    }
    commands[args.command](args, config)
