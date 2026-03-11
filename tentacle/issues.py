"""GitHub issue creation via `gh` CLI."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import UTC, datetime

from tentacle.models import Analysis, Article, Issue

logger = logging.getLogger(__name__)

_CC_PREFIX_RE = re.compile(r"^\w+(\([^)]*\))?:\s*")


def _strip_cc_prefix(title: str) -> str:
    """Strip a leading conventional commit prefix (e.g. 'feat(llm): ') from a title."""
    return _CC_PREFIX_RE.sub("", title)


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two issue titles, ignoring conventional commit prefixes."""
    tokens_a = set(_strip_cc_prefix(a).lower().split())
    tokens_b = set(_strip_cc_prefix(b).lower().split())
    if not tokens_a and not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


_GH_SEARCH_SPECIAL_RE = re.compile(r'["\':()<>|]|(?:^|\s)(?:AND|OR|NOT)(?:\s|$)')


def _sanitize_search_query(title: str) -> str:
    """Strip characters and operators that could be misinterpreted by GitHub's search syntax."""
    # Remove GitHub search operators and shell-unfriendly characters; keep plain words.
    sanitized = _GH_SEARCH_SPECIAL_RE.sub(" ", title)
    return " ".join(sanitized.split())


def check_duplicate(title: str, *, repo: str) -> str | None:
    """Check GitHub for an open issue with a similar title.

    Returns the URL of a matching issue if similarity > 0.8, else None.
    Fails open: returns None on any subprocess or parse error so the pipeline
    is not blocked by transient GitHub API errors.
    """
    search_query = _sanitize_search_query(title)
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--search",
                search_query,
                "--json",
                "title,number,url",
                "--limit",
                "50",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("check_duplicate: gh issue list timed out")
        return None
    except FileNotFoundError:
        logger.warning("check_duplicate: gh CLI not found")
        return None

    if result.returncode != 0:
        logger.warning("check_duplicate: gh issue list failed: %s", result.stderr[:200])
        return None

    try:
        issues = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning("check_duplicate: could not parse gh output")
        return None

    for issue in issues:
        if _title_similarity(title, issue.get("title", "")) > 0.8:
            logger.info("Duplicate detected: '%s' matches existing issue %s", title, issue["url"])
            return str(issue["url"])

    return None


def create_issue(
    article: Article,
    analysis: Analysis,
    *,
    repo: str,
    label: str,
    dry_run: bool = False,
) -> Issue | None:
    """Create a GitHub issue from an analysis. Returns Issue or None on failure."""
    if not analysis.suggested_title or not analysis.suggested_body:
        logger.warning("Analysis missing title or body for article '%s'", article.title[:60])
        return None

    title = analysis.suggested_title
    body = _format_body(article, analysis)

    if dry_run:
        logger.info("DRY RUN: would create issue '%s'", title)
        return None

    dup_url = check_duplicate(title, repo=repo)
    if dup_url is not None:
        logger.info("Skipping duplicate issue '%s' (existing: %s)", title, dup_url)
        return None

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                title,
                "--body",
                body,
                "--label",
                label,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.exception("Failed to create issue")
        return None

    if result.returncode != 0:
        logger.warning("gh issue create failed: %s", result.stderr[:200])
        return None

    # Parse issue URL and number from output
    issue_url = result.stdout.strip()
    try:
        issue_number = int(issue_url.rsplit("/", 1)[-1])
    except ValueError:
        logger.warning("Could not parse issue number from: %s", issue_url)
        return None

    logger.info("Created issue #%d: %s", issue_number, title)

    analysis_id = analysis.id if analysis.id is not None else 0
    return Issue(
        article_id=article.id,
        analysis_id=analysis_id,
        github_number=issue_number,
        github_url=issue_url,
        title=title,
        created_at=datetime.now(UTC),
        maturity_score=analysis.maturity_score,
        current_maturity=analysis.maturity_score,
    )


def _format_body(article: Article, analysis: Analysis) -> str:
    """Format the issue body with source attribution and maturity tag."""
    body = analysis.suggested_body or ""

    # Ensure source section exists
    if "## Source" not in body:
        authors = ", ".join(article.authors) if article.authors else "Unknown"
        published = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
        body += f"""

## Source
- **Paper/Article:** [{article.title}]({article.url})
- **Authors:** {authors}
- **Published:** {published}
- **Relevance Score:** {analysis.relevance_score:.2f}
- **Maturity:** {analysis.maturity_score}/5"""

    body += f"""

---
*Generated by tentacle. Maturity {analysis.maturity_score}/5.*"""

    return body
