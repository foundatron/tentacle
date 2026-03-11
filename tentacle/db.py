"""SQLite catalog store."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from tentacle.models import Analysis, Article, DecayEntry, Issue, ScanRun

logger = logging.getLogger(__name__)


class Stats(TypedDict):
    """Aggregate counts for the catalog."""

    total_articles: int
    total_analyses: int
    total_issues: int
    open_issues: int
    total_scan_runs: int
    latest_scan_at: str | None


class MonthlyCost(TypedDict):
    """Cost summary for a calendar month."""

    total_cost: float
    scan_count: int
    avg_cost_per_scan: float


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT,
    title           TEXT NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    url             TEXT NOT NULL,
    pdf_url         TEXT,
    published_at    TEXT,
    discovered_at   TEXT NOT NULL,
    tags            TEXT,
    full_text       TEXT,
    access_status   TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      TEXT NOT NULL REFERENCES articles(id) UNIQUE,
    relevance_score REAL NOT NULL,
    relevance_reasoning TEXT,
    key_insights    TEXT,
    applicable_scopes TEXT,
    suggested_type  TEXT,
    suggested_title TEXT,
    suggested_body  TEXT,
    maturity_score  INTEGER NOT NULL,
    maturity_reasoning TEXT,
    model_used      TEXT NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    analyzed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      TEXT NOT NULL REFERENCES articles(id),
    analysis_id     INTEGER NOT NULL REFERENCES analyses(id),
    github_number   INTEGER NOT NULL,
    github_url      TEXT NOT NULL,
    title           TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    maturity_score  INTEGER NOT NULL,
    current_maturity INTEGER NOT NULL,
    last_decay_at   TEXT,
    status          TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS decay_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id        INTEGER NOT NULL REFERENCES issues(id),
    old_maturity    INTEGER NOT NULL,
    new_maturity    INTEGER NOT NULL,
    reason          TEXT NOT NULL,
    decayed_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    source          TEXT NOT NULL,
    articles_found  INTEGER DEFAULT 0,
    articles_new    INTEGER DEFAULT 0,
    articles_relevant INTEGER DEFAULT 0,
    issues_created  INTEGER DEFAULT 0,
    total_cost_usd  REAL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'running'
);
"""


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class Store:
    """SQLite-backed catalog store."""

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # -- Articles --

    def insert_article(self, article: Article) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO articles
            (id, source, source_id, title, authors, abstract, url, pdf_url,
             published_at, discovered_at, tags, full_text, access_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                article.id,
                article.source,
                article.source_id,
                article.title,
                json.dumps(article.authors) if article.authors is not None else None,
                article.abstract,
                article.url,
                article.pdf_url,
                _iso(article.published_at),
                _iso(article.discovered_at),
                json.dumps(article.tags) if article.tags is not None else None,
                article.full_text,
                article.access_status,
            ),
        )
        self._conn.commit()

    def get_article(self, article_id: str) -> Article | None:
        row = self._conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if row is None:
            return None
        return _row_to_article(row)

    def article_exists(self, article_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,)).fetchone()
        return row is not None

    def get_unanalyzed_articles(self) -> list[Article]:
        rows = self._conn.execute(
            """SELECT a.* FROM articles a
            LEFT JOIN analyses an ON a.id = an.article_id
            WHERE an.id IS NULL
            ORDER BY a.discovered_at DESC"""
        ).fetchall()
        return [_row_to_article(r) for r in rows]

    def get_articles_by_source(self, source: str) -> list[Article]:
        rows = self._conn.execute(
            "SELECT * FROM articles WHERE source = ? ORDER BY discovered_at DESC",
            (source,),
        ).fetchall()
        return [_row_to_article(r) for r in rows]

    # -- Analyses --

    def insert_analysis(self, analysis: Analysis) -> int:
        cursor = self._conn.execute(
            """INSERT INTO analyses
            (article_id, relevance_score, relevance_reasoning, key_insights,
             applicable_scopes, suggested_type, suggested_title, suggested_body,
             maturity_score, maturity_reasoning, model_used, input_tokens,
             output_tokens, cost_usd, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis.article_id,
                analysis.relevance_score,
                analysis.relevance_reasoning,
                json.dumps(analysis.key_insights) if analysis.key_insights is not None else None,
                (
                    json.dumps(analysis.applicable_scopes)
                    if analysis.applicable_scopes is not None
                    else None
                ),
                analysis.suggested_type,
                analysis.suggested_title,
                analysis.suggested_body,
                analysis.maturity_score,
                analysis.maturity_reasoning,
                analysis.model_used,
                analysis.input_tokens,
                analysis.output_tokens,
                analysis.cost_usd,
                _iso(analysis.analyzed_at),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_analysis_for_article(self, article_id: str) -> Analysis | None:
        row = self._conn.execute(
            "SELECT * FROM analyses WHERE article_id = ?", (article_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_analysis(row)

    def get_issueable_analyses(self, min_maturity: int) -> list[Analysis]:
        """Get analyses with sufficient maturity that don't have issues yet."""
        rows = self._conn.execute(
            """SELECT an.* FROM analyses an
            LEFT JOIN issues i ON an.id = i.analysis_id
            WHERE an.maturity_score >= ? AND i.id IS NULL
            ORDER BY an.relevance_score DESC""",
            (min_maturity,),
        ).fetchall()
        return [_row_to_analysis(r) for r in rows]

    # -- Issues --

    def insert_issue(self, issue: Issue) -> int:
        cursor = self._conn.execute(
            """INSERT INTO issues
            (article_id, analysis_id, github_number, github_url, title,
             created_at, maturity_score, current_maturity, last_decay_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                issue.article_id,
                issue.analysis_id,
                issue.github_number,
                issue.github_url,
                issue.title,
                _iso(issue.created_at),
                issue.maturity_score,
                issue.current_maturity,
                _iso(issue.last_decay_at),
                issue.status,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_issues_by_source_url(self, url: str) -> list[Issue]:
        """Return all issues whose source article matches the given URL."""
        rows = self._conn.execute(
            """SELECT i.id, i.article_id, i.analysis_id, i.github_number, i.github_url,
                      i.title, i.created_at, i.maturity_score, i.current_maturity,
                      i.last_decay_at, i.status
               FROM issues i
               JOIN articles a ON i.article_id = a.id
               WHERE a.url = ?""",
            (url,),
        ).fetchall()
        return [_row_to_issue(r) for r in rows]

    def get_open_issues(self) -> list[Issue]:
        rows = self._conn.execute(
            "SELECT * FROM issues WHERE status = 'open' ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_issue(r) for r in rows]

    def update_issue_maturity(self, issue_id: int, new_maturity: int) -> None:
        now = datetime.now(UTC)
        self._conn.execute(
            "UPDATE issues SET current_maturity = ?, last_decay_at = ? WHERE id = ?",
            (new_maturity, _iso(now), issue_id),
        )
        self._conn.commit()

    def update_issue_status(self, issue_id: int, status: str) -> None:
        self._conn.execute("UPDATE issues SET status = ? WHERE id = ?", (status, issue_id))
        self._conn.commit()

    # -- Decay log --

    def insert_decay(self, entry: DecayEntry) -> None:
        self._conn.execute(
            """INSERT INTO decay_log
            (issue_id, old_maturity, new_maturity, reason, decayed_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                entry.issue_id,
                entry.old_maturity,
                entry.new_maturity,
                entry.reason,
                _iso(entry.decayed_at),
            ),
        )
        self._conn.commit()

    def get_decay_log_for_issue(self, issue_id: int) -> list[DecayEntry]:
        rows = self._conn.execute(
            "SELECT * FROM decay_log WHERE issue_id = ? ORDER BY decayed_at DESC",
            (issue_id,),
        ).fetchall()
        return [_row_to_decay(r) for r in rows]

    # -- Scan runs --

    def start_scan_run(self, source: str) -> int:
        now = datetime.now(UTC)
        cursor = self._conn.execute(
            "INSERT INTO scan_runs (started_at, source) VALUES (?, ?)",
            (_iso(now), source),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def finish_scan_run(
        self,
        run_id: int,
        *,
        articles_found: int = 0,
        articles_new: int = 0,
        articles_relevant: int = 0,
        issues_created: int = 0,
        total_cost_usd: float = 0.0,  # per-run delta cost, NOT cumulative monthly total
        status: str = "complete",
    ) -> None:
        now = datetime.now(UTC)
        self._conn.execute(
            """UPDATE scan_runs SET
            finished_at = ?, articles_found = ?, articles_new = ?,
            articles_relevant = ?, issues_created = ?, total_cost_usd = ?, status = ?
            WHERE id = ?""",
            (
                _iso(now),
                articles_found,
                articles_new,
                articles_relevant,
                issues_created,
                total_cost_usd,
                status,
                run_id,
            ),
        )
        self._conn.commit()

    def get_recent_scan_runs(self, limit: int = 10) -> list[ScanRun]:
        rows = self._conn.execute(
            "SELECT * FROM scan_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_scan_run(r) for r in rows]

    def get_monthly_cost(self, year: int, month: int) -> MonthlyCost:
        """Return cost totals for completed scan_runs started in the given UTC month."""
        start = datetime(year, month, 1, tzinfo=UTC)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end = datetime(year, month + 1, 1, tzinfo=UTC)

        row = self._conn.execute(
            """SELECT COALESCE(SUM(total_cost_usd), 0.0), COUNT(*)
               FROM scan_runs
               WHERE status != 'running'
                 AND started_at >= ?
                 AND started_at < ?""",
            (_iso(start), _iso(end)),
        ).fetchone()

        total_cost: float = row[0]
        scan_count: int = row[1]
        avg_cost = total_cost / scan_count if scan_count > 0 else 0.0

        return MonthlyCost(
            total_cost=total_cost,
            scan_count=scan_count,
            avg_cost_per_scan=avg_cost,
        )

    def get_stats(self) -> Stats:
        row = self._conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM articles),
                (SELECT COUNT(*) FROM analyses),
                (SELECT COUNT(*) FROM issues),
                (SELECT COUNT(*) FROM issues WHERE status = 'open'),
                (SELECT COUNT(*) FROM scan_runs),
                (SELECT started_at FROM scan_runs ORDER BY started_at DESC LIMIT 1)"""
        ).fetchone()
        return Stats(
            total_articles=row[0],
            total_analyses=row[1],
            total_issues=row[2],
            open_issues=row[3],
            total_scan_runs=row[4],
            latest_scan_at=row[5],
        )


# -- Row converters --


def _row_to_article(row: sqlite3.Row) -> Article:
    return Article(
        id=row["id"],
        source=row["source"],
        source_id=row["source_id"],
        title=row["title"],
        authors=json.loads(row["authors"]) if row["authors"] else None,
        abstract=row["abstract"],
        url=row["url"],
        pdf_url=row["pdf_url"],
        published_at=_parse_dt(row["published_at"]),
        discovered_at=_parse_dt(row["discovered_at"]) or datetime.now(UTC),
        tags=json.loads(row["tags"]) if row["tags"] else None,
        full_text=row["full_text"],
        access_status=row["access_status"],
    )


def _row_to_analysis(row: sqlite3.Row) -> Analysis:
    return Analysis(
        id=row["id"],
        article_id=row["article_id"],
        relevance_score=row["relevance_score"],
        relevance_reasoning=row["relevance_reasoning"],
        key_insights=json.loads(row["key_insights"]) if row["key_insights"] else None,
        applicable_scopes=(
            json.loads(row["applicable_scopes"]) if row["applicable_scopes"] else None
        ),
        suggested_type=row["suggested_type"],
        suggested_title=row["suggested_title"],
        suggested_body=row["suggested_body"],
        maturity_score=row["maturity_score"],
        maturity_reasoning=row["maturity_reasoning"],
        model_used=row["model_used"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cost_usd=row["cost_usd"],
        analyzed_at=_parse_dt(row["analyzed_at"]) or datetime.now(UTC),
    )


def _row_to_issue(row: sqlite3.Row) -> Issue:
    return Issue(
        id=row["id"],
        article_id=row["article_id"],
        analysis_id=row["analysis_id"],
        github_number=row["github_number"],
        github_url=row["github_url"],
        title=row["title"],
        created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
        maturity_score=row["maturity_score"],
        current_maturity=row["current_maturity"],
        last_decay_at=_parse_dt(row["last_decay_at"]),
        status=row["status"],
    )


def _row_to_scan_run(row: sqlite3.Row) -> ScanRun:
    return ScanRun(
        id=row["id"],
        started_at=_parse_dt(row["started_at"]) or datetime.now(UTC),
        finished_at=_parse_dt(row["finished_at"]),
        source=row["source"],
        articles_found=row["articles_found"],
        articles_new=row["articles_new"],
        articles_relevant=row["articles_relevant"],
        issues_created=row["issues_created"],
        total_cost_usd=row["total_cost_usd"],
        status=row["status"],
    )


def _row_to_decay(row: sqlite3.Row) -> DecayEntry:
    decayed_at = _parse_dt(row["decayed_at"])
    if decayed_at is None:
        msg = f"decay_log row {row['id']}: decayed_at is NULL or unparseable"
        logger.warning(msg)
        raise ValueError(msg)
    return DecayEntry(
        id=row["id"],
        issue_id=row["issue_id"],
        old_maturity=row["old_maturity"],
        new_maturity=row["new_maturity"],
        reason=row["reason"],
        decayed_at=decayed_at,
    )
