"""Data models for tentacle."""

from __future__ import annotations

import dataclasses
import datetime


@dataclasses.dataclass
class Article:
    """A discovered article/paper."""

    id: str  # SHA-256(canonical_url)[:16]
    source: str  # 'arxiv', 'semantic_scholar', 'hn', 'rss'
    title: str
    url: str
    discovered_at: datetime.datetime
    source_id: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    published_at: datetime.datetime | None = None
    tags: list[str] | None = None
    full_text: str | None = None
    access_status: str = "unknown"
    metadata: dict[str, str | int | float | bool | None] | None = None


@dataclasses.dataclass
class Analysis:
    """LLM analysis of an article."""

    article_id: str
    relevance_score: float  # 0.0-1.0
    maturity_score: int  # 1-5
    model_used: str
    analyzed_at: datetime.datetime
    id: int | None = None
    relevance_reasoning: str | None = None
    key_insights: list[str] | None = None
    applicable_scopes: list[str] | None = None
    suggested_type: str | None = None  # feat, fix, perf, etc.
    suggested_title: str | None = None
    suggested_body: str | None = None
    maturity_reasoning: str | None = None
    confidence_score: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclasses.dataclass
class Issue:
    """A GitHub issue created from an analysis."""

    article_id: str
    analysis_id: int
    github_number: int
    github_url: str
    title: str
    created_at: datetime.datetime
    maturity_score: int
    current_maturity: int
    id: int | None = None
    last_decay_at: datetime.datetime | None = None
    status: str = "open"


@dataclasses.dataclass
class DecayEntry:
    """Audit log entry for maturity decay."""

    issue_id: int
    old_maturity: int
    new_maturity: int
    reason: str
    decayed_at: datetime.datetime
    id: int | None = None


@dataclasses.dataclass
class ScanRun:
    """Record of a scan execution."""

    started_at: datetime.datetime
    source: str
    id: int | None = None
    finished_at: datetime.datetime | None = None
    articles_found: int = 0
    articles_new: int = 0
    articles_relevant: int = 0
    issues_created: int = 0
    total_cost_usd: float = 0.0
    status: str = "running"


@dataclasses.dataclass
class ContextEntry:
    """Cached context file from octopusgarden."""

    filename: str
    content: str
    checksum: str
    fetched_at: datetime.datetime
    id: int | None = None
