"""Fetch and cache OctopusGarden context for LLM analysis."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tentacle.db import Store
from tentacle.models import ContextEntry

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("~/.cache/tentacle").expanduser()
_CONTEXT_FILES = [
    "CLAUDE.md",
    "docs/architecture.md",
]


@dataclasses.dataclass
class ContextResult:
    """Result of fetching context files."""

    context: str
    changed_files: list[str]


def _checksum(content: str) -> str:
    """Compute SHA-256 checksum of content.

    Uses surrogateescape so content from external sources (gh CLI, file reads)
    with surrogate characters doesn't raise UnicodeEncodeError.
    """
    return hashlib.sha256(content.encode(errors="surrogateescape")).hexdigest()


def fetch_context(repo_path: str | None = None, store: Store | None = None) -> ContextResult:
    """Fetch OctopusGarden context from local repo or GitHub.

    Tries local path first, then falls back to `gh` CLI, then cached content.
    Writes fresh content to filesystem and DB caches for future fallback.
    """
    sections: list[str] = []
    changed_files: list[str] = []

    for filename in _CONTEXT_FILES:
        content = _read_file(filename, repo_path)

        if content is not None:
            # Write to filesystem cache for future fallback.
            # Wrapped in try/except so a failed write (disk full, permissions)
            # doesn't prevent the DB cache from being updated below.
            cache_file = _CACHE_DIR / filename
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache_file.write_text(content)
            except OSError as exc:
                logger.warning("Failed to write filesystem cache for %s: %s", filename, exc)

            # Update DB cache and detect changes
            if store is not None:
                checksum = _checksum(content)
                existing = store.get_context(filename)
                if existing is None:
                    logger.debug("New context file seen: %s", filename)
                    changed_files.append(filename)
                elif existing.checksum != checksum:
                    logger.debug("Context file changed: %s", filename)
                    changed_files.append(filename)
                store.upsert_context(
                    ContextEntry(
                        filename=filename,
                        content=content,
                        checksum=checksum,
                        fetched_at=datetime.now(UTC),
                    )
                )
        else:
            # Fetch failed — try DB cache, then filesystem cache
            logger.warning("Could not fetch %s fresh, trying cache", filename)

            if store is not None:
                cached = store.get_context(filename)
                if cached is not None:
                    content = cached.content
                    age = datetime.now(UTC) - cached.fetched_at
                    logger.warning("Using DB-cached content for %s (age: %s)", filename, age)

            if content is None:
                cache_file = _CACHE_DIR / filename
                if cache_file.exists():
                    content = cache_file.read_text()
                    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
                    age = datetime.now(UTC) - mtime
                    logger.warning(
                        "Using filesystem-cached content for %s (age: %s)", filename, age
                    )

        if content is not None:
            sections.append(f"### {filename}\n\n{content}")

    if not sections:
        logger.warning("No OctopusGarden context files found")
        return ContextResult(context="(no context available)", changed_files=changed_files)

    return ContextResult(
        context="\n\n---\n\n".join(sections),
        changed_files=changed_files,
    )


def _read_file(filename: str, repo_path: str | None) -> str | None:
    """Read a context file from local path or GitHub. Does not use cache."""
    # Try local path first
    if repo_path:
        local_path = Path(repo_path) / filename
        if local_path.exists():
            return local_path.read_text()

    # Try common local paths
    for base in ["~/src/foundatron/octopusgarden", "."]:
        path = Path(base).expanduser() / filename
        if path.exists():
            return path.read_text()

    # Fall back to GitHub via gh CLI
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "repos/foundatron/octopusgarden/contents/" + filename,
                "--jq",
                ".content",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            import base64

            return base64.b64decode(result.stdout.strip()).decode()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None
