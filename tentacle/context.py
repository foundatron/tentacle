"""Fetch and cache OctopusGarden context for LLM analysis."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("~/.cache/tentacle").expanduser()
_CONTEXT_FILES = [
    "CLAUDE.md",
    "docs/architecture.md",
]


def fetch_context(repo_path: str | None = None) -> str:
    """Fetch OctopusGarden context from local repo or GitHub.

    Tries local path first, then falls back to `gh` CLI.
    """
    sections: list[str] = []

    for filename in _CONTEXT_FILES:
        content = _read_file(filename, repo_path)
        if content:
            sections.append(f"### {filename}\n\n{content}")

    if not sections:
        logger.warning("No OctopusGarden context files found")
        return "(no context available)"

    return "\n\n---\n\n".join(sections)


def _read_file(filename: str, repo_path: str | None) -> str | None:
    """Read a context file from local path or GitHub."""
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

    # Try cache
    cache_file = _CACHE_DIR / filename
    if cache_file.exists():
        return cache_file.read_text()

    logger.warning("Could not fetch %s", filename)
    return None
