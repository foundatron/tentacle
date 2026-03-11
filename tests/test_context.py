"""Tests for context fetching and caching."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from tentacle.context import ContextResult, _checksum, fetch_context
from tentacle.db import Store
from tentacle.models import ContextEntry

_CLAUDE_CONTENT = "# CLAUDE.md\n\nThis is the project context."
_ARCH_CONTENT = "# Architecture\n\nThis describes the architecture."


def _make_entry(filename: str, content: str) -> ContextEntry:
    return ContextEntry(
        filename=filename,
        content=content,
        checksum=_checksum(content),
        fetched_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


class TestFetchContextNoStore(unittest.TestCase):
    @patch("tentacle.context._read_file")
    def test_fetch_fresh_no_store(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [_CLAUDE_CONTENT, _ARCH_CONTENT]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tentacle.context._CACHE_DIR", new=Path(tmp)),
        ):
            result = fetch_context()

        assert isinstance(result, ContextResult)
        assert "CLAUDE.md" in result.context
        assert _CLAUDE_CONTENT in result.context
        assert "docs/architecture.md" in result.context
        assert _ARCH_CONTENT in result.context
        assert result.changed_files == []

    @patch("tentacle.context._read_file")
    def test_all_sources_fail_no_store_no_cache(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [None, None]

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tentacle.context._CACHE_DIR", new=Path(tmp)),
        ):
            result = fetch_context()

        assert result.context == "(no context available)"
        assert result.changed_files == []


class TestFetchContextWithStore(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @patch("tentacle.context._read_file")
    def test_fetch_caches_to_store(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [_CLAUDE_CONTENT, _ARCH_CONTENT]

        with patch("tentacle.context._CACHE_DIR", new=Path(self.tmp.name)):
            fetch_context(store=self.store)

        cached = self.store.get_context("CLAUDE.md")
        assert cached is not None
        assert cached.content == _CLAUDE_CONTENT
        assert cached.checksum == _checksum(_CLAUDE_CONTENT)

        cached_arch = self.store.get_context("docs/architecture.md")
        assert cached_arch is not None
        assert cached_arch.content == _ARCH_CONTENT

    @patch("tentacle.context._read_file")
    def test_cache_hit_no_change(self, mock_read_file: MagicMock) -> None:
        # Pre-populate store with matching content
        self.store.upsert_context(_make_entry("CLAUDE.md", _CLAUDE_CONTENT))
        self.store.upsert_context(_make_entry("docs/architecture.md", _ARCH_CONTENT))

        mock_read_file.side_effect = [_CLAUDE_CONTENT, _ARCH_CONTENT]

        with patch("tentacle.context._CACHE_DIR", new=Path(self.tmp.name)):
            result = fetch_context(store=self.store)

        assert result.changed_files == []

    @patch("tentacle.context._read_file")
    def test_cache_miss_detects_change(self, mock_read_file: MagicMock) -> None:
        old_content = "# CLAUDE.md\n\nOld content."
        self.store.upsert_context(_make_entry("CLAUDE.md", old_content))
        self.store.upsert_context(_make_entry("docs/architecture.md", _ARCH_CONTENT))

        mock_read_file.side_effect = [_CLAUDE_CONTENT, _ARCH_CONTENT]

        with patch("tentacle.context._CACHE_DIR", new=Path(self.tmp.name)):
            result = fetch_context(store=self.store)

        assert "CLAUDE.md" in result.changed_files
        assert "docs/architecture.md" not in result.changed_files

        # Verify store was updated
        updated = self.store.get_context("CLAUDE.md")
        assert updated is not None
        assert updated.content == _CLAUDE_CONTENT

    @patch("tentacle.context._read_file")
    def test_fallback_to_store_on_fetch_failure(self, mock_read_file: MagicMock) -> None:
        self.store.upsert_context(_make_entry("CLAUDE.md", _CLAUDE_CONTENT))
        self.store.upsert_context(_make_entry("docs/architecture.md", _ARCH_CONTENT))

        mock_read_file.side_effect = [None, None]

        with (
            patch("tentacle.context._CACHE_DIR", new=Path(self.tmp.name)),
            self.assertLogs("tentacle.context", level="WARNING") as log,
        ):
            result = fetch_context(store=self.store)

        assert _CLAUDE_CONTENT in result.context
        assert _ARCH_CONTENT in result.context
        assert any("cache" in msg.lower() for msg in log.output)

    @patch("tentacle.context._read_file")
    def test_fallback_to_filesystem_on_fetch_failure(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [None, None]

        # Write files to filesystem cache
        cache_dir = Path(self.tmp.name)
        (cache_dir / "CLAUDE.md").write_text(_CLAUDE_CONTENT)
        (cache_dir / "docs").mkdir(parents=True)
        (cache_dir / "docs" / "architecture.md").write_text(_ARCH_CONTENT)

        with (
            patch("tentacle.context._CACHE_DIR", new=cache_dir),
            self.assertLogs("tentacle.context", level="WARNING") as log,
        ):
            result = fetch_context()

        assert _CLAUDE_CONTENT in result.context
        assert _ARCH_CONTENT in result.context
        assert any("filesystem" in msg.lower() for msg in log.output)

    @patch("tentacle.context._read_file")
    def test_all_sources_fail_empty_store(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [None, None]

        with patch("tentacle.context._CACHE_DIR", new=Path(self.tmp.name)):
            result = fetch_context(store=self.store)

        assert result.context == "(no context available)"

    @patch("tentacle.context._read_file")
    def test_filesystem_cache_written(self, mock_read_file: MagicMock) -> None:
        mock_read_file.side_effect = [_CLAUDE_CONTENT, _ARCH_CONTENT]

        cache_dir = Path(self.tmp.name)
        with patch("tentacle.context._CACHE_DIR", new=cache_dir):
            fetch_context()

        assert (cache_dir / "CLAUDE.md").read_text() == _CLAUDE_CONTENT
        # Verify subdirectory was created for docs/architecture.md
        assert (cache_dir / "docs" / "architecture.md").exists()
        assert (cache_dir / "docs" / "architecture.md").read_text() == _ARCH_CONTENT


class TestChecksum(unittest.TestCase):
    def test_checksum_deterministic(self) -> None:
        content = "Some content for checksumming."
        result1 = _checksum(content)
        result2 = _checksum(content)
        assert result1 == result2

    def test_checksum_is_sha256(self) -> None:
        content = "hello"
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert _checksum(content) == expected

    def test_different_content_different_checksum(self) -> None:
        assert _checksum("content A") != _checksum("content B")


if __name__ == "__main__":
    unittest.main()
