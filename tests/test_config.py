"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from tentacle.config import Config, ConfigError, SourceConfig, load_config, validate


class TestValidate(unittest.TestCase):
    def _valid(self) -> Config:
        c = Config()
        c.anthropic_api_key = "dummy"
        return c

    def test_defaults_pass_validation(self) -> None:
        validate(self._valid())

    def test_empty_api_key_raises(self) -> None:
        config = Config()
        config.anthropic_api_key = ""
        with self.assertRaises(ConfigError):
            validate(config)

    def test_source_max_results_zero_raises(self) -> None:
        config = self._valid()
        config.arxiv = SourceConfig(max_results=0)
        with self.assertRaises(ConfigError):
            validate(config)

    def test_range_checks(self) -> None:
        cases: list[tuple[str, object]] = [
            ("relevance_threshold", 1.5),
            ("relevance_threshold", -0.1),
            ("min_maturity_for_issue", 0),
            ("min_maturity_for_issue", 6),
            ("scan_budget", -1.0),
            ("monthly_budget", -1.0),
            ("max_issues_per_cycle", 0),
            ("decay_grace_days", -1),
            ("decay_interval_days", 0),
        ]
        for field, bad_value in cases:
            with self.subTest(field=field, value=bad_value):
                config = self._valid()
                setattr(config, field, bad_value)
                with self.assertRaises(ConfigError):
                    validate(config)

    def test_boundary_values_pass(self) -> None:
        cases: list[tuple[str, object]] = [
            ("relevance_threshold", 0.0),
            ("relevance_threshold", 1.0),
            ("min_maturity_for_issue", 1),
            ("min_maturity_for_issue", 5),
            ("scan_budget", 0.0),
            ("monthly_budget", 0.0),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                config = self._valid()
                setattr(config, field, value)
                validate(config)

    def test_wrong_type_raises_config_error(self) -> None:
        """TOML wrong-type assignments should raise ConfigError, not TypeError."""
        cases: list[tuple[str, object]] = [
            ("max_issues_per_cycle", "three"),
            ("issue_creation_delay", "sixty"),
            ("relevance_threshold", "high"),
            ("scan_budget", "none"),
            ("monthly_budget", "unlimited"),
            ("decay_grace_days", "thirty"),
            ("decay_interval_days", "sixty"),
            ("min_maturity_for_issue", "four"),
        ]
        for field, bad_value in cases:
            with self.subTest(field=field, value=bad_value):
                config = self._valid()
                setattr(config, field, bad_value)
                with self.assertRaises(ConfigError):
                    validate(config)


class TestLoadFromToml(unittest.TestCase):
    def test_load_from_toml(self) -> None:
        toml_content = b"""
anthropic_api_key = "test-key"
max_issues_per_cycle = 5
relevance_threshold = 0.7
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            self.assertEqual(config.anthropic_api_key, "test-key")
            self.assertEqual(config.max_issues_per_cycle, 5)
            self.assertAlmostEqual(config.relevance_threshold, 0.7)
        finally:
            tmp_path.unlink()

    def test_env_var_override(self) -> None:
        toml_content = b'anthropic_api_key = "from-toml"\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}):
                config = load_config(tmp_path)
                self.assertEqual(config.anthropic_api_key, "from-env")
        finally:
            tmp_path.unlink()


class TestWarnings(unittest.TestCase):
    def test_unknown_top_level_key_warns(self) -> None:
        toml_content = b'anthropic_api_key = "key"\ntypo_key = true\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_config(tmp_path)

            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(
                any("typo_key" in m for m in messages),
                f"Expected warning about typo_key, got: {messages}",
            )
        finally:
            tmp_path.unlink()

    def test_unknown_source_key_warns(self) -> None:
        toml_content = b'anthropic_api_key = "key"\n[sources.arxiv]\ntyop = true\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_config(tmp_path)

            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(
                any("tyop" in m for m in messages),
                f"Expected warning about tyop, got: {messages}",
            )
        finally:
            tmp_path.unlink()

    def test_unknown_source_name_warns(self) -> None:
        toml_content = b'anthropic_api_key = "key"\n[sources.reddit]\nenabled = true\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_config(tmp_path)

            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(
                any("reddit" in m for m in messages),
                f"Expected warning about unknown source 'reddit', got: {messages}",
            )
        finally:
            tmp_path.unlink()

    def test_valid_config_no_warnings(self) -> None:
        toml_content = b"""
anthropic_api_key = "key"
max_issues_per_cycle = 3
relevance_threshold = 0.3

[sources.arxiv]
enabled = true
max_results = 50
"""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(toml_content)
            tmp_path = Path(f.name)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_config(tmp_path)

            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(user_warnings, [])
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
