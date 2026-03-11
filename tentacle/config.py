"""TOML configuration loading."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("~/.config/tentacle/config.toml").expanduser()


@dataclasses.dataclass
class SourceConfig:
    """Configuration for a single source."""

    enabled: bool = True
    queries: list[str] = dataclasses.field(default_factory=list)
    max_results: int = 50


@dataclasses.dataclass
class Config:
    """Application configuration."""

    # API keys
    anthropic_api_key: str = ""

    # Target repo for issue creation
    target_repo: str = "foundatron/octopusgarden"
    issue_label: str = "tentacle"

    # Scan settings
    scan_interval: str = "daily"
    backlog_interval: str = "weekly"
    max_issues_per_cycle: int = 3
    issue_creation_delay: int = 60  # seconds to wait between issue creations
    min_maturity_for_issue: int = 3
    relevance_threshold: float = 0.3
    scan_budget: float = 2.0
    monthly_budget: float = 10.0

    # LLM models
    filter_model: str = "claude-haiku-4-5-20251001"
    analyze_model: str = "claude-sonnet-4-5-20250514"

    # Database
    db_path: str = "~/.local/share/tentacle/tentacle.db"

    # Sources
    arxiv: SourceConfig = dataclasses.field(default_factory=SourceConfig)
    semantic_scholar: SourceConfig = dataclasses.field(default_factory=SourceConfig)
    hackernews: SourceConfig = dataclasses.field(default_factory=SourceConfig)
    rss: SourceConfig = dataclasses.field(default_factory=SourceConfig)

    # Decay
    decay_grace_days: int = 30
    decay_interval_days: int = 60


def load_config(path: Path | None = None) -> Config:
    """Load configuration from TOML file, with env var overrides."""
    config = Config()

    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        _apply_toml(config, data)

    # Env var overrides
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        config.anthropic_api_key = api_key

    return config


def _apply_toml(config: Config, data: dict[str, object]) -> None:
    """Apply TOML data to config, handling nested source configs."""
    for key, value in data.items():
        if key == "sources" and isinstance(value, dict):
            for source_name, source_data in value.items():
                if isinstance(source_data, dict) and hasattr(config, source_name):
                    source_config = getattr(config, source_name)
                    for sk, sv in source_data.items():
                        if hasattr(source_config, sk):
                            setattr(source_config, sk, sv)
        elif hasattr(config, key):
            setattr(config, key, value)
