"""TOML configuration loading."""

from __future__ import annotations

import dataclasses
import os
import tomllib
import warnings
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("~/.config/tentacle/config.toml").expanduser()

# Default config template written by `tentacle init`. Kept here alongside the
# config logic so it stays in sync with the Config dataclass.
DEFAULT_CONFIG_TEMPLATE = """\
# Tentacle configuration
# Copy to ~/.config/tentacle/config.toml

# API keys (prefer env vars: ANTHROPIC_API_KEY)
# anthropic_api_key = "sk-ant-..."

# Target repo for issue creation
target_repo = "foundatron/octopusgarden"
issue_label = "tentacle"

# Scan settings
scan_interval = "daily"
backlog_interval = "weekly"
max_issues_per_cycle = 3
# Seconds to wait between issue creations to avoid GitHub rate limits. Set to 0 to disable.
issue_creation_delay = 60
min_maturity_for_issue = 3
relevance_threshold = 0.3
# Maximum LLM cost (USD) for a single scan run. 0.0 means no limit.
scan_budget = 2.0
monthly_budget = 10.0

# LLM models
filter_model = "claude-haiku-4-5-20251001"
analyze_model = "claude-sonnet-4-6"

# Database
db_path = "~/.local/share/tentacle/tentacle.db"

# Decay settings
decay_grace_days = 30
decay_interval_days = 60

[sources.arxiv]
enabled = true
queries = [
    "autonomous code generation",
    "LLM software engineering",
    "self-improving AI systems",
    "automated software testing LLM",
]
max_results = 50
# days_back = 30
# sort_order = "descending"

[sources.semantic_scholar]
enabled = true
queries = [
    "autonomous code generation large language models",
    "LLM as judge evaluation",
    "self-improving software systems",
]
max_results = 50

[sources.hackernews]
enabled = true
queries = [
    "autonomous code generation",
    "LLM code review",
    "software dark factory",
    "AI software engineering",
]
max_results = 30

[sources.rss]
enabled = false
queries = []
max_results = 50
"""


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclasses.dataclass
class SourceConfig:
    """Configuration for a single source."""

    enabled: bool = True
    queries: list[str] = dataclasses.field(default_factory=list)
    max_results: int = 50
    # arXiv-specific fields; ignored by other adapters
    days_back: int | None = None
    sort_order: str = "descending"


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
    analyze_model: str = "claude-sonnet-4-6"

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


def _check_type(value: object, expected: type | tuple[type, ...], name: str) -> None:
    """Raise ConfigError if value is not an instance of expected type(s)."""
    if not isinstance(value, expected):
        type_name = (
            " or ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ConfigError(f"{name} must be {type_name}, got {type(value).__name__!r} ({value!r})")


def validate(config: Config) -> None:
    """Validate configuration values, raising ConfigError on invalid input."""
    if not config.anthropic_api_key:
        raise ConfigError(
            "anthropic_api_key must be set (use ANTHROPIC_API_KEY env var or set it in config)"
        )

    # Guard numeric fields against wrong types (e.g. TOML string instead of int/float)
    _check_type(config.relevance_threshold, (int, float), "relevance_threshold")
    _check_type(config.min_maturity_for_issue, int, "min_maturity_for_issue")
    _check_type(config.scan_budget, (int, float), "scan_budget")
    _check_type(config.monthly_budget, (int, float), "monthly_budget")
    _check_type(config.max_issues_per_cycle, int, "max_issues_per_cycle")
    _check_type(config.issue_creation_delay, int, "issue_creation_delay")
    _check_type(config.decay_grace_days, int, "decay_grace_days")
    _check_type(config.decay_interval_days, int, "decay_interval_days")

    if not (0.0 <= config.relevance_threshold <= 1.0):
        raise ConfigError(
            f"relevance_threshold must be in [0.0, 1.0], got {config.relevance_threshold}"
        )
    if not (1 <= config.min_maturity_for_issue <= 5):
        raise ConfigError(
            f"min_maturity_for_issue must be in [1, 5], got {config.min_maturity_for_issue}"
        )
    if config.scan_budget < 0:
        raise ConfigError(f"scan_budget must be >= 0, got {config.scan_budget}")
    if config.monthly_budget < 0:
        raise ConfigError(f"monthly_budget must be >= 0, got {config.monthly_budget}")
    if config.max_issues_per_cycle < 1:
        raise ConfigError(f"max_issues_per_cycle must be >= 1, got {config.max_issues_per_cycle}")
    if config.issue_creation_delay < 0:
        raise ConfigError(f"issue_creation_delay must be >= 0, got {config.issue_creation_delay}")
    if config.decay_grace_days < 0:
        raise ConfigError(f"decay_grace_days must be >= 0, got {config.decay_grace_days}")
    if config.decay_interval_days < 1:
        raise ConfigError(f"decay_interval_days must be >= 1, got {config.decay_interval_days}")
    for source_name in ("arxiv", "semantic_scholar", "hackernews", "rss"):
        source_config: SourceConfig = getattr(config, source_name)
        if source_config.max_results < 1:
            raise ConfigError(
                f"sources.{source_name}.max_results must be >= 1, got {source_config.max_results}"
            )

    if config.arxiv.days_back is not None and config.arxiv.days_back < 1:
        raise ConfigError(f"sources.arxiv.days_back must be >= 1, got {config.arxiv.days_back}")
    if config.arxiv.sort_order not in ("ascending", "descending"):
        raise ConfigError(
            f"sources.arxiv.sort_order must be 'ascending' or 'descending', "
            f"got {config.arxiv.sort_order!r}"
        )


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

    validate(config)
    return config


def _apply_toml(config: Config, data: dict[str, object]) -> None:
    """Apply TOML data to config, handling nested source configs."""
    config_fields = {f.name for f in dataclasses.fields(config)}
    source_fields = {f.name for f in dataclasses.fields(SourceConfig)}

    for key, value in data.items():
        if key == "sources" and isinstance(value, dict):
            for source_name, source_data in value.items():
                if not hasattr(config, source_name) or not isinstance(
                    getattr(config, source_name), SourceConfig
                ):
                    warnings.warn(
                        f"Unknown config source: sources.{source_name}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                if isinstance(source_data, dict):
                    source_config = getattr(config, source_name)
                    for sk, sv in source_data.items():
                        if sk in source_fields:
                            setattr(source_config, sk, sv)
                        else:
                            warnings.warn(
                                f"Unknown config key: sources.{source_name}.{sk}",
                                UserWarning,
                                stacklevel=2,
                            )
        elif key in config_fields:
            setattr(config, key, value)
        else:
            warnings.warn(
                f"Unknown config key: {key}",
                UserWarning,
                stacklevel=2,
            )
