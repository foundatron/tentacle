# Tentacle

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Research scout for autonomous code generation literature. Tentacle continuously scans arXiv, Semantic Scholar, Hacker News, and RSS feeds for papers, posts, and articles about autonomous code generation and software dark factories. It uses Claude to filter and analyze findings, then creates actionable GitHub issues in the [OctopusGarden](https://github.com/foundatron/octopusgarden) repo.

OctopusGarden is a system for autonomous software development. Tentacle keeps it informed about the state of the art by surfacing relevant research as structured, prioritized issues — so new techniques and ideas get evaluated without manual literature review.

**See example issues:** [octopusgarden issues labeled `tentacle`](https://github.com/foundatron/octopusgarden/issues?q=label:tentacle)

## Pipeline

```
Sources (arXiv, Semantic Scholar, HN, RSS)
  -> Dedup (SHA-256 fingerprint)
  -> Filter (Haiku: title+abstract -> relevance 0-1)
  -> Analyze (Sonnet: full content + octopusgarden context -> maturity 1-5)
  -> Create Issues (gh CLI, max N per cycle)
  -> Backlog Review (time-based decay, status sync)
```

Each stage is independent and cheap to run. The filter stage uses Haiku to keep costs low — only articles that pass the relevance threshold get a full Sonnet analysis.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # install dependencies
cp config.example.toml ~/.config/tentacle/config.toml
# edit config.toml and set ANTHROPIC_API_KEY
```

You'll also need the [GitHub CLI](https://cli.github.com/) (`gh`) authenticated for issue creation.

## Usage

```bash
make run        # run scan pipeline
tentacle run    # or use the CLI directly
```

### CLI Commands

```
tentacle run              # run the full scan pipeline
tentacle review-backlog   # re-evaluate existing tickets with decay
tentacle status           # show scan history and stats
tentacle health           # check configuration and connectivity
tentacle daemon           # run continuously on a schedule
```

Options:

- `--dry-run` — run the pipeline without writing to the database or creating issues
- `--days-back N` — override how far back to scan for new articles

## Architecture

### Key Modules

- `tentacle/sources/` — Source adapters (arXiv, Semantic Scholar, Hacker News, RSS) with a shared ABC
- `tentacle/llm/` — Anthropic client wrapper, filter stage, analysis stage, prompt templates
- `tentacle/db.py` — SQLite catalog (articles, analyses, issues, decay log, scan runs)
- `tentacle/models.py` — Dataclasses: Article, Analysis, Issue, DecayEntry, ScanRun
- `tentacle/dedup.py` — SHA-256 fingerprinting for article deduplication
- `tentacle/config.py` — TOML config with env var overrides
- `tentacle/context.py` — Fetches OctopusGarden docs to give the analyzer project context
- `tentacle/issues.py` — GitHub issue creation via `gh` CLI
- `tentacle/decay.py` — Time-based maturity decay with LLM recheck at threshold crossings
- `tentacle/cli.py` — CLI entry point (run, review-backlog, status, health, daemon)

### Maturity Scale

Articles are scored on a 1–5 maturity scale that determines whether they're ready to become issues:

| Score | Label   | Readiness           |
|-------|---------|---------------------|
| 1     | Seed    | Not actionable      |
| 2     | Sketch  | Needs human input   |
| 3     | Draft   | Risky for autoissue |
| 4     | Ready   | Good for autoissue  |
| 5     | Perfect | Ideal for autoissue |

Only articles scoring 4+ are automatically filed as issues. The backlog review process rechecks articles over time — scores can decay as research ages or accelerate if follow-up work appears.

## Development

```bash
make test       # run tests
make lint       # ruff check
make typecheck  # mypy --strict
make fmt        # ruff fix + format
```

Python 3.12+, strict mypy, ruff for linting/formatting. Tests use unittest with in-memory SQLite and mocked HTTP/LLM calls.

## Deployment

Designed to run as a systemd timer on a home server (oneshot service + daily timer). See the `systemd/` directory for unit files.

## License

[MIT](LICENSE)
