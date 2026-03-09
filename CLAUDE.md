# CLAUDE.md -- Tentacle

Research scout service for OctopusGarden. Scans research literature, blog posts, and articles about
autonomous code generation and software dark factories, then creates actionable GitHub issues in the
octopusgarden repo.

## Commands

```bash
make test       # run tests
make lint       # ruff check
make typecheck  # mypy strict
make fmt        # ruff fix + format
make run        # run scan pipeline
uv sync         # install/update dependencies
```

Commits must follow [Conventional Commits](https://www.conventionalcommits.org/) -- enforced by
commit-msg hook. Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`,
`build`, `ci`, `revert`.

## Architecture

Python 3.12+, managed with `uv`. Minimal dependencies: `anthropic` SDK + stdlib.

### Pipeline

```
Sources (arXiv, Semantic Scholar, HN, RSS)
  -> Dedup (SHA-256 fingerprint)
  -> Filter (Haiku, cheap: title+abstract -> relevance 0-1)
  -> Analyze (Sonnet, thorough: full content + octopusgarden context -> maturity 1-5)
  -> Create Issues (gh CLI, conventional commits format, max N per cycle)
  -> Backlog Review (time-based decay, status sync)
```

### Key Modules

- `tentacle/sources/` -- Source adapters (ABC in `base.py`): arxiv, semantic_scholar, hackernews, rss
- `tentacle/llm/` -- Anthropic client wrapper (`client.py`), filter stage, analysis stage, prompts
- `tentacle/db.py` -- SQLite catalog (articles, analyses, issues, decay_log, scan_runs)
- `tentacle/models.py` -- Dataclasses: Article, Analysis, Issue, DecayEntry, ScanRun
- `tentacle/dedup.py` -- SHA-256 fingerprinting
- `tentacle/config.py` -- TOML config loading with env var overrides
- `tentacle/context.py` -- Fetch/cache octopusgarden CLAUDE.md + architecture.md
- `tentacle/issues.py` -- GitHub issue creation via `gh` CLI
- `tentacle/decay.py` -- Ticket maturity decay logic
- `tentacle/cli.py` -- argparse CLI: run, review-backlog, status

### Maturity Scale

| Score | Label   | Readiness           |
|-------|---------|---------------------|
| 1     | Seed    | Not actionable      |
| 2     | Sketch  | Needs human input   |
| 3     | Draft   | Risky for autoissue |
| 4     | Ready   | Good for autoissue  |
| 5     | Perfect | Ideal for autoissue |

## Coding Standards

- Python 3.12+, type hints everywhere
- `ruff` for linting and formatting (strict config in pyproject.toml)
- `mypy --strict` for type checking
- Tests: `unittest`, table-driven where possible, in-memory SQLite, mocked HTTP/LLM
- Logging: `logging` module, structured messages
- No global mutable state
- Stdlib preferred (urllib, xml.etree, sqlite3, json, hashlib)

## Configuration

TOML at `~/.config/tentacle/config.toml`. Env vars override (ANTHROPIC_API_KEY).
See `config.example.toml` for all options.

## Deployment

systemd timer on home server (oneshot service + daily timer). See `systemd/` directory.
