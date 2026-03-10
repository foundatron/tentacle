# Tentacle

Research scout service for [OctopusGarden](https://github.com/foundatron/octopusgarden). Scans research literature, blog posts, and articles about autonomous code generation and software dark factories, then creates actionable GitHub issues in the octopusgarden repo.

## Pipeline

```
Sources (arXiv, Semantic Scholar, HN, RSS)
  -> Dedup (SHA-256 fingerprint)
  -> Filter (Haiku: title+abstract -> relevance 0-1)
  -> Analyze (Sonnet: full content + octopusgarden context -> maturity 1-5)
  -> Create Issues (gh CLI, max N per cycle)
  -> Backlog Review (time-based decay, status sync)
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # install dependencies
cp config.example.toml ~/.config/tentacle/config.toml
# edit config.toml and set ANTHROPIC_API_KEY
```

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
```

## Development

```bash
make test       # run tests
make lint       # ruff check
make typecheck  # mypy --strict
make fmt        # ruff fix + format
```

## Maturity Scale

| Score | Label   | Readiness           |
|-------|---------|---------------------|
| 1     | Seed    | Not actionable      |
| 2     | Sketch  | Needs human input   |
| 3     | Draft   | Risky for autoissue |
| 4     | Ready   | Good for autoissue  |
| 5     | Perfect | Ideal for autoissue |

## Deployment

systemd timer on home server (oneshot service + daily timer). See `systemd/` directory.
