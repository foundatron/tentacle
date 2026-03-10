.PHONY: test lint typecheck fmt run sync

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check tentacle/ tests/ scripts/

typecheck:
	uv run mypy tentacle/

fmt:
	uv run ruff check --fix tentacle/ tests/
	uv run ruff format tentacle/ tests/ scripts/

run:
	uv run python -m tentacle run

sync:
	uv sync
