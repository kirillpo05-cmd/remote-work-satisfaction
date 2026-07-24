.PHONY: install test lint train run

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app
	uv run mypy src

train:
	uv run python -m rwsat.cli train

run:
	docker compose up
