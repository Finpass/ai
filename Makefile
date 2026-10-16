.PHONY: run test lint fmt typecheck check

run:
	uv run python main.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

check: lint typecheck test
