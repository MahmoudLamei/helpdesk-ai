.PHONY: install lint format test clean train

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

clean:
	uv run python -m helpdesk.data.clean

train: clean
	uv run python -m helpdesk.train --label intent --analyzer char_wb --ngram-max 5
