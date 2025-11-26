PY_PACKAGES := retrieval-kit graph-memory cite-check

.PHONY: install test test-python test-rust test-ts lint example

install:
	uv sync

test: test-python test-rust test-ts

test-python:
	@for p in $(PY_PACKAGES); do (cd packages/$$p && uv run pytest -q) || exit 1; done
	uv run pytest -q tests/integration

test-rust:
	cargo test --manifest-path packages/retrieval-kit/tantivy-ext/Cargo.toml

test-ts:
	cd packages/notes-reader && npm test

lint:
	uv run ruff check .

example:
	uv run python examples/end_to_end.py
