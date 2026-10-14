# Contributing to hana-knowledge

Thank you for helping. This file says how to set up, what a pull request needs, and how changes are reviewed.

## Set up

```bash
git clone https://github.com/hanaml-lab/hana-knowledge && cd hana-knowledge
uv sync                       # Python 3.10+; every package in one environment
make test                     # Python, the integration tests, the Rust extension (cargo) and the TypeScript reader (node 22.18+)
make lint                     # ruff
make example                  # both examples, offline
```

The Rust extension and the TypeScript reader are optional for Python-only changes: `make test-python` runs the
Python packages and the integration tests on their own.

## What a pull request needs

- **A test first.** Add a failing test that states the behaviour through the package's public interface, then the
  change that makes it pass. Expected values come from a worked example or the spec, not from the code under test.
- **Offline by default.** Tests and examples must run without network, GPUs or model weights; use the stand-ins
  (`HashingEncoder`, the term-overlap reranker, `ExtractiveModel`, `LexicalNLI`, `PatternExtractor`) or a small
  scripted adapter at the same seam.
- **A CHANGELOG line** under the next version in `CHANGELOG.md`.
- **Green CI.** Lint, tests on Python 3.10 and 3.12, both examples, clippy and the Rust tests, and the TypeScript
  tests (see `.github/workflows/ci.yml`).
- **Results that can be reproduced.** A number in a README names the command that prints it.

## Commit messages

Conventional and lowercase, as in the history: `type(scope): summary`, for example
`feat(cite-check): convert physical units through pint before comparing figures`. Types: `feat`, `fix`, `test`,
`docs`, `ci`, `refactor`, `chore`. The scope is the package (`retrieval-kit`, `graph-memory`, `cite-check`,
`hana-knowledge`, `notes-reader`) or `examples`.

## Security

Do not open a public issue for a vulnerability. Follow the
[hanaML security policy](https://github.com/hanaml-lab/.github/blob/main/SECURITY.md).

## Licence

By contributing you agree that your contribution is licensed under the licence of the package it touches
(inbound = outbound): Apache-2.0, or MIT for `packages/graph-memory` and `packages/notes-reader`.
