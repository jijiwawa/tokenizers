# Repository Guidelines

## Project Structure & Module Organization
This repository is multi-language, with Rust as the core implementation.

- `tokenizers/`: core Rust crate (`src/`), integration tests (`tests/`), benchmarks (`benches/`), and examples (`examples/`).
- `bindings/python/`: PyO3-based Python package, Python sources in `py_src/tokenizers/`, Rust bridge code in `src/`, tests in `tests/`.
- `bindings/node/`: N-API Node.js bindings with TypeScript tooling and Jest tests.
- `docs/`: Sphinx/doc-builder sources and documentation build config.
- Root-level performance notes/logs (`*.json`, `log_*.txt`, `*.md`) are experiment artifacts; keep changes scoped to your task.

## Build, Test, and Development Commands
Run commands from the relevant subdirectory:

- Rust core (`tokenizers/`):
  - `make build` (build all targets)
  - `make lint` (rustfmt check + clippy with warnings denied)
  - `make test` (runs Rust tests; downloads required test data)
  - `make bench` (criterion benches)
- Python bindings (`bindings/python/`):
  - `make style` (stub generation + `ruff` auto-fix/format + `ty`)
  - `make check-style` (format/lint/type checks without auto-fix)
  - `make test` (pytest + Rust tests for Python binding)
- Node bindings (`bindings/node/`):
  - `yarn build`, `yarn test`, `yarn lint`, `yarn format`

## Coding Style & Naming Conventions
- Rust: format with `cargo fmt`; pass `cargo clippy --all-targets --all-features -D warnings`.
- Python: `ruff` line length is 119 (configured in `bindings/python/pyproject.toml`); keep typed public APIs and generated stubs consistent.
- Node/TS: Prettier + ESLint rules from `bindings/node/package.json`.
- Naming: keep modules/files aligned with existing patterns (`snake_case` in Rust/Python, `camelCase` for JS/TS variables, descriptive benchmark/test file names).

## Testing Guidelines
- Prefer targeted tests while iterating, then run full suites before PR.
- Rust tests live in `tokenizers/tests/*.rs`; add regression tests near affected components.
- Python tests live in `bindings/python/tests/` (pytest). Name files `test_*.py`.
- For performance changes, include reproducible benchmark commands and before/after numbers.

## Commit & Pull Request Guidelines
- Follow existing commit style: imperative, concise subject lines (e.g., `Optimize single-thread Python tokenizer benchmark`), optionally reference issues/PRs (`(#1938)`).
- Keep commits focused; separate refactors from behavior changes.
- PRs should include:
  - what changed and why,
  - impacted modules (`tokenizers`, `bindings/python`, `bindings/node`),
  - test/benchmark evidence,
  - migration notes if public behavior or APIs change.
