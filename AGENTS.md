# Repository Guidelines

## Project Structure & Module Organization

`doccl/` is the live Python package. Continual-learning methods are in `doccl/methods/`,
model wrappers in `doccl/models/`, dataset and scenario code in `doccl/data/`, and metrics in
`doccl/eval/`. Ignore the legacy `src/` tree; it is untracked and unused. Hydra configuration
is under `configs/`, executable workflows are in `scripts/`, and tests mirror the package in
`tests/`. Research notes live in `docs/`; thesis sources are in `thesis/`. Generated experiment
artifacts belong in gitignored `results/<run>/` directories.

Read `STATE.md` and `ROADMAP.md` before starting research work. Use `CLAUDE.md` for the full
architecture and experiment workflow.

## Build, Test, and Development Commands

- `uv sync --extra dev` installs Python 3.10+ dependencies and development tools.
- `uv run python scripts/train.py method=naive scenario=single_funsd seed=42` runs a Hydra
  training job; use `group=option` and `key=value` overrides.
- `DRY_RUN=1 bash scripts/run_grid_multigpu.sh` previews the resume-safe experiment grid.
- `uv run ruff check .` checks lint rules; `uv run black --check .` verifies formatting.
- `uv run pytest -m "not slow and not gpu"` is the fast, CI-safe test gate.
- `cd thesis && latexmk -xelatex main.tex` builds the thesis PDF.

Run heavy grids on rented GPUs. Locally, keep VRAM below 5 GB and RAM below 14 GB; use batch
size 1–2, gradient checkpointing, and zero data-loader workers for heavy datasets.

## Coding Style & Naming Conventions

Use four-space indentation, Black formatting, and a 100-character line length. Ruff enforces
imports, naming, bug-risk, and modernization rules configured in `pyproject.toml`. Follow
existing `snake_case` modules/functions and `PascalCase` classes. Never edit an existing
`configs/` option: add a new YAML file or use a CLI override. Register new methods or models in
`scripts/train.py`; standard-forward methods also require an `_STD_FORWARD` entry.

## Testing Guidelines

Pytest markers are `slow`, `gpu`, and `integration`. Name files `test_*.py` and tests
`test_*`. Add focused tests near the affected subsystem, then run the smallest relevant test
before the fast suite. No coverage percentage is mandated; regressions and non-trivial logic
must have a runnable check.

## Commit & Pull Request Guidelines

Use Conventional Commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`). Changes to
`doccl/` use `AGENT IMPL:` or `AGENT FIX:`. Pull requests should state the motivation, list
validation commands, link the issue or roadmap item, and include metric/table evidence for
experiment-affecting changes. Do not commit datasets, checkpoints, W&B state, or generated
`results/` artifacts.
