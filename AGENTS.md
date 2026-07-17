# Agent Instructions

Compact guide for OpenCode sessions. Comprehensive guide is `CLAUDE.md` (gitignored, loaded into session) — read it for architecture, CL loop, checklists for adding methods/backbones/scenarios. This file holds only high-signal facts an agent would otherwise get wrong.

## Research context (diagnostic + falsification paper, AAAI 2027)

This is a **diagnostic + falsification** paper, NOT a method paper (frame locked 2026-07-04, re-locked after SLR method pivot falsified 2026-07-10). See `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` for the full argument spine.

**Core claims:**
1. Forgetting in multimodal document encoders is readout/head-localized; Fisher/CKA locate the main displacement in the classifier head and late functional representations.
2. The locus **migrates** under naive protection: freezing/pinning the head redirects damage into the trunk without eliminating forgetting.
3. **No buffer-free method works.** Five families are falsified: merge, subspace transfer, parametric slots, input-anchored memory, and feature-Gaussian replay. Real past-task gradients ground the head.
4. **Finding 3b:** whole-document `(feature, position, label)` consistency is necessary for replay; marginal summaries fail (+27 AA in the consistency control).
5. **RCA (2026-07-17, provisional n=1 DIL/LayoutLMv3):** failing heads undergo a readout-marginal snap to the latest task's label marginal. Under-exercised classes extinguish; modality-pathway and late-trunk mechanisms are refuted. See `docs/RCA_FORGETTING_BASELINES_2026-07.md`.

**Current status (2026-07-17):** CoLaR is the constructive control: per-document SVD replay preserves whole-document consistency (headline DIL k4/d50/r128: 87.6 AA at 60 MB; conservation seeds show no selection/reweighting lever improves AA). PLaR is a bounded, zero-private-byte negative: public proxies reach about 59 AA but lack SROIE coverage. Read-side kNN blending is closed (no material signal); do not build MbPA. Pre-registered RCA kill tests are queued: logit/marginal corrections, frozen-trunk naive, `marginal_kl`, and `logit_adjust` (`docs/RCA_KILLTESTS_PREREG_2026-07.md`). The B+ critical path remains the three-seed, three-scenario, four-backbone generalization grid. `bd` tracking is write-blocked; use `STATE.md` and `ROADMAP.md`.

## Toolchain

- UV-managed, Python ≥3.10. Install: `uv sync --extra dev`.
- Lint/format: `uv run ruff check .` and `uv run black .` (line-length 100; ruff ignores E501, selects E,F,W,I,N,UP,B,C4,SIM).
- Tests (markers in `pyproject.toml`: `slow`, `gpu`, `integration`):
  - Fast no-GPU sanity (CI-safe gate): `uv run pytest -m "not slow and not gpu"`
  - All: `uv run pytest tests/`
  - Single test: `uv run pytest tests/methods/test_early_stopping.py::test_name`
- Quality-gate order before commit: **lint → fast tests** (no typecheck).
- CI: only `build-thesis.yml` — GitHub Actions builds `thesis/main.pdf` with XeLaTeX on push touching `thesis/`.

## Non-obvious repo facts

- **Live package is `doccl/`** (installable name `doccl`). `src/` tree sits on disk but is **dead** (untracked, unimported) — ignore it and any docs mentioning `cl4ie`/`STRATEGY_MAP`/`ExperimentConfig`.
- `scripts/train.py` is a **Hydra entry point, not a console script**. Run directly with `group=option` / `a.b=value` overrides: `uv run python scripts/train.py method=naive scenario=single_funsd seed=42`. Only installed console scripts: `doccl-pilot`, `doccl-pilot-analyze`.
- Two registries in `scripts/train.py`: `MODEL_REGISTRY` (`cfg.model.family` → wrapper class) and `METHOD_REGISTRY` (`cfg.method.name` → method class). Adding backbone/method = new registry entry + config file. Standard-forward methods require `_STD_FORWARD` set entry for per-class F1 saving.
- `configs/` are **immutable ground truth** — never edit existing config; add new option file. CLI overrides for one-off changes.
- `LayoutLMv3Wrapper` deliberately does **not** inherit shared `TokenClassificationWrapper` base — 54 validated runs must stay reproducible.
- Classifier heads forced to plain `Linear`. MLP head saturates and collapses to all-O after CIL expansion (real, fixed bug) — don't "fix" it.
- Results in `results/<run>/{metrics.json, matrix.npy, tb/, per_class_f1.json}` (gitignored). `scripts/analyze_results.py` reads from disk (NOT W&B) for offline operation.
- `STATE.md` (current snapshot) + `ROADMAP.md` (ordered plan) — read at session start, update at session end.
- `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` — the full research argument spine. `docs/` has ~25 design/validation/review documents; `docs/superpowers/` contains spec/plan documents for candidate methods.

## Operating constraints (read before running anything heavy)

- **Local box VRAM/RAM-constrained** (RTX 2060 6 GB; ~14 GB host RAM). Hard rule: VRAM < 5 GB, RAM < 14 GB or OOM. Use `training.batch_size=1..2 training.gradient_checkpointing=true`, `training.num_workers=0` on heavy datasets (XFUND, WildReceipt) — workers fork dataset working set and trip OOM killer.
- **Never** run second dataset-building job alongside a grid — box fits exactly one dataset builder. Parallel sub-agents OK only for non-dataset work (edits, synthetic-tensor tests, analysis of saved JSON).
- `fp16`/`gradient_accumulation_steps` config flags exist but are **NOT wired** into training loops — `batch_size` + `gradient_checkpointing` are the memory knobs. `training.amp=true` does enable bf16/fp16 autocast.
- Grid is resume-safe: run skipped once `results/<run>/.done` exists. Delete `.done` to re-run.
- Heavy runs belong on rented GPU (Vast.ai). See `RUNBOOK.md`, `A6000_RUNBOOK.md`, `docs/MULTI_MACHINE_RUN.md`, `scripts/setup_remote.sh`.
- `wandb.mode=offline` for local runs — sync later or skip.

## Main operational tool: the grid

`scripts/run_grid_multigpu.sh` is a resume-safe multi-slot scheduler driven entirely by env vars.

```bash
# Local single-GPU, VRAM-safe, offline W&B
GPUS=0 JOBS_PER_GPU=1 BATCH_SIZE=2 GRAD_CKPT=true WANDB_MODE=offline \
    bash scripts/run_grid_multigpu.sh
DRY_RUN=1 bash scripts/run_grid_multigpu.sh   # print job plan and exit
```

Default runs full 873-job multi-backbone sweep (LayoutLMv3 + LiLT + BROS + BERT). Set `BACKBONES=""` for LayoutLMv3-only.

Key env knobs: `GPUS`, `JOBS_PER_GPU`, `BATCH_SIZE`, `EPOCHS_CAP`, `SEEDS`, `SCENARIOS`, `CORE_METHODS`, `PROMPT_METHODS`, `CURRENCY_METHODS`, `RUN_BERT`, `RUN_DOCCL`, `RUN_ABLATION`, `DEPTH_TARGETS`, `AMP`, `NUM_WORKERS`, `GPU_VRAM_BUDGET_GB`, `SYNC_REMOTE`.

Results → thesis:
```bash
uv run python scripts/analyze_results.py --source local   # results/ → CSVs, .tex tables, figures
uv run python scripts/ingest_to_thesis.py                  # copy artifacts into thesis/
cd thesis && latexmk -xelatex main.tex                     # XeLaTeX; latexmkrc present
```

## Conventions

- Task tracking via **bd (beads)** when operational. **Currently write-blocked** (schema-migration fork, tracked in `ROADMAP.md`) — track remaining work in STATE/ROADMAP. Run `bd ready --json` for ready work.
- Patches to `doccl/` committed with `AGENT IMPL:` / `AGENT FIX:` prefix; other commits use conventional-commit types (feat, fix, refactor, perf).
- Do NOT spawn sub-agents or parallel task agents unless explicitly requested.
- `cp`/`mv`/`rm` may be aliased to `-i` — always use `-f` flags. `ssh`/`scp` with `-o BatchMode=yes`.

## Session completion (mandatory)

Work is **not complete until `git push` succeeds**. At session end:
1. File beads issues for remaining work (or log in ROADMAP/STATE if bd blocked).
2. Run quality gates if code changed: `uv run pytest -m "not slow and not gpu" && uv run ruff check . && uv run black --check .`
3. Push: `git pull --rebase && git push && git status` (must show "up to date with origin").
4. Never stop before pushing — work stranded locally. If push fails, resolve and retry.
