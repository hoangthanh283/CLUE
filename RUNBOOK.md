# DocCL Experiment Runbook

Sequence to run the experiments that fill the thesis result placeholders (Chapter 6
tables/figures, §3.3.6, abstract/conclusion margins). Runs execute on the local GPU
server with W&B **offline**; results land in `results/` and are then ingested into
the thesis.

Pipeline: **setup → pilot → [GATE A: Week-4 decision] → main grid + ablation → aggregate → ingest → write prose**.

### Memory profile (limited-VRAM GPU)

Full fine-tuning of LayoutLMv3-base is memory-heavy, so every full-FT run
(naive / joint / ewc / lwf / er / der_pp, and the pilot) must use **gradient
checkpointing** + a **small batch**. Prompt/LoRA methods (l2p / dualprompt /
coda_prompt / o_lora, DocCL-A/C) freeze the backbone and are much lighter. The
recipe is threaded through the commands below via env vars:

- Grid:  `EXTRA="training.batch_size=2 training.gradient_checkpointing=true"`
- Pilot: `GRAD_CKPT=1 BATCH_SIZE=2 CKA_N=200 FISHER_N=100`
- Still OOM → drop to `batch_size=1` (and `CKA_N=100 FISHER_N=50`).

Runs are slow and the grid is large, so go in order: pilot → GATE A → core baselines
→ rest. (`fp16` and `gradient_accumulation_steps` config flags exist but are not yet
wired into the training loops — `batch_size` + checkpointing are the memory knobs.)

---

## 0. Environment + data (one-time)

```bash
conda create -n doccl python=3.10 -y && conda activate doccl
pip install -e ".[dev]"

# Sanity (no GPU/network needed):
pytest tests/test_smoke.py -v -m "not slow and not gpu"

# FUNSD + CORD auto-download from HuggingFace on first use. SROIE needs one of:
#   (A) Manual (canonical): download ICDAR-2019 SROIE Task 3, then
python scripts/prepare_sroie.py --raw_dir /path/to/SROIE_raw --output_dir data/sroie
#   (B) HF mirror (one-command, confirm provenance — see doccl/data/sroie.py):
#       set source="hf" for the SROIE loader (verify its BIO scheme matches LABEL_NAMES).
```

W&B is offline by default in `run_grid.sh` (`WANDB_MODE=offline`); the pilot runner
doesn't touch W&B. A first GPU sanity check (limited-VRAM settings):

```bash
python scripts/train.py method=naive scenario=single_funsd wandb.mode=offline \
    training.batch_size=2 training.gradient_checkpointing=true   # expect FUNSD F1 ≈ 88–92
```

## 1. Pilot study  → fills §6.1, Figs 6.1/6.2, Ch1 headline, abstract diagnosis

```bash
GRAD_CKPT=1 BATCH_SIZE=2 CKA_N=200 FISHER_N=100 bash scripts/run_pilot.sh   # 4 conditions × 3 seeds
GRAD_CKPT=1 BATCH_SIZE=2 ORDER="2 1 0" CONDITIONS="c4_full" bash scripts/run_pilot.sh  # alt-order (§6.1.3)
python -m doccl.pilot.analyze --pilot_dir results/pilot        # figures + findings_summary.md
```

Review `results/pilot/findings_summary.md` — it reports the Mann-Whitney U verdicts
(H0/H1, Bonferroni-corrected) and the dominant component.

## ⛔ GATE A — Week-4 advisor decision (do NOT pre-lock)

From the pilot verdict, select the proposed method per the decision rule:

| Diagnosis | Method |
|---|---|
| Fusion-dominant forgetting (+ layout matters) | Candidate **A** (`doccl_a`, H-LoRA) |
| 2D layout-position drift dominates | Candidate **B** (`doccl_b`, Layout-Protected EWC) |
| Scenario-dependent per-modality patterns | Candidate **C** (`doccl_c`, Modality-Routed Prompts) |
| No clear pattern (fail to reject H0) | **Characterization-only fallback** (see CLAUDE.md) |

Then point the `doccl` alias at the winner (2 edits):
1. `scripts/train.py` → `METHOD_REGISTRY["doccl"] = DocCL_<X>`
2. `configs/method/doccl.yaml` → copy the body of `doccl_<x>.yaml` (keep `name: doccl`)

## 2. Main grid + ablation  → fills Table 6.1/6.2/6.3, Fig 6.3

```bash
G6='training.batch_size=2 training.gradient_checkpointing=true'   # limited-VRAM settings
PHASE=3 EXTRA="$G6" bash scripts/run_grid.sh        # core baselines (can run during the pilot)
PHASE=4 EXTRA="$G6" bash scripts/run_grid.sh        # prompt + LoRA baselines (lighter)
PHASE=5 EXTRA="$G6" bash scripts/run_grid.sh        # proposed method (after GATE A)
PHASE=ablation EXTRA="$G6" bash scripts/run_grid.sh # component-targeting ablation (Table 6.2)
```

## 3. Aggregate + ingest into the thesis

```bash
python scripts/analyze_results.py                 # reads results/*/metrics.json (offline)
python scripts/ingest_to_thesis.py                # copies figures/tables into thesis/
# Rebuild the thesis (see thesis/README.md, e.g. `latexmk` in thesis/).
```

`chapter6.tex` uses `\IfFileExists`, so it shows the TODO scaffold until artifacts
exist and the real tables/figures afterwards — the build is always green.

## 4. Prose (Claude, gated on returned results)

- After the pilot: Claude writes §6.1 interpretation + captions, Ch1 headline, abstract diagnosis.
- After the grid: Claude writes §3.3.6 (selected method + Algorithm 2 + complexity),
  §6.2–6.4, BWT/FWT, Ch7 conclusions, abstract margin, Appendix A extended tables.
  **No fabricated numbers** — prose follows the real `results/`.

## 5. Generalization (optional, Stage 6)

Implement `lilt_wrapper.py`/`bros_wrapper.py` (+ model configs), then
`bash scripts/run_generalization.sh`; fills the §6.4 LiLT/BROS paragraph.

---

## Troubleshooting

- **CUDA OOM** → keep `training.gradient_checkpointing=true` and drop
  `training.batch_size=1`; for the pilot also `CKA_N=100 FISHER_N=50`. Prompt/LoRA
  phases tolerate a larger batch than the full-FT phases.
- **W&B blocks** → grid is offline by default; else `export WANDB_MODE=offline`.
- **SROIE missing** → run `prepare_sroie.py` or use the HF mirror (Step 0).
- **Resume** → grid runs touch `results/<run>/.done`; pilot skips existing JSONs. Delete to re-run.
