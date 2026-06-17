# DocCL

**Continual Learning for Document Understanding**

## TL;DR

This repository implements a **diagnostic + remedy** paper:

1. **Characterize** how catastrophic forgetting manifests in a multimodal document encoder (LayoutLMv3) via a controlled pilot study (4 modality conditions × seeds × FUNSD→CORD→SROIE), capturing per-layer CKA + per-component Fisher at each task boundary.
2. **Localize** the forgetting: it concentrates in the **classifier head + late layers**, not uniformly across the network.
3. **Remedy** it with **DocCL** — a depth/head-targeted method derived from that diagnosis (`method.target_depth ∈ {all, head_only, late_only, uniform}` drives the component-targeting ablation).
4. **Validate** across CIL / DIL / mixed scenarios on FUNSD / CORD / SROIE / WildReceipt / XFUND, against classical (EWC, LwF, ER, DER++), prompt/LoRA (L2P, DualPrompt, CODA-P, O-LoRA), and 2025 (C-Flat++, CL-LoRA) baselines, plus LiLT / BROS / text-only-BERT backbones for a generalization study.

---

## Quickstart

### 1. Install (UV)

```bash
uv sync --extra dev
```

### 2. Verify setup (no GPU/network needed)

```bash
uv run pytest -m "not slow and not gpu"
```

### 3. Run a single-task baseline (sanity check)

```bash
uv run python scripts/train.py method=naive scenario=single_funsd seed=42
```

Expected: FUNSD test F1 ≈ 88–92 (LayoutLMv3-base). On a small-VRAM GPU add
`training.batch_size=2 training.gradient_checkpointing=true wandb.mode=offline`.

### 4. Run the pilot study (the diagnostic)

```bash
bash scripts/run_pilot.sh
uv run python -m doccl.pilot.analyze --pilot_dir results/pilot
```

### 5. Run the experiment grid

`scripts/train.py` is a Hydra entry point (`group=option` / `a.b=value` overrides). The full
sweep behind the thesis tables is run by the resume-safe multi-slot scheduler:

```bash
# Local single-GPU, VRAM-safe, offline W&B
GPUS=0 JOBS_PER_GPU=1 BATCH_SIZE=2 GRAD_CKPT=true WANDB_MODE=offline \
    bash scripts/run_grid_multigpu.sh

DRY_RUN=1 bash scripts/run_grid_multigpu.sh   # print the job plan and exit
```

Tunable via env: `GPUS`, `JOBS_PER_GPU`, `BATCH_SIZE`, `EPOCHS_CAP`, `SEEDS`, `SCENARIOS`,
`CORE_METHODS`, `PROMPT_METHODS`, `CURRENCY_METHODS`, `RUN_BERT`, `RUN_DOCCL`, `RUN_ABLATION`,
`DEPTH_TARGETS`, `AMP`, `NUM_WORKERS`. Remote/rented-GPU bootstrap is `scripts/setup_remote.sh`;
see `RUNBOOK.md` and `docs/MULTI_MACHINE_RUN.md` for the end-to-end pipeline.

### 6. Build the result tables / thesis

```bash
uv run python scripts/analyze_results.py --source local   # results/ → CSVs, .tex tables, figures
uv run python scripts/ingest_to_thesis.py                  # copy artifacts into thesis/
cd thesis && latexmk -xelatex main.tex
```

`analyze_results.py` reads each run's `results/<run>/{metrics.json,matrix.npy}` (not W&B), so it
works on an offline box.

---

## Repository layout

```
doccl/                     Live package (installable name `doccl`)
├── data/                  Dataset loaders (FUNSD, CORD, SROIE, WildReceipt, XFUND),
│                          scenario builders, CIL/DIL label remappers, per-backbone encoders
├── methods/               CL methods (ContinualMethod ABC + EarlyStopper / AMP in base.py)
│   ├── naive.py           Lower bound (sequential FT) + Joint upper bound
│   ├── ewc.py, lwf.py     Regularization
│   ├── er.py, der.py      Replay
│   ├── er_cflat.py        C-Flat++ (SAM / flat-minima, 2025)
│   ├── l2p.py, dualprompt.py, coda_prompt.py   Prompt-based
│   ├── o_lora.py, cl_lora.py                    LoRA-based (CL-LoRA = 2025)
│   └── doccl.py           DocCL — the proposed depth/head-targeted method (+ ablation variants)
├── models/                Backbone wrappers: LayoutLMv3 (primary) + LiLT/BROS/BERT
│                          (TokenClassificationWrapper base) with extension points
├── eval/                  Metrics: AA/BWT/FWT/AF, seqeval F1, CKA, Fisher displacement
├── pilot/                 Pilot study runner + analysis
└── utils/                 TensorBoard logger, etc.

configs/                   Hydra config groups: scenario/ method/ model/ training/ (+ default.yaml)
scripts/
├── train.py               Hydra training entry point
├── run_grid_multigpu.sh   Resume-safe multi-slot grid scheduler (main tool)
├── run_pilot.sh           Pilot study runner
├── setup_remote.sh        Rented-GPU bootstrap
├── analyze_results.py     results/ → LaTeX tables + figures
└── ingest_to_thesis.py    Copy tables/figures into thesis/
tests/                     Fast unit/smoke tests (markers: slow, gpu, integration)
thesis/                    LaTeX thesis (XeLaTeX; latexmk)
docs/                      Design notes (DIL schema, CIL splits, multi-machine run)
```

> A `src/` directory remains on disk but is **dead** (untracked, unimported) — ignore it.

---

## Scenarios

- **CIL** (`cil_funsd`, `cil_cord`, `cil_wildreceipt`, `*_long`) — class-incremental: the
  classifier head grows as sessions introduce new BIO labels.
- **DIL** (`dil` form↔receipt, `dil_receipts`, `dil_xlingual`) — domain-incremental with a fixed
  unified label space; forgetting is pure representation drift. `dil_xlingual` sweeps XFUND
  languages (de→es→fr→it→zh) to test whether the output-side forgetting finding holds under
  language shift.
- **mixed** — interleaves class-IL within a domain and domain shifts.
- **single_\*** — single-task baselines (also the FWT reference).

## Pilot study design (the diagnostic core)

| Condition | Active modalities     | Purpose                               |
| --------- | --------------------- | ------------------------------------- |
| C1        | text only             | Unimodal text reference               |
| C2        | image + layout        | Isolate visual+layout-only forgetting |
| C3        | text + layout         | Isolate text+layout-only forgetting   |
| C4        | text + image + layout | Main subject (full LayoutLMv3)        |

Captured at every task boundary: per-layer **CKA**, per-parameter-group **Fisher information**,
and the per-task F1 forgetting matrix R[i, j]. The pattern (forgetting localized to the head +
late layers) is what `doccl` targets.

## References

- LayoutLMv3 — Huang et al., ACM MM 2022, arXiv:2204.08387; LiLT — Wang et al., ACL 2022; BROS — Hong et al., AAAI 2022.
- Diagnostic foundations: Ramasesh et al., "Anatomy of CF", ICLR 2021; Zhai et al., "CF in MLLMs", 2023; Kornblith et al., "NN Representation Similarity (CKA)", ICML 2019.
- CL baselines: Kirkpatrick (EWC) PNAS 2017; Li & Hoiem (LwF) ECCV 2016; Rolnick (ER) NeurIPS 2019; Buzzega (DER++) NeurIPS 2020; Wang (L2P) CVPR 2022 & (DualPrompt) ECCV 2022; Smith (CODA-P) CVPR 2023; Wang (O-LoRA) EMNLP-F 2023; C-Flat++ and CL-LoRA (2025 currency baselines).

---

## License

MIT (code). Datasets retain their original licenses.
