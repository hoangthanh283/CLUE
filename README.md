# DocCL

**Continual Learning for Document Understanding** — Master's thesis project at HUST + AAAI 2027 submission target.

> 📍 **Status**: Phase 1 (Infrastructure + Reading + Sketching). See [PLAN.md](PLAN.md) for the 14-week roadmap.

---

## TL;DR

This repository implements a **diagnostic + remedy** paper:

1. **Characterize** how catastrophic forgetting manifests in multimodal document encoders (LayoutLMv3) compared to unimodal/modality-ablated baselines via a controlled pilot study (4 conditions × 3 seeds × 3 tasks).
2. **Hypothesize** which architectural component(s) drive forgetting (fusion vs. text vs. visual).
3. **Exploit** the finding via a method selected from 3 pre-sketched candidates.
4. **Validate** on the **FCS-CL benchmark** (FUNSD/CORD/SROIE) across 3 CL scenarios.

Method emerges from data; not pre-locked.

---

## Quickstart

### 1. Install

```bash
conda create -n doccl python=3.10 -y && conda activate doccl
pip install -e ".[dev]"
```

### 2. Verify setup

```bash
pytest tests/test_smoke.py -v -m "not slow and not gpu"
```

All non-slow tests should pass without a GPU. Slow/GPU tests are skipped automatically.

### 3. Run a single-task baseline (sanity check)

```bash
python scripts/train.py method=naive scenario=single_funsd seed=42
```

Expected: F1 ≥ 88 on FUNSD test set after 10 epochs (LayoutLMv3-base, RTX 4090).

### 4. Run the pilot study (Week 3-4 deliverable)

```bash
bash scripts/run_pilot.sh
```

This runs 4 conditions × 3 seeds = 12 sequential training runs (FUNSD → CORD → SROIE), capturing CKA + Fisher metrics at task boundaries. Results land in `results/pilot/`.

Then aggregate:

```bash
python -m doccl.pilot.analyze --pilot_dir results/pilot
```

### 5. Run the main grid (Week 5-8)

```bash
bash scripts/run_grid.sh                  # all baselines
PHASE=3 bash scripts/run_grid.sh          # only EWC, LwF, ER, DER++
PHASE=4 bash scripts/run_grid.sh          # only L2P, DualPrompt, CODA-P, O-LoRA
```

---

## Repository layout

```
doccl/
├── data/                  Dataset loaders (FUNSD, CORD, SROIE) + scenario builders
├── methods/               CL method implementations
│   ├── base.py            ContinualMethod ABC (lifecycle hooks)
│   ├── naive.py           Lower bound (sequential FT) + Joint upper bound
│   ├── ewc.py, lwf.py     Regularization (W5)
│   ├── er.py, der.py      Replay (W6)
│   ├── l2p.py, ...        Prompt-based (W7-8)
│   ├── o_lora.py          LoRA-based (W8)
│   └── doccl.py           Selected method (W9-11, post-pilot)
├── models/                LayoutLMv3 wrapper with extension points
├── eval/                  Metrics: CKA, Fisher, AA/BWT/FWT/AF, seqeval F1
├── pilot/                 Pilot study runner + analysis
└── utils/

configs/                   Hydra configs (composable)
scripts/
├── train.py               Main entry point
├── run_pilot.sh           Pilot study grid runner
├── run_grid.sh            Baseline grid runner
├── run_generalization.sh  LiLT + BROS subset (W12-13)
├── prepare_sroie.py       SROIE preprocessing (manual download required)
├── setup_vastai.sh        Vast.ai instance bootstrap
└── analyze_results.py     W&B → LaTeX tables
tests/                     Smoke tests
docs/                      Design notes (DIL schema, CIL splits, etc.)
```

---

## Pilot study design (the diagnostic core)

| Condition | Model | Active modalities | Purpose |
|---|---|---|---|
| C1 | LayoutLMv3 / TEXT_ONLY mask | text only | Unimodal text reference |
| C2 | LayoutLMv3 / IMAGE_LAYOUT mask | image + layout | Isolate visual+layout-only forgetting |
| C3 | LayoutLMv3 / TEXT_LAYOUT mask | text + layout | Isolate text+layout-only forgetting |
| C4 | LayoutLMv3 / FULL | text + image + layout | Main subject |

**Metrics captured at every task boundary:**

- Per-layer **CKA** (Kornblith et al. 2019) between consecutive checkpoints
- Per-parameter-group **Fisher information**
- Per-task **F1** (forgetting matrix R[i, j])

**Hypothesis test:**

- **H₀:** Forgetting is uniform across components.
- **H₁:** Forgetting concentrates in specific components (e.g., fusion).

The Week 4 advisor meeting decides which of three pre-sketched method candidates to implement based on the pilot pattern. If H₀ cannot be rejected, the paper pivots to a characterization-only framing.

---

## Three candidate methods (pre-sketched W1-2, selected W4)

1. **Candidate A — LAPP + H-LoRA**: Layout-Aware Prompt Pool + Hierarchical LoRA banks (text/visual/fusion). Selected if pilot shows fusion-layer forgetting + layout-sensitivity.
2. **Candidate B — Layout-Protected Regularization**: Selective EWC with high importance weight on 2D position embeddings. Selected if pilot shows uniform forgetting + position-embedding drift.
3. **Candidate C — Modality-Routed Prompts**: Three prompt sub-pools (text/visual/layout) with router. Selected if pilot shows scenario-dependent patterns.

---

## Compute budget

~340 GPU-hours total on RTX 4090 (Vast.ai), ≈ $200-250.

| Phase | GPU-hours | Cost @ $0.35/h |
|---|---|---|
| Setup + Pilot | 40 | $14 |
| Core baselines (W5-6) | 54 | $19 |
| Advanced baselines (W7-8) | 43 | $15 |
| Selected method + ablations (W9-11) | 74 | $26 |
| Generalization (LiLT + BROS, parallel W12-13) | 130 | $46 |
| Final reruns + buffer | 30+ | ~$80 |

---

## Key design decisions (locked May 6, 2026)

See [CLAUDE.md](CLAUDE.md) for the full decision log. Highlights:

- **Insight-driven framing** (characterize → exploit), not pure method paper
- **3 backbones**: LayoutLMv3-base (primary, full grid) + LiLT + BROS (subset, generalization)
- **3 scenarios**: CIL-CORD (5 sessions × 6 classes), DIL (FUNSD→SROIE→CORD with unified schema), Mixed (6 sessions interleaving)
- **CORD label space**: 30 fine-grained classes (61 BIO tags), not 5 super-classes
- **Pilot conditions**: BERT-style + 3 LayoutLMv3 modality variants (apples-to-apples)
- **TIL deferred** to NeurIPS extension; **DSR also deferred**
- **Fallback**: characterization-only paper if Week-11 selected method fails

---

## References

- LayoutLMv3 — Huang et al., ACM MM 2022, arXiv:2204.08387
- Pilot foundations:
  - Ramasesh et al., "Anatomy of CF", ICLR 2021, arXiv:2007.07400
  - Zhai et al., "Investigating CF in MLLMs", 2023, arXiv:2309.10313
  - Kornblith et al., "NN Representation Similarity", ICML 2019, arXiv:1905.00414
- CL methods: Kirkpatrick (EWC) PNAS 2017; Li & Hoiem (LwF) ECCV 2016; Rolnick (ER) NeurIPS 2019; Buzzega (DER++) NeurIPS 2020; Wang (L2P) CVPR 2022, (DualPrompt) ECCV 2022; Smith (CODA-P) CVPR 2023; Wang (O-LoRA) EMNLP-F 2023.

---

## Author

Thanh Hoang — Hanoi University of Science and Technology  
Master's thesis advisor: [TBD]

## License

MIT (code). Datasets retain their original licenses.
