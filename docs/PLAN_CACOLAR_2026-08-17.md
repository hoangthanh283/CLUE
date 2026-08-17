# Research Plan — CA-CoLaR: Coverage-Adaptive Compressed Latent Replay (2026-08-17)

Post-AAAI follow-up paper plan for the memo's ranked-#1 direction
(`CL4IE/wiki/ideas/2026-08-17-coverage-adaptive-colar.md`,
`CL4IE/wiki/analyses/2026-08-17-colar-directions-memo.md`).

## 1. Research question

**Given a fixed replay byte budget, how should a compressed latent-replay method allocate it
across tasks and across the three dials — coverage (docs/task), fidelity (SVD rank), and
selection (which docs) — to maximize retention in continual document IE?**

Hypothesis: uniform allocation (CoLaR's d/r fixed globally) is Pareto-dominated by a
retention-need-driven allocation, because (a) per-task retention need is heterogeneous
(CLUE record: FUNSD collapses without coverage, CORD never forgets — final-row spread
44.3 vs 96.7 at d5) and (b) the marginal value of bytes falls steeply with coverage
(d5→d50 CoLaR control: FUNSD retention 44.3→85.3; auxiliary-capacity value +1.91 AA→+0.19 AA).

**Evidence status (2026-08-17):** the d5/d50 coverage law is 2-point but large and reproduced
across method variants; the shadow-branch GO is seed-fragile (seed 42 +1.91, seed 7 −0.66 AA;
seed 123 pending) — CA-CoLaR deliberately does NOT build on the shadow branch.

## 2. Methodology (4-stage)

### Stage A — Overall plan
1. **Measure the response surface** (observational, no training): retention(task) as a function
   of docs d ∈ {5,50}, rank r ∈ {64,128} from existing runs + the seed grid; per-doc SVD
   reconstruction-error vs retention correlation; per-task heterogeneity quantification.
2. **Define the allocator**: at each task boundary, allocate the next task's byte budget by an
   explicit rule (no learned router — the falsification record and SinglePrompt both say
   trained routing is a liability). Candidate rules, in prereg order:
   R1 *retention-proportional docs* (docs_t ∝ observed forgetting rate of task t's family, rank
   fixed at r64); R2 *residual-capped rank* (rank_t = smallest r with reconstruction error below
   a global cap, spend the saved bytes on docs); R3 *dual-geometry selection* (MERS-style:
   k-center over layout-signature ⊕ frozen-feature embeddings, replacing single-space kcenter).
3. **Equal-byte evaluation**: every comparison at matched total bytes; the claim is allocation,
   never budget.
4. **Scope**: dil (+ dil_o2/o3 orders), cil_cord (head-growth regime), 3 seeds, LayoutLMv3 +
   one secondary backbone (LiLT) — matches the thesis's multi-seed × multi-scenario ×
   multi-backbone bar.

### Stage B — Architecture (CLUE integration)
- `doccl/methods/colar_adaptive.py` — subclass of the `latent_replay.py` CoLaR class; overrides
  only the bank-time allocation step (`docs_per_task`, `rank_r` become per-task values computed
  by rule; selection pluggable). No training-path changes.
- `configs/method/colar_adaptive.yaml` — rule ∈ {r1,r2,r3,uniform}, byte_budget, caps.
- Registry entry in `scripts/train.py` + `_STD_FORWARD`; `tests/methods/test_colar_adaptive.py`
  (allocation arithmetic, byte-budget invariant, uniform-rule ≡ colar byte-identical check).
- Analysis: `scripts/analyze_cacolar.py` — response-surface fit + Pareto plots from metrics.json.

### Stage C — Logic / task graph
```
T1 response-surface fit (free, existing runs)          [no deps]
T2 allocator rules finalized from T1                    [T1]
T3 colar_adaptive impl + tests (uniform≡colar gate)     [can parallel T1]
T4 PREREG: equal-byte gate, dil seed42, R-winner vs uniform   [T2,T3]
T5 multi-seed × dil orders × cil_cord expansion         [T4 pass]
T6 LiLT backbone slice                                  [T5]
T7 Pareto sweep (bytes × rule) — rented GPU             [T5]
T8 paper writing (motivation = falsification chain + coverage law)  [T5+, parallel]
```
Parallelizable: T1∥T3; T5/T6/T7 grid-scheduled. Local box: T1–T4; T5–T7 rented GPU.

### Stage D — Configuration
Fixed from CoLaR: k=4 split, lr 5e-5, 5 epochs, batch 1, grad-ckpt, seeds {42,7,123}.
Budget grid: bytes(d5,r64) ≈ 3.27 MB (floor), bytes(d50,r128) ≈ 61 MB (ceiling), one midpoint.
Guards (CoLaR gate style, preregistered before T4 runs): AA ≥ +0.5, old-domain mean ≥ +1.0,
no domain < −0.5 vs equal-byte uniform; runtime within +10% (allocation is bank-time only).

## 3. Paper structure (target ~8 pp + appendix)

| Section | Content | Assets |
|---|---|---|
| Abstract/Intro | Replay budgets are spent uniformly; document streams have heterogeneous retention need; equal-byte allocation gains | Fig 1: response surface |
| Background | Compressed-replay lineage: Pellegrini → REMIND → ACAE → GFR → MRDC; consistency law (whole-doc unit) | Table 1: lineage vs compression unit |
| Method | The three dials + rules R1–R3; no-learned-router principle | Fig 2: allocator diagram |
| Experiments | Equal-byte gates, multi-seed/scenario/backbone; Pareto | Fig 3: AA-vs-bytes Pareto; Table 2: main |
| Analysis | Where bytes go (per-task allocation traces); failure ablations (each rule alone) | Fig 4: allocation heatmap |
| Related work | MRDC (raw-input, quality-only), MERS (selection-only), Latent-LoRA/FSE (isolation frontier) | — |
| Limitations | 2-point coverage law extrapolation; token-BIO only; frozen trunk | — |

## 4. Baselines, datasets, metrics

- **Baselines**: CoLaR-uniform (equal-byte, the honest control), ER-{matched bytes}, DER++,
  latent_replay (uncompressed ceiling), joint (oracle), naive (floor); currency: er_cflat.
- **Datasets/scenarios**: FUNSD/SROIE/CORD (dil + orders), CORD sessions (cil_cord),
  WildReceipt (dil_receipts robustness).
- **Metrics**: AA, BWT/AF, per-domain final row, old-domain micro-precision (the chain's
  false-positive guard), bytes, wall-time; optional stability-gap trace (memo #3) as an
  evaluation contribution.

## 5. Risks

1. **Response surface too flat between d5 and d50** → midpoint runs first (cheap); if
   retention saturates by ~d15, the interesting budget region is narrow → reframe as
   "how little replay suffices" (still a paper, weaker).
2. **Heterogeneity is scenario-specific** (CORD-never-forgets may be a dataset artifact) →
   dil orders + cil_cord in scope from the start.
3. **Rule gains < guard threshold** → the prereg kills it in one gate; falls back to memo #2
   (task-free) with the sweep infrastructure reused.
4. **Seed fragility** (as just observed for shadow-replay) → every claim 3-seed from T4 on;
   no single-seed headline.
5. Novelty collision window: MRDC/MERS teams extending to latent replay → cite-and-differ
   already drafted in wiki source pages; move T4 fast.

## 6. Output JSON (skill schema)

```json
{
  "research_question": "Under a fixed replay byte budget, does per-task allocation across coverage (docs), fidelity (SVD rank), and selection beat uniform allocation for compressed latent replay in continual document IE?",
  "methodology": "Equal-byte comparisons of rule-based (non-learned) allocators over CoLaR's per-document SVD replay; observational response-surface fit first, then preregistered gates; multi-seed x multi-scenario x multi-backbone",
  "paper_structure": {
    "sections": ["Abstract","Introduction","Background","Method","Experiments","Analysis","Related Work","Limitations","Conclusion"],
    "section_plans": {"Introduction": "uniform budgets vs heterogeneous retention need; d5/d50 coverage law as motivating measurement"}
  },
  "task_list": [
    {"task": "T1 response-surface fit from existing runs", "depends_on": [], "priority": 1},
    {"task": "T3 colar_adaptive impl + uniform-equivalence test", "depends_on": [], "priority": 1},
    {"task": "T2 finalize allocation rules R1-R3", "depends_on": ["T1"], "priority": 2},
    {"task": "T4 preregistered equal-byte gate (dil seed42)", "depends_on": ["T2","T3"], "priority": 2},
    {"task": "T5 multi-seed/scenario expansion", "depends_on": ["T4"], "priority": 3},
    {"task": "T6 LiLT backbone slice", "depends_on": ["T5"], "priority": 4},
    {"task": "T7 Pareto byte sweep (rented GPU)", "depends_on": ["T5"], "priority": 4},
    {"task": "T8 paper draft", "depends_on": ["T5"], "priority": 4}
  ],
  "baselines": ["CoLaR-uniform (equal bytes)","ER (matched bytes)","DER++","latent_replay (uncompressed)","joint","naive","er_cflat"],
  "datasets": ["FUNSD","SROIE","CORD","WildReceipt (dil, dil_o2/o3, cil_cord, dil_receipts)"],
  "evaluation_metrics": ["AA","BWT/AF","per-domain final F1","old-domain micro-precision","replay bytes","wall-time","optional per-iteration stability-gap trace"],
  "risks": ["flat response surface between d5/d50","dataset-specific heterogeneity","allocator gains below guard","seed fragility (observed in shadow-replay)","novelty window vs MRDC/MERS extensions"]
}
```
