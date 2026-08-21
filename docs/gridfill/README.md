# BROS/LiLT generality-grid fill (2026-08-21)

Closes the backbone gaps that bound the diagnostic paper's architecture-generality claim
(ROADMAP "B+ generalization grid"). Coverage before this work:

| backbone | dil | cil_cord | mixed |
|---|---|---|---|
| LayoutLMv3 | 6/6 | 6/6 | 6/6 |
| BERT | 6/6 | 6/6 | 6/6 |
| LiLT | 4/6 | **0/6** | 4/6 |
| BROS | **0/6** | **0/6** | **0/6** |

85 runs missing (core-6 methods × 3 scenarios × 3 seeds). Seeds were already complete
(42/7/123) in every populated cell.

## Files

- `local_jobs.txt` (67) — runnable on the 6 GB box. Format: `backbone scenario method seed`.
- `lwf_deferred_jobs.txt` (18) — **need >6 GB VRAM**, see below.
- `grid_fill.sh` — resume-safe runner (skips any run with an existing `metrics.json`,
  continues past failures, logs to `results/logs_gridfill_<run>.log`).

```bash
JOBS=docs/gridfill/local_jobs.txt bash docs/gridfill/grid_fill.sh
```

## Two protocol traps (both cost a restart; do not repeat)

1. **`EPOCHS_CAP=100` is mandatory.** Every method config declares `epochs: 10`, but the
   entire existing results table was produced by `run_grid_multigpu.sh`, which passes
   `method.epochs=${EPOCHS_CAP}` with a default of 100. Omitting it silently produces runs
   that are not comparable to the table. Verified real: `dil_joint_seed7_lilt` scored 84.27
   at the 10-epoch cap vs 85.12 at 100 (it early-stops at epoch 14).
2. **The 6 GB VRAM boundary has one rule.** A method that must hold a *second full-size,
   parameter-shaped tensor* alongside the live model fails on **LiLT at any batch size**;
   methods holding only activations or stored logits recover at bs=1. The recurring symptom is
   an identical **734 MiB** failed allocation — LiLT's 250k-row XLM-R embedding gradient buffer.

   | method | second tensor | verdict on LiLT |
   |---|---|---|
   | `lwf` | deepcopied frozen teacher (per task boundary) | **defer** — dies at bs=2 *and* bs=1, task 1, all scenarios |
   | `ewc` | Fisher diagonal over all params | **defer on `cil_cord` only** — 5 sessions + growing head; `dil`/`mixed` already completed fine |
   | `der_pp` | stored logits (not params) | local, bs=1; one seed needed a retry (marginal) |
   | `naive`/`joint`/`er` | none | local, bs=2 |

   BROS is expected to survive `ewc` because it uses BERT's ~30k WordPiece vocab (≈1/8 of
   LiLT's embedding table) — unverified at time of writing; the queue will settle it.
   **21 jobs deferred** to >6 GB hardware: 18 `lwf` + 3 LiLT `cil_cord` `ewc`
   (`lwf_deferred_jobs.txt`). Either rent, or disclose the gap in the reproducibility appendix —
   the rule above is a clean statement of the hardware boundary.

## Open finding for the methodology section

**The joint "oracle" is beaten by replay on LiLT dil, systematically across all three seeds:**
joint mean 83.79 (83.36 / 82.89 / 85.12) vs ER 85.65 and DER++ 84.35. Joint pools all domains
into one run and early-stops on the pooled val signal (seed 7 stopped at epoch 14 of 100),
while ER revisits each domain through its buffer. Since "joint upper bound" is load-bearing
language in the thesis, this needs either a protocol change (longer patience / pooled-val
split) or an explicit disclosure that joint is a fixed-protocol reference, not a true bound.
Worth checking whether the same inversion exists on LayoutLMv3 and BERT once the grid lands.

## Batch-size heterogeneity in the existing table (found 2026-08-22)

> Supersedes an earlier note here that called `mixed_der_pp_seed7_lilt` "marginal pressure,
> retry with gradient accumulation". That was wrong: the bs=1 retry failed identically
> (96 MiB, task 0 epoch 1). The real explanation is below — its siblings were never run at
> a batch size this box can reach.

`mixed_der_pp_seed7_lilt` cannot be filled locally: its two completed siblings
(`seed42`, `seed123`) were produced at **bs=16** on rented hardware, and the cell OOMs at
bs=1–2 on the 6 GB box. Deferred with the other >6 GB jobs (**22 total**).

Surveying the whole core-6 grid for recorded batch size:

| backbone | recorded batch sizes | reading |
|---|---|---|
| LayoutLMv3 | `None` (54 runs) | metadata absent — `training_hparams` logging was added in `c974159`; NOT evidence of variation |
| BERT | 2, plus `None` | same: the `None`s predate hparam logging |
| **LiLT** | **16 and 2, both explicitly recorded, in `dil` and `mixed`** | **genuine within-cell variation** |

Only the LiLT row is a real issue. Batch size affects optimization, so a bs=2 run does not
sit cleanly beside a bs=16 run of the same cell. Before the paper table is final, either
(a) re-run the LiLT bs=2 runs at bs=16 on adequate hardware, or (b) report the batch size
per cell and argue the within-cell method ordering is unaffected. Do not silently mix them.
