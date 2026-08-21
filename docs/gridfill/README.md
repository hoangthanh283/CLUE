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
2. **`lwf` cannot run on LiLT/BROS at 6 GB, at any batch size.** It deepcopies the model into
   a frozen teacher at each task boundary, so from task 1 onward two full backbones plus
   optimizer state coexist (~1.1 GB of parameters each for LiLT's 250k XLM-R vocab). Both
   bs=2 and bs=1 died on the identical 734 MiB allocation, at the same point (task 1, after
   task 0 completed). This is a capacity ceiling, not a tuning knob — run these 18 on rented
   hardware, or disclose the gap. `der_pp` does NOT deepcopy (it replays stored logits) and
   runs fine.

## Open finding for the methodology section

**The joint "oracle" is beaten by replay on LiLT dil, systematically across all three seeds:**
joint mean 83.79 (83.36 / 82.89 / 85.12) vs ER 85.65 and DER++ 84.35. Joint pools all domains
into one run and early-stops on the pooled val signal (seed 7 stopped at epoch 14 of 100),
while ER revisits each domain through its buffer. Since "joint upper bound" is load-bearing
language in the thesis, this needs either a protocol change (longer patience / pooled-val
split) or an explicit disclosure that joint is a fixed-protocol reference, not a true bound.
Worth checking whether the same inversion exists on LayoutLMv3 and BERT once the grid lands.
