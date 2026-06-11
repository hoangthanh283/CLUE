# Forward Transfer (FWT): why it is unavailable, and how to enable it

## Definition

Forward transfer measures how much *previously acquired* knowledge helps a new task
*before* that task is trained:

```
FWT = (1 / (T-1)) * Σ_{i>0} ( R[i-1, i] - b_i )
```

(`doccl/eval/metrics.py`, `CLMetricsTracker.forward_transfer`), where

- `R[i-1, i]` = the multi-task model's **zero-shot** entity-F1 on task `i`, evaluated
  *after training task `i-1` but before training task `i`* (a strictly **upper**-triangular
  entry of the accuracy matrix `R`); and
- `b_i` = the **single-task naive baseline** F1 on task `i`'s dataset (a from-scratch model
  trained on that dataset alone). This is what `scenario single_funsd / single_cord /
  single_sroie` produce, each storing the baseline as `AA` with `matrix = [[f1]]`.

## Why true FWT is currently unavailable (not just zero)

Two independent reasons, both in `scripts/train.py`:

1. **The future-task term `R[i-1, i]` is never measured.** In the standard CL loop, after
   training task `task_idx` the model is evaluated only on `eval_loaders_seen`, which is
   populated incrementally (`eval_loaders_seen[task_idx] = eval_loader`) and therefore
   contains *only tasks `0..task_idx`*:

   ```python
   eval_loaders_seen[task_idx] = eval_loader          # only up to current task
   ...
   results = method.evaluate(eval_loaders_seen)        # seen tasks only
   tracker.update(task_idx, {tid: {"f1": r.f1} for tid, r in results.items()})
   ```

   The accuracy matrix is thus **strictly lower-triangular**: `R[i-1, i]` (a *future* task,
   column `i` > row `i-1`) stays `NaN`. There is no zero-shot evaluation of an unseen task.

2. **The tracker is never seeded with `baseline_perf`.** `CLMetricsTracker(num_tasks=...)`
   is constructed without `baseline_perf`, so `forward_transfer()` short-circuits to `0.0`.

Because the zero-shot term is `NaN` in every stored `metrics.json["matrix"]`, **true FWT
cannot be reconstructed offline.** Any "FWT" derived from the saved matrices would be a
different (weaker) quantity, not the metric defined above. `scripts/analyze_results.py`
therefore reports FWT as unavailable (`"--"`) and does **not** fabricate a value. A thesis
must not present a proxy as if it were the defined metric.

## What `analyze_results.py` *does* provide

It emits the **single-task baseline table** — `b_i` per dataset, mean ± std over seeds —
from the `single_*` runs:

- `results/table_single_task_baselines.tex` (citeable LaTeX, a single-task upper-reference)
- `results/table_single_task_baselines.csv` (machine-readable)

This is exactly the `b_i` term that true FWT subtracts, so it is both useful on its own and
ready to plug in once the zero-shot term is available. It degrades gracefully: if some
`single_*` runs are missing, only the available datasets are reported (no crash).

## How to enable true FWT later (requires a `train.py` change + re-run)

This change is **not** applied here (the grid is running live). To enable true FWT:

1. **Measure the zero-shot future-task term.** In the standard CL loop, *before* calling
   `method.train_task` for task `task_idx`, evaluate the current model on the *next* task's
   eval loader and record it into the upper triangle:

   ```python
   if task_idx > 0:
       prev_state_results = method.evaluate({task_idx: eval_loader})   # zero-shot on task i
       tracker.matrix[task_idx - 1, task_idx] = prev_state_results[task_idx].f1
   ```

   (Equivalently: after each task, also evaluate the *immediately following* unseen task.)

2. **Seed the tracker with the single-task baselines.** Load the `b_i` vector (this script's
   `table_single_task_baselines.csv`, mapped through `SCENARIO_TASK_DATASETS`) and pass it:

   ```python
   tracker = CLMetricsTracker(num_tasks=len(scenario.tasks), baseline_perf=baseline_vec)
   ```

   where `baseline_vec[i]` is the single-task F1 on task `i`'s dataset. The scenario→dataset
   mapping (`dil`, `cil_cord`, `mixed`) lives in `scripts/analyze_results.py`
   (`SCENARIO_TASK_DATASETS`) and mirrors `doccl/data/scenarios.py`.

With both in place, `tracker.forward_transfer()` returns the defined FWT, `save_run_metrics`
persists a non-`NaN` upper triangle, and `analyze_results.py` can aggregate a real FWT
column. Until then, FWT remains honestly reported as unavailable.
