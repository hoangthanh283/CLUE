"""RCA-A2: retention-matrix dynamics across all saved runs.

Mines every ``results/<run>/matrix.npy`` (R[i,j] = F1 on task j after training task i).
Decomposes each task's forgetting into the IMMEDIATE one-boundary drop
(R[j+1,j] − R[j,j]) vs the GRADUAL later decay (R[T−1,j] − R[j+1,j]) and clusters
method families by that signature — a penalty method that leaks slowly and a naive run
that collapses at the first boundary have different root causes even at equal AF.

Also cross-checks AA recomputed from the matrix against metrics.json (artifact sanity).

CPU-only, writes results/rca/a2_*.csv + a2_signature.png.
Run from CLUE/:  uv run python scripts/rca_matrix_dynamics.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rca_per_class import FAMILY, SKIP_DIR_PREFIXES  # same run-dir conventions

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_all_matrices(results_dir: str | Path = "results") -> pd.DataFrame:
    """One row per run: metadata + the retention matrix (as object column)."""
    rows = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith(SKIP_DIR_PREFIXES):
            continue
        mat_p, met_p = run_dir / "matrix.npy", run_dir / "metrics.json"
        if not (mat_p.exists() and met_p.exists()):
            continue
        try:
            matrix = np.load(mat_p)
            metrics = json.loads(met_p.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as e:
            log.warning("skipping %s: %s", run_dir.name, e)
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
            continue
        method = metrics.get("method", "?")
        rows.append(
            {
                "run": run_dir.name,
                "method": method,
                "family": FAMILY.get(method, "other"),
                "scenario": metrics.get("scenario", "?"),
                "seed": metrics.get("seed", -1),
                "AA_reported": metrics.get("AA"),
                "matrix": matrix,
                "T": matrix.shape[0],
            }
        )
    return pd.DataFrame(rows)


def per_task_trajectory(runs: pd.DataFrame) -> pd.DataFrame:
    """Long rows: (run, task j, after_task i >= j, f1 = R[i,j])."""
    rows = []
    for r in runs.itertuples():
        for j in range(r.T):
            for i in range(j, r.T):
                f1 = float(r.matrix[i, j])
                if np.isnan(f1):
                    continue
                rows.append(
                    {
                        "run": r.run,
                        "method": r.method,
                        "family": r.family,
                        "scenario": r.scenario,
                        "seed": r.seed,
                        "task": j,
                        "after_task": i,
                        "f1": f1,
                    }
                )
    return pd.DataFrame(rows)


def immediate_vs_gradual(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (run, task j<T-1): immediate one-boundary drop vs later gradual decay."""
    rows = []
    for r in runs.itertuples():
        t = r.T
        for j in range(t - 1):
            peak, after_one, final = (
                float(r.matrix[j, j]),
                float(r.matrix[j + 1, j]),
                float(r.matrix[t - 1, j]),
            )
            if any(np.isnan(x) for x in (peak, after_one, final)):
                continue
            rows.append(
                {
                    "run": r.run,
                    "method": r.method,
                    "family": r.family,
                    "scenario": r.scenario,
                    "seed": r.seed,
                    "task": j,
                    "peak": peak,
                    "immediate_drop": after_one - peak,
                    "gradual_drop": final - after_one,
                    "total_drop": final - peak,
                }
            )
    return pd.DataFrame(rows)


def family_signature(drops: pd.DataFrame, scenario: str = "dil") -> pd.DataFrame:
    """Per method: mean immediate vs gradual drop and the immediate share of total."""
    d = drops[drops.scenario == scenario]
    g = (
        d.groupby(["method", "family"])[["immediate_drop", "gradual_drop", "total_drop"]]
        .mean()
        .reset_index()
    )
    denom = g["total_drop"].where(g["total_drop"].abs() > 1e-6)
    g["immediate_share"] = g["immediate_drop"] / denom
    return g.sort_values("total_drop")


def aa_crosscheck(runs: pd.DataFrame, tol: float = 0.1) -> pd.DataFrame:
    """AA recomputed from the matrix's final row must match metrics.json AA."""
    rows = []
    for r in runs.itertuples():
        final_row = r.matrix[r.T - 1]
        aa = float(np.nanmean(final_row))
        rows.append(
            {
                "run": r.run,
                "AA_from_matrix": aa,
                "AA_reported": r.AA_reported,
                "mismatch": r.AA_reported is not None and abs(aa - r.AA_reported) > tol,
            }
        )
    out = pd.DataFrame(rows)
    n_bad = int(out.mismatch.sum())
    if n_bad:
        log.warning("AA cross-check: %d/%d runs mismatch (>%.1f)", n_bad, len(out), tol)
    return out


def plot_signature(sig: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, g in sig.groupby("family"):
        ax.scatter(-g.immediate_drop, -g.gradual_drop, label=fam, s=60)
        for r in g.itertuples():
            ax.annotate(r.method, (-r.immediate_drop, -r.gradual_drop), fontsize=7)
    ax.set_xlabel("immediate drop at first boundary (F1 pts)")
    ax.set_ylabel("gradual later decay (F1 pts)")
    ax.set_title("Forgetting signature per method (dil, mean over tasks/seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path("results/rca")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_all_matrices()
    log.info("loaded %d matrices", len(runs))

    per_task_trajectory(runs).to_csv(out_dir / "a2_trajectories.csv", index=False)
    drops = immediate_vs_gradual(runs)
    drops.to_csv(out_dir / "a2_drop_decomposition.csv", index=False)
    sig = family_signature(drops)
    sig.to_csv(out_dir / "a2_family_signature.csv", index=False)
    aa_crosscheck(runs).to_csv(out_dir / "a2_aa_crosscheck.csv", index=False)
    plot_signature(sig, out_dir / "a2_signature.png")

    log.info("dil forgetting signatures:\n%s", sig.round(1).to_string(index=False))
    log.info("saved -> %s/a2_*.csv + a2_signature.png", out_dir)


if __name__ == "__main__":
    main()
