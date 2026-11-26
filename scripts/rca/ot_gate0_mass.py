"""Gate 0 of docs/OT_COMPLETION_PREREG_2026-11.md: ranking and assigned mass, offline.

Reads the mass_probe_{cur,mem}_t*.pt dumps of a ``colar_otc_probe`` run and compares the
assignment rules at equal information. No model, no GPU.

    uv run python scripts/rca/ot_gate0_mass.py results/gates/ot_gate0/<run_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
from sklearn.metrics import average_precision_score

from doccl.methods.colar_otc import quota_assign, sinkhorn_assign, threshold_assign

TAUS = (0.5, 0.7, 0.9, 0.99)
RATIOS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)  # post hoc control, see budget/ratio below
MIN_SUPPORT = 10  # classes with fewer hidden tokens make the macro AP of a boundary noisy


def budget_assign(p: torch.Tensor, k: float) -> torch.Tensor:
    """Post hoc control (added after the first reading): one number instead of a class prior.
    The k tokens least likely to be background take their best class."""
    pred = torch.zeros(p.shape[0], dtype=torch.long)
    k = min(int(round(k)), p.shape[0])
    if k > 0:
        idx = (1 - p[:, 0]).topk(k).indices
        pred[idx] = p[idx, 1:].argmax(-1) + 1
    return pred


def ratio_assign(p: torch.Tensor, t: float) -> torch.Tensor:
    """Post hoc control: one scalar shift of the background score, no prior at all.
    A token takes its best class when that score is at least t times the background score."""
    best, cls = p[:, 1:].max(-1)
    return torch.where(best >= t * p[:, 0], cls + 1, torch.zeros_like(cls))


def side_problem(dump: dict, side: str) -> dict:
    """Candidate columns [O, hidden classes], truth in column space, both marginals."""
    probs = dump["probs"].float()
    lo, hi = (1, dump["n_old"]) if side == "cur" else (dump["n_prev"], dump["n_cur"])
    cols = torch.tensor([0, *range(lo, hi)])
    p = probs[:, cols]
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)  # renormalise over the candidate set
    full = dump["labels_full"].long()
    truth = torch.zeros_like(full)
    for j, c in enumerate(cols.tolist()):
        if c:
            truth[full == c] = j
    n = p.shape[0]
    oracle = torch.bincount(truth, minlength=len(cols)).float()
    est = torch.tensor(
        [0.0] + [dump["class_freq"].get(c, 0.0) * dump["n_valid"] for c in cols.tolist()[1:]]
    )
    est[0] = max(n - est.sum().item(), 0.0)
    return {"p": p, "truth": truth, "oracle": oracle, "est": est, "n": n}


def prf(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    hit = int(((pred == truth) & (truth != 0)).sum())
    n_pred, n_true = int((pred != 0).sum()), int((truth != 0).sum())
    prec, rec = hit / max(n_pred, 1), hit / max(n_true, 1)
    return {
        "assigned": n_pred,
        "hidden": n_true,
        "hit": hit,
        "P": 100 * prec,
        "R": 100 * rec,
        "F1": 100 * 2 * prec * rec / max(prec + rec, 1e-12),
    }


def macro_ap(p: torch.Tensor, truth: torch.Tensor, min_support: int = 1) -> float:
    aps = [
        average_precision_score((truth == j).numpy(), p[:, j].numpy())
        for j in range(1, p.shape[1])
        if (truth == j).sum() >= min_support
    ]
    return 100 * sum(aps) / max(len(aps), 1)


def rules(prob: dict) -> dict[str, torch.Tensor]:
    p, logp = prob["p"], prob["p"].clamp_min(1e-12).log()
    out = {f"threshold@{t}": threshold_assign(p, t) for t in TAUS}
    for tag in ("oracle", "est"):
        m = prob[tag]
        out[f"quota[{tag}]"] = quota_assign(p, m)
        out[f"sinkhorn[{tag}]"] = sinkhorn_assign(logp, m).argmax(-1)
    for lam in (3.0, 30.0):
        out[f"sinkhorn[est,lam={lam:g}]"] = sinkhorn_assign(logp, prob["est"], lam=lam).argmax(-1)
    for relax in (0.9, 0.5):
        out[f"sinkhorn[est,relax={relax}]"] = sinkhorn_assign(
            logp, prob["est"], col_relax=relax
        ).argmax(-1)
    for tag in ("oracle", "est"):  # post hoc: total entity budget only
        out[f"budget[{tag}]"] = budget_assign(p, prob[tag][1:].sum().item())
    for t in RATIOS:  # post hoc: scalar background shift
        out[f"ratio@{t:g}"] = ratio_assign(p, t)
    return out


def main(run_dir: str) -> None:
    run = pathlib.Path(run_dir)
    report: dict = {}
    for side in ("cur", "mem"):
        pooled: dict[str, list] = {}
        rows = []
        for f in sorted(run.glob(f"mass_probe_{side}_t*.pt")):
            dump = torch.load(f, weights_only=False)
            if dump.get("probs") is None:
                continue  # task 0 on the current side: frequency table only
            prob = side_problem(dump, side)
            ap = macro_ap(prob["p"], prob["truth"])
            est_total = float(prob["est"][1:].sum())
            rows.append(
                {
                    "boundary": dump["task_id"],
                    "O_tokens": prob["n"],
                    "hidden": int((prob["truth"] != 0).sum()),
                    "est_hidden": round(est_total),
                    "macro_AP": round(ap, 1),
                    f"macro_AP_support{MIN_SUPPORT}": round(
                        macro_ap(prob["p"], prob["truth"], MIN_SUPPORT), 1
                    ),
                }
            )
            for name, pred in rules(prob).items():
                pooled.setdefault(name, []).append((pred, prob["truth"]))
        table = {
            name: {
                **prf(torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])),
                "F1_by_boundary": [round(prf(a, b)["F1"], 1) for a, b in pairs],
            }
            for name, pairs in pooled.items()
        }
        report[side] = {"boundaries": rows, "rules": table}
        print(f"\n=== {side} side ===")
        for r in rows:
            print("  ", r)
        print(f"   {'rule':<28}{'assigned':>9}{'hidden':>8}{'P':>7}{'R':>7}{'F1':>7}  by boundary")
        for name, m in table.items():
            print(
                f"   {name:<28}{m['assigned']:>9}{m['hidden']:>8}"
                f"{m['P']:>7.1f}{m['R']:>7.1f}{m['F1']:>7.1f}  {m['F1_by_boundary']}"
            )
    (run / "ot_gate0_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {run / 'ot_gate0_report.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
