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


def macro_ap(p: torch.Tensor, truth: torch.Tensor) -> float:
    aps = [
        average_precision_score((truth == j).numpy(), p[:, j].numpy())
        for j in range(1, p.shape[1])
        if (truth == j).any()
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
                }
            )
            for name, pred in rules(prob).items():
                pooled.setdefault(name, []).append((pred, prob["truth"]))
        table = {
            name: prf(torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs]))
            for name, pairs in pooled.items()
        }
        report[side] = {"boundaries": rows, "rules": table}
        print(f"\n=== {side} side ===")
        for r in rows:
            print("  ", r)
        print(f"   {'rule':<28}{'assigned':>9}{'hidden':>8}{'P':>7}{'R':>7}{'F1':>7}")
        for name, m in table.items():
            print(
                f"   {name:<28}{m['assigned']:>9}{m['hidden']:>8}"
                f"{m['P']:>7.1f}{m['R']:>7.1f}{m['F1']:>7.1f}"
            )
    (run / "ot_gate0_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {run / 'ot_gate0_report.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
