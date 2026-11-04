"""Offline calibration check for CoLaR-SBR's block-selection rule (reads eval_dump.pt).

The deployed rule: the block with the largest entity mass e_b = 1 - p_b(O_b) claims a
token if e_b > tau (0.5), else O; then the corrected-scorer restriction. This script asks
how much AA a better rule could buy, on the final model's TEST scores:
  default   tau=0.5, no bias (must reproduce metrics.json — sanity check)
  tau*      best single global threshold
  oracle    per-block log-odds bias + tau by coordinate ascent ON TEST (an upper bound,
            not a deployable number; if it is small, calibration is not worth building)

  uv run python scripts/rca/sbr_calibrate.py results/gates/cil_protocol_2026-11/*sbr*/**/eval_dump.pt
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from doccl.eval.metrics import compute_eval_metrics

TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BIASES = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def predict(d: dict, bias: np.ndarray, tau: float) -> torch.Tensor:
    e = d["e"].float().clamp(1e-6, 1 - 1e-6)
    score = torch.logit(e) + torch.as_tensor(bias[: e.shape[1]], dtype=torch.float32)
    best_s, best_b = score.max(-1)
    tag = d["tag"].long().gather(-1, best_b.unsqueeze(-1)).squeeze(-1)
    preds = torch.where(torch.sigmoid(best_s) > tau, tag, torch.zeros_like(tag))
    if d["allowed"] is not None:
        keep = torch.zeros(d["n_head"], dtype=torch.bool)
        keep[[i for i in d["allowed"] if i < d["n_head"]]] = True
        preds = torch.where(keep[preds], preds, torch.zeros_like(preds))
    return preds


def aa(dump: dict, bias: np.ndarray, tau: float) -> float:
    f1s = []
    for d in dump.values():
        preds = predict(d, bias, tau)
        f1s.append(
            compute_eval_metrics(preds.tolist(), d["gold"].long().tolist(), d["id_to_label"])["f1"]
        )
    return float(np.mean(f1s))


def main(paths: list[str]) -> None:
    for path in paths:
        dump = torch.load(path, weights_only=False)
        nb = max(d["e"].shape[1] for d in dump.values())
        zero = np.zeros(nb)
        base = aa(dump, zero, 0.5)
        tau_aa = {t: aa(dump, zero, t) for t in TAUS}
        t_best = max(tau_aa, key=tau_aa.get)
        bias, tau, best = zero.copy(), t_best, tau_aa[t_best]
        for _ in range(2):  # coordinate ascent: each block's bias, then tau
            for b in range(nb):
                for v in BIASES:
                    trial = bias.copy()
                    trial[b] = v
                    s = aa(dump, trial, tau)
                    if s > best + 1e-9:
                        best, bias = s, trial
            for t in TAUS:
                s = aa(dump, bias, t)
                if s > best + 1e-9:
                    best, tau = s, t
        print(
            f"{path}\n  default AA {base:.1f} | best tau {t_best} AA {tau_aa[t_best]:.1f} | "
            f"oracle AA {best:.1f} (bias {np.round(bias, 1).tolist()}, tau {tau})"
        )
        print("  tau sweep: " + " ".join(f"{t}:{v:.1f}" for t, v in tau_aa.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
