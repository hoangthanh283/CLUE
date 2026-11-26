"""Gate R of docs/OT_COMPLETION_PREREG_2026-11.md: transport readout on saved test outputs.

Decodes the full_label_dump.pt of finished runs under a class prior. No model, no GPU.

    uv run python scripts/rca/ot_readout.py <gate0_run_dir> <results_root> <run_prefix> [...]

<gate0_run_dir> supplies the estimated class-frequency table (mass_probe_cur_t*.pt, last one).
Each <run_prefix> is matched as <results_root>/<run_prefix>_seed* (three runs).
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
from seqeval.metrics import f1_score

from doccl.methods.colar_otc import quota_assign

O_SHIFTS = torch.linspace(-12, 4, 33).tolist()  # rule 5, tuned on the truth


def potentials(
    logp: torch.Tensor, prior: torch.Tensor, lam: float = 1.0, relax: float = 1.0
) -> torch.Tensor:
    """Column potentials g of the entropic plan between tokens (mass 1) and classes
    (mass prior * N). Decoded label = argmax(lam * logp + g)."""
    kernel = lam * logp.double()
    log_mass = (prior.double() * logp.shape[0]).clamp_min(1e-300).log()
    g = torch.zeros(kernel.shape[1], dtype=torch.float64)
    for _ in range(200):
        f = -torch.logsumexp(kernel + g[None], dim=1)
        g = relax * (log_mass - torch.logsumexp(kernel + f[:, None], dim=0))
    return g


def estimated_prior(gate0_run: pathlib.Path, n_cls: int) -> torch.Tensor:
    freq = torch.load(sorted(gate0_run.glob("mass_probe_cur_t*.pt"))[-1], weights_only=False)[
        "class_freq"
    ]
    prior = torch.tensor([0.0] + [freq.get(c, 0.0) for c in range(1, n_cls)], dtype=torch.float64)
    prior[0] = max(1.0 - prior.sum().item(), prior[1:][prior[1:] > 0].min().item())
    return prior / prior.sum()


def span_f1(pred: torch.Tensor, gold: torch.Tensor, doc: torch.Tensor, i2l: dict) -> float:
    seqs = lambda x: [[i2l[int(t)] for t in x[doc == d]] for d in doc.unique()]  # noqa: E731
    return float(100 * f1_score(seqs(gold), seqs(pred)))


def decode_all(d: dict, est: torch.Tensor | None) -> dict[str, float]:
    gold, doc = d["gold"].long(), d["doc"].long()
    i2l = {int(k): v for k, v in d["id_to_label"].items()}
    logp = d["logits"].float().log_softmax(-1)
    p, n_cls = logp.exp(), logp.shape[1]
    f1 = lambda pred: span_f1(pred, gold, doc, i2l)  # noqa: E731
    out = {"argmax": f1(logp.argmax(-1)), "stored": float(d["full_label_f1"])}
    priors = {"oracle": torch.bincount(gold, minlength=n_cls).double() / len(gold)}
    if est is not None and len(est) == n_cls:
        priors["est"] = est
    folds = [doc % 2 == 0, doc % 2 == 1]
    for tag, prior in priors.items():
        out[f"transport[{tag}]"] = f1((logp.double() + potentials(logp, prior)).argmax(-1))
        out[f"transport[{tag},lam=10]"] = f1(
            (10 * logp.double() + potentials(logp, prior, lam=10.0)).argmax(-1)
        )
        out[f"transport[{tag},relax=0.5]"] = f1(
            (logp.double() + potentials(logp, prior, relax=0.5)).argmax(-1)
        )
        held = torch.zeros_like(gold)
        for fit, apply in (folds, folds[::-1]):
            held[apply] = (logp[apply].double() + potentials(logp[fit], prior)).argmax(-1)
        out[f"transport[{tag}] held-out potentials"] = f1(held)
        out[f"top-k[{tag}]"] = f1(quota_assign(p, prior.float() * len(gold)))
        adj = prior.clamp_min(1e-12).log() - p.double().mean(0).clamp_min(1e-12).log()
        out[f"one-step[{tag}]"] = f1((logp.double() + adj).argmax(-1))
    shifted = lambda s: (  # noqa: E731
        logp + torch.nn.functional.one_hot(torch.tensor(0), n_cls) * s
    ).argmax(-1)
    out["O-shift (tuned on truth)"] = max(f1(shifted(s)) for s in O_SHIFTS)
    return out


def main(gate0_run: str, root: str, prefixes: list[str]) -> None:
    out_file = pathlib.Path(gate0_run, "ot_readout_report.json")
    report = json.loads(out_file.read_text()) if out_file.exists() else {}  # merged over calls
    for prefix in prefixes:
        runs = sorted(pathlib.Path(root).glob(f"{prefix}_seed*/full_label_dump.pt"))
        rows = []
        for f in runs:
            d = torch.load(f, weights_only=False)
            if "logits" not in d:
                continue
            est = (
                estimated_prior(pathlib.Path(gate0_run), d["logits"].shape[1])
                if prefix.startswith("cil_cord")
                else None
            )
            rows.append(decode_all(d, est))
        if not rows:
            print(f"\n{prefix}: no shared-softmax dump")
            continue
        report[prefix] = rows
        print(f"\n=== {prefix} ({len(rows)} runs) ===")
        base = [float(r["argmax"]) for r in rows]
        for name in rows[0]:
            vals = [float(r[name]) for r in rows]
            gain = [v - b for v, b in zip(vals, base, strict=True)]
            print(
                f"   {name:<40}{sum(vals) / len(vals):>7.1f}   runs {[round(v, 1) for v in vals]}"
                f"   gain {[round(g, 1) for g in gain]}"
            )
    out_file.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
