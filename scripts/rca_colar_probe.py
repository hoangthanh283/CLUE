"""RCA probes for CoLaR on ViT image CIL (prereg Amendment 7, Phase R).

Loads a run's ``final_model.pt`` + ``final_store.pt`` (written with method.dump_final=true)
and decomposes the final accuracy gap into readout vs representation loss:

  head          the trained CE head on current features (what the run reports)
  ncm_full      NCM, class means from the FULL train set, current trunk   (representation)
  ncm_stored    NCM, class means from the stored latents, current trunk   (memorisation)
  rp_full       RP-ridge fit on the FULL train set, current trunk         (best readout)
  rp_stored     RP-ridge fit on stored latents only, current trunk        (S1, stored)
  rp_stored_cur RP-ridge on stored latents + last task's full data        (S1 as run)
  simplecil     NCM on the FROZEN pre-trained trunk (full train means)    (reference)
  ranpac        RP-ridge on the FROZEN pre-trained trunk                  (reference)
  drift         mean cos(current, pre-trained) CLS feature of test images, per task

  uv run python scripts/rca_colar_probe.py --run results/<run_dir> --k 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.data.vision import VisionCILDataset, cifar100_bases, imagenet_r_bases
from doccl.methods.colar_pp import CoLaRPP
from doccl.methods.frozen_ptm import RanPAC, cls_features
from doccl.models.vit_wrapper import ViTWrapper


def per_task_acc(pred: torch.Tensor, y: torch.Tensor, per: int) -> list[float]:
    t = (y // per).cpu()
    ok = (pred.cpu() == y.cpu()).float()
    return [100 * ok[t == i].mean().item() for i in range(int(t.max()) + 1)]


def ncm(train_x, train_y, x, n_cls):
    means = torch.stack([train_x[train_y == c].mean(0) for c in range(n_cls)])
    return (F.normalize(x, dim=-1) @ F.normalize(means, dim=-1).T).argmax(-1)


def rp(train_x, train_y, x, model, weights=None):
    r = RanPAC(model, {"rp_dim": 10000, "rp_seed": 0})
    r.device = x.device
    r.fit(train_x, train_y, weights)
    return r._predict(x).argmax(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--scenario", default="cil_cifar100")
    ap.add_argument("--sessions", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_tr, base_te, split, n_cls = (
        cifar100_bases() if args.scenario == "cil_cifar100" else imagenet_r_bases()
    )
    tr_idx, te_idx = split if split is not None else (None, None)
    perm = np.random.RandomState(1993).permutation(n_cls).tolist()
    head_index = {int(c): i for i, c in enumerate(perm)}
    per = n_cls // args.sessions
    mk = lambda base, idx: DataLoader(  # noqa: E731
        VisionCILDataset(base, perm, head_index, train=False, allowed_idx=idx),
        batch_size=128,
        num_workers=4,
    )
    train_loader, test_loader = mk(base_tr, tr_idx), mk(base_te, te_idx)

    pre = ViTWrapper(num_labels=n_cls).to(dev).eval()
    cur = ViTWrapper(num_labels=n_cls)
    cur.load_state_dict(torch.load(args.run / "final_model.pt", map_location="cpu"))
    cur = cur.to(dev).eval()

    xtr_c, ytr = cls_features(cur, train_loader, dev)
    xte_c, yte = cls_features(cur, test_loader, dev)
    xtr_p, _ = cls_features(pre, train_loader, dev)
    xte_p, _ = cls_features(pre, test_loader, dev)

    m = CoLaRPP(cur, {"split_layer_k": args.k, "rank_r": 0})
    m.device = dev
    m._pixel_shape = (3, 224, 224)
    m.store = torch.load(args.run / "final_store.pt", map_location="cpu")
    xs, ys = m._stored_features()

    last = ytr // per == (n_cls // per - 1)
    xsc, ysc = torch.cat([xs, xtr_c[last]]), torch.cat([ys, ytr[last]])
    w = (torch.bincount(ysc).float().max() / torch.bincount(ysc).float())[ysc]

    with torch.no_grad():
        head_pred = cur.model.classifier(xte_c).argmax(-1)
    preds = {
        "head": head_pred,
        "ncm_full": ncm(xtr_c, ytr, xte_c, n_cls),
        "ncm_stored": ncm(xs, ys, xte_c, n_cls),
        "rp_full": rp(xtr_c, ytr, xte_c, cur),
        "rp_stored": rp(xs, ys, xte_c, cur),
        "rp_stored_cur": rp(xsc, ysc, xte_c, cur, w),
        "simplecil": ncm(xtr_p, ytr, xte_p, n_cls),
        "ranpac": rp(xtr_p, ytr, xte_p, pre),
    }
    out = {k: per_task_acc(v, yte, per) for k, v in preds.items()}
    cos = F.cosine_similarity(xte_c, xte_p)
    t = (yte // per).cpu()
    out["drift_cos"] = [cos.cpu()[t == i].mean().item() for i in range(n_cls // per)]
    summary = {k: float(np.mean(v)) for k, v in out.items()}
    (args.run / "probes.json").write_text(json.dumps({"per_task": out, "AA": summary}, indent=2))
    print("probe            AA     per-task")
    for k, v in out.items():
        print(f"{k:14s} {np.mean(v):6.2f}   {[round(x, 1) for x in v]}")
    print(
        f"\nreadout gap (rp_full - head)          = {summary['rp_full'] - summary['head']:+.2f}"
        f"\nrepresentation gap (ranpac - rp_full) = {summary['ranpac'] - summary['rp_full']:+.2f}"
        f"\nmemorisation (ncm_full - ncm_stored)  = {summary['ncm_full'] - summary['ncm_stored']:+.2f}"
    )


if __name__ == "__main__":
    main()
