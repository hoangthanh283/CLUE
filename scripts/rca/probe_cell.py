"""Pass 2b — ridge decomposition for one CoLaR cell.

Supports image (ViT, cifar/imagenet-r) and document (LayoutLMv3, cil_cord/mixed/dil) runs.
Needs final_model.pt + final_store.pt in the cell dir (dump_final=true).

Produces results/<run>/probes.json with:
  head         — current plastic head on test features
  rp_full      — RP-ridge solved on all seen train data through CURRENT trunk
  rp_stored    — RP-ridge on 50/class (or token) stored latents only
  ncm_full     — NCM full-data class means, current trunk
  ncm_stored   — NCM, stored latents, current trunk
  ranpac       — RP-ridge on FROZEN pre-trained trunk (image only)
  simplecil    — NCM on frozen trunk (image only)

Then derives:
  head_gap     = rp_full - head               (coverage + statistics headroom)
  trunk_gap    = rp_full(cur) - rp_full(pre)  (image only, else NaN)
  coverage_gap = rp_full - rp_stored          (what fresh features of all docs would buy)
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def ridge(X, Y, lam=1.0):
    """Solve (X^T X + lam I) B = X^T Y; returns (D, C) weight matrix."""
    X = X.float(); Y = Y.float()
    D = X.shape[1]
    A = X.T @ X + lam * torch.eye(D, device=X.device, dtype=X.dtype)
    B = X.T @ Y
    return torch.linalg.solve(A, B)


def rp_project(X, W, mode="relu", bs=2048):
    """Chunked X @ W + ReLU to keep peak VRAM small (CIFAR N=50k × M=10k = ~2 GB)."""
    outs = []
    for i in range(0, X.shape[0], bs):
        Z = X[i:i+bs].float() @ W
        outs.append(F.relu(Z) if mode == "relu" else Z)
    return torch.cat(outs, 0)


def onehot(y, C):
    O = torch.zeros(y.shape[0], C, device=y.device)
    O.scatter_(1, y.view(-1, 1), 1.0)
    return O


def ridge_eval(Xtr, ytr, Xte, yte, C, M=10000, lam=10.0, seed=42):
    # Build RP matrix on CPU first, then slice to each stage's device
    g = torch.Generator().manual_seed(seed)
    W_cpu = torch.randn(Xtr.shape[1], M, generator=g) / math.sqrt(Xtr.shape[1])
    W = W_cpu.to(Xtr.device)
    # Project train + eval in chunks; accumulate Gram on-device
    D = M
    A = lam * torch.eye(D, device=Xtr.device)
    Yh = onehot(ytr, C)
    rhs = torch.zeros(D, C, device=Xtr.device)
    for i in range(0, Xtr.shape[0], 2048):
        Z = rp_project(Xtr[i:i+2048], W, bs=2048)
        A = A + Z.T @ Z
        rhs = rhs + Z.T @ Yh[i:i+2048]
        del Z
    B = torch.linalg.solve(A, rhs)
    # Eval in chunks
    preds = []
    for i in range(0, Xte.shape[0], 2048):
        Z = rp_project(Xte[i:i+2048], W, bs=2048)
        preds.append((Z @ B).argmax(-1))
        del Z
    pred = torch.cat(preds, 0)
    del A, rhs, B, W
    torch.cuda.empty_cache() if Xtr.is_cuda else None
    return (pred == yte).float().mean().item() * 100


def ncm_eval(Xtr, ytr, Xte, yte, C):
    mu = torch.zeros(C, Xtr.shape[1], device=Xtr.device)
    cnt = torch.zeros(C, device=Xtr.device)
    for c in range(C):
        mask = ytr == c
        if mask.any():
            mu[c] = Xtr[mask].mean(0); cnt[c] = 1
    valid = cnt > 0
    d = torch.cdist(F.normalize(Xte, dim=1), F.normalize(mu[valid], dim=1))
    pred = valid.nonzero().squeeze(-1)[d.argmin(-1)]
    return (pred == yte).float().mean().item() * 100


def probe_image(run: Path, scenario: str, k: int, sessions: int):
    from doccl.data.vision import VisionCILDataset, cifar100_bases, imagenet_r_bases
    from doccl.methods.colar_pp import CoLaRPP
    from doccl.methods.frozen_ptm import cls_features
    from doccl.models.vit_wrapper import ViTWrapper

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_tr, base_te, split, n_cls = (
        cifar100_bases() if scenario == "cil_cifar100" else imagenet_r_bases()
    )
    tr_idx, te_idx = (split or (None, None))
    perm = np.random.RandomState(1993).permutation(n_cls).tolist()
    head_index = {int(c): i for i, c in enumerate(perm)}
    mk = lambda base, idx: DataLoader(
        VisionCILDataset(base, perm, head_index, train=False, allowed_idx=idx),
        batch_size=32, num_workers=2)
    train_loader, test_loader = mk(base_tr, tr_idx), mk(base_te, te_idx)

    pre = ViTWrapper(num_labels=n_cls).to(dev).eval()
    cur = ViTWrapper(num_labels=n_cls)
    cur.load_state_dict(torch.load(run / "final_model.pt", map_location="cpu"))
    cur = cur.to(dev).eval()

    xtr_c, ytr = cls_features(cur, train_loader, dev)
    xte_c, yte = cls_features(cur, test_loader, dev)
    xtr_p, _ = cls_features(pre, train_loader, dev)
    xte_p, _ = cls_features(pre, test_loader, dev)

    m = CoLaRPP(cur, {"split_layer_k": k, "rank_r": 0})
    m.device = dev; m._pixel_shape = (3, 224, 224)
    m.store = torch.load(run / "final_store.pt", map_location="cpu")
    xs, ys = m._stored_features()

    with torch.no_grad():
        head_pred = cur.model.classifier(xte_c).argmax(-1)
    head_acc = (head_pred == yte).float().mean().item() * 100
    rp_full_cur = ridge_eval(xtr_c, ytr, xte_c, yte, n_cls)
    rp_stored   = ridge_eval(xs.to(dev), ys.to(dev), xte_c, yte, n_cls)
    rp_full_pre = ridge_eval(xtr_p, ytr, xte_p, yte, n_cls)
    ncm_full    = ncm_eval(xtr_c, ytr, xte_c, yte, n_cls)
    ncm_stored  = ncm_eval(xs.to(dev), ys.to(dev), xte_c, yte, n_cls)
    ncm_pre     = ncm_eval(xtr_p, ytr, xte_p, yte, n_cls)
    return dict(head=head_acc, rp_full=rp_full_cur, rp_stored=rp_stored,
                rp_full_frozen=rp_full_pre, simplecil=ncm_pre,
                ncm_full=ncm_full, ncm_stored=ncm_stored,
                head_gap=rp_full_cur - head_acc,
                trunk_gap=rp_full_cur - rp_full_pre,
                coverage_gap=rp_full_cur - rp_stored,
                scenario=scenario, modality="image")


def probe_doc(run: Path, scenario: str, k: int):
    from doccl.data.scenarios import SCENARIO_REGISTRY
    from doccl.data.encoders import build_encoder
    from doccl.methods.colar_pp import CoLaRPP
    from doccl.models.layoutlmv3_wrapper import LayoutLMv3Wrapper
    from omegaconf import OmegaConf

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Need encoder before scenario build (per CLAUDE.md)
    cfg_model = OmegaConf.create({"family": "layoutlmv3",
                                   "pretrained": "microsoft/layoutlmv3-base"})
    build_encoder(cfg_model)
    scen = SCENARIO_REGISTRY[scenario]()
    # Determine total label count from final head
    sd = torch.load(run / "final_model.pt", map_location="cpu")
    head_w = sd.get("model.classifier.weight") or sd.get("classifier.weight")
    n_cls = int(head_w.shape[0])
    cur = LayoutLMv3Wrapper(num_labels=n_cls); cur.load_state_dict(sd); cur = cur.to(dev).eval()

    # Build stored features
    m = CoLaRPP(cur, {"split_layer_k": k, "rank_r": 0})
    m.device = dev
    m.store = torch.load(run / "final_store.pt", map_location="cpu")
    xs, ys = m._stored_features()
    xs = xs.to(dev); ys = ys.to(dev)

    # Build FULL per-token features of all seen tasks through current trunk
    # Reuse the method's _features_of on the full train streams.
    all_feats, all_labs = [], []
    for t in scen.tasks:
        ds = t.train_dataset
        # Build minimal doc list for _features_of; subsample up to 400 docs per task
        docs = []
        for i in range(min(len(ds), 400)):
            d = ds[i]; docs.append({k: v for k, v in d.items()})
        xf, yf = m._features_of(docs)
        all_feats.append(xf.to(dev)); all_labs.append(yf.to(dev))
    xtr_full = torch.cat(all_feats); ytr_full = torch.cat(all_labs)

    # Eval set: union of all task test sets, up to 100 docs per task
    xte_list, yte_list = [], []
    for t in scen.tasks:
        ds = t.eval_dataset
        docs = []
        for i in range(min(len(ds), 100)):
            d = ds[i]; docs.append({k: v for k, v in d.items()})
        xe, ye = m._features_of(docs)
        xte_list.append(xe.to(dev)); yte_list.append(ye.to(dev))
    xte = torch.cat(xte_list); yte = torch.cat(yte_list)

    with torch.no_grad():
        head_pred = cur.model.classifier(xte).argmax(-1)
    head_acc = (head_pred == yte).float().mean().item() * 100
    rp_full_cur = ridge_eval(xtr_full, ytr_full, xte, yte, n_cls, M=4096, lam=10.0)
    rp_stored   = ridge_eval(xs, ys, xte, yte, n_cls, M=4096, lam=10.0)
    ncm_full    = ncm_eval(xtr_full, ytr_full, xte, yte, n_cls)
    ncm_stored  = ncm_eval(xs, ys, xte, yte, n_cls)
    return dict(head=head_acc, rp_full=rp_full_cur, rp_stored=rp_stored,
                rp_full_frozen=None, simplecil=None,
                ncm_full=ncm_full, ncm_stored=ncm_stored,
                head_gap=rp_full_cur - head_acc,
                trunk_gap=None,
                coverage_gap=rp_full_cur - rp_stored,
                scenario=scenario, modality="document",
                n_classes=n_cls,
                n_stored=int(ys.numel()), n_full_train=int(ytr_full.numel()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=10)
    args = ap.parse_args()

    if args.scenario.startswith("cil_cifar") or args.scenario.startswith("cil_imagenet"):
        out = probe_image(args.run, args.scenario, args.k, args.sessions)
    else:
        out = probe_doc(args.run, args.scenario, args.k)

    (args.run / "probes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v
                      for k, v in out.items()}, indent=2))
