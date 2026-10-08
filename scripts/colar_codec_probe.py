"""A (Amendment 9): learned per-token codec vs per-document SVD, measured downstream.

Loads a document run's ``final_model.pt`` + ``final_store.pt`` (colar_pp, dump_final=true),
decodes every stored latent, trains a small MLP autoencoder (bottleneck c) on task-0 tokens,
and reports token-F1 of the frozen upper layers + head on (a) exact latents, (b) SVD r=16/32,
(c) AE c=16/32/64 reconstructions. Bytes/doc are reported alongside.

  uv run python scripts/colar_codec_probe.py --run results/dil_docD1_seed42...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from doccl.eval.metrics import compute_token_f1
from doccl.methods.colar_pp import CoLaRPP
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper


class AE(torch.nn.Module):
    def __init__(self, d: int, c: int):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, c)
        )
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(c, 256), torch.nn.GELU(), torch.nn.Linear(256, d)
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(x: torch.Tensor, c: int, epochs: int = 30) -> AE:
    ae = AE(x.shape[1], c).to(x.device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in torch.randperm(len(x)).split(1024):
            loss = F.mse_loss(ae(x[i]), x[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return ae.eval()


def svd_recon(h: torch.Tensor, r: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vh[:r]


@torch.no_grad()
def f1_through_model(m: CoLaRPP, docs: list[dict], hidden_fn, id_to_label) -> float:
    preds, golds = [], []
    for i in range(0, len(docs), 4):
        chunk = docs[i : i + 4]
        replay = m._stack_replay(chunk)
        h = replay["hidden"].float()
        replay["hidden"] = torch.stack([hidden_fn(x) for x in h]).to(torch.float16)
        out = m._replay_forward(
            {k: v for k, v in replay.items() if k != "labels"} | {"labels": replay["labels"]}
        )
        p = out.logits.argmax(-1).cpu()
        lab = replay["labels"]
        mask = lab != -100
        preds += p[mask].tolist()
        golds += lab[mask].tolist()
    return compute_token_f1(preds, golds, id_to_label)["f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = torch.load(args.run / "final_store.pt", map_location="cpu")
    meta = json.loads((args.run / "metrics.json").read_text())
    n_labels = max(int(d["labels"].max()) for d in store) + 1
    model = LayoutLMv3Wrapper(num_labels=n_labels)
    model.load_state_dict(torch.load(args.run / "final_model.pt", map_location="cpu"))
    model = model.to(dev).eval()
    model.id_to_label = {i: lbl for i, lbl in enumerate(meta.get("label_names", []))} or {
        i: str(i) for i in range(n_labels)
    }
    m = CoLaRPP(model, {"split_layer_k": args.k, "rank_r": 0})
    m.device = dev
    m._pixel_shape = (3, 224, 224)
    m.store = store

    n_tasks = 3
    per = len(store) // n_tasks
    dec = [m._decode(d).to(dev) for d in store]
    tok0 = torch.cat(
        [
            x[: d["n_text"]] if "n_text" in d else x
            for x, d in zip(dec[:per], store[:per], strict=True)
        ]
    )
    rows = []
    ident = lambda x: x  # noqa: E731
    rows.append(("exact", f1_through_model(m, store, ident, model.id_to_label), dec[0].numel() * 2))
    for r in (16, 32, 64):
        rows.append(
            (
                f"svd r={r}",
                f1_through_model(m, store, lambda x, r=r: svd_recon(x, r), model.id_to_label),
                (dec[0].shape[0] + dec[0].shape[1]) * r * 2,
            )
        )
    for c in (16, 32, 64):
        ae = train_ae(tok0, c)
        rows.append(
            (
                f"ae c={c}",
                f1_through_model(m, store, lambda x, ae=ae: ae(x), model.id_to_label),
                dec[0].shape[0] * c * 2,
            )
        )
    print(f"{'codec':10s} {'token F1':>9s} {'bytes/doc':>10s}")
    for name, f1, b in rows:
        print(f"{name:10s} {f1:9.2f} {b:10d}")
    (args.run / "codec_probe.json").write_text(
        json.dumps([{"codec": n, "f1": f, "bytes": b} for n, f, b in rows], indent=2)
    )


if __name__ == "__main__":
    main()
