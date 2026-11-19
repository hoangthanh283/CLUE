"""Qualitative figure of the appendix: one CORD test receipt with its word boxes coloured by
whether the final model labels the word correctly, for CoLaR with a shared softmax and for
CoLaR-SBR (the two runs that saved their predictions on the unmasked test set).

The receipt is not picked by hand: it is the test receipt whose word accuracies on entity words
are closest to the test-set averages of the two models (15 to 45 words, at least 8 entity words).

Writes paper_cvpr/figures/qualitative.pdf and paper_cvpr/qual_vals.tex (the counts the caption
quotes). CPU only; loads the CORD test split (100 receipts).

  uv run python scripts/rca/make_qualitative_fig.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_analysis import blocked_pred  # noqa: E402

CLEAN = ROOT / "results" / "clean_2026-11"
RUNS = {  # panel label -> run directory of the prediction dump
    "CoLaR, shared softmax": CLEAN / "cil_cord_colar_dump_seed42_k4_d50_r128",
    "CoLaR-SBR": CLEAN / "cil_cord_colar_sbr_dump_seed42",
}
OUT = ROOT / "paper_cvpr"
RIGHT, WRONG, FALSE_POS = "#0072B2", "#D55E00", "#E69F00"  # Okabe-Ito: blue, vermillion, orange
MIN_WORDS, MAX_WORDS, MIN_ENTITY_WORDS = 15, 45, 8


def load(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(doc index, gold tag, predicted tag) of every labelled word of the unmasked test set."""
    d = torch.load(run / "full_label_dump.pt", map_location="cpu", weights_only=False)
    if "logits" in d:
        pred = d["logits"].float().argmax(-1)
    else:
        pred = blocked_pred(d["e"], d["tag"], d["threshold"])[0]
    return d["doc"].numpy(), d["gold"].long().numpy(), pred.long().numpy()


def pick(doc, gold, preds, n_words) -> int:
    """The receipt closest to the test-set average entity-word accuracy of both models."""
    ent = gold > 0
    avg = [float((p[ent] == gold[ent]).mean()) for p in preds]
    best, best_d = None, None
    for d in np.unique(doc):
        m = doc == d
        if not (MIN_WORDS <= m.sum() <= MAX_WORDS) or m.sum() != n_words[d]:
            continue
        e = m & ent
        if e.sum() < MIN_ENTITY_WORDS:
            continue
        dist = sum(
            abs(float((p[e] == gold[e]).mean()) - a) for p, a in zip(preds, avg, strict=True)
        )
        if best is None or dist < best:
            best, best_d = dist, int(d)
    assert best_d is not None, "no receipt satisfies the size constraints"
    return best_d


def main() -> None:
    from doccl.data.cord import CORDDataset

    dumps = {k: load(v) for k, v in RUNS.items()}
    docs = [v[0] for v in dumps.values()]
    golds = [v[1] for v in dumps.values()]
    assert all((d == docs[0]).all() for d in docs) and all((g == golds[0]).all() for g in golds)
    doc, gold = docs[0], golds[0]
    preds = [v[2] for v in dumps.values()]

    ds = CORDDataset(split="test", granularity="fine")
    n_words = {i: len(ex["tokens"]) for i, ex in enumerate(ds.data)}
    d = pick(doc, gold, preds, n_words)
    ex = ds.data[d]
    img = ds._ds[ex["row"]]["image"].convert("RGB")
    w, h = img.size
    boxes = np.array(ex["bboxes"], dtype=float) * np.array([w, h, w, h]) / 1000.0
    m = doc == d
    g = gold[m]
    pad = 0.03 * max(w, h)
    x0, y0 = max(boxes[:, 0].min() - pad, 0), max(boxes[:, 1].min() - pad, 0)
    x1, y1 = min(boxes[:, 2].max() + pad, w), min(boxes[:, 3].max() + pad, h)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 6.6 * 0.5 * (y1 - y0) / (x1 - x0) + 0.55))
    counts = {}
    for ax, (label, _), p_all, tag in zip(axes, RUNS.items(), preds, "ab", strict=True):
        p = p_all[m]
        ax.imshow(img)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")
        ok = wrong = fp = 0
        for (bx0, by0, bx1, by1), gi, pi in zip(boxes, g, p, strict=True):
            if gi > 0:
                col = RIGHT if pi == gi else WRONG
                ok += pi == gi
                wrong += pi != gi
            elif pi > 0:
                col, fp = FALSE_POS, fp + 1
            else:
                continue
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=True, fc=col, alpha=0.28,
                                   ec="none"))  # fmt: skip
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=col, lw=0.9))
        counts[tag] = (int(ok), int(ok + wrong), int(fp))
        ax.set_title(f"({tag}) {label}: {ok} of {ok + wrong} correct", fontsize=9,
                     pad=3)  # fmt: skip
    keys = [(RIGHT, "entity word, correct tag"), (WRONG, "entity word, wrong tag")]
    if any(c[2] for c in counts.values()):  # only when a panel shows such a word
        keys.append((FALSE_POS, "background word tagged as an entity"))
    fig.legend(
        [Patch(fc=c, alpha=0.5) for c, _ in keys], [t for _, t in keys],
        loc="lower center", ncol=3, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.2,
    )  # fmt: skip
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.07, wspace=0.03)
    fig.savefig(OUT / "figures" / "qualitative.pdf", dpi=220)
    plt.close(fig)

    ent = gold > 0
    acc = [100 * float((p[ent] == gold[ent]).mean()) for p in preds]
    (a_ok, n_ent, a_fp), (b_ok, _, b_fp) = counts["a"], counts["b"]
    v = "% AUTO-GENERATED by scripts/rca/make_qualitative_fig.py -- do not edit.\n"
    for k, x in (("qualEnt", n_ent), ("qualColarOk", a_ok), ("qualSbrOk", b_ok),
                 ("qualColarFp", a_fp), ("qualSbrFp", b_fp), ("qualWords", int(m.sum())),
                 ("qualColarAcc", f"{acc[0]:.0f}"), ("qualSbrAcc", f"{acc[1]:.0f}")):  # fmt: skip
        v += f"\\newcommand{{\\{k}}}{{{x}}}\n"
    (OUT / "qual_vals.tex").write_text(v)
    print(f"receipt {d}: {int(m.sum())} words, {n_ent} entity words; correct {a_ok} / {b_ok}; "
          f"false positives {a_fp} / {b_fp}; test-set entity-word accuracy {acc[0]:.1f} / {acc[1]:.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
