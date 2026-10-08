"""Report workbook for the Google "Reports" sheet: forgetting diagnosis, approaches, metrics,
plus the two image-CL per-scenario sheets (from report_image_xlsx). Every sheet carries charts;
import into Sheets via File > Import > Upload > "Insert new sheet(s)".

    uv run --with openpyxl python scripts/report_addendum_xlsx.py [--out results/report_addendum.xlsx]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).parent))
from report_image_xlsx import SCENARIOS, load_runs, write_scenario_sheet  # noqa: E402

from doccl.pilot.analyze import load_pilot_results, to_long_dataframe  # noqa: E402

COND = (
    {  # pilot condition -> label (naive sequential; docs FUNSD→CORD→SROIE, images Split CIFAR-100)
        "c4_full": "LayoutLMv3 full (text+layout+vision)",
        "c3_no_image": "LayoutLMv3 text+layout",
        "c2_no_text": "LayoutLMv3 layout+vision",
        "c1_text": "LayoutLMv3 text-only",
        "cb_bert": "BERT (text-only)",
        "cl_lilt": "LiLT",
        "cr_bros": "BROS",
        "cv_vit_fast": "ViT-B/16 CIFAR-100 (fast, AdamW)",
        "cv_vit_slow": "ViT-B/16 CIFAR-100 (slow trunk)",
    }
)
DEPTHS = ["input", "early", "mid", "late", "head"]
CKA_DEPTH = {
    "embeddings": "embeddings",
    "layer.0": "L0",
    "layer.5": "L6",
    "layer.6": "L6",
    "layer.11": "L11",
    "classifier": "head",
}
BOLD = Font(bold=True)


def _bar(ws, title, cats, data, anchor, y_title="", log=False, stacked=False, width=26, height=9):
    ch = BarChart()
    ch.type, ch.title, ch.y_axis.title = "col", title, y_title
    if stacked:
        ch.grouping, ch.overlap = "percentStacked", 100
    if log:
        ch.y_axis.scaling.logBase = 10
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.width, ch.height = width, height
    ws.add_chart(ch, anchor)


def _line(ws, title, cats, data, anchor, y_title="", x_title="", width=26, height=10):
    ch = LineChart()
    ch.title, ch.y_axis.title, ch.x_axis.title = title, y_title, x_title
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.width, ch.height = width, height
    ws.add_chart(ch, anchor)


def _title(ws, text):
    ws.append([text])
    ws.cell(row=ws.max_row, column=1).font = BOLD


def _header(ws, cols):
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        ws.cell(row=ws.max_row, column=c).font = BOLD


def _rows(ws, df):
    for r in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in r])


def pilot_tables():
    results = [
        r
        for r in load_pilot_results(Path("results/pilot"))
        if sorted(r["task_order"]) == r["task_order"]
    ]
    df = to_long_dataframe(results)
    df = df[df.condition.isin(COND)]
    # Forgetting metrics from the per-run F1 matrix (AA = last row mean; BWT = mean R[T,i]-R[i,i]).
    f1 = df[df.metric == "f1"]
    rows = []
    for (c, s), g in f1.groupby(["condition", "seed"]):
        M = g.pivot(index="task_idx", columns="evaluated_task", values="value").to_numpy()
        T = M.shape[0]
        rows.append(
            dict(
                condition=c,
                seed=s,
                AA=np.nanmean(M[T - 1, :T]),
                BWT=np.mean([M[T - 1, i] - M[i, i] for i in range(T - 1)]),
            )
        )
    forget = pd.DataFrame(rows).groupby("condition")[["AA", "BWT"]].mean()
    # Fisher-weighted displacement summed over boundaries, per depth bucket; share in %.
    dd = (
        df[df.metric == "displacement_depth"]
        .groupby(["condition", "seed", "group"])
        .value.sum()
        .reset_index()
    )
    dd["pct"] = 100 * dd.value / dd.groupby(["condition", "seed"]).value.transform("sum")
    share = dd.groupby(["condition", "group"]).pct.mean().unstack()[DEPTHS]
    absd = dd.groupby(["condition", "group"]).value.mean().unstack()[DEPTHS]
    n_seeds = dd.groupby("condition").seed.nunique()
    # CKA by canonical depth, mean over seeds and boundaries (ViT embeddings CKA is undefined).
    ck = df[df.metric == "cka"].copy()
    ck["depth"] = ck.layer.map(
        lambda L: next((v for k, v in CKA_DEPTH.items() if L.endswith(k)), None)
    )
    ck = ck.dropna(subset=["depth"])
    ck = ck[~(ck.condition.str.startswith("cv_") & (ck.depth == "embeddings"))]
    cka = (
        ck.groupby(["condition", "depth"])
        .value.mean()
        .unstack()[["embeddings", "L0", "L6", "L11", "head"]]
    )
    order = [c for c in COND if c in share.index]
    return (
        forget.reindex(order),
        share.reindex(order),
        absd.reindex(order),
        n_seeds.reindex(order),
        cka.reindex(order),
    )


def write_diagnosis(wb):
    ws = wb.create_sheet("Diagnosis (forgetting locus)")
    forget, share, absd, n, cka = pilot_tables()
    _title(
        ws,
        "Where forgetting lives — naive sequential fine-tuning, no CL method (pilot study, mean over seeds)",
    )
    ws.append(
        [
            "Documents: FUNSD → CORD → SROIE (3 tasks). Images: Split CIFAR-100 (10 tasks, ViT-B/16 IN-21k). "
            "Displacement = old-task-Fisher-weighted parameter displacement across task boundaries, bucketed by depth."
        ]
    )
    ws.append([])
    _header(
        ws,
        [
            "condition",
            "n_seeds",
            "AA",
            "BWT",
            "share input %",
            "share early %",
            "share mid %",
            "share late %",
            "share head %",
            "disp input",
            "disp early",
            "disp mid",
            "disp late",
            "disp head",
            "head / max non-head",
        ],
    )
    t1_first = ws.max_row + 1
    for c in share.index:
        s, a = share.loc[c], absd.loc[c]
        ws.append(
            [
                COND[c],
                int(n[c]),
                round(forget.loc[c, "AA"], 1),
                round(forget.loc[c, "BWT"], 1),
                *[round(float(v), 2) for v in s],
                *[float(v) for v in a],
                round(float(a["head"] / a[["input", "early", "mid", "late"]].max()), 1),
            ]
        )
    t1_last = ws.max_row
    cats = Reference(ws, min_col=1, min_row=t1_first, max_row=t1_last)
    _bar(
        ws,
        "Displacement share by depth (100 % stacked) — head dominates on every backbone",
        cats,
        Reference(ws, min_col=5, max_col=9, min_row=t1_first - 1, max_row=t1_last),
        "Q4",
        "% of displacement",
        stacked=True,
    )
    _bar(
        ws,
        "Fisher-weighted displacement by depth (log scale)",
        cats,
        Reference(ws, min_col=10, max_col=14, min_row=t1_first - 1, max_row=t1_last),
        "Q24",
        "displacement",
        log=True,
    )

    ws.append([])
    ws.append([])
    _title(
        ws,
        "Representation drift — linear CKA between consecutive task checkpoints, by depth (1 = unchanged)",
    )
    _header(ws, ["condition", "embeddings", "L0", "L6", "L11", "head"])
    t2_first = ws.max_row + 1
    for c in cka.index:
        ws.append([COND[c], *[None if np.isnan(v) else round(float(v), 3) for v in cka.loc[c]]])
    t2_last = ws.max_row
    # Transposed block so the line chart runs over depth (x) with one series per condition.
    ws.append([])
    _header(ws, ["depth", *[COND[c] for c in cka.index]])
    t3_first = ws.max_row
    for d in cka.columns:
        ws.append(
            [
                d,
                *[
                    None if np.isnan(cka.loc[c, d]) else round(float(cka.loc[c, d]), 3)
                    for c in cka.index
                ],
            ]
        )
    t3_last = ws.max_row
    _line(
        ws,
        "CKA drift increases monotonically with depth",
        Reference(ws, min_col=1, min_row=t3_first + 1, max_row=t3_last),
        Reference(ws, min_col=2, max_col=1 + len(cka.index), min_row=t3_first, max_row=t3_last),
        "Q44",
        "CKA",
        "depth",
    )

    ws.append([])
    ws.append([])
    _title(
        ws,
        "Protecting the locus relocates forgetting (DIL, LayoutLMv3; freeze arms seed 42, consolidation probe 3 seeds)",
    )
    _header(ws, ["arm", "AA", "BWT", "reading"])
    t4_first = ws.max_row + 1
    for r in [
        ("naive (nothing frozen)", 41.3, -73.1, "reference"),
        ("freeze all (no learning)", 41.1, 0.0, "no forgetting, no acquisition"),
        ("freeze head + late layers", 38.8, -76.8, "drift migrates to early/mid (CKA 1.00 → 0.19)"),
        ("no-memory control", 38.9, -75.6, "same as freeze: memory, not freezing, is what matters"),
        ("consolidation penalty: uniform over depth", 42.6, None, "cannot fit the tasks"),
        ("consolidation penalty: head + late", 84.7, None, "near-oracle"),
        ("consolidation penalty: head only", 86.1, None, "near-oracle"),
        ("consolidation penalty: late only", 86.2, None, "near-oracle"),
        ("joint (oracle)", 88.7, 0.0, "upper bound"),
    ]:
        ws.append(list(r))
    t4_last = ws.max_row
    _bar(
        ws,
        "Depth-targeted protection: where a fixed budget attaches decides whether it is usable",
        Reference(ws, min_col=1, min_row=t4_first, max_row=t4_last),
        Reference(ws, min_col=2, min_row=t4_first - 1, max_row=t4_last),
        "Q66",
        "AA",
    )
    ws.column_dimensions["A"].width = 42


FAMILY = {
    "naive": "bound",
    "joint": "bound",
    "ewc": "regularisation",
    "lwf": "distillation",
    "er": "replay (raw samples)",
    "der_pp": "replay (raw samples + logits)",
    "er_cflat": "replay + SAM (C-Flat++)",
    "er_b2000": "replay (raw images, 2000)",
    "der_pp_b2000": "replay (raw images + logits, 2000)",
    "er_b2000amp": "replay (raw images, 2000, AMP)",
    "l2p": "prompt pool",
    "dualprompt": "prompt pool",
    "coda_prompt": "prompt pool",
    "o_lora": "LoRA isolation",
    "cl_lora": "LoRA isolation",
    "sd_lora": "LoRA isolation",
    "latent_replay": "latent replay (raw bank)",
    "colar": "latent replay (CoLaR, per-doc SVD)",
    "colar_adaptive": "latent replay (CA-CoLaR)",
    "lexslot": "lexical slot memory (+200-doc buffer)",
    "doccl": "DocCL (diagnosis-guided consolidation)",
    "lca": "weight merging + realignment",
    "slca": "slow trunk + Gaussian head alignment",
    "slca_noca": "slow trunk only",
    "slca_noca_pub": "slow trunk only (published lr)",
    "ranpac": "frozen trunk + RP ridge head",
    "simplecil": "frozen trunk + prototypes",
    "bank500": "latent replay (raw bank, k=4)",
    "aglr_replay": "per-class Gaussian replay",
    "hrp": "hybrid routed prompt",
    "magmax": "weight merging",
    "cpfd": "NER-specific",
    "is3": "NER-specific",
}
BUFFER = {
    "naive": "none",
    "joint": "all data",
    "ewc": "none",
    "lwf": "none",
    "er": "raw",
    "der_pp": "raw + logits",
    "er_cflat": "raw",
    "er_b2000": "raw (2000)",
    "der_pp_b2000": "raw (2000)",
    "er_b2000amp": "raw (2000)",
    "l2p": "none",
    "dualprompt": "none",
    "coda_prompt": "none",
    "o_lora": "none",
    "cl_lora": "none",
    "sd_lora": "none",
    "latent_replay": "layer-k latents",
    "colar": "SVD latents",
    "colar_adaptive": "SVD latents",
    "lexslot": "raw (200 docs)",
    "doccl": "none",
    "lca": "none",
    "slca": "class Gaussians",
    "slca_noca": "none",
    "slca_noca_pub": "none",
    "ranpac": "Gram statistics",
    "simplecil": "prototypes",
    "bank500": "layer-4 latents",
    "aglr_replay": "class Gaussians",
}
DOC_SCENARIOS = [
    ("dil", "DIL — FUNSD → SROIE → CORD (fixed 9-tag schema)"),
    ("cil_cord", "CIL-CORD — 5 sessions × 6 classes (growing head)"),
    ("cil_wildreceipt", "CIL-WildReceipt — 4 sessions"),
    ("mixed", "Mixed — 6 sessions alternating class/domain shifts"),
    ("dil_xlingual", "DIL cross-lingual — XFUND de → es → fr → it → zh"),
]
IMG_SCENARIOS = [
    ("cil_cifar100", "Split CIFAR-100 — 10 × 10 classes (ViT-B/16 IN-21k)"),
    ("cil_imagenet_r", "Split ImageNet-R — 10 × 20 classes (ViT-B/16 IN-21k)"),
]
LADDER = [  # single-seed diagnostics and falsified lines from docs/RESULTS_REPORT_2026-10.md §4–5
    (
        "CoLaR r=128, k=4, 50 docs/task",
        "latent replay (CoLaR)",
        "SVD latents, 60 MB",
        87.6,
        -1.7,
        "lossless at 2.7× vs raw latent bank",
    ),
    (
        "raw latent replay, k=4, 50 docs/task",
        "latent replay (raw bank)",
        "latents, ≈163 MB",
        87.3,
        -2.2,
        "whole-document latents retain",
    ),
    ("CoLaR r=64", "latent replay (CoLaR)", "SVD latents, 32 MB", 80.6, -12.3, "rank floor"),
    (
        "latent replay, k=8, 5 docs/task (3 seeds)",
        "latent replay (raw bank)",
        "latents",
        66.5,
        -30.5,
        "small-buffer point",
    ),
    (
        "LexMem v3b (feature Gaussians)",
        "feature-Gaussian replay",
        "class statistics",
        66.0,
        None,
        "partial; dense-label tasks only",
    ),
    (
        "LexMem v5 (+relational)",
        "feature-Gaussian replay",
        "class statistics",
        63.2,
        None,
        "marginal summary breaks whole-doc binding",
    ),
    (
        "PLaR public-proxy replay",
        "proxy replay",
        "0 private bytes",
        58.9,
        None,
        "coverage-limited (FUNSD 76 / SROIE ≈7)",
    ),
    (
        "LexSlot, no buffer",
        "parametric slots",
        "none",
        42.2,
        None,
        "isolated slots cannot re-ground a shared head",
    ),
    ("spectral summary r=16", "compressed latent summary", "≈0.4 MB", 41.9, None, "collapses"),
    (
        "LCA (ICLR'26, merge + realignment)",
        "weight merging",
        "none",
        41.9,
        None,
        "Gaussian realignment destroys acquisition",
    ),
    (
        "per-class Gaussians (full dim)",
        "compressed latent summary",
        "class statistics",
        39.4,
        None,
        "collapses",
    ),
    (
        "k-means centroids",
        "compressed latent summary",
        "centroids",
        38.2,
        None,
        "collapses — not a fidelity problem",
    ),
    (
        "TIES / Fisher head merge",
        "weight merging",
        "none",
        21.6,
        None,
        "merged head vectors cancel (no-merge 21.9)",
    ),
]


def _method_table(d: pd.DataFrame, label_col: str) -> pd.DataFrame:
    g = (
        d.groupby(label_col)
        .agg(n=("AA", "size"), AA=("AA", "mean"), AA_std=("AA", "std"), BWT=("BWT", "mean"))
        .round(2)
    )
    return g.sort_values("AA", ascending=False)


def _block(ws, title, table, note=None):
    """One per-dataset comparison block (table + AA bar + BWT bar); returns nothing."""
    _title(ws, title)
    if note:
        ws.append([note])
    _header(ws, ["method", "family", "buffer", "n_seeds", "AA", "AA_std", "BWT"])
    first = ws.max_row + 1
    for m, r in table.iterrows():
        base = m.split(" [")[0]
        ws.append(
            [
                m,
                FAMILY.get(base, "variant / ablation"),
                BUFFER.get(base, ""),
                int(r.n),
                r.AA,
                None if np.isnan(r.AA_std) else r.AA_std,
                None if np.isnan(r.BWT) else r.BWT,
            ]
        )
    last = ws.max_row
    cats = Reference(ws, min_col=1, min_row=first, max_row=last)
    h = max(7, min(16, 0.33 * len(table)))
    _bar(
        ws,
        f"{title.split(' — ')[0]}: AA",
        cats,
        Reference(ws, min_col=5, min_row=first - 1, max_row=last),
        f"J{first - 1}",
        "AA",
        width=26,
        height=h,
    )
    _bar(
        ws,
        f"{title.split(' — ')[0]}: BWT",
        cats,
        Reference(ws, min_col=7, min_row=first - 1, max_row=last),
        f"X{first - 1}",
        "BWT",
        width=26,
        height=h,
    )
    ws.append([])
    ws.append([])
    ws.append([])


def write_approaches(wb):
    ws = wb.create_sheet("Approaches by dataset")
    _title(
        ws,
        "Approach comparison, one table per dataset (mean over seeds; sorted by AA; joint = oracle upper bound)",
    )
    ws.append(
        [
            "Documents: LayoutLMv3 primary backbone, results/all_runs.csv. Images: ViT-B/16, results/cil_*_vit. "
            "Family/buffer columns say what each approach stores."
        ]
    )
    ws.append([])
    df = pd.read_csv("results/all_runs.csv")
    doc = df[(df.model_family == "layoutlmv3") & df.method.isin(FAMILY)]
    for sc, title in DOC_SCENARIOS:
        d = doc[doc.scenario == sc]
        if d.empty:
            continue
        _block(ws, title, _method_table(d, "method"))
        if sc == "dil":
            _title(
                ws,
                "DIL — replay-content ladder and falsified buffer-free lines (what the buffer must contain)",
            )
            _header(ws, ["instance", "family", "store", "AA", "BWT", "reading"])
            f2 = ws.max_row + 1
            for r in LADDER:
                ws.append(list(r))
            l2 = ws.max_row
            _bar(
                ws,
                "DIL: whole-document latents retain; every marginal summary collapses",
                Reference(ws, min_col=1, min_row=f2, max_row=l2),
                Reference(ws, min_col=4, min_row=f2 - 1, max_row=l2),
                f"J{f2 - 1}",
                "AA",
                width=26,
                height=11,
            )
            ws.append([])
            ws.append([])
            ws.append([])
    img = load_runs()
    for sc, title in IMG_SCENARIOS:
        d = img[img.scenario == sc]
        if d.empty:
            continue
        _block(
            ws,
            title,
            _method_table(d, "method_label"),
            note="pp* rows are CoLaR++ screening variants (seed 42); see the per-scenario sheets for the learning curves.",
        )
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 36
    ws.column_dimensions["F"].width = 48


METRICS = [
    (
        "AA",
        "Average accuracy after the last task",
        "mean_i R[T, i] over all T seen tasks",
        "0–100 (higher better)",
        "Entity-level F1 (seqeval, BIO) on documents; top-1 accuracy on images.",
    ),
    (
        "BWT",
        "Backward transfer",
        "mean_{i<T} (R[T, i] − R[i, i])",
        "≤ 0 means forgetting",
        "Joint has BWT 0 by construction.",
    ),
    ("AF", "Average forgetting", "−BWT (reported positive)", "≥ 0", "Same information as BWT."),
    (
        "FWT",
        "Forward transfer",
        "mean_{i>0} (R[i−1, i] − b_i), b_i = single-task from-scratch score",
        "",
        "Left NaN when single-task baselines are missing (never imputed).",
    ),
    (
        "R[i, j]",
        "Accuracy matrix",
        "score on task j after training task i (i ≥ j); R[i−1, i] = zero-shot on the next task",
        "",
        "Saved as matrix.npy per run.",
    ),
    (
        "mean seen-task accuracy",
        "Learning curve",
        "mean_{j≤i} R[i, j] after each task i",
        "",
        "The line charts on the per-scenario sheets.",
    ),
    (
        "replay_memory_bytes",
        "Buffer size",
        "bytes held by the method after the last task (raw docs / latents / statistics)",
        "lower better",
        "0 or empty = buffer-free.",
    ),
    ("mean_time_per_task_s", "Compute", "wall-clock seconds per task (training + eval)", "", ""),
    ("peak_gpu_mem_mb", "Compute", "torch.cuda.max_memory_allocated", "", ""),
    (
        "linear CKA",
        "Representation drift",
        "CKA(X_before, X_after) of per-token (documents) / CLS (images) activations at a layer, between consecutive task checkpoints",
        "0–1 (1 = unchanged)",
        "Monotone in depth: input ≈1.00 → head ≈0.16 on LayoutLMv3.",
    ),
    (
        "Fisher-weighted displacement",
        "Where forgetting lives",
        "Σ_p F_old(p) · (θ_after(p) − θ_before(p))², F_old = empirical Fisher diagonal on the old task",
        "≥ 0",
        "Bucketed by component (text/layout/vision embeddings, encoder layers, head) and by depth (input/early/mid/late/head).",
    ),
    (
        "displacement share",
        "Locus",
        "bucket displacement / total displacement × 100",
        "%",
        "Head carries 96–100 % on every backbone (documents and images).",
    ),
    (
        "head / max non-head",
        "Locus",
        "head displacement / max over non-head buckets",
        "ratio",
        "1–4 orders of magnitude on documents.",
    ),
    (
        "readout-marginal cosine",
        "Root cause",
        "cos(predicted label marginal, new-task label marginal) after each task",
        "0–1",
        "0.997–0.998 for naive/LwF: the head snaps to the latest task's label prior.",
    ),
]


def write_metrics(wb):
    ws = wb.create_sheet("Metrics")
    _title(ws, "Metrics used across the report")
    ws.append([])
    _header(ws, ["metric", "what it measures", "definition", "range / direction", "notes"])
    for r in METRICS:
        ws.append(list(r))
    for col, w in zip("ABCDE", (28, 28, 70, 22, 60)):
        ws.column_dimensions[col].width = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/report_addendum.xlsx")
    args = ap.parse_args()
    wb = Workbook()
    wb.remove(wb.active)
    write_diagnosis(wb)
    write_approaches(wb)
    write_metrics(wb)
    runs = load_runs()
    for sc in SCENARIOS:
        write_scenario_sheet(wb, runs, sc)
    wb.save(args.out)
    print(f"wrote {args.out}: {wb.sheetnames}")


if __name__ == "__main__":
    main()
