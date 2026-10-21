"""Pass 1: build the CoLaR / LexSlot RCA ledger from results/*/metrics.json.

One row per cell. Columns: family, scenario, seed, backbone, variant,
AA, BWT, AF, FWT, final_row_mean_old, diag_mean, replay_bytes,
peak_gpu_mb, trainable_params, epochs, docs_per_task, rank_r,
has_dump, dir.

Pass-3 derivatives (slope, recovery count, boundary vs long-run drop)
are computed here too — same read, no second pass.
"""
from __future__ import annotations
import json, re, csv, math, pathlib, sys
import numpy as np

ROOT = pathlib.Path("results")
OUT = pathlib.Path("rca/colar_lexslot_ledger.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

PAT = re.compile(r"(colar|lexslot|lexmem|docR|docU|docB|docD)", re.I)

def classify(name: str) -> tuple[str, str, int|None, str, str]:
    """Return (family, scenario, seed, backbone, variant)."""
    s = int(m.group(1)) if (m := re.search(r"seed(\d+)", name)) else None
    # Backbone from the run-dir family suffix (train.py appends _<family> for non-LayoutLMv3).
    backbone = next((f for f in ("vit", "bert", "lilt", "bros") if f"_{f}" in name), "layoutlmv3")
    if name.startswith("cil_cifar100_"):   scenario = "cil_cifar100"
    elif name.startswith("cil_imagenet_r_"): scenario = "cil_imagenet_r"
    elif name.startswith("cil_cord_"):     scenario = "cil_cord"
    elif name.startswith("mixed_"):        scenario = "mixed"
    elif name.startswith("dil_"):          scenario = "dil"
    else: scenario = name.split("_", 1)[0]
    rest = re.sub(rf"^{scenario}_", "", name)
    rest = re.sub(r"_seed\d+.*", "", rest)
    fam = next((f for f in ("colar_r_sliced_pseudo","colar_r_sliced","colar_r_oracle",
                            "colar_r_pseudo","colar_meta","colar_knn","colar_wsvd",
                            "colar_bal","colar_cb","colar_adaptive","colar_e10","colar",
                            "lexslot","lexmem","docRfar","docRnear","docRtask","docU",
                            "docB2b","docB2","docB1","docD2","docD1","colar500r128",
                            "bank500") if f in rest), rest)
    variant = rest  # keep full tail for disambiguation
    return (fam, scenario, s, backbone, variant)

def final_row_metrics(matrix):
    M = np.array(matrix, dtype=float)
    T = M.shape[0]
    diag = [M[i,i] for i in range(T)]
    final = M[-1]
    old = final[:-1] if T > 1 else final
    # Recency slope on the final row vs task index
    xs = np.arange(T, dtype=float)
    try: slope = float(np.polyfit(xs, final, 1)[0])
    except Exception: slope = float("nan")
    # Recovery events: later readout exceeds diagonal
    rec = 0
    for t in range(1, T):
        for k in range(t):
            if not math.isnan(M[t,k]) and not math.isnan(M[k,k]) and M[t,k] > M[k,k] + 1.0:
                rec += 1
    # Boundary drop (immediate) vs long-run drop
    imm, longr = [], []
    for k in range(T-1):
        if k+1 < T and not math.isnan(M[k+1,k]):
            imm.append(M[k,k] - M[k+1,k])
        if not math.isnan(M[-1,k]):
            longr.append(M[k,k] - M[-1,k])
    return dict(diag_mean=float(np.nanmean(diag)),
                final_old_mean=float(np.nanmean(old)) if len(old) else float("nan"),
                final_new=float(final[-1]),
                recency_slope=slope, recoveries=rec,
                boundary_drop=float(np.nanmean(imm)) if imm else float("nan"),
                longrun_drop=float(np.nanmean(longr)) if longr else float("nan"))

def row(d: pathlib.Path) -> dict | None:
    name = d.name
    if not PAT.search(name) or name.startswith("_") or "_invalid" in str(d): return None
    mj = d / "metrics.json"
    if not mj.exists(): return None
    try: m = json.load(open(mj))
    except Exception: return None
    fam, sc, seed, bb, var = classify(name)
    r = dict(dir=str(d), family=fam, scenario=sc, seed=seed, backbone=bb, variant=var,
             AA=m.get("AA"), BWT=m.get("BWT"), AF=m.get("AF"), FWT=m.get("FWT"),
             num_tasks=m.get("num_tasks"),
             replay_bytes=m.get("replay_memory_bytes"),
             peak_gpu_mb=m.get("peak_gpu_mem_mb"),
             trainable_params=m.get("trainable_params"))
    hp = m.get("method_hparams", {}) or {}
    r["epochs"] = hp.get("epochs"); r["docs_per_task"] = hp.get("docs_per_task")
    r["rank_r"] = hp.get("rank_r"); r["k"] = hp.get("split_layer_k")
    r["dump_final"] = bool(hp.get("dump_final", False)) or (d/"final_model.pt").exists()
    if m.get("matrix") is not None:
        r.update(final_row_metrics(m["matrix"]))
    return r

rows = []
for d in sorted(ROOT.iterdir()):
    if d.is_dir() and (r := row(d)) is not None:
        rows.append(r)

cols = ["family","scenario","seed","backbone","variant","AA","BWT","AF","FWT",
        "diag_mean","final_old_mean","final_new","recency_slope","recoveries",
        "boundary_drop","longrun_drop","num_tasks","epochs","docs_per_task",
        "rank_r","k","replay_bytes","peak_gpu_mb","trainable_params","dump_final","dir"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r.get(c) for c in cols})
print(f"wrote {len(rows)} rows -> {OUT}")
