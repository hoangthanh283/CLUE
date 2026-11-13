"""Every chart of the CVPR paper and its appendix, from saved result files (CPU only).

Reads the same sources with the same selection rule as make_clean_tables.py (imported, never
re-implemented): a cell is clean only with all three clean seeds, else its legacy source (dagger,
hatched); fewer than three seeds -> per-seed dots, no bar, n in the label. Run make_clean_tables.py
first (the self-check compares the plotted values with the tables it wrote).

    uv run python scripts/rca/make_clean_tables.py
    PYTHONPATH=scripts/rca uv run python scripts/rca/make_paper_figs.py

Writes paper_cvpr/figures/:
  findings_a.pdf  findings_b.pdf  findings_c.pdf   (main paper, fig:findings)
  heatmaps.pdf  perseed.pdf  trajectory.pdf  routing.pdf  spans.pdf  perclass.pdf  (appendix)
"""

import json
import re
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from make_clean_tables import (  # noqa: E402
    CLEAN,
    LADDER,
    MECH_JSON,
    MECH_ROWS,
    OUT,
    SEEDS,
    C,
    _mech_runs,
    load_seeds,
    matrix_choice,
    mech_stats,
    pick,
    pick_clean_only,
)
from mechanism_analysis import blocked_pred, owner_map  # noqa: E402
from seqeval.metrics import classification_report, f1_score  # noqa: E402

FIG = OUT / "figures"
COL, WIDE = 3.31, 6.89  # CVPR column and text width, inches
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42,
                     "axes.titlesize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5})  # fmt: skip

# One colour per method in every figure (Okabe-Ito; distinct in greyscale by lightness).
STYLE = {
    "naive": ("Naive", "#8C8C8C"),
    "er": ("ER", "#F0E442"),
    "colar": ("CoLaR", "#D55E00"),
    "oracle": ("Relabelled memory", "#CC79A7"),
    "sliced2": ("Sliced losses", "#E69F00"),
    "nobank": ("SBR, no memory", "#56B4E9"),
    "sbr": ("CoLaR-SBR", "#0072B2"),
    "recall": ("SBR-Recall", "#009E73"),
    "joint": ("Joint training", "#000000"),
    # ladder controls not run yet: drawn only once their three clean seeds exist (Tol muted)
    "er_sbr": ("Raw memory + SBR", "#DDCC77"),
    "oracle_pre": ("Relabel before each session", "#882255"),
    "colar_mib": ("MiB-style loss", "#AA4499"),
    "sbr_constbg": ("Constant background", "#44AA99"),
    "ice_o": ("ICE-O", "#332288"),
}
SEED_MARK = {42: "o", 7: "s", 123: "^"}
HATCH = "////"
SLOT_LABEL = {"er_sbr": "raw mem. + SBR", "oracle_pre": "relabel before", "colar_mib": "MiB loss",
              "sbr_constbg": "const. bg", "ice_o": "ICE-O"}  # fmt: skip
SHORT = {"colar": "CoLaR", "er": "ER", "oracle": "relabelled", "sliced2": "sliced losses",
         "sbr": "CoLaR-SBR", "nobank": "SBR, no mem.", "recall": "SBR-Recall"}  # fmt: skip
MECH_KEY = {"naive": "naive", "colar": "colar", "colar_r_oracle": "oracle",
            "colar_sbr_noreplay": "nobank", "colar_sbr": "sbr", "colar_sbr_recall_r64": "recall",
            "colar_dump": "colar", "naive_dump": "naive", "colar_sbr_dump": "sbr", "joint_dump": "joint"}  # fmt: skip
CHECKS: list[str] = []  # what the self-check verified, printed at the end


def _sources(key: str):
    """Sources of a method key: the registry C, or the ladder row's (oracle, sliced2)."""
    if key in C:
        return C[key]
    return next(row[5] for row in LADDER if row[4] == key)


def mats(key: str) -> tuple[dict, bool]:
    """{seed: R} and is_clean, by the tables' rule; a clean-only control (oracle) with fewer than
    three seeds falls back to the clean seeds that exist (labelled with n, never a legacy mix)."""
    sources = _sources(key)
    source, by_seed = matrix_choice(sources)
    if source is not None:
        return by_seed, source[0] == CLEAN
    clean = [s for s in sources if s[0] == CLEAN]
    if clean:
        from make_clean_tables import _run_file

        r, n, sfx = clean[0]
        found = {sd: np.load(f) for sd in SEEDS if (f := _run_file(r, n, sd, sfx, "matrix.npy"))}
        return found, True
    return {}, False


def tag(n: int, clean: bool) -> str:
    return ("" if clean else r"$\dagger$") + (f" n={n}" if n < 3 else "")


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


# ---------------------------------------------------------------- F2(a): the ladder as bars
def ladder_cells() -> list[dict]:
    """CIL-CORD rows of the ladder, read exactly as write_ladder reads them."""
    out = []
    for rung, _, _, interv, key, aa_s, full_s, kind, _, min_n in LADDER:
        if key in ("naive", "joint", "mix_nomem", "mix_mem"):
            continue
        if kind == "slot":
            aa = pick_clean_only(aa_s, "AA")
            if len(aa[0]) < min_n:  # R11-M3: a slot appears only with its three clean seeds,
                continue  # the rule tab_ladder.tex follows (no placeholder bar)
            full = pick_clean_only(full_s, "full_label_f1")
        elif kind == "clean":
            aa, full = pick_clean_only(aa_s, "AA"), pick_clean_only(full_s, "full_label_f1")
        else:
            aa = pick(aa_s)
            full = pick_clean_only(full_s, "full_label_f1") if full_s else ([], True, False)
        out.append({"key": key, "rung": rung, "interv": interv,
                    "aa": aa[0], "aa_clean": aa[1], "full": full[0]})  # fmt: skip
    return out


def _bar(ax, x, vals, clean, color, light, w):
    """Mean bar (n = 3 only) with per-seed dots; legacy hatched; AA lighter than full-label."""
    if not vals:
        return
    if len(vals) >= 3:
        ax.bar(x, np.mean(vals), w, color=color, alpha=0.45 if light else 1.0,
               edgecolor="black", lw=0.4, hatch=None if clean else HATCH)  # fmt: skip
    ax.scatter(x + np.linspace(-w / 4, w / 4, len(vals)), vals, s=3, color="black", zorder=3,
               lw=0)  # fmt: skip


def fig_findings_a() -> list[dict]:
    cells = ladder_cells()
    fig, ax = plt.subplots(figsize=(3.45, 1.8))
    w = 0.38
    for i, c in enumerate(cells):
        col = STYLE[c["key"]][1]
        _bar(ax, i - w / 2, c["full"], True, col, False, w)
        _bar(ax, i + w / 2, c["aa"], c["aa_clean"], col, True, w)
        if not c["full"]:
            ax.text(i - w / 2, 2, "--", ha="center", va="bottom", fontsize=6.5)
        n = max(len(c["full"]), len(c["aa"]))
        if n < 3:
            ax.text(i, 101, f"n={n}", ha="center", va="bottom", fontsize=6.5)
    refs = []
    for key, ls in (("joint", "-"), ("naive", "--")):
        v = np.mean(pick_clean_only(C[key], "full_label_f1")[0])
        ax.axhline(v, color=STYLE[key][1], lw=0.7, ls=ls, zorder=0)
        refs.append((plt.Line2D([], [], color=STYLE[key][1], lw=0.7, ls=ls),
                     f"{STYLE[key][0].split()[0].lower()} {v:.1f}"))  # fmt: skip
    labels = [SHORT.get(c["key"], SLOT_LABEL.get(c["key"], c["key"])) for c in cells]
    ax.set_xticks(range(len(cells)), labels, rotation=50, ha="right", rotation_mode="anchor")
    rung_x = {}
    for i, c in enumerate(cells):
        rung_x.setdefault(c["rung"] or list(rung_x)[-1], []).append(i)
    for r, xs in rung_x.items():
        ax.text(np.mean(xs), 108, r, ha="center", va="bottom", fontsize=7, weight="bold")
        if xs[0] > 0:
            ax.axvline(xs[0] - 0.5, color="#BBBBBB", lw=0.4, zorder=0)
    ax.set_ylim(0, 117)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylabel("CIL-CORD score", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#666666", ec="black", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="#666666", alpha=0.45, ec="black", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.4, hatch=HATCH)]  # fmt: skip
    labs = ["full-label F1", "AA", r"legacy$\dagger$ (AA)"]
    fig.legend([h[0], refs[0][0], h[1], refs[1][0], h[2]],
               [labs[0], refs[0][1], labs[1], refs[1][1], labs[2]], loc="upper left",
               bbox_to_anchor=(0.1, 1.0), ncol=3, frameon=False, handlelength=1.4,
               columnspacing=1.0, borderaxespad=0.1, labelspacing=0.15)  # fmt: skip
    fig.subplots_adjust(left=0.115, right=0.995, top=0.84, bottom=0.36)
    save(fig, "findings_a.pdf")
    return cells


# ---------------------------------------------------------------- F2(b): where old tokens go
CAT = (("correct", "#333333"), ("to_O", "#E6E6E6"), ("to_same_session", "#A6A6A6"),
       ("to_other_session", "#FFFFFF"))  # fmt: skip
CAT_LABEL = {"correct": "correct", "to_O": r"$\to$ O", "to_same_session": r"$\to$ same session",
             "to_other_session": r"$\to$ other session"}  # fmt: skip


def mech_cols() -> list[dict]:
    """One column per MECH_ROWS run present in mechanism_analysis.json (the tab:mech columns)."""
    res = json.loads(MECH_JSON.read_text()) if MECH_JSON.exists() else {}
    cols = []
    for label, method, section, readout in MECH_ROWS:
        s = mech_stats(res, method, section)
        if s is None:
            continue
        runs = _mech_runs(res, method, section)
        errs = [r[section]["old"].get("errors") for r in runs]
        col = {"label": label, "method": method, "readout": readout, "stats": s, "n": s["n"]}
        if all(errs):
            col["frac"] = {k: [e[k] / e["n_gold_tokens"] for e in errs] for k, _ in CAT}
        cols.append(col)
    return cols


def fig_findings_b() -> list[dict]:
    cols = mech_cols()
    fig, ax = plt.subplots(figsize=(1.72, 1.8))
    for i, c in enumerate(cols):
        if "frac" in c:  # per-token split (blocked readouts, dumps): mean over seeds
            bottom = 0.0
            for k, fc in CAT:
                v = 100 * np.mean(c["frac"][k])
                ax.bar(i, v, 0.7, bottom=bottom, color=fc, ec="black", lw=0.4)
                bottom += v
        else:  # shared softmax: only span recall is recorded
            v = np.mean(c["stats"]["recall"])
            ax.bar(i, v, 0.7, color=CAT[0][1], ec="black", lw=0.4)
            ax.bar(i, 100 - v, 0.7, bottom=v, color="white", ec="#999999", lw=0.4, ls=":")
            if 100 - v > 40:
                ax.text(i, 100 - (100 - v) / 2, "not split", rotation=90, ha="center",
                        va="center", fontsize=6.5, color="#555555")  # fmt: skip
    ax.set_xticks(range(len(cols)),
                  [re.sub(r"\$\^\{[a-z]\}\$", "", c["label"]) + tag(c["n"], True) for c in cols],
                  rotation=40, ha="right", rotation_mode="anchor")  # fmt: skip
    n_sh = sum(c["readout"] == "shared" for c in cols)
    ax.axvline(n_sh - 0.5, color="black", lw=0.5)
    ax.text((n_sh - 1) / 2, 103, "shared\n(spans)", ha="center", va="bottom", fontsize=6.5)
    ax.text(n_sh + (len(cols) - n_sh - 1) / 2, 103, "blocked\n(tokens)", ha="center",
            va="bottom", fontsize=6.5)  # fmt: skip
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("old-session entities (%)", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.97, bottom=0.36)
    save(fig, "findings_b.pdf")
    # the legend is drawn once, below the three panels, by the caption's key: a separate strip
    fig, ax = plt.subplots(figsize=(3.9, 0.15))
    ax.axis("off")
    h = [plt.Rectangle((0, 0), 1, 1, fc=fc, ec="black", lw=0.4) for _, fc in CAT]
    ax.legend(h, [CAT_LABEL[k] for k, _ in CAT], loc="center", ncol=4, frameon=False,
              handlelength=1.2, columnspacing=1.2, borderaxespad=0)  # fmt: skip
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "findings_b_key.pdf")
    return cols


# ---------------------------------------------------------------- F2(c): stability vs plasticity
STAB_KEYS = ("naive", "nobank", "colar", "oracle", "sbr", "recall", "joint")
SHORT_C = {**SHORT, "naive": "naive", "joint": "joint"}


def stab_points() -> dict:
    """key -> {seed: (newest-session F1 R[n_s,n_s], mean old-session F1 R[n_s,:n_s])}, is_clean."""
    out = {}
    for k in STAB_KEYS:
        by_seed, clean = mats(k)
        out[k] = ({s: (float(m[-1, -1]), float(m[-1, :-1].mean())) for s, m in by_seed.items()},
                  clean)  # fmt: skip
    return out


def fig_findings_c() -> dict:
    pts = stab_points()
    fig, ax = plt.subplots(figsize=(1.72, 1.8))
    for k, (by_seed, clean) in pts.items():
        lab, col = STYLE[k]
        for s, (x, y) in by_seed.items():
            ax.scatter(x, y, s=12, marker=SEED_MARK[s], color=col, ec="black", lw=0.3, zorder=3, alpha=0.75,
                       label=SHORT_C[k] + tag(len(by_seed), clean)
                       if s == next(iter(by_seed)) else None)  # fmt: skip
    ax.plot([0, 100], [0, 100], color="#CCCCCC", lw=0.4, zorder=0)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("newest session F1", labelpad=1)
    ax.set_ylabel("mean F1, old sessions", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.12, 0.08), frameon=False, handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.12, fontsize=6.5, markerscale=0.8)  # fmt: skip
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.18)
    save(fig, "findings_c.pdf")
    return pts


# ---------------------------------------------------------------- appendix
HEAT_KEYS = ("naive", "colar", "oracle", "nobank", "sbr", "recall", "joint")


def fig_heatmaps() -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(WIDE, 3.25))
    means = {}
    for ax, k in zip(axes.flat, HEAT_KEYS, strict=False):
        by_seed, clean = mats(k)
        if not by_seed:
            ax.set_title(f"{STYLE[k][0]}: no run")
            ax.axis("off")
            continue
        m = np.mean(list(by_seed.values()), axis=0)
        means[k] = m
        n = m.shape[0]
        shown = np.where(np.tril(np.ones((n, n), bool)) & np.isfinite(m), m, np.nan)
        ax.imshow(shown, vmin=0, vmax=100, cmap="cividis")
        for i in range(n):
            for j in range(n):
                if np.isfinite(shown[i, j]):
                    v = shown[i, j]
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                            color="black" if v > 45 else "white")  # fmt: skip
        proto = "clean" if clean else r"legacy $\dagger$"
        ax.set_title(f"{STYLE[k][0]} ({proto}, n={len(by_seed)})", pad=2)
        ax.set_xticks(range(n), [str(j + 1) for j in range(n)])
        ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
        ax.tick_params(length=0, pad=1)
        for s in ax.spines.values():
            s.set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.5, "row $i$: after session $i$\ncolumn $j$: session $j$\n"
                       "session-restricted F1,\nmean over the seeds", va="center", fontsize=7)  # fmt: skip
    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.05, wspace=0.18, hspace=0.32)
    save(fig, "heatmaps.pdf")
    return means


DOT_KEYS = ("naive", "nobank", "colar", "oracle", "sbr", "recall", "joint")


def fig_perseed() -> dict:
    vals = {}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
    for ax, (metric, title) in zip(axes, (("full_label_f1", "full-label F1"), ("AA", "AA")),
                                   strict=True):  # fmt: skip
        for y, k in enumerate(DOT_KEYS):
            src = [s for s in _sources(k) if s[0] == CLEAN]
            d = load_seeds(*src[0][:2], key=metric, suffix=src[0][2]) if src else {}
            vals[(k, metric)] = d
            for s, v in d.items():
                ax.scatter(v, y, s=10, marker=SEED_MARK[s], color=STYLE[k][1], ec="black",
                           lw=0.3, zorder=3)  # fmt: skip
            if len(d) >= 3:
                ax.plot([np.mean(list(d.values()))] * 2, [y - 0.32, y + 0.32], color="black",
                        lw=0.8)  # fmt: skip
        ax.set_title(title, pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75])
        ax.grid(axis="x", alpha=0.3, lw=0.3)
        ax.tick_params(length=1.5, pad=1)
    axes[0].set_yticks(range(len(DOT_KEYS)),
                       [STYLE[k][0] + tag(len(vals[(k, "AA")]), True) for k in DOT_KEYS])  # fmt: skip
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker=m, ls="", color="#BBBBBB", mec="black", mew=0.3, ms=3.5)
         for m in SEED_MARK.values()]  # fmt: skip
    axes[0].legend(h, [f"seed {s}" for s in SEED_MARK], loc="lower left", frameon=False,
                   handletextpad=0.1, borderaxespad=0.1, labelspacing=0.1)  # fmt: skip
    fig.subplots_adjust(left=0.29, right=0.98, top=0.9, bottom=0.12, wspace=0.12)
    save(fig, "perseed.pdf")
    return vals


TRAJ_KEYS = ("naive", "er", "colar", "oracle", "nobank", "sbr", "recall")


def fig_trajectory() -> dict:
    traj = {}
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for k in TRAJ_KEYS:
        by_seed, clean = mats(k)
        if not by_seed:
            continue
        per = [[np.mean(m[i, : i + 1]) for i in range(m.shape[0])] for m in by_seed.values()]
        y = np.mean(per, axis=0)
        traj[k] = (by_seed, y)
        ax.plot(range(1, len(y) + 1), y, marker="o", ms=2.5, lw=0.9, color=STYLE[k][1],
                ls="-" if clean else "--", mec="black", mew=0.2,
                label=STYLE[k][0] + tag(len(by_seed), clean))  # fmt: skip
    jb, jc = mats("joint")
    if jb:
        jv = np.mean([np.mean(m[-1]) for m in jb.values()])
        traj["joint"] = (jb, [jv])
        ax.scatter(5, jv, marker="*", s=30, color="black", zorder=3,
                   label="Joint (one pooled run)" + tag(len(jb), jc))  # fmt: skip
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("sessions trained", labelpad=1)
    ax.set_ylabel("AA over sessions seen", labelpad=1)
    ax.grid(alpha=0.25, lw=0.3)
    ax.tick_params(length=1.5, pad=1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6.5,
              handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)  # fmt: skip
    fig.subplots_adjust(left=0.12, right=0.58, top=0.97, bottom=0.2)
    save(fig, "trajectory.pdf")
    return traj


ROUTE_KEYS = ("nobank", "sbr")


def routing_counts(key: str) -> dict:
    """{seed: counts (sessions x blocks+1)} of gold entity tokens of each split by the block that
    claims them (last column: unclaimed), from eval_dump.pt (the decision of the SBR readout)."""
    r, n, sfx = _sources(key)[0]
    out = {}
    for s in SEEDS:
        hits = sorted(glob(f"{r}/{n}_seed{s}{sfx}/eval_dump.pt"))
        if not hits:
            continue
        d = torch.load(hits[0], map_location="cpu", weights_only=False)
        n_s = len(d)
        own = owner_map([d[t]["allowed"] for t in range(n_s)])
        cnt = np.zeros((n_s, n_s + 1), dtype=int)
        for t in range(n_s):
            x = d[t]
            _, best_b, claimed = blocked_pred(x["e"], x["tag"], x["threshold"])
            ent = x["gold"].long() > 0
            assert all(own[int(g)] == t for g in x["gold"][ent].tolist())  # masked split
            for b in range(n_s):
                cnt[t, b] = int((ent & claimed & (best_b == b)).sum())
            cnt[t, n_s] = int((ent & ~claimed).sum())
        out[s] = cnt
    return out


def fig_routing() -> dict:
    counts = {k: routing_counts(k) for k in ROUTE_KEYS}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75))
    for ax, k in zip(axes, ROUTE_KEYS, strict=True):
        if not counts[k]:
            ax.axis("off")
            continue
        tot = sum(counts[k].values())
        pct = 100 * tot / tot.sum(1, keepdims=True)
        ax.imshow(pct, vmin=0, vmax=100, cmap="cividis", aspect="auto")
        n_s = pct.shape[0]
        for i in range(n_s):
            for j in range(n_s + 1):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="black" if pct[i, j] > 45 else "white")  # fmt: skip
        ax.set_xticks(range(n_s + 1), [str(b + 1) for b in range(n_s)] + ["none"])
        ax.set_yticks(range(n_s), [str(t + 1) for t in range(n_s)])
        ax.set_title(f"{STYLE[k][0]} (n={len(counts[k])})", pad=2)
        ax.set_xlabel("claiming block", labelpad=1)
        ax.tick_params(length=0, pad=1)
    axes[0].set_ylabel("session of the gold entity", labelpad=1)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.2, wspace=0.18)
    save(fig, "routing.pdf")
    return counts


def fig_spans() -> list[int]:
    res = json.loads(MECH_JSON.read_text())
    spans = res["gold_spans_per_session_test_split"]
    fig, ax = plt.subplots(figsize=(COL, 1.4))
    x = np.arange(1, len(spans) + 1)
    ax.bar(x, spans, 0.6, color="#8C8C8C", ec="black", lw=0.4)
    for xi, v in zip(x, spans, strict=True):
        ax.text(xi, v + 15, f"{v} ({100 * v / sum(spans):.0f}%)", ha="center", va="bottom",
                bbox={"fc": "white", "ec": "none", "pad": 0.6}, zorder=3,
                fontsize=6.5)  # fmt: skip
    ax.axhline(sum(spans) / len(spans), color="black", lw=0.5, ls="--", zorder=1)
    ax.text(1.45, sum(spans) / len(spans) + 12, "equal share (AA's weight)",
            ha="left", va="bottom", fontsize=6.5)  # fmt: skip
    ax.set_xticks(x)
    ax.set_ylim(0, max(spans) * 1.25)
    ax.set_xlabel("CIL-CORD session", labelpad=1)
    ax.set_ylabel("gold test spans", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.27)
    save(fig, "spans.pdf")
    return spans


def dump_pred(d: dict) -> list[int]:
    if "logits" in d:
        return d["logits"].float().argmax(-1).tolist()
    return blocked_pred(d["e"], d["tag"], d["threshold"])[0].tolist()


def fig_perclass() -> dict:
    """Per-class full-label F1 of the seed-42 prediction-dump re-runs that exist (one run each)."""
    res = json.loads(MECH_JSON.read_text())
    order = [c for sess in res["session_types"] for c in sess]
    found = {}
    for f in sorted(glob(f"{CLEAN}/cil_cord_*dump*/full_label_dump.pt")):
        meta = json.loads((Path(f).parent / "metrics.json").read_text())
        key = MECH_KEY.get(meta["method"], meta["method"].removesuffix("_dump"))
        d = torch.load(f, map_location="cpu", weights_only=False)
        i2l = {int(k): v for k, v in d["id_to_label"].items()}
        seq: dict[int, tuple[list, list]] = {}
        for p, g, doc in zip(dump_pred(d), d["gold"].tolist(), d["doc"].tolist(), strict=True):
            seq.setdefault(doc, ([], []))
            seq[doc][0].append(i2l[p])
            seq[doc][1].append(i2l[g])
        preds, golds = [v[0] for v in seq.values()], [v[1] for v in seq.values()]
        rep = classification_report(golds, preds, output_dict=True, zero_division=0)
        found[key] = {"seed": meta["seed"], "micro": 100 * f1_score(golds, preds),
                      "dump_f1": d["full_label_f1"],
                      "f1": [100 * rep[c]["f1-score"] if c in rep else np.nan for c in order]}  # fmt: skip
    fig, ax = plt.subplots(figsize=(WIDE, 2.25))
    x = np.arange(len(order))
    w = 0.8 / max(len(found), 1)
    for i, (k, v) in enumerate(found.items()):
        ax.bar(
            x - 0.4 + w * (i + 0.5),
            v["f1"],
            w,
            color=STYLE.get(k, ("", "#999999"))[1],
            ec="black",
            lw=0.3,
            label=f"{STYLE.get(k, (k,))[0]} (seed {v['seed']}; {v['dump_f1']:.1f})",
        )
    edges = np.cumsum([len(s) for s in res["session_types"]])[:-1] - 0.5
    for e in edges:
        ax.axvline(e, color="#999999", lw=0.5)
    for t, sess in enumerate(res["session_types"]):
        lo = sum(len(s) for s in res["session_types"][:t])
        ax.text(lo + (len(sess) - 1) / 2, 103, f"session {t + 1}", ha="center", va="bottom",
                fontsize=6.5)  # fmt: skip
    ax.set_xticks(x, order, rotation=50, ha="right", rotation_mode="anchor", fontsize=6.5)
    # R11-L7: a class absent from the test set has no F1 (n/a), not F1 = 0
    absent = [i for i in range(len(order)) if all(np.isnan(v["f1"][i]) for v in found.values())]
    for i in absent:
        ax.text(i, 1.5, "n/a", ha="center", va="bottom", rotation=90, fontsize=6.5, color="#777777")
        ax.get_xticklabels()[i].set_color("#999999")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_ylabel("full-label F1", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), frameon=False, ncol=4,
              borderaxespad=0.1)  # fmt: skip
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.47)
    save(fig, "perclass.pdf")
    return found


# ---------------------------------------------------------------- self-check
def _macro(text: str, name: str) -> str:
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", text)
    assert m, f"macro {name} missing from vals.tex"
    return m.group(1)


def _fmt_cell(vs: list[float]) -> str:
    """How the tables print a cell's value(s): mean at n = 3, else the per-seed values."""
    return f"{np.mean(vs):.1f}" if len(vs) >= 3 else "/".join(f"{v:.1f}" for v in vs)


def selfcheck(cells, cols, pts, means, dots, traj, counts, spans, perclass) -> None:
    vals = (OUT / "vals.tex").read_text()
    lad = (OUT / "tab_ladder.tex").read_text().splitlines()
    for c in cells:  # F2(a): every bar (or dot set) is the cell tab_ladder.tex prints
        line = next(x for x in lad if c["interv"] in x)
        for vs in (c["aa"], c["full"]):
            if vs:
                assert _fmt_cell(vs) in line, (c["key"], _fmt_cell(vs), line)
    for key in ("joint", "naive"):  # F2(a) reference lines
        v = np.mean(pick_clean_only(C[key], "full_label_f1")[0])
        assert f"{v:.1f}" == _macro(vals, f"n{key.capitalize()}Full"), key
    CHECKS.append(f"F2(a): {len(cells)} ladder cells = tab_ladder.tex")
    res = json.loads(MECH_JSON.read_text())
    for c in cols:  # F2(b): token fractions reproduce the shares of tab:mech; recall = R
        s = mech_stats(res, c["method"], next(r[2] for r in MECH_ROWS if r[1] == c["method"]))
        assert np.allclose(c["stats"]["recall"], s["recall"])
        if "frac" in c:
            fr = c["frac"]
            for i in range(c["n"]):
                assert abs(sum(fr[k][i] for k, _ in CAT) - 1) < 1e-9
                wrong = 1 - fr["correct"][i]
                for k in ("to_O", "to_same_session", "to_other_session"):
                    assert abs(fr[k][i] / wrong - s[f"share_{k}"][i]) < 1e-9, (c["method"], k)
    CHECKS.append(f"F2(b): {len(cols)} columns = mechanism_analysis.json (tab:mech)")
    # P14b: the seed-42 dump re-runs feed only the per-class figure, never a cell, range or mean
    assert not any(c["method"].endswith("_dump") for c in cols)
    assert not any(r[1].endswith("_dump") for r in MECH_ROWS)
    for k in set(STAB_KEYS) | set(HEAT_KEYS) | set(DOT_KEYS) | set(TRAJ_KEYS):
        assert not any("dump" in s[1] for s in _sources(k)), k
    CHECKS.append("dump re-runs: absent from every table-backed figure")
    by_run = {(r["method"], r["seed"]): r["aa_vs_full"] for r in res["runs"].values()}
    meth = {"naive": "naive", "nobank": "colar_sbr_noreplay", "colar": "colar",
            "oracle": "colar_r_oracle", "sbr": "colar_sbr", "recall": "colar_sbr_recall_r64",
            "joint": "joint"}  # fmt: skip
    n_c = 0
    for k, (by_seed, _) in pts.items():  # F2(c) = aa_vs_full of the mechanism analysis
        for s, (x, y) in by_seed.items():
            a = by_run.get((meth[k], s))
            if a:
                assert (
                    abs(a["last_session_f1"] - x) < 1e-9 and abs(a["old_sessions_mean"] - y) < 1e-9
                )
                n_c += 1
    CHECKS.append(f"F2(c): {n_c} points = aa_vs_full in mechanism_analysis.json")
    for k, f in (("colar", "supp_mat_colar.tex"), ("sbr", "supp_mat_sbr.tex"),
                 ("recall", "supp_mat_recall.tex")):  # fmt: skip
        rows = [ln for ln in (OUT / f).read_text().splitlines() if re.match(r"^\d+ &", ln)]
        for i, ln in enumerate(rows):
            got = [x.strip() for x in ln.split("&")[1:]]
            got[-1] = got[-1].replace("\\\\", "").strip()
            want = [f"{means[k][i, j]:.1f}" for j in range(i + 1)]
            assert got[: i + 1] == want, (k, i, got, want)
    CHECKS.append("heatmaps: CoLaR, CoLaR-SBR, SBR-Recall = supp_mat_*.tex")
    for k, nm in (("naive", "Naive"), ("nobank", "NoBank"), ("colar", "Colar"), ("sbr", "Sbr"),
                  ("recall", "Recall"), ("joint", "Joint")):  # fmt: skip
        assert f"{np.mean(list(dots[(k, 'AA')].values())):.1f}" == _macro(vals, f"n{nm}")
        assert f"{np.mean(list(dots[(k, 'full_label_f1')].values())):.1f}" == _macro(
            vals, f"n{nm}Full"
        )
    o = dots[("oracle", "AA")]
    if len(o) == 3:
        assert f"{np.mean(list(o.values())):.1f}" == _macro(vals, "nOracle")
    CHECKS.append("per-seed dots: means = vals.tex macros")
    for k, (by_seed, y) in traj.items():  # trajectory end point = AA of metrics.json
        if k in ("naive", "colar", "nobank", "sbr", "recall", "joint", "oracle"):
            src = [s for s in _sources(k) if s[0] == CLEAN][0]
            aa = load_seeds(*src[:2], suffix=src[2])
            assert abs(y[-1] - np.mean([aa[s] for s in by_seed])) < 1e-6, k
    CHECKS.append("trajectory: final points = AA in metrics.json")
    rt = {"nobank": "colar_sbr_noreplay", "sbr": "colar_sbr"}
    for k, by_seed in counts.items():  # routing = per-session routing of the analysis
        for s, cnt in by_seed.items():
            r = res["runs"][f"cil_cord_{rt[k]}_seed{s}"]["masked_splits"]["sessions"]
            for t, ses in enumerate(r):
                ro = ses["routing"]
                assert cnt[t, t] == ro["routed_to_owner"] and cnt[t, -1] == ro["unclaimed"]
                assert cnt[t, :-1].sum() - cnt[t, t] == ro["claimed_by_other_block"]
                assert cnt[t].sum() == ro["n_entity_tokens"]
    CHECKS.append("routing: counts = routing in mechanism_analysis.json")
    total = f"{sum(spans):,}".replace(",", "{,}")
    assert f"\\newcommand{{\\mTestSpans}}{{{total}}}" in vals
    assert f"{100 * spans[0] / sum(spans):.0f}" == _macro(vals, "mSessOneShare")
    CHECKS.append("spans: total and session-1 share = vals.tex")
    for k, v in perclass.items():  # per-class: our scoring reproduces the run's full-label F1
        assert abs(v["micro"] - v["dump_f1"]) < 1e-6, k
        # P17: each dump re-run reproduces the full-label F1 of its clean run of the same seed
        src_clean = [x for x in _sources(k) if x[0] == CLEAN][0]
        ref = load_seeds(*src_clean[:2], key="full_label_f1", suffix=src_clean[2])
        assert abs(ref[v["seed"]] - v["dump_f1"]) < 1e-6, (k, ref, v["dump_f1"])
    CHECKS.append(
        f"per-class: {len(perclass)} dump(s) = clean run of the same seed (full-label F1)"
    )


PC_NAME = {"colar": "Colar", "sbr": "Sbr", "joint": "Joint", "naive": "Naive"}


def write_perclass_vals(perclass: dict) -> None:
    """supp_perclass_vals.tex: the counts the App. I per-class paragraph states (seed 42 only).
    Found = full-label F1 above 0 on a class that occurs in the test set; "--" for a missing run."""
    t = "% AUTO-GENERATED by scripts/rca/make_paper_figs.py -- do not edit.\n"
    present = None
    for k, nm in PC_NAME.items():
        f1 = np.array(perclass[k]["f1"]) if k in perclass else None
        if f1 is not None:
            present = ~np.isnan(f1)
        found = str(int((f1[~np.isnan(f1)] > 0).sum())) if f1 is not None else "--"
        t += f"\\newcommand{{\\pcFound{nm}}}{{{found}}}\n"
    t += f"\\newcommand{{\\pcPresent}}{{{int(present.sum()) if present is not None else '--'}}}\n"
    both = "--"
    if "sbr" in perclass and "joint" in perclass:
        a, b = np.array(perclass["sbr"]["f1"]), np.array(perclass["joint"]["f1"])
        both = str(int(((a == 0) & (b == 0)).sum()))
    t += f"\\newcommand{{\\pcBothMissed}}{{{both}}}\n"
    (OUT / "supp_perclass_vals.tex").write_text(t)
    print("wrote", OUT / "supp_perclass_vals.tex")


# ---------------------------------------------------------------- P18: ablation grid (App. K)
ABL_COLOR = {"naive": "naive", "colar": "colar", "nobank": "nobank", "sbr": "sbr",
             "recall": "recall", "oracle": "oracle", "oracle_pre": "oracle_pre",
             "joint": "joint"}  # fmt: skip


def fig_ablation() -> dict:
    """tab:ablation as bars: full-label F1 (dark) and AA (light), per-seed dots."""
    from make_clean_tables import ABLATION, _abl_seeds

    rows = [r for r in ABLATION
            if len(_abl_seeds(r[1], "full_label_f1")) == 3 and len(_abl_seeds(r[1], "AA")) == 3]  # fmt: skip
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    w, out = 0.38, {}
    for i, (_label, key, *_) in enumerate(rows):
        col = STYLE[ABL_COLOR[key]][1]
        fl, aa = list(_abl_seeds(key, "full_label_f1").values()), list(
            _abl_seeds(key, "AA").values()
        )
        out[key] = (fl, aa)
        _bar(ax, i - w / 2, fl, True, col, False, w)
        _bar(ax, i + w / 2, aa, True, col, True, w)
    ax.set_xticks(range(len(rows)), [re.sub(r"\$\^\{r\}\$", "", r[0]) for r in rows], rotation=35,
                  ha="right", rotation_mode="anchor")  # fmt: skip
    ax.set_ylim(0, 100)
    ax.set_ylabel("CIL-CORD score", labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, fc="#666666", ec="black", lw=0.4),
         plt.Rectangle((0, 0), 1, 1, fc="#666666", alpha=0.45, ec="black", lw=0.4)]  # fmt: skip
    ax.legend(h, ["full-label F1", "AA"], loc="upper left", frameon=False, handlelength=1.2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.37)
    save(fig, "ablation.pdf")
    return out


# ---------------------------------------------------------------- P18: image appendix (App. L)
IMG_DS = (("cil_cifar100", "CIFAR-100"), ("cil_imagenet_r", "ImageNet-R"))
IMG_STYLE = {  # key of make_image_table.VAL_RUNS -> (label, colour); one colour per method
    "Naive": ("Naive", "#8C8C8C"), "Lwf": ("LwF", "#BBBBBB"),
    "ErTwoK": ("ER, 2,000", "#F0E442"), "DerTwoK": ("DER++, 2,000", "#DDCC77"),
    "Coda": ("CODA-Prompt", "#AA4499"), "Cllora": ("CL-LoRA", "#882255"),
    "Slca": ("SLCA", "#332288"), "Ranpac": ("RanPAC", "#88CCEE"), "Lca": ("LCA", "#44AA99"),
    "Bank": ("Latent memory", "#999933"), "Colar": ("CoLaR", "#D55E00"),
    "ErTwoH": ("ER, 200", "#F0E442"), "DerTwoH": ("DER++, 200", "#DDCC77"),
    "Joint": ("Joint training", "#000000"),
}  # fmt: skip
IMG_EXTRA_RUNS = {"Lwf": "{sc}_lwf_seed{sd}_vit", "Coda": "{sc}_coda_prompt_seed{sd}_vit",
                  "Cllora": "{sc}_cl_lora_seed{sd}_vit"}  # fmt: skip
IMG_AA_KEYS = ("Naive", "Lwf", "Coda", "Cllora", "ErTwoK", "DerTwoK", "Lca", "Ranpac", "Slca",
               "Bank", "Colar")  # fmt: skip


def _img_tmpl(key):
    from make_image_table import VAL_RUNS

    return VAL_RUNS.get(key) or IMG_EXTRA_RUNS[key]


def _img_seeds(key, sc):
    from make_image_table import seeds_of

    return seeds_of(_img_tmpl(key), sc)


def fig_image_aa() -> dict:
    """AA per method on both datasets (bars = mean of three seeds, dots = seeds), joint a line."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.9), sharey=True)
    out = {}
    for ax, (sc, title) in zip(axes, IMG_DS, strict=True):
        for i, k in enumerate(IMG_AA_KEYS):
            d = _img_seeds(k, sc)
            out[(k, sc)] = d
            if len(d) == 3:
                ax.barh(i, np.mean(list(d.values())), 0.7, color=IMG_STYLE[k][1], ec="black",
                        lw=0.4)  # fmt: skip
            ax.scatter(list(d.values()), [i] * len(d), s=4, color="black", zorder=3, lw=0)
        j = _img_seeds("Joint", sc)
        out[("Joint", sc)] = j
        ax.axvline(np.mean(list(j.values())), color="black", lw=0.8, ls="--")
        ax.text(np.mean(list(j.values())) - 1, -0.9, "joint", ha="right", va="center",
                fontsize=6.5)  # fmt: skip
        ax.set_title(title, pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("AA after ten sessions", labelpad=1)
        ax.tick_params(length=1.5, pad=1)
        ax.grid(axis="x", alpha=0.3, lw=0.3)
    axes[0].set_yticks(range(len(IMG_AA_KEYS)), [IMG_STYLE[k][0] for k in IMG_AA_KEYS])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.2, wspace=0.05)
    save(fig, "image_aa.pdf")
    return out


IMG_TRAJ_KEYS = ("Naive", "ErTwoK", "Bank", "Colar", "Slca")


def _img_matrix(key, sc, sd):
    run = _img_tmpl(key).format(sc=sc, sd=sd)
    f = Path("results") / run / "matrix.npy"
    if f.exists():
        return np.load(f)
    return np.array(json.loads((Path("results") / run / "metrics.json").read_text())["matrix"],
                    dtype=float)  # fmt: skip


def fig_image_traj() -> dict:
    """AA over the sessions seen after each session (mean over seeds); joint: one pooled run."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.7), sharey=True)
    out = {}
    for ax, (sc, title) in zip(axes, IMG_DS, strict=True):
        for k in IMG_TRAJ_KEYS:
            mats = [_img_matrix(k, sc, sd) for sd in SEEDS]
            y = np.mean([[np.nanmean(m[i, : i + 1]) for i in range(m.shape[0])] for m in mats], 0)
            out[(k, sc)] = y
            ax.plot(range(1, len(y) + 1), y, marker="o", ms=2, lw=0.9, color=IMG_STYLE[k][1],
                    mec="black", mew=0.2, label=IMG_STYLE[k][0])  # fmt: skip
        jv = np.mean([np.mean(_img_matrix("Joint", sc, sd)[-1]) for sd in SEEDS])
        out[("Joint", sc)] = jv
        ax.scatter(10, jv, marker="*", s=30, color="black", zorder=3, label="Joint (pooled run)")
        ax.set_title(title, pad=2)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 100)
        ax.set_xlabel("sessions trained", labelpad=1)
        ax.grid(alpha=0.25, lw=0.3)
        ax.tick_params(length=1.5, pad=1)
    axes[0].set_ylabel("AA over sessions seen", labelpad=1)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=6.5,
                   labelspacing=0.25)  # fmt: skip
    fig.subplots_adjust(left=0.07, right=0.83, top=0.88, bottom=0.22, wspace=0.06)
    save(fig, "image_traj.pdf")
    return out


IMG_BUDGET = (("ErTwoH", "200 raw images"), ("ErTwoK", "2,000 raw images"),
              ("DerTwoH", "200 raw images"), ("DerTwoK", "2,000 raw images"),
              ("Bank", "5,000 latents"), ("Colar", "5,000 latents, rank 128"))  # fmt: skip


def fig_image_budget() -> dict:
    """ER / DER++ at 200 and 2,000 raw exemplars against the latent memory (500 per session)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.6), sharey=True)
    out = {}
    for ax, (sc, title) in zip(axes, IMG_DS, strict=True):
        for i, (k, _lab) in enumerate(IMG_BUDGET):
            d = _img_seeds(k, sc)
            out[(k, sc)] = d
            ax.barh(i, np.mean(list(d.values())), 0.7, color=IMG_STYLE[k][1], ec="black", lw=0.4,
                    alpha=0.5 if k.endswith("TwoH") else 1.0)  # fmt: skip
            ax.scatter(list(d.values()), [i] * len(d), s=4, color="black", zorder=3, lw=0)
        ax.set_title(title, pad=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_xlabel("AA after ten sessions", labelpad=1)
        ax.grid(axis="x", alpha=0.3, lw=0.3)
        ax.tick_params(length=1.5, pad=1)
    axes[0].set_yticks(range(len(IMG_BUDGET)),
                       [f"{IMG_STYLE[k][0].split(',')[0]}: {lab}" for k, lab in IMG_BUDGET])  # fmt: skip
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.25, right=0.99, top=0.88, bottom=0.24, wspace=0.05)
    save(fig, "image_budget.pdf")
    return out


def _tab_cell(fname: str, label_start: str, col: int) -> str:
    """The mean printed in column ``col`` (0-based, after the label) of a generated image table."""
    line = next(x for x in (OUT / fname).read_text().splitlines() if x.startswith(label_start))
    return re.match(r"\s*([0-9.]+)", line.split("&")[1 + col]).group(1)


def selfcheck_p18(abl, img_aa, img_traj, img_budget) -> None:
    tab = (OUT / "tab_ablation.tex").read_text().splitlines()
    from make_clean_tables import ABLATION

    for label, key, *_ in ABLATION:  # ablation bars = tab_ablation.tex cells
        if key in abl:
            line = next(x for x in tab if x.startswith(label + " &"))
            for vs in abl[key]:
                assert f"{np.mean(vs):.1f}" in line, (key, line)
    CHECKS.append(f"ablation: {len(abl)} rows = tab_ablation.tex")
    core = {"Naive": "Naive &", "Lwf": "LwF &", "ErTwoK": "ER (2", "DerTwoK": "DER++ (2",
            "Coda": "CODA-Prompt &", "Cllora": "CL-LoRA", "Slca": "SLCA (2023)",
            "Ranpac": "RanPAC", "Lca": "LCA", "Bank": "\\textbf{Latent memory",
            "Colar": "\\textbf{CoLaR", "Joint": "Joint training"}  # fmt: skip
    supp = {"ErTwoH": "ER (200", "DerTwoH": "DER++ (200"}
    n = 0
    for (k, sc), d in {**img_aa, **img_budget}.items():
        ci = [s for s, _ in IMG_DS].index(sc)
        if k in core:
            assert (
                _tab_cell("tab_image_core.tex", core[k], ci) == f"{np.mean(list(d.values())):.1f}"
            ), k
        else:
            assert (
                _tab_cell("tab_image_supp.tex", supp[k], 2 * ci)
                == f"{np.mean(list(d.values())):.1f}"
            ), k
        n += 1
    for sc, _ in IMG_DS:  # trajectory end point = AA of metrics.json (mean over seeds)
        for k in IMG_TRAJ_KEYS:
            assert abs(img_traj[(k, sc)][-1] - np.mean(list(_img_seeds(k, sc).values()))) < 1e-6, k
    CHECKS.append(f"image: {n} bars = tab_image_core/supp cells; trajectories end at AA")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    cells = fig_findings_a()
    cols = fig_findings_b()
    pts = fig_findings_c()
    means = fig_heatmaps()
    dots = fig_perseed()
    traj = fig_trajectory()
    counts = fig_routing()
    spans = fig_spans()
    perclass = fig_perclass()
    write_perclass_vals(perclass)
    selfcheck(cells, cols, pts, means, dots, traj, counts, spans, perclass)
    selfcheck_p18(fig_ablation(), fig_image_aa(), fig_image_traj(), fig_image_budget())
    print("self-check passed:\n  " + "\n  ".join(CHECKS))


if __name__ == "__main__":
    main()
