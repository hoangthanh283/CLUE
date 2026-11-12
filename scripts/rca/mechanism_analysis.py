"""AGENT IMPL (2026-11-12, D16b): direct measurements of the two proposed causes of forgetting on
clean CIL-CORD runs, from SAVED files only (CPU; no model, no dataset).

Inputs per run dir under results/clean_2026-11/cil_cord_*:
  matrix.npy, metrics.json                      every run
  eval_dump.pt   blocked readout (colar_sbr*): final model, per-session test split (other sessions'
                 entities masked to O), per first-subword token: e (n, blocks) = 1 - p(O_b),
                 tag (n, blocks) = best in-block head id, gold (masked), allowed, threshold
  per_class_f1.json   shared softmax: seqeval per-class P/R/support, unrestricted argmax, per split
  full_label_dump.pt  (training.full_label_dump, from v9 on): final model on the UNMASKED test set

Units. "token" = first-subword word token with a label (-100 positions excluded). "span" = seqeval
entity (BIO, default mode). Session of a class = the session that introduced it.

  uv run python scripts/rca/mechanism_analysis.py              # writes the JSON, prints a table
  uv run python scripts/rca/mechanism_analysis.py --selfcheck  # asserts on a hand-made example
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities

ROOT = Path("results/clean_2026-11")
OUT = ROOT / "mechanism_analysis.json"


# ------------------------------------------------------------------ primitives
def owner_map(session_ids: list[list[int]]) -> dict[int, int]:
    """head id -> session that introduced it (O = 0 excluded)."""
    own: dict[int, int] = {}
    for s, ids in enumerate(session_ids):
        for c in ids:
            if c:
                own.setdefault(int(c), s)
    return own


def blocked_pred(e: torch.Tensor, tag: torch.Tensor, thr: float):
    """SBR decision: the block with the largest entity mass claims the token if mass > thr."""
    best_e, best_b = e.float().max(-1)
    claimed = best_e > thr
    pred = torch.where(claimed, tag.long().gather(1, best_b[:, None])[:, 0], 0)
    return pred, best_b, claimed


def decompose(pred, gold, owner: dict[int, int], session: int) -> dict:
    """Gold entity tokens of ``session``: correct / predicted O / another class of the same session
    (includes B-/I- swaps of the same type) / a class of a different session."""
    pred, gold = pred.tolist(), gold.tolist()
    c = {"n_gold_tokens": 0, "correct": 0, "to_O": 0, "to_same_session": 0, "to_other_session": 0}
    for p, g in zip(pred, gold, strict=True):
        if g <= 0 or owner.get(g) != session:
            continue
        c["n_gold_tokens"] += 1
        if p == g:
            c["correct"] += 1
        elif p == 0:
            c["to_O"] += 1
        elif owner.get(p) == session:
            c["to_same_session"] += 1
        else:
            c["to_other_session"] += 1
    return c


def span_counts(pred, gold, i2l: dict[int, str], types: set[str]) -> dict:
    """Span TP / FP / FN restricted to entity ``types`` (one flat sequence, like the scorer)."""
    g = {x for x in get_entities([i2l.get(int(v), "O") for v in gold]) if x[0] in types}
    p = {x for x in get_entities([i2l.get(int(v), "O") for v in pred]) if x[0] in types}
    return {"tp": len(g & p), "fp": len(p - g), "fn": len(g - p)}


def prf(c: dict) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    pr = 100 * tp / (tp + fp) if tp + fp else 0.0
    rc = 100 * tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {**c, "precision": pr, "recall": rc, "f1": f1}


def add(dicts: list[dict]) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def shares(c: dict) -> dict:
    """Error shares (fraction of WRONG gold tokens) next to the raw counts."""
    wrong = c["n_gold_tokens"] - c["correct"]
    out = {**c, "n_wrong": wrong}
    for k in ("to_O", "to_same_session", "to_other_session"):
        out[f"share_{k}"] = c[k] / wrong if wrong else None
    return out


def types_of(ids, i2l) -> set[str]:
    return {i2l[int(c)][2:] for c in ids if int(c) and int(c) in i2l}


# ------------------------------------------------------------------ blocked readout (eval_dump)
def routing(e, tag, gold, owner, session: int, thr: float) -> dict:
    """Per split: on gold entity tokens, does the owning block claim? On gold-O tokens, does the
    split's OWN block claim (a clean false positive) / ANY block claim (NOT clean on masked splits:
    a gold-O token may be another session's masked entity)."""
    _, best_b, claimed = blocked_pred(e, tag, thr)
    g = gold.long()
    ent = g > 0
    own = torch.tensor([owner.get(int(x), -1) for x in g.tolist()])
    rows = torch.arange(len(g))
    owner_e = e.float()[rows, own.clamp(min=0)]
    n_ent, bg = int(ent.sum()), ~ent
    return {
        "n_entity_tokens": n_ent,
        "routed_to_owner": int((ent & claimed & (best_b == own)).sum()),
        "unclaimed": int((ent & ~claimed).sum()),
        "claimed_by_other_block": int((ent & claimed & (best_b != own)).sum()),
        "owner_block_fires": int((ent & (owner_e > thr)).sum()),
        "n_gold_O_tokens": int(bg.sum()),
        "gold_O_claimed_by_own_block": int((bg & claimed & (best_b == session)).sum()),
        "gold_O_claimed_by_any_block_masked_split": int((bg & claimed).sum()),
    }


def analyse_eval_dump(path: Path, matrix: np.ndarray) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_sess = len(d)
    i2l = {int(k): v for k, v in d[0]["id_to_label"].items()}
    owner = owner_map([d[t]["allowed"] for t in range(n_sess)])
    sessions = []
    for t in range(n_sess):
        x = d[t]
        pred, _, _ = blocked_pred(x["e"], x["tag"], x["threshold"])
        restricted = torch.where(
            torch.tensor([owner.get(int(p), -1) == t for p in pred.tolist()]), pred, 0
        )
        sc = prf(span_counts(restricted, x["gold"], i2l, types_of(x["allowed"], i2l)))
        sessions.append(
            {
                "session": t,
                "errors_unrestricted": decompose(pred, x["gold"].long(), owner, t),
                "spans_restricted": sc,
                "matrix_f1": float(matrix[-1, t]),
                "repro_minus_matrix": sc["f1"] - float(matrix[-1, t]),
                "routing": routing(x["e"], x["tag"], x["gold"], owner, t, x["threshold"]),
            }
        )
    return {"source": "eval_dump.pt", "sessions": sessions, **_pool(sessions, n_sess)}


def _pool(sessions: list[dict], n_sess: int) -> dict:
    old, last = sessions[: n_sess - 1], sessions[n_sess - 1 :]
    out = {}
    for name, grp in (("old", old), ("last", last)):
        blk = (
            {"errors": shares(add([s["errors_unrestricted"] for s in grp]))}
            if "errors_unrestricted" in grp[0]
            else {}
        )
        if "spans_restricted" in grp[0]:
            blk["spans"] = prf(
                add([{k: s["spans_restricted"][k] for k in ("tp", "fp", "fn")} for s in grp])
            )
        if "routing" in grp[0]:
            blk["routing"] = add([s["routing"] for s in grp])
        out[name] = blk
    if "spans_restricted" in sessions[0]:
        out["pooled_all_sessions_spans"] = prf(
            add([{k: s["spans_restricted"][k] for k in ("tp", "fp", "fn")} for s in sessions])
        )
    return out


# ------------------------------------------------------------------ shared softmax (per_class_f1)
def analyse_per_class(path: Path, session_types: list[set[str]]) -> dict:
    """Own-class span counts per split. TP = recall x support exactly; predicted spans = TP /
    precision only when TP > 0 (seqeval reports P = 0 for TP = 0 whether or not it predicted), so
    FP is a LOWER bound when ``n_classes_fp_unknown`` > 0 (precision then an upper bound)."""
    pc = json.loads(path.read_text())
    sessions = []
    for t, types in enumerate(session_types):
        tp = fp = fn = unknown = 0
        for cls, v in pc.get(str(t), {}).items():
            if cls not in types:
                continue
            sup = int(round(v["support"]))
            ctp = int(round(v["recall"] * sup / 100))
            tp, fn = tp + ctp, fn + sup - ctp
            if ctp > 0 and v["precision"] > 0:
                fp += int(round(ctp * 100 / v["precision"])) - ctp
            else:
                unknown += 1
        sessions.append(
            {
                "session": t,
                "spans_unrestricted_own_classes": {
                    **prf({"tp": tp, "fp": fp, "fn": fn}),
                    "n_classes_fp_unknown": unknown,
                },
            }
        )
    n_sess = len(session_types)
    out = {"source": "per_class_f1.json", "sessions": sessions}
    for name, grp in (("old", sessions[: n_sess - 1]), ("last", sessions[n_sess - 1 :])):
        c = add(
            [{k: s["spans_unrestricted_own_classes"][k] for k in ("tp", "fp", "fn")} for s in grp]
        )
        out[name] = {
            "spans": {
                **prf(c),
                "fp_is_lower_bound": any(
                    s["spans_unrestricted_own_classes"]["n_classes_fp_unknown"] for s in grp
                ),
            }
        }
    all_c = add(
        [{k: s["spans_unrestricted_own_classes"][k] for k in ("tp", "fp", "fn")} for s in sessions]
    )
    out["pooled_all_sessions_spans"] = prf(all_c)
    return out


# ------------------------------------------------------------------ full-label dump (v9 runs)
def analyse_full_dump(path: Path, session_ids: list[list[int]]) -> dict:
    """Unmasked test set: gold is complete, so background errors and background claims are clean."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    i2l = {int(k): v for k, v in d["id_to_label"].items()}
    gold = d["gold"].long()
    if "logits" in d:
        pred = d["logits"].float().argmax(-1)
        claimed = pred > 0
    else:
        pred, _, claimed = blocked_pred(d["e"], d["tag"], d["threshold"])
    owner = owner_map(session_ids)
    n_sess = len(session_ids)
    sessions = []
    for t in range(n_sess):
        sessions.append(
            {
                "session": t,
                "errors_unrestricted": decompose(pred, gold, owner, t),
                "spans_restricted": span_counts(pred, gold, i2l, types_of(session_ids[t], i2l)),
            }
        )
    for s in sessions:
        s["spans_restricted"] = prf(s["spans_restricted"])
    bg = gold == 0
    out = {"source": "full_label_dump.pt", "sessions": sessions, **_pool(sessions, n_sess)}
    allc = span_counts(pred, gold, i2l, set().union(*[types_of(i, i2l) for i in session_ids]))
    out["full_label_f1_repro"] = prf(allc)["f1"]
    out["background"] = {
        "n_gold_O_tokens": int(bg.sum()),
        "claimed_non_O": int((bg & claimed).sum()),
    }
    return out


# ------------------------------------------------------------------ driver
def session_layout() -> tuple[list[list[int]], dict[int, str], list[int]]:
    """Session head ids and the id->label map, read from any saved blocked-readout dump (the
    CIL-CORD layout is a dataset property, identical across runs)."""
    for f in sorted(glob(f"{ROOT}/cil_cord_*/eval_dump.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        i2l = {int(k): v for k, v in d[0]["id_to_label"].items()}
        ids = [d[t]["allowed"] for t in range(len(d))]
        spans = [
            sum(1 for g in d[t]["gold"].tolist() if i2l.get(g, "O").startswith("B-"))
            for t in range(len(d))
        ]
        return ids, i2l, spans
    raise SystemExit("no eval_dump.pt found: cannot recover the session layout")


def aa_vs_full(matrix: np.ndarray, metrics: dict, span_w: list[int]) -> dict:
    row = matrix[-1]
    w = np.asarray(span_w, dtype=float)
    return {
        "AA_unweighted_mean_final_row": float(row.mean()),
        "span_weighted_mean_final_row": float(row @ w / w.sum()),
        "old_sessions_mean": float(row[:-1].mean()),
        "last_session_f1": float(row[-1]),
        "full_label_f1": metrics.get("full_label_f1"),
        "final_row": [float(x) for x in row],
    }


def run_all() -> dict:
    ids, i2l, span_w = session_layout()
    types = [types_of(i, i2l) for i in ids]
    out = {
        "units": {
            "tokens": "first-subword word tokens with a label; gold entity tokens = gold != O",
            "spans": "seqeval BIO entities (default mode) over one flat sequence per split",
            "errors_unrestricted": "final model's decision before the per-session restriction",
            "spans_restricted": "the per-session scorer (out-of-session predictions -> O), = matrix",
        },
        "gold_spans_per_session_test_split": span_w,
        "session_types": [sorted(t) for t in types],
        "runs": {},
    }
    for mf in sorted(glob(f"{ROOT}/cil_cord_*/metrics.json")):
        rd = Path(mf).parent
        if "smoke" in rd.name or not (rd / "matrix.npy").exists():
            continue
        m = json.loads(Path(mf).read_text())
        if m.get("scenario") != "cil_cord":
            continue
        mat = np.load(rd / "matrix.npy")
        r = {"method": m["method"], "seed": m["seed"], "aa_vs_full": aa_vs_full(mat, m, span_w)}
        if (rd / "eval_dump.pt").exists():
            r["masked_splits"] = analyse_eval_dump(rd / "eval_dump.pt", mat)
        elif (rd / "per_class_f1.json").exists():
            r["masked_splits"] = analyse_per_class(rd / "per_class_f1.json", types)
        if (rd / "full_label_dump.pt").exists():
            r["full_label"] = analyse_full_dump(rd / "full_label_dump.pt", ids)
        pooled = (r.get("masked_splits") or {}).get("pooled_all_sessions_spans")
        if pooled:
            r["aa_vs_full"]["pooled_own_class_f1_from_splits"] = pooled["f1"]
        out["runs"][rd.name] = r
    return out


def _pct(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def print_table(res: dict) -> None:
    print(f"gold spans per session split: {res['gold_spans_per_session_test_split']}")
    hdr = (
        f"{'run':44s} {'AA':>5s} {'wAA':>5s} {'pool':>5s} {'full':>5s} | old: {'P':>5s} {'R':>5s}"
        f" {'wrong':>5s} {'%O':>5s} {'%same':>5s} {'%oth':>5s} {'route':>5s} | last: {'P':>5s} {'R':>5s}"
    )
    print(hdr)
    for name, r in res["runs"].items():
        a, ms = r["aa_vs_full"], r.get("masked_splits") or {}
        old, last = ms.get("old", {}), ms.get("last", {})
        oe, osp, osr = old.get("errors"), old.get("spans", {}), old.get("routing")
        fl = a["full_label_f1"]
        pool = a.get("pooled_own_class_f1_from_splits")
        route = osr["routed_to_owner"] / osr["n_entity_tokens"] if osr else None
        print(
            f"{name[:44]:44s} {a['AA_unweighted_mean_final_row']:5.1f} {a['span_weighted_mean_final_row']:5.1f}"
            f" {pool if pool is not None else float('nan'):5.1f} {fl if fl is not None else float('nan'):5.1f} |"
            f" {osp.get('precision', float('nan')):5.1f} {osp.get('recall', float('nan')):5.1f}"
            f" {oe['n_wrong'] if oe else '-':>5} {_pct(oe and oe['share_to_O'])} {_pct(oe and oe['share_to_same_session'])}"
            f" {_pct(oe and oe['share_to_other_session'])} {_pct(route)} |"
            f" {last.get('spans', {}).get('precision', float('nan')):5.1f} {last.get('spans', {}).get('recall', float('nan')):5.1f}"
        )


# ------------------------------------------------------------------ self-check
def selfcheck() -> None:
    i2l = {0: "O", 1: "B-a", 2: "I-a", 3: "B-b", 4: "B-c"}
    owner = owner_map([[0, 1, 2, 3], [0, 4]])  # session 0: a, b; session 1: c
    assert owner == {1: 0, 2: 0, 3: 0, 4: 1}
    gold = torch.tensor([1, 2, 0, 3, 3, 4, 0])
    pred = torch.tensor([1, 0, 0, 1, 4, 0, 4])  # B-a ok, I-a->O, B-b->B-a, B-b->c, c->O
    c = decompose(pred, gold, owner, 0)
    assert c == {
        "n_gold_tokens": 4,
        "correct": 1,
        "to_O": 1,
        "to_same_session": 1,
        "to_other_session": 1,
    }, c
    assert decompose(pred, gold, owner, 1)["to_O"] == 1
    sc = prf(span_counts(pred, gold, i2l, {"a", "b"}))
    # gold spans a[0:1], b[3], b[4]; pred spans a[0], a[3] -> a[0] != a[0:1]: tp 0, fp 2, fn 3
    assert (sc["tp"], sc["fp"], sc["fn"]) == (0, 2, 3), sc
    # blocked readout: 2 blocks, threshold .5
    e = torch.tensor([[0.9, 0.1], [0.2, 0.3], [0.6, 0.8], [0.7, 0.2]])
    tag = torch.tensor([[1, 4], [3, 4], [3, 4], [3, 4]])
    p, b, cl = blocked_pred(e, tag, 0.5)
    assert (
        p.tolist() == [1, 0, 4, 3]
        and b.tolist() == [0, 1, 1, 0]
        and cl.tolist() == [True, False, True, True]
    )
    g = torch.tensor([1, 3, 3, 0])
    r = routing(e, tag, g, owner, 0, 0.5)
    assert r["routed_to_owner"] == 1 and r["unclaimed"] == 1 and r["claimed_by_other_block"] == 1
    assert r["owner_block_fires"] == 2 and r["gold_O_claimed_by_own_block"] == 1
    w = aa_vs_full(np.array([[50.0, 0], [80.0, 20.0]]), {"full_label_f1": 70.0}, [3, 1])
    assert w["AA_unweighted_mean_final_row"] == 50.0 and w["span_weighted_mean_final_row"] == 65.0
    # full-label dump (shared softmax): logits whose argmax is ``pred`` above
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "full_label_dump.pt"
        logits = torch.nn.functional.one_hot(pred, 5).half()
        torch.save({"gold": gold.short(), "logits": logits, "id_to_label": i2l}, f)
        fd = analyse_full_dump(f, [[0, 1, 2, 3], [0, 4]])
    assert fd["old"]["errors"]["to_O"] == 1 and fd["last"]["errors"]["to_O"] == 1
    assert fd["background"] == {"n_gold_O_tokens": 2, "claimed_non_O": 1}
    print("selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    res = run_all()
    OUT.write_text(json.dumps(res, indent=1))
    print_table(res)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
