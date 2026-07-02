"""Zero-training gate experiment for the Lexical Ledger idea.

Question: how much old-task entity F1 does an INPUT-ANCHORED, APPEND-ONLY memory
recover on its own — no model, no gradients, no features? The ledger maps
(subword lexeme x quantized layout cell) -> per-label evidence counts, appended
one task at a time (never overwritten). Because keys live in input space, they
cannot go stale under encoder drift — the failure mode that killed every
parametric LexSlot/LexMem variant (measured CKA 1.0 -> 0.19).

Protocol (dil: FUNSD -> SROIE -> CORD, unified 9-label space):
  1. Build the ledger from each task's TRAIN split in sequence (append-only).
  2. Evaluate the ledger ALONE on every task's EVAL split: per token, look up
     (tok, cell) then back off to (tok,); predict the majority label if support
     >= smin, else abstain -> "O".
  3. Report per task: coverage, covered-token accuracy, entity F1 (seqeval) —
     for both the task's OWN ledger (interference-free ceiling) and the
     CUMULATIVE ledger (the continual-learning end state). The delta is the
     append-only interference. Also report cross-task key-conflict rates.

GATE: cumulative-ledger entity F1 >= ~60 on old tasks => the idea is alive and
the only thing left to build is the ledger<->trunk fusion gate.

CPU-only; run from CLUE/:  uv run python scripts/lexical_ledger_gate.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch

from doccl.data.dil_remapping import DIL_UNIFIED_LABELS
from doccl.data.scenarios import get_scenario
from doccl.eval.metrics import compute_token_f1

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

O_ID = 0  # DIL_UNIFIED_LABELS[0] == "O"
ID_TO_LABEL = dict(enumerate(DIL_UNIFIED_LABELS))
CELL_GRID = 4  # 4x4 layout cells over the 0..1000 bbox space
SMIN_SWEEP = [1, 2, 5]


def _cell(bbox_row) -> int:
    cx = (int(bbox_row[0]) + int(bbox_row[2])) // 2
    cy = (int(bbox_row[1]) + int(bbox_row[3])) // 2
    gx = min(cx * CELL_GRID // 1001, CELL_GRID - 1)
    gy = min(cy * CELL_GRID // 1001, CELL_GRID - 1)
    return gy * CELL_GRID + gx


def _iter_tokens(ds):
    """Yield (tok, prev_tok, cell, label_id) for every supervised token.

    ``prev_tok`` is the previous supervised token in reading order (the encoder
    serializes in reading order), enabling relational context keys: the FUNSD
    KEY->VALUE structure is literally "value tokens follow a key lexeme".
    """
    for i in range(len(ds)):
        item = ds[i]
        ids = item["input_ids"]
        bbox = item["bbox"]
        labels = item["labels"]
        mask = item.get("attention_mask")
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        labels = labels.tolist() if torch.is_tensor(labels) else list(labels)
        prev = -1
        for j, (tok, lab) in enumerate(zip(ids, labels)):
            if lab == -100:
                continue
            if mask is not None and int(mask[j]) == 0:
                continue
            yield tok, prev, _cell(bbox[j]), lab
            prev = tok


def build_ledger(ds, ledger=None, task_id=None, key_tasks=None):
    """Append a dataset into a ledger: {key: Counter{label: count}}.

    Key families, most-specific first: bigram ("b", prev, tok), lexeme x layout
    cell ("uc", tok, cell), unigram ("u", tok), after-context ("a", prev) — the
    label of whatever FOLLOWS lexeme ``prev`` (the relational KEY->VALUE key).
    """
    ledger = ledger if ledger is not None else defaultdict(Counter)
    key_tasks = key_tasks if key_tasks is not None else defaultdict(set)
    for tok, prev, cell, lab in _iter_tokens(ds):
        ledger[("b", prev, tok)][lab] += 1
        ledger[("uc", tok, cell)][lab] += 1
        ledger[("u", tok)][lab] += 1
        ledger[("a", prev)][lab] += 1
        if task_id is not None:
            key_tasks[("uc", tok, cell)].add(task_id)
    return ledger, key_tasks


def predict(ledger, tok, prev, cell, smin):
    """Ledger lookup with specificity-ordered back-off. None = abstain."""
    for key in (("b", prev, tok), ("uc", tok, cell), ("u", tok), ("a", prev)):
        ev = ledger.get(key)
        if ev is not None and sum(ev.values()) >= smin:
            return ev.most_common(1)[0][0]
    return None


def evaluate(ledger, ds, smin):
    preds, labels, covered = [], [], 0
    for tok, prev, cell, lab in _iter_tokens(ds):
        p = predict(ledger, tok, prev, cell, smin)
        if p is not None:
            covered += 1
        preds.append(p if p is not None else O_ID)
        labels.append(lab)
    n = max(len(labels), 1)
    acc_cov = (
        sum(1 for p, y in zip(preds, labels) if p == y and p != O_ID or (p == y == O_ID))
        / n
    )
    m = compute_token_f1(preds, labels, ID_TO_LABEL)
    return {
        "n_tokens": len(labels),
        "coverage": round(covered / n, 4),
        "token_acc": round(acc_cov, 4),
        "entity_f1": round(m["f1"], 2),
        "precision": round(m["precision"], 2),
        "recall": round(m["recall"], 2),
    }


def main() -> None:
    scenario = get_scenario("dil")
    names = [t.task_name for t in scenario.tasks]
    log.info("dil tasks: %s", names)

    # Per-task ledgers (interference-free ceilings) + the cumulative ledger.
    per_task_ledgers = []
    cum_ledger, key_tasks = defaultdict(Counter), defaultdict(set)
    for tid, train_ds in enumerate(scenario.train_datasets):
        own, _ = build_ledger(train_ds)
        per_task_ledgers.append(own)
        build_ledger(train_ds, cum_ledger, task_id=tid, key_tasks=key_tasks)
        log.info("task %d (%s): ledger has %d keys cumulative", tid, names[tid], len(cum_ledger))

    # Cross-task conflict: shared (tok, cell) keys whose majority label flips when
    # another task's counts are appended (the append-only interference channel).
    shared = [k for k, ts in key_tasks.items() if len(ts) > 1]
    conflicts = 0
    for k in shared:
        # Majority under cumulative counts vs any single task would need per-task
        # ledgers; approximate: a key conflicts if >1 label has counts.
        if len(cum_ledger[k]) > 1:
            conflicts += 1
    conflict_rate = conflicts / max(len(shared), 1)
    log.info(
        "shared (tok,cell) keys across tasks: %d (%.1f%% of %d); multi-label among shared: %.1f%%",
        len(shared),
        100 * len(shared) / max(len(key_tasks), 1),
        len(key_tasks),
        100 * conflict_rate,
    )

    results = {
        "scenario": "dil",
        "tasks": names,
        "cell_grid": CELL_GRID,
        "n_keys_cumulative": len(cum_ledger),
        "shared_keys": len(shared),
        "shared_multilabel_rate": round(conflict_rate, 4),
        "eval": {},
    }
    for smin in SMIN_SWEEP:
        block = {}
        for tid, eval_ds in enumerate(scenario.eval_datasets):
            own = evaluate(per_task_ledgers[tid], eval_ds, smin)
            cum = evaluate(cum_ledger, eval_ds, smin)
            block[names[tid]] = {"own_ledger": own, "cumulative_ledger": cum}
            log.info(
                "smin=%d %s: OWN f1=%.1f cov=%.2f | CUMULATIVE f1=%.1f cov=%.2f "
                "(interference delta %.1f)",
                smin,
                names[tid],
                own["entity_f1"],
                own["coverage"],
                cum["entity_f1"],
                cum["coverage"],
                own["entity_f1"] - cum["entity_f1"],
            )
        results["eval"][f"smin_{smin}"] = block

    out = Path("results/ledger_gate.json")
    out.write_text(json.dumps(results, indent=2))
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
