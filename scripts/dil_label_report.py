"""Report per-task DIL label frequencies (model-free) — review M5 evidence.

Surfaces the VALUE-dominance / KEY-sparsity of the unified DIL schema: KEY
originates only from FUNSD (QUESTION) and SROIE (COMPANY), while CORD-super
contributes only VALUE/OTHER. This is the class-balance context the AC review
asks for before interpreting the DIL result (e.g. ER's near-oracle AA), so a
VALUE-dominant schema is not mistaken for genuine cross-domain retention.

Reads native ``ner_tags`` from each dataset's ``.data`` and maps them through the
``DIL_LabelRemapper`` translation table — no model and no image decode needed.

Usage:  python scripts/dil_label_report.py
Output: results/dil_label_frequencies.json  (+ a printed entity-span table)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from doccl.data.dil_remapping import DIL_UNIFIED_LABELS
from doccl.data.scenarios import build_dil


def task_frequencies(remapper) -> dict:
    """BIO-tag and entity-span counts for one DIL_LabelRemapper-wrapped task."""
    id_to_name = dict(enumerate(DIL_UNIFIED_LABELS))
    trans = remapper._id_translation  # native_id -> unified_id
    bio: Counter[str] = Counter()
    entity: Counter[str] = Counter()
    for ex in remapper.underlying.data:
        for nid in ex["ner_tags"]:
            name = id_to_name.get(trans.get(nid, 0), "O")
            bio[name] += 1
            if name.startswith("B-"):
                entity[name[2:]] += 1
    return {"bio": dict(bio), "entity": dict(entity)}


def main() -> None:
    scenario = build_dil()
    report = {
        task.task_name: task_frequencies(ds)
        for task, ds in zip(scenario.tasks, scenario.train_datasets)
    }
    out = Path("results/dil_label_frequencies.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    entities = ["HEADER", "KEY", "VALUE", "OTHER"]
    print("DIL per-task entity-span frequencies (train):")
    print(f"{'task':<16} " + " ".join(f"{e:>8}" for e in entities))
    for tname, freq in report.items():
        print(f"{tname:<16} " + " ".join(f"{freq['entity'].get(e, 0):>8}" for e in entities))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
