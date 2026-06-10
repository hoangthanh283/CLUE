"""SROIE preprocessing: materialize SROIE into LayoutLMv3-friendly local JSON.

Two input sources are supported:

(A) ``--source hf`` (default, one command, no registration):
    Pull a parquet HuggingFace mirror (default ``mp-02/sroie``) and convert it to
    the local schema below. That mirror tags every entity token with a flat
    ``S-<FIELD>`` scheme + ``O``; we convert it to the 9-tag BIO scheme used by
    ``doccl.data.sroie.SROIEDataset`` by detecting runs of identical tags (first
    token -> ``B-``, rest -> ``I-``). Its bboxes are already normalized to 0-1000,
    so they are copied through verbatim (NOT re-normalized).
    Provenance note: ``mp-02/sroie`` is a community mirror (626 train / 347 test,
    matching canonical SROIE). Its BIO labels (after remap) align with
    ``SROIEDataset.LABEL_NAMES``; re-confirm before citing in the thesis.

(B) ``--raw_dir /path/to/SROIE_raw`` (canonical, needs ICDAR-2019 registration):
    Convert a manual Task-3 download. Raw structure:
        raw_dir/{train,test}/img/*.jpg, box/*.txt, entities/*.txt
    Here pixel bboxes ARE normalized to 0-1000 against each image size.

Output (written to ``--output_dir``, default ``data/sroie``), identical for both:
    data_root/sroie/
        train.json    list of {image_filename, tokens, bboxes, ner_tags}
        test.json     same format
        train/images/ copies of the receipt images
        test/images/

Usage:
    # HF mirror (default):
    python scripts/prepare_sroie.py --output_dir data/sroie
    # Manual ICDAR download:
    python scripts/prepare_sroie.py --raw_dir /path/to/SROIE_raw --output_dir data/sroie
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image


# Mapping from SROIE entity field name to BIO tags (0-indexed)
LABEL_TO_ID = {
    "O": 0,
    "B-COMPANY": 1, "I-COMPANY": 2,
    "B-DATE": 3,    "I-DATE": 4,
    "B-ADDRESS": 5, "I-ADDRESS": 6,
    "B-TOTAL": 7,   "I-TOTAL": 8,
}


def parse_box_file(box_path: Path) -> list[tuple[list[int], str]]:
    """Parse SROIE box .txt file. Each line:
        x1,y1,x2,y2,x3,y3,x4,y4,transcribed_text

    Returns list of (axis-aligned bbox [x0,y0,x1,y1], text).
    """
    items = []
    with open(box_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                xs = [int(parts[i]) for i in range(0, 8, 2)]
                ys = [int(parts[i]) for i in range(1, 8, 2)]
            except ValueError:
                continue
            text = parts[8].strip()
            if not text:
                continue
            box = [min(xs), min(ys), max(xs), max(ys)]
            items.append((box, text))
    return items


def parse_entities_file(ent_path: Path) -> dict[str, str]:
    """Parse SROIE entities JSON: {company, date, address, total}."""
    try:
        with open(ent_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def assign_labels(
    boxes_with_text: list[tuple[list[int], str]],
    entities: dict[str, str],
) -> list[int]:
    """Heuristic label assignment by string matching against entity values.

    Walks through the OCR'd boxes left-to-right, top-to-bottom, and tags
    contiguous runs of words that appear in each entity value.

    Note: this is the standard "weakly supervised" labeling used by most
    SROIE LayoutLM implementations. Spans are imperfect — known limitation.
    """
    n = len(boxes_with_text)
    labels = [0] * n  # default O

    # Normalize entity values for matching
    field_to_class = {
        "company": "COMPANY",
        "date": "DATE",
        "address": "ADDRESS",
        "total": "TOTAL",
    }
    for field, cls in field_to_class.items():
        target = (entities.get(field) or "").strip()
        if not target:
            continue
        target_norm = re.sub(r"\s+", " ", target.lower())
        target_words = target_norm.split()
        if not target_words:
            continue

        # Greedy left-to-right match
        i = 0
        while i < n:
            if labels[i] != 0:  # already labeled
                i += 1
                continue
            # Try to match starting at i
            matched_len = 0
            j = i
            while j < n and matched_len < len(target_words):
                word = boxes_with_text[j][1].lower().strip()
                if word == target_words[matched_len]:
                    matched_len += 1
                    j += 1
                else:
                    break
            if matched_len == len(target_words) and matched_len > 0:
                # Assign B- to first, I- to rest
                labels[i] = LABEL_TO_ID[f"B-{cls}"]
                for k in range(i + 1, i + matched_len):
                    labels[k] = LABEL_TO_ID[f"I-{cls}"]
                i = i + matched_len
            else:
                i += 1
    return labels


def normalize_box_to_1000(box: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


def process_split(raw_split_dir: Path, out_dir: Path, split_name: str) -> int:
    """Process one split (train/test). Returns number of examples written."""
    img_dir = raw_split_dir / "img"
    box_dir = raw_split_dir / "box"
    ent_dir = raw_split_dir / "entities"

    if not img_dir.exists():
        raise FileNotFoundError(f"Expected images at {img_dir}")
    if not box_dir.exists():
        raise FileNotFoundError(f"Expected boxes at {box_dir}")

    out_split = out_dir / split_name
    out_img = out_split / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    examples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        box_path = box_dir / f"{stem}.txt"
        ent_path = ent_dir / f"{stem}.txt" if ent_dir.exists() else None
        if not box_path.exists():
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        bxt = parse_box_file(box_path)
        if not bxt:
            continue

        entities = parse_entities_file(ent_path) if ent_path else {}
        labels = assign_labels(bxt, entities)

        tokens = [t for _, t in bxt]
        bboxes = [normalize_box_to_1000(b, W, H) for b, _ in bxt]

        # Copy image
        new_img_name = img_path.name
        shutil.copy2(img_path, out_img / new_img_name)

        examples.append({
            "image_filename": new_img_name,
            "tokens": tokens,
            "bboxes": bboxes,
            "ner_tags": labels,
        })

    out_json = out_dir / f"{split_name}.json"
    with open(out_json, "w") as f:
        json.dump(examples, f, indent=2)
    return len(examples)


def _remap_span_tags_to_bio(tag_names: list[str]) -> list[int]:
    """Convert a flat ``S-<FIELD>`` / ``O`` tag sequence to 9-tag BIO ids.

    Consecutive tokens sharing the same field are one entity run: the first
    token becomes ``B-<FIELD>`` and the rest ``I-<FIELD>``. ``O`` stays ``O``.
    Already-BIO mirrors (tags starting ``B-``/``I-``) pass through unchanged.
    """
    out: list[int] = []
    prev_cls: str | None = None
    for tag in tag_names:
        if tag == "O" or tag is None:
            out.append(LABEL_TO_ID["O"])
            prev_cls = None
            continue
        prefix, _, cls = tag.partition("-")
        cls = cls or tag
        if prefix in ("B", "I"):  # mirror is already BIO — trust it
            out.append(LABEL_TO_ID.get(tag, LABEL_TO_ID["O"]))
            prev_cls = cls
            continue
        # prefix == "S" (or anything else): derive B/I from run position
        bio = f"I-{cls}" if prev_cls == cls else f"B-{cls}"
        out.append(LABEL_TO_ID.get(bio, LABEL_TO_ID["O"]))
        prev_cls = cls
    return out


def export_split_from_hf(ds, out_dir: Path, split_name: str) -> int:
    """Convert one HF-mirror split to the local schema. Returns examples written.

    Expects each example to expose word tokens, bboxes (already 0-1000), an
    ``ner_tags`` sequence, and an in-memory PIL image. Field-name aliases are
    handled so several community mirrors work without code changes.
    """
    tag_names = ds.features["ner_tags"].feature.names

    out_split = out_dir / split_name
    out_img = out_split / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    def pick(ex: dict, *aliases: str):
        for a in aliases:
            if a in ex:
                return ex[a]
        raise KeyError(f"None of {aliases} in HF example (keys={list(ex)}).")

    examples = []
    for idx, ex in enumerate(ds):
        tokens = pick(ex, "tokens", "words")
        bboxes = pick(ex, "bboxes", "boxes", "bbox")
        tag_ids = pick(ex, "ner_tags", "labels", "tags")
        labels = _remap_span_tags_to_bio([tag_names[t] for t in tag_ids])

        image = pick(ex, "image", "img").convert("RGB")
        img_name = f"{split_name}_{idx:05d}.png"
        image.save(out_img / img_name)

        examples.append({
            "image_filename": img_name,
            "tokens": list(tokens),
            "bboxes": [list(b) for b in bboxes],  # already normalized 0-1000
            "ner_tags": labels,
        })

    out_json = out_dir / f"{split_name}.json"
    with open(out_json, "w") as f:
        json.dump(examples, f, indent=2)
    return len(examples)


def run_hf_export(hf_name: str, output_dir: Path) -> None:
    """Materialize a HuggingFace SROIE mirror into ``output_dir``."""
    from datasets import load_dataset

    for split_name in ("train", "test"):
        ds = load_dataset(hf_name, split=split_name)
        n = export_split_from_hf(ds, output_dir, split_name)
        print(f"  {split_name}: {n} examples written to {output_dir / split_name}.json")


def run_manual(raw_dir: Path, output_dir: Path) -> None:
    """Convert a manual ICDAR Task-3 download into ``output_dir``."""
    for split_name in ("train", "test"):
        raw_split = raw_dir / split_name
        if not raw_split.exists():
            print(f"WARNING: {raw_split} not found — skipping")
            continue
        n = process_split(raw_split, output_dir, split_name)
        print(f"  {split_name}: {n} examples written to {output_dir / split_name}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("hf", "manual"), default="hf",
                        help="'hf' (default): export a parquet HF mirror. "
                             "'manual': convert a local ICDAR Task-3 download (needs --raw_dir).")
    parser.add_argument("--raw_dir", type=Path, default=None,
                        help="Raw SROIE Task 3 dir (train/test subdirs). Required for --source manual.")
    parser.add_argument("--hf_name", type=str, default="mp-02/sroie",
                        help="HuggingFace SROIE mirror id (used when --source hf).")
    parser.add_argument("--output_dir", type=Path, default=Path("data/sroie"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "manual":
        if args.raw_dir is None:
            parser.error("--source manual requires --raw_dir")
        run_manual(args.raw_dir, args.output_dir)
    else:
        run_hf_export(args.hf_name, args.output_dir)

    print(f"\nDone. SROIE data ready at {args.output_dir}")
    print("Verify with: python -c \"from doccl.data.sroie import SROIEDataset; print(len(SROIEDataset('train')))\"")


if __name__ == "__main__":
    main()
