"""SROIE preprocessing: download raw data and convert to LayoutLMv3-friendly JSON.

SROIE has no canonical HuggingFace mirror. The official source is the ICDAR2019
robust reading challenge website which now requires registration. We expect the
user to have downloaded SROIE manually (Task 3) and to point this script at the
extracted files.

Expected raw structure:
    raw_dir/
        train/
            img/             *.jpg
            box/             *.txt   (bbox + transcribed text per line)
            entities/        *.txt   (JSON: {company, date, address, total})
        test/
            img/, box/, entities/

Output (written to data_root/sroie/):
    data_root/sroie/
        train.json    list of {image_filename, tokens, bboxes, ner_tags}
        test.json     same format
        train/images/ symlinks or copies of raw images
        test/images/

Usage:
    python scripts/prepare_sroie.py \
        --raw_dir /path/to/SROIE_raw \
        --output_dir data/sroie
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True,
                        help="Directory containing raw SROIE Task 3 data (train/test subdirs)")
    parser.add_argument("--output_dir", type=Path, default=Path("data/sroie"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        raw_split = args.raw_dir / split_name
        if not raw_split.exists():
            print(f"WARNING: {raw_split} not found — skipping")
            continue
        n = process_split(raw_split, args.output_dir, split_name)
        print(f"  {split_name}: {n} examples written to {args.output_dir / split_name}.json")

    print(f"\nDone. SROIE data ready at {args.output_dir}")
    print("Verify with: python -c \"from doccl.data.sroie import SROIEDataset; print(len(SROIEDataset('train')))\"")


if __name__ == "__main__":
    main()
