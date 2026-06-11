"""SROIE (Scanned Receipt OCR and Information Extraction) loader.

Reference:
    Huang et al., "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction",
    ICDAR 2019.

SROIE has 4 KIE fields (company, date, address, total) → 9 BIO tags.
973 receipts (626 train / 347 test).

Note: SROIE is not on HuggingFace officially. We use the community mirror
`mlpc-lab/sroie` or expect locally-prepared data in `data/sroie/`.

Format expected per example:
    {
        "image": PIL.Image,
        "tokens": list[str],
        "bboxes": list[[x0, y0, x1, y1]] (normalized to 0-1000),
        "ner_tags": list[int],
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


class SROIEDataset(Dataset):
    """SROIE 4-field KIE dataset.

    Labels: O, B-COMPANY, I-COMPANY, B-DATE, I-DATE, B-ADDRESS, I-ADDRESS,
            B-TOTAL, I-TOTAL (9 tags).
    """

    LABEL_NAMES = [
        "O",
        "B-COMPANY",
        "I-COMPANY",
        "B-DATE",
        "I-DATE",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-TOTAL",
        "I-TOTAL",
    ]

    NUM_LABELS = 9

    def __init__(
        self,
        split: str = "train",
        data_root: str | Path = "data/sroie",
        processor: LayoutLMv3Processor | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
        source: str = "local",
        hf_name: str = "mp-02/sroie",
    ):
        """Args:
            split: "train" or "test".
            data_root: local dir holding ``{split}.json`` (default source).
            source: "local" (default; reads the artifact produced by
                ``scripts/prepare_sroie.py``, which materializes the canonical
                626 train / 347 test split with proper BIO labels) or "hf"
                (load a parquet mirror directly at runtime — convenient for
                cloud runs, but ``scenarios.py`` uses the local artifact).
            hf_name: HuggingFace dataset id used when source="hf". Default
                ``mp-02/sroie`` is a parquet mirror with the canonical
                626/347 split. It tags entity tokens with a flat ``S-<FIELD>``
                scheme + ``O``; ``_load_hf`` converts those to the 9-tag BIO
                scheme by detecting runs (first token ``B-``, rest ``I-``).
                String-BIO mirrors (e.g. ``B_company``) are also normalised.
        """
        self.split = split
        self.max_length = max_length
        self.source = source
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        self.label_to_id = {l: i for i, l in enumerate(self.LABEL_NAMES)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}
        self._image_root: Path | None = None

        if source == "hf":
            self.data = self._load_hf(hf_name, split)
        elif source == "local":
            data_root = Path(data_root)
            json_path = data_root / f"{split}.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"SROIE {split} data not found at {json_path}. "
                    "Run `python scripts/prepare_sroie.py` first, or pass source='hf'."
                )
            with open(json_path) as f:
                self.data = json.load(f)
            self._image_root = data_root / split / "images"
        else:
            raise ValueError(f"Unknown SROIE source {source!r} (expected 'local' or 'hf').")

        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _load_hf(self, hf_name: str, split: str) -> list[dict]:
        """Load + normalise a HuggingFace SROIE mirror into our example schema.

        Normalises common field aliases (tokens/words, bboxes/boxes/bbox,
        ner_tags/labels/tags) and keeps the PIL image in-memory. Asserts the tag
        space matches the expected 9-BIO scheme so a mismatched mirror fails loud.
        """
        from datasets import get_dataset_split_names, load_dataset

        # Resolve the requested split against what the mirror actually publishes.
        # Several SROIE mirrors (incl. the default ``jinho8345/sroie-bio``) ship
        # ``train``/``val`` only, so fall back ``test`` -> ``val`` when needed.
        try:
            available = set(get_dataset_split_names(hf_name))
        except Exception:
            available = set()
        hf_split = split
        if split == "test" and "test" not in available and "val" in available:
            hf_split = "val"
        ds = load_dataset(hf_name, split=hf_split)

        # Some mirrors (e.g. ``mp-02/sroie``) encode ner_tags as int ids into a
        # flat ``S-<FIELD>`` / ``O`` ClassLabel rather than BIO. Detect that and
        # expose a mapper from int id -> ``S-FIELD`` string so the per-example
        # loop below can run-encode it into BIO (first token B-, rest I-).
        s_scheme_names: list[str] | None = None
        tag_feature = ds.features.get("ner_tags")
        try:
            cand = tag_feature.feature.names
            if any(n.upper().startswith("S-") for n in cand):
                s_scheme_names = cand
        except AttributeError:
            s_scheme_names = None

        def pick(ex: dict, *aliases: str):
            for a in aliases:
                if a in ex:
                    return ex[a]
            raise KeyError(f"None of {aliases} present in HF SROIE example (keys={list(ex)}).")

        def s_ids_to_bio(tag_ids: list[int]) -> list[int]:
            """Convert ``S-<FIELD>`` int ids to BIO ids via run detection."""
            out: list[int] = []
            prev_cls: str | None = None
            for tid in tag_ids:
                name = s_scheme_names[tid]
                if name == "O":
                    out.append(self.label_to_id["O"])
                    prev_cls = None
                    continue
                cls = name.split("-", 1)[1]
                bio = f"I-{cls}" if prev_cls == cls else f"B-{cls}"
                out.append(self.label_to_id.get(bio, self.label_to_id["O"]))
                prev_cls = cls
            return out

        def norm_label(t: str) -> str:
            """Normalise a mirror label string to our canonical ``B-COMPANY`` form.

            Mirrors use varied surface forms (``B_company``, ``b-company``,
            ``B-Company``). We uppercase and unify the prefix separator to ``-`` so
            they index into ``label_to_id``; unknown strings map to ``O`` (id 0).
            """
            s = t.strip().upper().replace("_", "-")
            return s if s in self.label_to_id else t

        out: list[dict] = []
        for ex in ds:
            tags = pick(ex, "ner_tags", "labels", "tags")
            if s_scheme_names is not None:  # int ids into an S-<FIELD> ClassLabel
                tags = s_ids_to_bio(tags)
            elif tags and isinstance(tags[0], str):  # string labels → normalise → ids
                norm = [norm_label(t) for t in tags]
                unknown = {orig for orig, n in zip(tags, norm) if n not in self.label_to_id}
                if unknown:
                    raise ValueError(
                        f"HF mirror {hf_name!r} has label strings {sorted(unknown)} that do "
                        f"not map onto LABEL_NAMES {self.LABEL_NAMES} — confirm provenance."
                    )
                tags = [self.label_to_id[n] for n in norm]
            if tags and max(tags) >= self.NUM_LABELS:
                raise ValueError(
                    f"HF mirror {hf_name!r} has tag id {max(tags)} >= {self.NUM_LABELS}; "
                    "its BIO scheme does not match LABEL_NAMES — confirm provenance."
                )
            bboxes = pick(ex, "bboxes", "boxes", "bbox")
            # LayoutLMv3 requires bboxes in the 0-1000 range; mirrors occasionally
            # emit values a hair over 1000 from rounding — clamp to be safe.
            bboxes = [[min(1000, max(0, int(c))) for c in box] for box in bboxes]
            out.append(
                {
                    "image": pick(ex, "image", "img"),
                    "tokens": pick(ex, "tokens", "words"),
                    "bboxes": bboxes,
                    "ner_tags": tags,
                }
            )
        return out

    def _filter_by_labels(self, data: list, label_filter: list[str]) -> list:
        """Class-incremental split: keep any example with >=1 in-session entity, and
        mask out-of-session entity tokens to 'O'.

        SROIE receipts carry company/date/address/total fields together, so a
        strict-subset filter keeps almost no documents. Standard CIL token-classification
        practice is to retain documents with at least one target-class entity and relabel
        every out-of-session entity token as background ('O') for this session.
        """
        o_id = self.label_to_id["O"]
        target_entity_ids = {
            self.label_to_id[l] for l in label_filter if l in self.label_to_id
        } - {o_id}

        filtered = []
        for ex in data:
            if not (set(ex["ner_tags"]) & target_entity_ids):
                continue
            masked_tags = [t if t in target_entity_ids else o_id for t in ex["ner_tags"]]
            filtered.append({**ex, "ner_tags": masked_tags})
        return filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.data[idx]
        if "image" in ex:  # HF source: PIL image already in-memory
            image = ex["image"].convert("RGB")
        else:  # local source: open from disk
            image = Image.open(self._image_root / ex["image_filename"]).convert("RGB")
        encoding = self.processor(
            image,
            ex["tokens"],
            boxes=ex["bboxes"],
            word_labels=ex["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
