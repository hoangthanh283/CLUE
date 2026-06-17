"""XFUND dataset loader (multilingual forms).

Source: ``nnul/xfund-multilingual`` on HuggingFace — XFUND preprocessed into the same
token-classification schema as FUNSD, WITH page images bundled. 7 languages
(de, es, fr, it, ja, pt, zh); each example's ``id`` is ``<lang>_<split>_<n>``.

XFUND is the non-English sibling of FUNSD (English). It shares FUNSD's entity schema
exactly (HEADER / QUESTION / ANSWER + O → 7 BIO tags), which is why it slots into the
DIL unified label space via ``DIL_NAME_MAPPING["xfund"] = DIL_NAME_MAPPING["funsd"]``.
Used here for the cross-lingual domain-incremental scenario (``build_dil_xlingual``):
the label space is fixed across languages, so forgetting there is pure representation
drift under language shift — a clean test of the output-side forgetting finding.

Reference:
    Xu et al., "XFUND: A Benchmark Dataset for Multilingual Visually Rich Form
    Understanding", ACL 2022 Findings.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from doccl.data.encoders import KIEEncoder, get_default_encoder

# ISO code → the language whose docs we keep. XFUND has no English (English == FUNSD).
XFUND_LANGS = ("de", "es", "fr", "it", "ja", "pt", "zh")

# Process-wide cache keyed by (lang, split) so a language's split reused across scenarios
# (e.g. the same lang appearing in build_dil_xlingual and a single-task baseline) shares
# one underlying HF Arrow handle instead of re-materialising every form.
_RAW_DS_CACHE: dict[tuple[str, str], Any] = {}


def _normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """Scale an absolute-pixel box to LayoutLMv3's 0–1000 space, clamped."""
    if not width or not height:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = box
    norm = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [min(1000, max(0, c)) for c in norm]


class XFUNDDataset(Dataset):
    """XFUND multilingual form dataset wrapper (one language at a time).

    Labels (BIO scheme, identical to FUNSD):
        0: O
        1: B-HEADER     2: I-HEADER
        3: B-QUESTION   4: I-QUESTION
        5: B-ANSWER     6: I-ANSWER
    """

    LABEL_NAMES = [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]
    NUM_LABELS = 7

    def __init__(
        self,
        split: str = "train",
        lang: str = "fr",
        encoder: KIEEncoder | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
        hf_name: str = "nnul/xfund-multilingual",
    ):
        """Args:
        split: "train" or "test" ("test" maps to XFUND's "val" split).
        lang: one of XFUND_LANGS (de/es/fr/it/ja/pt/zh).
        encoder: per-backbone KIEEncoder (defaults to LayoutLMv3Encoder).
        max_length: max token length (LayoutLMv3 supports up to 512).
        label_filter: keep only examples with an in-set entity, masking the rest
                      to O (class-incremental split building).
        hf_name: HuggingFace dataset id.
        """
        if lang not in XFUND_LANGS:
            raise ValueError(f"Unknown XFUND lang {lang!r}; expected one of {XFUND_LANGS}.")
        self.split = split
        self.lang = lang
        self.max_length = max_length
        self.encoder = encoder or get_default_encoder()

        self.label_to_id = {l: i for i, l in enumerate(self.LABEL_NAMES)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        cache_key = (lang, split)
        if cache_key not in _RAW_DS_CACHE:
            # XFUND has train + val; we expose val as our "test".
            hf_split = "val" if split == "test" else "train"
            full = load_dataset(hf_name, split=hf_split)
            # Keep only this language's rows (id prefix is "<lang>_...").
            full = full.filter(lambda ex, lg=lang: str(ex["id"]).startswith(f"{lg}_"))
            _RAW_DS_CACHE[cache_key] = full
        self._ds = _RAW_DS_CACHE[cache_key]

        # The upstream ner_tags ClassLabel order differs from ours; map by NAME.
        self._native_names = self._ds.features["ner_tags"].feature.names

        self.data = self._parse_examples(self._ds)
        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _parse_examples(self, ds) -> list[dict]:
        """Build lightweight rows (no decoded image): row idx + tokens + 0–1000 boxes
        + ner_tags remapped into OUR label-id space by tag name.

        Image dimensions for normalisation are read from the (already-decoded) image
        size; XFUND boxes are absolute pixels. We read size via the image column's
        ``.size`` without keeping the decoded image in ``self.data``.
        """
        data = []
        for row in range(len(ds)):
            ex = ds[row]
            img = ex["image"]
            w, h = img.size  # (width, height)
            native_tags = ex["ner_tags"]
            tags = [self.label_to_id.get(self._native_names[t], 0) for t in native_tags]
            boxes = [_normalize_box(b, w, h) for b in ex["bboxes"]]
            data.append(
                {
                    "row": row,
                    "tokens": ex["words"],
                    "bboxes": boxes,
                    "ner_tags": tags,
                }
            )
        return data

    def _filter_by_labels(self, data: list, label_filter: list[str]) -> list:
        """CIL split: keep examples with ≥1 in-session entity; mask the rest to O.

        Mirrors FUNSDDataset._filter_by_labels (XFUND forms also contain multiple
        entity types per page, so a strict-subset filter would keep almost nothing).
        """
        o_id = self.label_to_id["O"]
        target_entity_ids = {self.label_to_id[l] for l in label_filter if l in self.label_to_id} - {
            o_id
        }

        filtered = []
        for ex in data:
            if not (set(ex["ner_tags"]) & target_entity_ids):
                continue
            masked = [t if t in target_entity_ids else o_id for t in ex["ner_tags"]]
            filtered.append({**ex, "ner_tags": masked})
        return filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.data[idx]
        image = self._ds[ex["row"]]["image"] if self.encoder.has_image else None
        return self.encoder.encode(
            image, ex["tokens"], ex["bboxes"], ex["ner_tags"], self.max_length
        )
