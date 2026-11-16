"""cil_wildreceipt training documents carry their unmasked labels (no real dataset)."""

import torch

from doccl.data import scenarios
from doccl.data.wildreceipt import WildReceiptDataset

NAMES = ["O"] + [f"{p}-{c}" for c in ("a", "b", "c", "d") for p in ("B", "I")]


class _FakeWildReceipt:
    LABEL_NAMES = NAMES

    def __init__(self, split="train", label_filter=None):
        self.split = split
        self.id_to_label = dict(enumerate(NAMES))
        keep = None if label_filter is None else {NAMES.index(x) for x in label_filter}
        self.full = torch.tensor([0, 1, 3, 5, 7, -100])  # O, B-a, B-b, B-c, B-d
        self.filtered = keep is not None
        self.labels = (
            self.full
            if keep is None
            else torch.tensor([t if t in keep or t == -100 else 0 for t in self.full.tolist()])
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        out = {"labels": self.labels.clone()}
        if self.filtered:
            out["labels_full"] = self.full.clone()
        return out


def test_cil_wildreceipt_train_items_carry_labels_full(monkeypatch):
    monkeypatch.setattr(scenarios, "WildReceiptDataset", _FakeWildReceipt)

    scen = scenarios.build_cil_wildreceipt(num_sessions=2)

    # head: O, B-a, B-b, I-a, I-b, B-c, B-d, I-c, I-d; session 2 annotates c and d only
    item = scen.train_datasets[1][0]
    assert item["labels"].tolist() == [0, 0, 0, 5, 6, -100]
    assert item["labels_full"].tolist() == [0, 1, 2, 5, 6, -100]
    assert "labels_full" not in scen.eval_datasets[1][0]


def test_wildreceipt_filter_keeps_the_unmasked_tags():
    stub = type("Stub", (), {"label_to_id": {n: i for i, n in enumerate(NAMES)}})()

    out = WildReceiptDataset._filter_by_labels(stub, [{"ner_tags": [0, 1, 5]}], ["B-c"])

    assert out == [{"ner_tags": [0, 0, 5], "ner_tags_full": [0, 1, 5]}]
