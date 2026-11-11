"""cil_funsd supplies a full-label joint pool and an unmasked test set (no real dataset)."""

import torch

from doccl.data import scenarios
from doccl.data.funsd import FUNSDDataset


class _FakeFUNSD:
    LABEL_NAMES = FUNSDDataset.LABEL_NAMES  # O, B/I-HEADER, B/I-QUESTION, B/I-ANSWER

    def __init__(self, split="train", label_filter=None):
        self.split = split
        self.id_to_label = dict(enumerate(self.LABEL_NAMES))
        keep = None if label_filter is None else {self.LABEL_NAMES.index(x) for x in label_filter}
        tags = torch.tensor([0, 1, 2, 3, 4, 5, 6, -100])
        self.labels = (
            tags
            if keep is None
            else torch.tensor([t if t.item() in keep or t.item() == -100 else 0 for t in tags])
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"labels": self.labels.clone()}


def test_cil_funsd_has_full_label_joint_pool_and_eval(monkeypatch):
    monkeypatch.setattr(scenarios, "FUNSDDataset", _FakeFUNSD)
    scen = scenarios.build_cil_funsd()
    full = [0, 1, 2, 3, 4, 5, 6, -100]  # final head = O, HEADER, QUESTION, ANSWER (BIO)
    assert scen.full_eval_dataset.underlying.split == "test"
    assert scen.full_eval_dataset[0]["labels"].tolist() == full
    assert len(scen.joint_train_datasets) == 1
    assert scen.joint_train_datasets[0].underlying.split == "train"
    assert scen.joint_train_datasets[0][0]["labels"].tolist() == full
    # the per-session copies stay masked: session 1 sees HEADER/QUESTION as O
    assert scen.train_datasets[1][0]["labels"].tolist() == [0, 0, 0, 0, 0, 5, 6, -100]
