"""Methods must accept image-classification batches: {"pixel_values", "labels"(B,)}, 2-D logits."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from doccl.methods.der import DERpp
from doccl.methods.lca import LCA
from doccl.types import TaskInfo


class _Out:
    def __init__(self, logits, labels):
        self.logits = logits
        self.loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None


class _TinyImageModel(nn.Module):
    """Stand-in for ViTWrapper: (B,3,8,8) -> CLS feature (B,D) -> classifier."""

    task_type = "image"
    hidden_size = 16

    def __init__(self, n=4):
        super().__init__()
        self.model = nn.Module()
        self.model.classifier = nn.Linear(16, n)
        self.model.config = type("C", (), {"num_labels": n})()
        self.enc = nn.Linear(3 * 8 * 8, 16)
        self.id_to_label = {i: f"c{i}" for i in range(n)}

    def forward(self, pixel_values, labels=None, **kw):
        return _Out(self.model.classifier(self.enc(pixel_values.flatten(1))), labels)

    def token_features(self, batch):
        return self.enc(batch["pixel_values"].flatten(1))[:, None]


def _loader(n=8):
    ds = TensorDataset(torch.randn(n, 3, 8, 8), torch.randint(0, 4, (n,)))
    return DataLoader([{"pixel_values": x, "labels": y} for x, y in ds], batch_size=4)


def test_lca_evaluate_and_align_on_image_batches():
    model = _TinyImageModel()
    m = LCA(
        model,
        {
            "epochs": 1,
            "merge": False,
            "ca_epochs": 1,
            "ca_skip_O": False,
            "ca_feature_n_batches": 2,
            "ca_samples_per_cls": 8,
        },
    )
    m.device = torch.device("cpu")
    task = TaskInfo(task_id=1, task_name="t1", label_set=list(model.id_to_label.values()))
    m.after_task(task, _loader())  # class stats on (B,1,D) features + align
    res = m.evaluate({0: _loader()})
    assert 0.0 <= res[0].f1 <= 100.0


def test_lca_skips_align_when_ca_epochs_zero():
    model = _TinyImageModel()
    m = LCA(model, {"merge": False, "ca_epochs": 0, "ca_skip_O": False, "ca_feature_n_batches": 1})
    m.device = torch.device("cpu")
    before = model.model.classifier.weight.clone()
    m.after_task(TaskInfo(task_id=1, task_name="t1", label_set=[]), _loader())
    assert torch.equal(before, model.model.classifier.weight)


def test_derpp_logit_mask_broadcasts_for_2d_logits():
    model = _TinyImageModel()
    m = DERpp(
        model, {"epochs": 1, "buffer_size": 8, "replay_batch_size": 4, "alpha": 0.5, "beta": 0.5}
    )
    m.device = torch.device("cpu")
    task = TaskInfo(task_id=0, task_name="t0", label_set=list(model.id_to_label.values()))
    m.train_task(task, _loader())
    m.after_task(task, _loader())
    metrics = m.train_task(TaskInfo(task_id=1, task_name="t1", label_set=[]), _loader())
    assert torch.isfinite(torch.tensor(metrics.loss))
