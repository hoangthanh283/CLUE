"""CoLaR-R label-refresh mechanics on synthetic tensors (fast, GPU-free)."""

from __future__ import annotations

from unittest.mock import patch

import torch

from doccl.data.cil_remapping import CIL_LabelRemapper
from doccl.data.encoders import encode_with_full_labels
from doccl.methods.colar import CoLaR
from doccl.methods.colar_r import (
    CoLaRRefresh,
    oracle_refresh,
    pseudo_refresh,
    pseudo_thresholds,
    sliced_replay_loss,
    token_entropy,
)


def test_oracle_refresh_unmasks_only_known_classes_and_keeps_ignore():
    labels = torch.tensor([0, 0, 1, -100, 0])
    labels_full = torch.tensor([3, 5, 1, -100, 0])  # 3 known now, 5 not yet introduced
    out = oracle_refresh(labels, labels_full, n_current=4)
    assert out.tolist() == [3, 0, 1, -100, 0]


def test_pseudo_refresh_relabels_confident_post_capture_predictions_only():
    labels = torch.tensor([0, 0, 0, 2])
    mask = torch.tensor([1, 1, 0, 1])
    probs = torch.tensor(
        [
            [0.05, 0.05, 0.05, 0.85],  # confident class 3 (post-capture) -> relabel
            [0.30, 0.30, 0.05, 0.35],  # class 3 but high entropy -> keep O
            [0.05, 0.05, 0.05, 0.85],  # masked position -> keep
            [0.05, 0.05, 0.85, 0.05],  # already labelled -> keep
        ]
    )
    thr = torch.tensor([0.0, 0.0, 0.0, 0.8])
    out = pseudo_refresh(labels, probs, mask, n_capture=3, thresholds=thr)
    assert out.tolist() == [3, 0, 0, 2]


def test_pseudo_thresholds_quantile_floor_and_absent_classes():
    ents = torch.tensor([0.1, 0.3, 0.5, 0.0002])
    preds = torch.tensor([3, 3, 3, 4])
    thr = pseudo_thresholds(ents, preds, n_classes=6, floor=1e-3, quantile=0.5)
    assert abs(thr[3].item() - 0.3) < 1e-6
    assert abs(thr[4].item() - 1e-3) < 1e-9  # clamped to floor
    assert thr[5].item() == 0.0 and thr[0].item() == 0.0  # never relabel into absent classes


def test_token_entropy_zero_for_one_hot():
    assert token_entropy(torch.tensor([[1.0, 0.0, 0.0]])).item() < 1e-6


class _Enc:
    has_image = False

    def encode(self, image, words, boxes, word_labels, max_length=512):
        # first-subword rule stand-in: every word -> one token, plus one pad (-100)
        return {"labels": torch.tensor(list(word_labels) + [-100])}


def test_encode_with_full_labels_unpacks_both_sequences():
    out = encode_with_full_labels(
        _Enc(), None, ["a", "b", "c"], [[0] * 4] * 3, [0, 2, 0], [5, 2, 7], 9
    )
    assert out["labels"].tolist() == [0, 2, 0, -100]
    assert out["labels_full"].tolist() == [5, 2, 7, -100]


class _Items(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_remapper_emits_labels_full_only_when_full_head_given():
    native = {0: "O", 1: "B-A", 2: "B-B"}
    sess0 = {"O": 0, "B-A": 1}
    full = {"O": 0, "B-A": 1, "B-B": 2}
    ds = _Items([{"labels": torch.tensor([0, 1, 0]), "labels_full": torch.tensor([0, 1, 2])}])
    with_full = CIL_LabelRemapper(ds, native, sess0, full)[0]
    assert with_full["labels"].tolist() == [0, 1, 0]
    assert with_full["labels_full"].tolist() == [0, 1, 2]
    assert "labels_full" not in CIL_LabelRemapper(ds, native, sess0)[0]
    # unfiltered dataset (no labels_full) -> labels themselves are full
    ds2 = _Items([{"labels": torch.tensor([0, 2])}])
    assert CIL_LabelRemapper(ds2, native, full, full)[0]["labels_full"].tolist() == [0, 2]


def _method(refresh):
    def fake_init(method, _model, _config):
        method.store = []
        method.replay_batch_size = 4
        method.model = type("M", (), {"label_to_id": {}})()

    with patch.object(CoLaR, "__init__", fake_init):
        return CoLaRRefresh(None, {"refresh": refresh})


def test_oracle_pass_refreshes_stale_docs_once_and_counts_tokens():
    m = _method("oracle")
    m.store = [
        {
            "labels": torch.tensor([0, 0, -100]),
            "labels_full": torch.tensor([2, 3, -100], dtype=torch.int16),
            "n_labels": torch.tensor(2),
        },
        {"labels": torch.tensor([0, 1]), "n_labels": torch.tensor(3)},  # fresh: untouched
    ]
    m.model.label_to_id = {"O": 0, "a": 1, "b": 2}
    m._refresh_memory(task_id=1)
    assert m.store[0]["labels"].tolist() == [2, 0, -100]
    assert int(m.store[0]["n_labels"]) == 3 and int(m.store[1]["n_labels"]) == 3
    assert m.refresh_log == [{"task_id": 1, "docs": 1, "tokens_relabelled": 1}]
    m._refresh_memory(task_id=1)  # nothing stale any more
    assert len(m.refresh_log) == 1


def test_capture_drops_labels_full_unless_oracle():
    for refresh, keep in (("oracle", True), ("pseudo", False), ("sliced", False), ("none", False)):
        m = _method(refresh)

        def fake_capture(self, loader):
            self.store.append({"labels_full": torch.tensor([1, 2]), "labels": torch.tensor([0, 0])})

        with patch.object(CoLaR, "_capture_task", fake_capture):
            m._capture_task(None)
        assert ("labels_full" in m.store[0]) is keep
        if keep:
            assert m.store[0]["labels_full"].dtype == torch.int16


def test_sliced_replay_loss_ignores_post_capture_classes():
    # one doc banked with 2 classes; class 2 was introduced later and gets a huge logit on an O token
    logits = torch.tensor([[[0.0, 0.0, 9.0], [0.0, 3.0, 9.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[0, 1, -100]])
    n_capture = torch.tensor([2])
    loss = sliced_replay_loss(logits, labels, n_capture)
    full = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100
    )
    assert loss < full  # the stale O no longer competes with class 2
    loss.backward()
    assert torch.all(logits.grad[0, :, 2] == 0)  # no gradient reaches the post-capture row
    # per-doc capture sizes: a doc banked with all 3 classes keeps the full softmax
    logits2 = torch.tensor([[[0.0, 0.0, 9.0]], [[0.0, 0.0, 9.0]]])
    labels2 = torch.tensor([[0], [0]])
    per_doc = sliced_replay_loss(logits2, labels2, torch.tensor([2, 3]))
    only_full = torch.nn.functional.cross_entropy(logits2[1], labels2[1])
    # doc 0's softmax is over its 2 capture-time classes only -> ln 2; doc 1 keeps all 3
    expected = (torch.log(torch.tensor(2.0)).item() + only_full.item()) / 2
    assert abs(per_doc.item() - expected) < 1e-5


def test_sliced_mode_accepted_and_refresh_skipped():
    m = _method("sliced")
    m.store = [{"labels": torch.tensor([0]), "n_labels": torch.tensor(1)}]
    m.model.label_to_id = {"O": 0, "a": 1}
    # after_task would refresh for oracle/pseudo; sliced must not touch labels
    with patch.object(CoLaR, "after_task", lambda self, _t, _l: None):
        m.after_task(type("T", (), {"task_id": 1})(), None)
    assert m.refresh_log == [] and m.store[0]["labels"].tolist() == [0]
