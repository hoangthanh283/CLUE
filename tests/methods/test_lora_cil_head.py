"""Regression test: PEFT-wrapped methods must survive class-IL head expansion.

PEFT registers the classifier in ``modules_to_save`` and routes the forward through
an internal COPY. The wrapper's ``expand_classifier`` widens only the proxy it sees,
leaving the saved copy (and the inner HF model's cached ``num_labels``) at the old
width -- so a forward on the next task's new-class label index overflowed the stale
head and the CUDA cross-entropy kernel asserted ``t < n_classes`` (a device-side
assert that took down O-LoRA and, by inheritance, CL-LoRA on every CIL scenario).

``OLoRA.before_task`` re-syncs both stale references. These tests pin that fix on
CPU (no GPU, no data) by expanding the head and running a forward whose labels
exercise the freshly added class.
"""

from __future__ import annotations

import torch

from doccl.methods.cl_lora import CLLoRA
from doccl.methods.o_lora import OLoRA
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.types import TaskInfo

_T0_LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"]
_T1_NEW = ["B-ANSWER", "I-ANSWER"]
_CFG = {
    "lr": 5e-5,
    "epochs": 1,
    "lora_rank": 8,
    "target_modules": ["query", "key", "value"],
    "kd_alpha": 1.0,
}


def _build(method_cls):
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(_T0_LABELS))
    model.label_to_id = {l: i for i, l in enumerate(_T0_LABELS)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}
    method = method_cls(model, dict(_CFG))
    return model, method


def _forward_with_new_class(model) -> float:
    """A forward whose every label is the last (newly added) class index."""
    n = len(model.id_to_label)
    ids = torch.randint(0, 100, (2, 16))
    bbox = torch.randint(0, 1000, (2, 16, 4))
    pixel_values = torch.zeros(2, 3, 224, 224)
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.full((2, 16), n - 1, dtype=torch.long)  # the new class
    out = model(
        input_ids=ids,
        bbox=bbox,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
    )
    return float(out.loss)


def test_olora_survives_cil_head_expansion():
    model, method = _build(OLoRA)
    model.expand_classifier(_T1_NEW)  # head 5 -> 7
    method.before_task(TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW), None)
    loss = _forward_with_new_class(model)  # would assert/crash pre-fix
    assert loss == loss  # not NaN
    assert model.model.classifier.out_features == len(_T0_LABELS) + len(_T1_NEW)


def test_cl_lora_survives_cil_head_expansion():
    model, method = _build(CLLoRA)
    model.expand_classifier(_T1_NEW)
    method.before_task(TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW), None)
    loss = _forward_with_new_class(model)
    assert loss == loss


def test_olora_survives_repeated_cil_expansions():
    """Multi-boundary CIL (cil_cord has 5 head growths): the sync must hold each time."""
    model, method = _build(OLoRA)
    # Three sequential expansions, mimicking task boundaries 1, 2, 3.
    for i, new in enumerate([["B-A", "I-A"], ["B-B", "I-B"], ["B-C", "I-C"]]):
        model.expand_classifier(new)
        method.before_task(
            TaskInfo(task_id=i + 1, task_name=f"t{i+1}", label_set=["O"] + new), None
        )
        loss = _forward_with_new_class(model)  # label = newest class each round
        assert loss == loss
    assert model.model.classifier.out_features == len(_T0_LABELS) + 6
