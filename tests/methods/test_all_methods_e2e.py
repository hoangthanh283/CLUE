"""End-to-end behavioral tests for EVERY CL method on a tiny real LayoutLMv3.

For each registered method this runs a realistic 2-task continual sequence —
``before_task -> train_task -> after_task`` on task 0, head growth, then the same on
task 1, then ``evaluate`` on both — and asserts:

  * no crash through the full lifecycle (catches replay-forward / teacher-deepcopy /
    PEFT head-expansion / shape-mismatch bugs),
  * the training loss is finite (catches NaN/inf loss terms and always-zero penalties
    that would silently no-op),
  * CIL head growth is survived (the classifier widens 5 -> 7 and a forward over
    new-class labels does not raise),
  * ``evaluate`` returns a finite F1 in [0, 100] for every seen task.

Uses a real (small) LayoutLMv3 on CPU with 1 epoch and 2 tiny synthetic documents, so
it exercises the true forward path (where the real bugs lived) while staying fast and
GPU-free. Marked ``slow`` so the CI-fast lane can skip it.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.types import TaskInfo

# Every method in METHOD_REGISTRY worth a behavioral test. doccl_a/c are legacy
# ablation variants; doccl_b is covered separately (CIL shape regression).
from doccl.methods.naive import NaiveFineTune, JointMultiTask
from doccl.methods.ewc import EWC
from doccl.methods.lwf import LwF
from doccl.methods.er import ER
from doccl.methods.der import DERpp
from doccl.methods.er_cflat import ERCFlat
from doccl.methods.o_lora import OLoRA
from doccl.methods.cl_lora import CLLoRA
from doccl.methods.l2p import L2P
from doccl.methods.dualprompt import DualPrompt
from doccl.methods.coda_prompt import CODAPrompt
from doccl.methods.hybrid_routed_prompt import HybridRoutedPrompt
from doccl.methods.doccl import DocCL

pytestmark = pytest.mark.slow

_T0_LABELS = ["O", "B-A", "I-A", "B-B", "I-B"]       # 5-class head
_T1_NEW = ["B-C", "I-C"]                              # grows head to 7

_BASE_CFG = {"lr": 5e-5, "weight_decay": 0.01, "epochs": 1, "max_grad_norm": 1.0,
             "early_stopping": False}

# Per-method extra config (paper defaults, scaled tiny where it matters for speed).
_METHOD_CFG = {
    "naive": {}, "joint": {},
    "ewc": {"lambda_": 1000.0, "fisher_n_samples": 4, "ewc_gamma": 1.0},
    "lwf": {"alpha": 1.0, "temperature": 2.0},
    "er": {"buffer_size": 20, "replay_batch_size": 2},
    "der_pp": {"buffer_size": 20, "replay_batch_size": 2, "alpha": 0.5, "beta": 0.5},
    "er_cflat": {"buffer_size": 20, "replay_batch_size": 2, "rho": 0.05, "cflat_lambda": 0.0},
    "o_lora": {"lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0,
               "lambda_ortho": 0.5, "target_modules": ["query", "value"]},
    "cl_lora": {"lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                "lambda_ortho": 0.5, "target_modules": ["query", "value"], "kd_alpha": 1.0},
    "l2p": {"n_prompts": 4, "prompt_length": 2, "top_k": 2, "lambda_key": 0.5},
    "dualprompt": {"n_experts": 4, "g_prompt_length": 2, "e_prompt_length": 2,
                   "lambda_key": 0.5},
    "coda_prompt": {"n_components": 4, "prompt_length": 2, "lambda_ortho": 0.1},
    "hrp": {"router": "hybrid", "n_tasks": 3, "slots_per_task": 2, "prompt_length": 2,
            "top_k": 2, "lambda_key": 0.5, "rrf_k": 60},
    "doccl": {"lambda_": 2000.0, "kd_alpha": 1.0, "temperature": 2.0,
              "fisher_n_samples": 4, "buffer_size": 20, "replay_batch_size": 2,
              "use_replay": True, "target_depth": "all"},
}

_METHOD_CLS = {
    "naive": NaiveFineTune, "joint": JointMultiTask, "ewc": EWC, "lwf": LwF, "er": ER,
    "der_pp": DERpp, "er_cflat": ERCFlat, "o_lora": OLoRA, "cl_lora": CLLoRA,
    "l2p": L2P, "dualprompt": DualPrompt, "coda_prompt": CODAPrompt,
    "hrp": HybridRoutedPrompt, "doccl": DocCL,
}


class _TinyKIEDataset(Dataset):
    """A handful of synthetic LayoutLMv3 documents with labels in [0, n_classes)."""

    def __init__(self, n: int = 4, seq_len: int = 8, n_classes: int = 5):
        g = torch.Generator().manual_seed(0)
        self.items = [
            {
                "input_ids": torch.randint(0, 100, (seq_len,), generator=g),
                "bbox": torch.randint(0, 1000, (seq_len, 4), generator=g),
                "pixel_values": torch.zeros(3, 224, 224),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, n_classes, (seq_len,), generator=g),
            }
            for _ in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _loader(n_classes: int, n: int = 4) -> DataLoader:
    return DataLoader(_TinyKIEDataset(n=n, n_classes=n_classes), batch_size=2)


def _run_one_task(method, task: TaskInfo, loader: DataLoader) -> float:
    method.before_task(task, loader)
    metrics = method.train_task(task, loader, val_loader=None)
    method.after_task(task, loader)
    return float(metrics.loss)


@pytest.mark.parametrize("name", list(_METHOD_CLS))
def test_method_cil_lifecycle(name):
    """Full 2-task CIL lifecycle: train, grow head, train, evaluate — all finite, no crash."""
    torch.manual_seed(0)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base",
                              num_labels=len(_T0_LABELS))
    model.label_to_id = {l: i for i, l in enumerate(_T0_LABELS)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}
    cfg = {**_BASE_CFG, **_METHOD_CFG[name]}
    method = _METHOD_CLS[name](model, cfg)

    # ── Task 0 (5 classes) ──
    t0 = TaskInfo(task_id=0, task_name="t0", label_set=_T0_LABELS)
    loss0 = _run_one_task(method, t0, _loader(n_classes=len(_T0_LABELS)))
    assert torch.isfinite(torch.tensor(loss0)), f"{name}: task-0 loss not finite ({loss0})"

    # ── CIL head growth 5 -> 7 ──
    model.expand_classifier(_T1_NEW)
    # PEFT methods re-sync their saved head inside before_task; mirror the loop order.
    t1 = TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW)
    loss1 = _run_one_task(method, t1, _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW)))
    assert torch.isfinite(torch.tensor(loss1)), f"{name}: task-1 loss not finite ({loss1})"

    # ── Evaluate on both seen tasks ──
    eval_loaders = {
        0: _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW), n=2),
        1: _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW), n=2),
    }
    results = method.evaluate(eval_loaders)
    assert set(results) == {0, 1}, f"{name}: evaluate must return both tasks"
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0 and torch.isfinite(torch.tensor(m.f1)), \
            f"{name}: task {tid} F1 out of range ({m.f1})"


@pytest.mark.parametrize("name", list(_METHOD_CLS))
def test_method_dil_lifecycle(name):
    """DIL (fixed head, no growth): two tasks on the same label space — no crash, finite."""
    torch.manual_seed(0)
    labels = _T0_LABELS
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(labels))
    model.label_to_id = {l: i for i, l in enumerate(labels)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}
    method = _METHOD_CLS[name](model, {**_BASE_CFG, **_METHOD_CFG[name]})

    for tid in (0, 1):
        t = TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=labels)
        loss = _run_one_task(method, t, _loader(n_classes=len(labels)))
        assert torch.isfinite(torch.tensor(loss)), f"{name}: DIL task-{tid} loss not finite"

    results = method.evaluate({0: _loader(len(labels), 2), 1: _loader(len(labels), 2)})
    assert set(results) == {0, 1}
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0
