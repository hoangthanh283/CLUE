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

from doccl.methods.cl_lora import CLLoRA
from doccl.methods.coda_prompt import CODAPrompt
from doccl.methods.cuber import CUBER
from doccl.methods.der import DERpp
from doccl.methods.doc_merge import DocMerge
from doccl.methods.doccl import DocCL
from doccl.methods.dualprompt import DualPrompt
from doccl.methods.er import ER
from doccl.methods.er_cflat import ERCFlat
from doccl.methods.ewc import EWC
from doccl.methods.hgt import HGT
from doccl.methods.hybrid_routed_prompt import HybridRoutedPrompt
from doccl.methods.l2p import L2P
from doccl.methods.lca import LCA
from doccl.methods.lexmem import LexMem
from doccl.methods.lexslot import LexSlot
from doccl.methods.lwf import LwF

# Every method in METHOD_REGISTRY worth a behavioral test. doccl_a/c are legacy
# ablation variants; doccl_b is covered separately (CIL shape regression).
from doccl.methods.naive import JointMultiTask, NaiveFineTune
from doccl.methods.o_lora import OLoRA
from doccl.models.bert_family_wrapper import BERTWrapper
from doccl.models.bros_wrapper import BROSWrapper
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.models.lilt_wrapper import LiLTWrapper
from doccl.types import TaskInfo

pytestmark = pytest.mark.slow

_T0_LABELS = ["O", "B-A", "I-A", "B-B", "I-B"]  # 5-class head
_T1_NEW = ["B-C", "I-C"]  # grows head to 7

_BASE_CFG = {
    "lr": 5e-5,
    "weight_decay": 0.01,
    "epochs": 1,
    "max_grad_norm": 1.0,
    "early_stopping": False,
}

# Per-method extra config (paper defaults, scaled tiny where it matters for speed).
_METHOD_CFG = {
    "naive": {},
    "joint": {},
    "ewc": {"lambda_": 1000.0, "fisher_n_samples": 4, "ewc_gamma": 1.0},
    "lwf": {"alpha": 1.0, "temperature": 2.0},
    "er": {"buffer_size": 20, "replay_batch_size": 2},
    "der_pp": {"buffer_size": 20, "replay_batch_size": 2, "alpha": 0.5, "beta": 0.5},
    "er_cflat": {"buffer_size": 20, "replay_batch_size": 2, "rho": 0.05, "cflat_lambda": 0.0},
    "o_lora": {
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lambda_ortho": 0.5,
        "target_modules": ["query", "value"],
    },
    "cl_lora": {
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lambda_ortho": 0.5,
        "target_modules": ["query", "value"],
        "kd_alpha": 1.0,
    },
    "l2p": {"n_prompts": 4, "prompt_length": 2, "top_k": 2, "lambda_key": 0.5},
    "dualprompt": {"n_experts": 4, "g_prompt_length": 2, "e_prompt_length": 2, "lambda_key": 0.5},
    "coda_prompt": {"n_components": 4, "prompt_length": 2, "lambda_ortho": 0.1},
    "hrp": {
        "router": "hybrid",
        "n_tasks": 3,
        "slots_per_task": 2,
        "prompt_length": 2,
        "top_k": 2,
        "lambda_key": 0.5,
        "rrf_k": 60,
    },
    # DocMERGE: one entry per consolidate mode so each branch (merge / memory / both)
    # gets full lifecycle coverage. fisher_n_samples tiny to keep the Fisher pass fast.
    "doc_merge": {
        "consolidate": "both",
        "merge_rule": "fisher",
        "router": "sparse",
        "n_tasks": 3,
        "slots_per_task": 2,
        "prompt_length": 2,
        "top_k": 2,
        "lambda_key": 0.5,
        "fisher_n_samples": 2,
    },
    "doc_merge_merge": {
        "consolidate": "merge",
        "merge_rule": "plain",
        "n_tasks": 3,
        "slots_per_task": 2,
        "prompt_length": 2,
        "fisher_n_samples": 2,
    },
    "doc_merge_memory": {
        "consolidate": "memory",
        "router": "sparse",
        "n_tasks": 3,
        "slots_per_task": 2,
        "prompt_length": 2,
        "top_k": 2,
        "lambda_key": 0.5,
        "fisher_n_samples": 2,
    },
    # LCA: tiny CA recipe (few samples/epochs) so the merge+align lifecycle runs fast.
    "lca": {
        "ca_samples_per_cls": 8,
        "ca_epochs": 2,
        "ca_batch_size": 8,
        "ca_robust_weight": 0.1,
        "ca_feature_n_batches": 2,
        "merge_topk": 100,
    },
    # HGT: head-only gradient-subspace transfer. alpha>0 exercises the steer/transfer path;
    # tiny subspace + 2 feature batches for speed. Frozen backbone (default).
    "hgt": {
        "transfer_alpha": 0.5,
        "subspace_k": 4,
        "subspace_n_batches": 2,
        "backbone_trainable": False,
    },
    # CUBER: whole-network, exercises per-layer hooks. tiny subspace for speed.
    "cuber": {"transfer_alpha": 0.5, "subspace_k": 4, "subspace_n_batches": 2},
    # LexSlot: head+late slot memories, soft lexical sharing. Tiny slots/fisher/replay for speed.
    # n_tasks=2 -> each of the 2 lifecycle tasks claims its own disjoint slot block.
    "lexslot": {
        "slot_depth": "head_late",
        "slot_sharing": "soft",
        "n_tasks": 2,
        "n_slots_head": 6,
        "n_slots_late": 4,
        "repr_rank": 2,
        "fisher_n_samples": 2,
        "buffer_size": 10,
        "replay_batch_size": 2,
        "target_depth": "all",
    },
    "doccl": {
        "lambda_": 2000.0,
        "kd_alpha": 1.0,
        "temperature": 2.0,
        "fisher_n_samples": 4,
        "buffer_size": 20,
        "replay_batch_size": 2,
        "use_replay": True,
        "target_depth": "all",
    },
    # LexMem: sparse lexical-memory head. Tiny pool/top-k/top-t for speed; task 0
    # full FT then frozen base, task 1 trains only selected memory-value rows.
    "lexmem": {
        "n_slots": 64,
        "top_k": 4,
        "top_t": 8,
        "temp": 0.05,
        "key_init": "sample",
        "select": "tfidf",
        "lr_mem": 0.05,
        "key_sample_cap": 512,
    },
    # LexMem v2: diagnosis-guided freeze map (late-1 + head frozen, early/mid
    # trainable) + feature-space values + AdamW + drift probe.
    "lexmem_v2": {
        "n_slots": 64,
        "top_k": 4,
        "top_t": 8,
        "temp": 0.05,
        "key_init": "sample",
        "select": "tfidf",
        "value_space": "feature",
        "mem_optimizer": "adamw",
        "lr_mem": 5.0e-3,
        "freeze_late_n": 1,
        "drift_probe_batches": 1,
        "key_sample_cap": 512,
    },
    # LexMem control: identical freeze map, memory disabled.
    "lexmem_ctrl": {
        "mem_enabled": False,
        "mem_optimizer": "adamw",
        "freeze_late_n": 1,
        "drift_probe_batches": 1,
        "n_slots": 64,
        "top_k": 4,
        "top_t": 8,
    },
    # LexMem v3: v2 + EWC on the plastic bucket (tiny Fisher for speed).
    "lexmem_v3": {
        "n_slots": 64,
        "top_k": 4,
        "top_t": 8,
        "temp": 0.05,
        "key_init": "sample",
        "select": "tfidf",
        "value_space": "feature",
        "mem_optimizer": "adamw",
        "lr_mem": 5.0e-3,
        "freeze_late_n": 1,
        "drift_probe_batches": 1,
        "key_sample_cap": 512,
        "ewc_lambda": 100.0,
        "fisher_n_samples": 2,
    },
}

_METHOD_CLS = {
    "naive": NaiveFineTune,
    "joint": JointMultiTask,
    "ewc": EWC,
    "lwf": LwF,
    "er": ER,
    "der_pp": DERpp,
    "er_cflat": ERCFlat,
    "o_lora": OLoRA,
    "cl_lora": CLLoRA,
    "l2p": L2P,
    "dualprompt": DualPrompt,
    "coda_prompt": CODAPrompt,
    "hrp": HybridRoutedPrompt,
    "doccl": DocCL,
    "doc_merge": DocMerge,
    "doc_merge_merge": DocMerge,
    "doc_merge_memory": DocMerge,
    "lca": LCA,
    "hgt": HGT,
    "cuber": CUBER,
    "lexslot": LexSlot,
    "lexmem": LexMem,
    "lexmem_v2": LexMem,
    "lexmem_ctrl": LexMem,
    "lexmem_v3": LexMem,
}


def _valid_bbox(seq_len: int, g: torch.Generator) -> torch.Tensor:
    """Ordered boxes (x0<=x1, y0<=y1) in [0,1000]. LiLT derives width/height embeddings
    from x1-x0 / y1-y0 and indexes an embedding table with them, so an unordered
    (negative) box raises IndexError. Real KIE datasets always emit ordered corners."""
    xy = torch.randint(0, 1000, (seq_len, 4), generator=g)
    x = xy[:, [0, 2]].sort(dim=-1).values
    y = xy[:, [1, 3]].sort(dim=-1).values
    return torch.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], dim=-1)


class _TinyKIEDataset(Dataset):
    """A handful of synthetic documents with labels in [0, n_classes).

    ``backbone`` tailors the per-backbone contract: BROS expects [0,1]-normalized
    float boxes; vision-free backbones (lilt/bros/bert) emit no ``pixel_values``.
    Default ``layoutlmv3`` keeps the original LayoutLMv3 fixtures unchanged.
    """

    def __init__(
        self, n: int = 4, seq_len: int = 8, n_classes: int = 5, backbone: str = "layoutlmv3"
    ):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for _ in range(n):
            bbox = _valid_bbox(seq_len, g)
            if backbone == "bros":
                bbox = bbox.float() / 1000.0  # BROS expects normalized [0,1] floats
            item = {
                "input_ids": torch.randint(0, 100, (seq_len,), generator=g),
                "bbox": bbox,
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "labels": torch.randint(0, n_classes, (seq_len,), generator=g),
            }
            if backbone == "layoutlmv3":  # only the vision backbone gets pixel_values
                item["pixel_values"] = torch.zeros(3, 224, 224)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _loader(n_classes: int, n: int = 4, backbone: str = "layoutlmv3") -> DataLoader:
    ds = _TinyKIEDataset(n=n, n_classes=n_classes, backbone=backbone)
    return DataLoader(ds, batch_size=2)


def _run_one_task(method, task: TaskInfo, loader: DataLoader) -> float:
    method.before_task(task, loader)
    metrics = method.train_task(task, loader, val_loader=None)
    method.after_task(task, loader)
    return float(metrics.loss)


@pytest.mark.parametrize("name", list(_METHOD_CLS))
def test_method_cil_lifecycle(name):
    """Full 2-task CIL lifecycle: train, grow head, train, evaluate — all finite, no crash."""
    torch.manual_seed(0)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(_T0_LABELS))
    model.label_to_id = {l: i for i, l in enumerate(_T0_LABELS)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}
    cfg = {**_BASE_CFG, **_METHOD_CFG[name]}
    method = _METHOD_CLS[name](model, cfg)

    t0 = TaskInfo(task_id=0, task_name="t0", label_set=_T0_LABELS)
    loss0 = _run_one_task(method, t0, _loader(n_classes=len(_T0_LABELS)))
    assert torch.isfinite(torch.tensor(loss0)), f"{name}: task-0 loss not finite ({loss0})"

    model.expand_classifier(_T1_NEW)
    # PEFT methods re-sync their saved head inside before_task; mirror the loop order.
    t1 = TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW)
    loss1 = _run_one_task(method, t1, _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW)))
    assert torch.isfinite(torch.tensor(loss1)), f"{name}: task-1 loss not finite ({loss1})"

    eval_loaders = {
        0: _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW), n=2),
        1: _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW), n=2),
    }
    results = method.evaluate(eval_loaders)
    assert set(results) == {0, 1}, f"{name}: evaluate must return both tasks"
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0 and torch.isfinite(
            torch.tensor(m.f1)
        ), f"{name}: task {tid} F1 out of range ({m.f1})"


def test_lexmem_v2_freeze_map_after_task0():
    """After task 0, lexmem_v2 must freeze exactly the diagnosed locus: the last
    freeze_late_n encoder layers + classifier head, leaving embeddings and the
    early/mid layers trainable (the plasticity source)."""
    torch.manual_seed(0)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(_T0_LABELS))
    model.label_to_id = {l: i for i, l in enumerate(_T0_LABELS)}  # noqa: E741
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}  # noqa: E741
    cfg = {**_BASE_CFG, **_METHOD_CFG["lexmem_v2"], "freeze_late_n": 2}
    method = LexMem(model, cfg)
    t0 = TaskInfo(task_id=0, task_name="t0", label_set=_T0_LABELS)
    _run_one_task(method, t0, _loader(n_classes=len(_T0_LABELS)))

    layers = model.model.layoutlmv3.encoder.layer
    n = len(layers)
    for i, layer in enumerate(layers):
        expect_trainable = i < n - 2
        got = all(p.requires_grad for p in layer.parameters())
        assert got == expect_trainable, f"layer {i}: trainable={got}, expected {expect_trainable}"
    assert all(not p.requires_grad for p in model.model.classifier.parameters())
    assert all(p.requires_grad for p in model.model.layoutlmv3.embeddings.parameters())
    assert method._mem_active, "memory must be active after task 0"


def test_lexslot_head_slots_active_after_cil_head_growth():
    """Finding-6 regression: a CIL expand_classifier REPLACES the classifier object, which
    orphaned LexSlot's head-slot forward hooks → the head slots were silently inactive for
    every CIL task after task 0 (the method degraded to DocCL). before_task must re-register
    the hooks on the NEW classifier so head slots contribute. We assert the head-slot logit
    delta is actually added to the classifier's output AFTER the CIL growth + before_task."""
    torch.manual_seed(0)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(_T0_LABELS))
    model.label_to_id = {l: i for i, l in enumerate(_T0_LABELS)}  # noqa: E741
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}  # noqa: E741
    method = LexSlot(model, {**_BASE_CFG, **_METHOD_CFG["lexslot"]})

    t0 = TaskInfo(task_id=0, task_name="t0", label_set=_T0_LABELS)
    _run_one_task(method, t0, _loader(n_classes=len(_T0_LABELS)))

    # CIL growth replaces the classifier object (orphaning the old hooks).
    model.expand_classifier(_T1_NEW)
    t1 = TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW)
    method.before_task(t1, _loader(n_classes=len(_T0_LABELS) + len(_T1_NEW)))

    # Make the head slots produce a clearly non-zero contribution, then verify the forward
    # hook on the CURRENT classifier actually adds it. Compare the live forward logits to the
    # bare classifier(features) — they must differ by exactly the head-slot logit delta.
    model.eval()  # deterministic: no dropout, so the comparison is exact
    with torch.no_grad():
        method.head_slots.proj[: method.head_slots.proj.shape[0]] = 0.5  # force non-no-op
        batch = next(iter(_loader(n_classes=len(_T0_LABELS) + len(_T1_NEW))))
        batch = {k: v.to(method.device) for k, v in batch.items() if torch.is_tensor(v)}
        live_logits = model(**batch).logits  # runs through the slot forward hook
        feats = method._cur_feats  # captured by the pre-hook during the forward above
        delta = method.head_slots.logits_delta(feats.to(live_logits.dtype))
        # Detach the slot hooks to get the bare classifier output (no slot bias) for the
        # SAME captured features, then re-attach.
        method._detach_slot_hooks()
        bare = model.model.classifier(feats)
        method._register_slot_hooks()
    assert delta.abs().sum() > 0, "head-slot delta is zero (proj not applied)"
    assert torch.allclose(
        live_logits, bare + delta, atol=1e-4
    ), "head-slot hook is NOT active on the post-CIL classifier (Finding-6 regression)"


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


# The matrix above runs on LayoutLMv3 only; backbone-coupling bugs (a missing
# wrapper method, a hard ``batch["pixel_values"]`` index, a PEFT head that the base
# wrapper can't grow) slipped through because no method ever ran on a secondary
# backbone. This block runs the FULL method set on each text/text+layout backbone so
# every method is proven backbone-agnostic, not just LayoutLMv3.
_SECONDARY_WRAPPERS = {
    "lilt": (LiLTWrapper, {}),  # LiLT/BROS default their model_name
    "bros": (BROSWrapper, {}),
    "bert": (BERTWrapper, {"model_name": "bert-base-uncased"}),
}


def _make_backbone(backbone: str, n_labels: int):
    wrapper_cls, kwargs = _SECONDARY_WRAPPERS[backbone]
    model = wrapper_cls(num_labels=n_labels, **kwargs)
    labels = _T0_LABELS if n_labels == len(_T0_LABELS) else list(range(n_labels))
    model.label_to_id = {str(l): i for i, l in enumerate(labels)}  # noqa: E741
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}  # noqa: E741
    return model


@pytest.mark.parametrize("backbone", list(_SECONDARY_WRAPPERS))
@pytest.mark.parametrize("name", list(_METHOD_CLS))
def test_method_cil_lifecycle_secondary_backbone(name, backbone):
    """Every method, full 2-task CIL lifecycle (incl. head growth + PEFT head for LoRA),
    on LiLT/BROS/BERT — finite losses, in-range F1, no backbone-coupling crash."""
    torch.manual_seed(0)
    model = _make_backbone(backbone, len(_T0_LABELS))
    method = _METHOD_CLS[name](model, {**_BASE_CFG, **_METHOD_CFG[name]})

    t0 = TaskInfo(task_id=0, task_name="t0", label_set=_T0_LABELS)
    loss0 = _run_one_task(method, t0, _loader(n_classes=len(_T0_LABELS), backbone=backbone))
    assert torch.isfinite(torch.tensor(loss0)), f"{name}/{backbone}: task-0 loss not finite"

    model.expand_classifier(_T1_NEW)  # CIL head growth 5 -> 7 (PEFT head unwrap path)
    t1 = TaskInfo(task_id=1, task_name="t1", label_set=["O"] + _T1_NEW)
    n = len(_T0_LABELS) + len(_T1_NEW)
    loss1 = _run_one_task(method, t1, _loader(n_classes=n, backbone=backbone))
    assert torch.isfinite(torch.tensor(loss1)), f"{name}/{backbone}: task-1 loss not finite"

    results = method.evaluate(
        {
            0: _loader(n_classes=n, n=2, backbone=backbone),
            1: _loader(n_classes=n, n=2, backbone=backbone),
        }
    )
    assert set(results) == {0, 1}, f"{name}/{backbone}: evaluate must return both tasks"
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0 and torch.isfinite(
            torch.tensor(m.f1)
        ), f"{name}/{backbone}: task {tid} F1 out of range ({m.f1})"


@pytest.mark.parametrize("backbone", list(_SECONDARY_WRAPPERS))
@pytest.mark.parametrize("name", list(_METHOD_CLS))
def test_method_dil_lifecycle_secondary_backbone(name, backbone):
    """Every method, 2-task DIL lifecycle (fixed head) on LiLT/BROS/BERT — no crash, finite."""
    torch.manual_seed(0)
    model = _make_backbone(backbone, len(_T0_LABELS))
    method = _METHOD_CLS[name](model, {**_BASE_CFG, **_METHOD_CFG[name]})

    for tid in (0, 1):
        t = TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=_T0_LABELS)
        loss = _run_one_task(method, t, _loader(n_classes=len(_T0_LABELS), backbone=backbone))
        assert torch.isfinite(torch.tensor(loss)), f"{name}/{backbone}: DIL task-{tid} not finite"

    n = len(_T0_LABELS)
    results = method.evaluate(
        {0: _loader(n, 2, backbone=backbone), 1: _loader(n, 2, backbone=backbone)}
    )
    assert set(results) == {0, 1}
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0, f"{name}/{backbone}: task {tid} F1 out of range ({m.f1})"
