"""Smoke tests — run with `pytest tests/test_smoke.py -v -m "not slow"`.

Slow tests (download model/dataset) marked with @pytest.mark.slow.
GPU tests marked with @pytest.mark.gpu.
"""
from __future__ import annotations

import pytest
import torch


def test_imports():
    """Verify all modules import cleanly."""
    from doccl.data.scenarios import get_scenario  # noqa
    from doccl.eval.cka import linear_cka  # noqa
    from doccl.eval.fisher import empirical_fisher_diagonal  # noqa
    from doccl.eval.metrics import CLMetricsTracker, compute_token_f1  # noqa
    from doccl.methods.base import ContinualMethod  # noqa
    from doccl.methods.naive import JointMultiTask, NaiveFineTune  # noqa
    from doccl.methods.ewc import EWC  # noqa
    from doccl.methods.lwf import LwF  # noqa
    from doccl.methods.er import ER  # noqa
    from doccl.methods.der import DERpp  # noqa
    from doccl.methods.o_lora import OLoRA  # noqa
    from doccl.methods.prompt_base import PromptBasedMethod, PromptPool  # noqa
    from doccl.methods.l2p import L2P  # noqa
    from doccl.methods.dualprompt import DualPrompt  # noqa
    from doccl.methods.coda_prompt import CODAPrompt  # noqa
    from doccl.methods.doccl import DocCL_A, DocCL_B, DocCL_C  # noqa
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper  # noqa
    from doccl.types import ScenarioType, ModalityMask, TaskInfo  # noqa


def test_method_registry_complete():
    """All 10 baselines + 3 DocCL candidates (+ alias) are registered."""
    import importlib.util
    from pathlib import Path

    pytest.importorskip("hydra")  # train.py is the Hydra entrypoint
    pytest.importorskip("wandb")

    spec = importlib.util.spec_from_file_location(
        "doccl_train", Path(__file__).resolve().parent.parent / "scripts" / "train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    for name in [
        "naive", "joint", "ewc", "lwf", "er", "der_pp",
        "l2p", "dualprompt", "coda_prompt", "o_lora",
        "doccl_a", "doccl_b", "doccl_c", "doccl",
    ]:
        assert name in train.METHOD_REGISTRY, f"{name} missing from METHOD_REGISTRY"


def test_metrics_tracker():
    """AA/BWT/AF computation on hand-crafted matrix."""
    from doccl.eval.metrics import CLMetricsTracker

    tracker = CLMetricsTracker(num_tasks=3)
    tracker.update(0, {0: {"f1": 90.0}})
    tracker.update(1, {0: {"f1": 75.0}, 1: {"f1": 85.0}})
    tracker.update(2, {0: {"f1": 70.0}, 1: {"f1": 78.0}, 2: {"f1": 88.0}})

    s = tracker.summary()
    # AA = (70 + 78 + 88) / 3 = 78.667
    assert abs(s["AA"] - 78.667) < 0.01
    # BWT = ((70-90) + (78-85)) / 2 = -13.5
    assert abs(s["BWT"] - (-13.5)) < 0.01
    # AF = 13.5
    assert abs(s["AF"] - 13.5) < 0.01


def test_linear_cka_self_similarity():
    """CKA(X, X) == 1, CKA(X, random) ≈ 0 for orthogonal random."""
    from doccl.eval.cka import linear_cka

    torch.manual_seed(42)
    X = torch.randn(2000, 64)
    assert abs(linear_cka(X, X) - 1.0) < 1e-5

    # CKA between independent random matrices is ~d/(n+d) for finite samples
    # (Kornblith et al. 2019); n must be >> d for it to be near 0. At n=100,
    # d=64 it concentrates around 0.39, so use n=2000 (expected ~0.03).
    Y = torch.randn(2000, 64)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka < 0.1, f"Expected low CKA between random, got {cka}"


def test_linear_cka_invariance():
    """CKA invariant to orthogonal transformation and isotropic scaling."""
    from doccl.eval.cka import linear_cka

    torch.manual_seed(42)
    X = torch.randn(200, 64)

    # Orthogonal transformation
    A = torch.randn(64, 64)
    Q, _ = torch.linalg.qr(A)
    XQ = X @ Q
    assert abs(linear_cka(X, XQ) - 1.0) < 1e-4, "CKA should be invariant to orthogonal transform"

    # Isotropic scaling
    cka_scaled = linear_cka(X, 5.0 * X)
    assert abs(cka_scaled - 1.0) < 1e-4, "CKA should be invariant to isotropic scaling"


def test_layout_signature():
    """Layout signature concentrates mass correctly."""
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

    boxes = torch.tensor(
        [
            [[0, 0, 100, 100], [50, 50, 150, 150], [0, 0, 0, 0]],  # top-left + padding
            [[800, 800, 1000, 1000], [700, 700, 900, 900], [0, 0, 0, 0]],  # bottom-right
        ]
    ).float()
    sig = LayoutLMv3Wrapper.get_layout_signature(boxes, grid_size=4)
    assert sig.shape == (2, 16)
    # First example: mass concentrated in top-left bin (idx 0)
    assert sig[0, 0] > 0.5
    # Second example: mass in bottom-right (idx 15)
    assert sig[1, 15] > 0.5
    assert torch.allclose(sig.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_reservoir_buffer_capacity():
    """Reservoir buffer respects capacity limit."""
    from doccl.methods.buffer import ReservoirBuffer

    buf = ReservoirBuffer(capacity=10)
    for _ in range(50):
        batch = {
            "input_ids": torch.zeros(4, 16, dtype=torch.long),
            "labels": torch.zeros(4, 16, dtype=torch.long),
        }
        buf.add_batch(batch)
    assert len(buf) == 10  # capacity respected
    assert buf.n_seen == 200  # 50 batches × 4 examples


def test_reservoir_buffer_sample():
    """Buffer sample returns correct batch shape."""
    from doccl.methods.buffer import ReservoirBuffer

    buf = ReservoirBuffer(capacity=20)
    for _ in range(5):
        batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "labels": torch.randint(0, 7, (4, 16)),
        }
        buf.add_batch(batch)

    sampled = buf.sample(batch_size=8)
    assert sampled is not None
    assert sampled["input_ids"].shape == (8, 16)
    assert sampled["labels"].shape == (8, 16)


def test_scenario_registry():
    """Scenario registry contains expected entries."""
    from doccl.data.scenarios import SCENARIO_REGISTRY

    expected = {"single_funsd", "single_cord", "single_sroie",
                "single_xfund", "single_wildreceipt",
                "cil_funsd", "cil_cord", "cil_wildreceipt",
                "dil", "dil_xlingual", "mixed", "pilot"}
    assert expected.issubset(set(SCENARIO_REGISTRY.keys()))


def test_prompt_pool_select_shapes():
    """PromptPool.select returns (B, top_k, L_p, D) and a scalar key-pull loss."""
    from doccl.methods.prompt_base import PromptPool

    pool = PromptPool(n_prompts=10, prompt_length=5, hidden_dim=32)
    sel, key_pull = pool.select(torch.randn(4, 32), top_k=3)
    assert sel.shape == (4, 3, 5, 32)
    assert key_pull.ndim == 0 and key_pull.item() >= 0.0


def test_coda_module_shapes():
    """CODA decomposed prompt is (B, L_p, D); orthogonality penalty is a scalar."""
    from doccl.methods.coda_prompt import _CodaModule

    coda = _CodaModule(n_components=8, prompt_length=5, hidden=32)
    prompt = coda(torch.randn(4, 32))
    assert prompt.shape == (4, 5, 32)
    pen = coda.ortho_penalty()
    assert pen.ndim == 0 and pen.item() >= 0.0


def test_router_softmax():
    """DocCL_C router emits per-pool weights that sum to one."""
    from doccl.methods.doccl import _Router

    weights = _Router(in_dim=48, n_pools=3)(torch.randn(4, 48))
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-5)


def test_dualprompt_modules_shapes():
    """DualPrompt holds one shared G-prompt + per-expert E-prompts and keys."""
    from doccl.methods.dualprompt import _DualPromptModules

    dp = _DualPromptModules(n_experts=5, g_len=4, e_len=6, hidden=32)
    assert dp.g_prompt.shape == (4, 32)
    assert len(dp.e_prompts) == 5 and dp.e_prompts[0].shape == (6, 32)
    assert dp.e_keys[0].shape == (32,)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["funsd", "cord", "sroie"])
def test_dataset_bbox_within_layoutlmv3_bounds(name):
    """Every bbox coordinate must lie in LayoutLMv3's 2D position-embedding range
    [0, 1023]; an out-of-range coord triggers a CUDA device-side assert at train
    time. CORD raw quads carry out-of-frame coords, so normalization must clamp.
    """
    from doccl.data.cord import CORDDataset
    from doccl.data.funsd import FUNSDDataset
    from doccl.data.sroie import SROIEDataset

    ds = {
        "funsd": lambda: FUNSDDataset("test"),
        "cord": lambda: CORDDataset("test", "fine"),
        "sroie": lambda: SROIEDataset("test"),
    }[name]()

    for j in range(min(len(ds), 40)):
        bbox = ds[j]["bbox"]
        assert int(bbox.min()) >= 0, f"{name}[{j}] has negative bbox coord {int(bbox.min())}"
        assert int(bbox.max()) <= 1023, f"{name}[{j}] has bbox coord {int(bbox.max())} > 1023"


@pytest.mark.slow
def test_forward_with_prompts_shape():
    """Prompt injection returns per-token logits over the (truncated) text length."""
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

    B, L, P = 2, 32, 5
    w = LayoutLMv3Wrapper(num_labels=7)
    w.eval()
    with torch.no_grad():
        logits = w.forward_with_prompts(
            input_ids=torch.randint(1, 1000, (B, L)),
            bbox=torch.randint(0, 1000, (B, L, 4)),
            pixel_values=torch.randn(B, 3, 224, 224),
            prompt_embeds=torch.randn(B, P, w.hidden_size),
            attention_mask=torch.ones(B, L, dtype=torch.long),
        )
    assert logits.shape[0] == B and logits.shape[2] == 7
    assert logits.shape[1] <= L  # prompt slots sliced off (text truncated to fit)


@pytest.mark.slow
def test_prompt_methods_train_eval():
    """L2P/DualPrompt/CODA/DocCL_C run a train step + eval end-to-end."""
    from doccl.methods.coda_prompt import CODAPrompt
    from doccl.methods.doccl import DocCL_C
    from doccl.methods.dualprompt import DualPrompt
    from doccl.methods.l2p import L2P
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
    from doccl.types import TaskInfo

    B, L = 2, 32

    def make_batch():
        return {
            "input_ids": torch.randint(1, 1000, (B, L)),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "pixel_values": torch.randn(B, 3, 224, 224),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
            "labels": torch.randint(0, 7, (B, L)),
        }

    task = TaskInfo(task_id=0, task_name="t0", label_set=[str(i) for i in range(7)])
    cfgs = {
        L2P: {"epochs": 1, "n_prompts": 5, "prompt_length": 3, "top_k": 2},
        DualPrompt: {"epochs": 1, "n_experts": 3, "g_prompt_length": 3, "e_prompt_length": 3},
        CODAPrompt: {"epochs": 1, "n_components": 5, "prompt_length": 3},
        DocCL_C: {"epochs": 1, "n_prompts": 5, "prompt_length": 3, "top_k": 2},
    }
    for cls, cfg in cfgs.items():
        m = cls(LayoutLMv3Wrapper(num_labels=7), cfg)
        m.model.id_to_label = {i: str(i) for i in range(7)}
        m.before_task(task, [make_batch()])
        tm = m.train_task(task, [make_batch()])
        assert tm.n_steps == 1
        assert 0 in m.evaluate({0: [make_batch()]})


@pytest.mark.slow
def test_doccl_ab_train_step():
    """DocCL_A (H-LoRA) and DocCL_B (Layout-Protected EWC) construct and step."""
    from doccl.methods.doccl import DocCL_A, DocCL_B
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
    from doccl.types import TaskInfo

    B, L = 2, 32

    def make_batch():
        return {
            "input_ids": torch.randint(1, 1000, (B, L)),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "pixel_values": torch.randn(B, 3, 224, 224),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
            "labels": torch.randint(0, 7, (B, L)),
        }

    task = TaskInfo(task_id=0, task_name="t0", label_set=[str(i) for i in range(7)])

    a = DocCL_A(
        LayoutLMv3Wrapper(num_labels=7),
        {"epochs": 1, "target_component": "fusion", "lora_rank": 4},
    )
    assert a.train_task(task, [make_batch()]).n_steps == 1

    b = DocCL_B(
        LayoutLMv3Wrapper(num_labels=7),
        {"epochs": 1, "target_component": "layout", "fisher_n_samples": 4},
    )
    assert b.train_task(task, [make_batch()]).n_steps == 1
    b.after_task(task, [make_batch()])  # Fisher snapshot path


@pytest.mark.slow
def test_doccl_depth_targeted_train_step():
    """The selected DocCL (depth/head-targeted) constructs, steps, and snapshots.

    Runs two consecutive tasks so the depth-scaled EWC penalty, the teacher
    distillation, and the head-replay paths all execute (the penalty is zero on
    the first task and active on the second).
    """
    from doccl.methods.doccl import DocCL
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
    from doccl.types import TaskInfo

    B, L = 2, 32

    def make_batch():
        return {
            "input_ids": torch.randint(1, 1000, (B, L)),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "pixel_values": torch.randn(B, 3, 224, 224),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
            "labels": torch.randint(0, 7, (B, L)),
        }

    task = TaskInfo(task_id=0, task_name="t0", label_set=[str(i) for i in range(7)])
    m = DocCL(
        LayoutLMv3Wrapper(num_labels=7),
        {"epochs": 1, "fisher_n_samples": 4, "buffer_size": 8, "target_depth": "all"},
    )
    m.model.id_to_label = {i: str(i) for i in range(7)}
    assert m.train_task(task, [make_batch()]).n_steps == 1
    m.after_task(task, [make_batch()])  # snapshot θ*, Fisher, teacher
    # Second task: penalty + KD + replay are now active.
    assert m.train_task(task, [make_batch()]).n_steps == 1


@pytest.mark.slow
@pytest.mark.gpu
def test_layoutlmv3_load_and_forward():
    """Verify LayoutLMv3 loads and runs a forward pass."""
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

    model = LayoutLMv3Wrapper(num_labels=7)
    assert model.hidden_size == 768
    assert model.num_layers == 12

    B, L = 2, 64
    out = model(
        input_ids=torch.zeros(B, L, dtype=torch.long),
        bbox=torch.zeros(B, L, 4, dtype=torch.long),
        pixel_values=torch.zeros(B, 3, 224, 224),
        attention_mask=torch.ones(B, L, dtype=torch.long),
    )
    assert out.logits.shape == (B, L, 7)


@pytest.mark.slow
@pytest.mark.gpu
def test_layoutlmv3_modality_masking():
    """Verify modality masking produces different outputs."""
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
    from doccl.types import ModalityMask

    torch.manual_seed(0)
    model = LayoutLMv3Wrapper(num_labels=7)
    model.eval()

    B, L = 2, 64
    inputs = dict(
        input_ids=torch.randint(1, 1000, (B, L), dtype=torch.long),
        bbox=torch.randint(0, 1000, (B, L, 4), dtype=torch.long),
        pixel_values=torch.randn(B, 3, 224, 224),
        attention_mask=torch.ones(B, L, dtype=torch.long),
    )
    with torch.no_grad():
        out_full = model(**inputs, modality_mask=ModalityMask.FULL).logits
        out_no_text = model(**inputs, modality_mask=ModalityMask.IMAGE_LAYOUT).logits
        out_no_image = model(**inputs, modality_mask=ModalityMask.TEXT_LAYOUT).logits

    # All outputs should have same shape
    assert out_full.shape == out_no_text.shape == out_no_image.shape
    # But outputs should differ (different modalities zeroed)
    assert not torch.allclose(out_full, out_no_text)
    assert not torch.allclose(out_full, out_no_image)


@pytest.mark.slow
def test_funsd_loader():
    """FUNSD loader returns expected shapes."""
    from doccl.data.funsd import FUNSDDataset

    ds = FUNSDDataset(split="test")
    assert len(ds) == 50
    item = ds[0]
    assert "input_ids" in item
    assert "bbox" in item
    assert "pixel_values" in item
    assert "labels" in item
    assert item["pixel_values"].shape == (3, 224, 224)


def test_classifier_expansion():
    """Expanding classifier preserves old logits."""
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

    pytest.importorskip("transformers")  # skip if transformers absent

    model = LayoutLMv3Wrapper(num_labels=4)
    model.label_to_id = {"O": 0, "A": 1, "B": 2, "C": 3}
    model.id_to_label = {v: k for k, v in model.label_to_id.items()}

    old_weight = model.model.classifier.weight.detach().clone()
    old_bias = model.model.classifier.bias.detach().clone()

    model.expand_classifier(["D", "E"])

    # Now 6 classes: O, A, B, C, D, E
    assert model.model.config.num_labels == 6
    assert "D" in model.label_to_id
    assert "E" in model.label_to_id

    # First 4 rows of new weight should match old weight
    new_weight = model.model.classifier.weight.detach()
    new_bias = model.model.classifier.bias.detach()
    assert torch.allclose(new_weight[:4], old_weight)
    assert torch.allclose(new_bias[:4], old_bias)
