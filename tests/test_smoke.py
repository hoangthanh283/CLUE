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
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper  # noqa
    from doccl.types import ScenarioType, ModalityMask, TaskInfo  # noqa


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
    X = torch.randn(100, 64)
    assert abs(linear_cka(X, X) - 1.0) < 1e-5

    # CKA between independent random matrices should be small (not 0 due to finite samples)
    Y = torch.randn(100, 64)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka < 0.3, f"Expected low CKA between random, got {cka}"


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
                "cil_funsd", "cil_cord", "dil", "mixed", "pilot"}
    assert expected.issubset(set(SCENARIO_REGISTRY.keys()))


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
