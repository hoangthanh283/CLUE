"""Unit tests for the AAAI-review fixes (REVIEW_AAAI_AC_2026-06-14.md).

Pure-logic tests (no GPU, no network) cover the instrument corrections:
  C1  real text-only masking          (test_apply_mask_*)
  C2/m2/m4 honest parameter groups    (test_param_groups_*)
  C4  forgetting-localizing Fisher     (test_fisher_weighted_displacement_*)
  M2  per-token CKA                     (test_select_vectors_*, test_linear_cka_*)
  C3  power floor                       (test_min_achievable_p)
  M3  depth/head-targeted DocCL knobs   (test_doccl_depth_lambda_*)
  M5  per-class F1 + label frequencies  (test_per_class_f1, test_label_frequencies)

Tests needing transformers/seqeval are skipped when those deps are absent.
"""

from __future__ import annotations

from math import comb

import pytest
import torch
import torch.nn as nn

from doccl.eval.cka import _select_vectors, linear_cka
from doccl.eval.fisher import (empirical_fisher_diagonal, fisher_drop,
                               fisher_weighted_displacement, snapshot_params)
from doccl.models import param_grouping as pg


class _Stub:
    """Minimal stand-in exposing ``named_parameters`` for grouping/fisher tests."""

    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return iter(self._named)


# ─── C2 / m2 / m4 — honest, populated parameter groups ────────────────────────
@pytest.mark.parametrize(
    "name,expected",
    [
        ("layoutlmv3.embeddings.word_embeddings.weight", "text_word_embed"),
        ("layoutlmv3.embeddings.x_position_embeddings.weight", "layout_2d_pos_embed"),
        ("layoutlmv3.embeddings.y_position_embeddings.weight", "layout_2d_pos_embed"),
        ("layoutlmv3.patch_embed.proj.weight", "image_patch_embed"),
        ("layoutlmv3.encoder.layer.0.attention.self.query.weight", "attn_qkv"),
        ("layoutlmv3.encoder.layer.3.attention.output.dense.weight", "attn_out"),
        ("layoutlmv3.encoder.layer.3.attention.output.LayerNorm.weight", "layernorm"),
        ("layoutlmv3.encoder.layer.3.intermediate.dense.weight", "ffn"),
        ("layoutlmv3.encoder.layer.3.output.dense.weight", "ffn"),
        ("layoutlmv3.encoder.rel_pos_bias.weight", "rel_pos_bias"),
        ("layoutlmv3.embeddings.position_embeddings.weight", "pos_1d_embed"),
        ("layoutlmv3.pooler.dense.weight", "cls_pooler"),
        ("classifier.weight", "classifier"),
        ("bert.encoder.layer.0.attention.self.key.weight", "attn_qkv"),
    ],
)
def test_classify_param(name, expected):
    assert pg.classify_param(name) == expected


def test_attn_out_not_in_ffn():
    """The attention output projection must NOT be bucketed into ffn (review m2)."""
    assert pg.classify_param("encoder.layer.0.attention.output.dense.weight") == "attn_out"
    assert pg.classify_param("encoder.layer.0.output.dense.weight") == "ffn"


def test_param_groups_no_empty_or_false_groups():
    names = [
        "layoutlmv3.embeddings.word_embeddings.weight",
        "layoutlmv3.encoder.layer.0.attention.self.query.weight",
        "layoutlmv3.encoder.layer.0.attention.output.dense.weight",
        "layoutlmv3.encoder.layer.0.intermediate.dense.weight",
        "classifier.weight",
    ]
    model = _Stub([(n, nn.Parameter(torch.randn(2, 2))) for n in names])
    groups = pg.param_groups(model)
    # No declared-but-unpopulated groups; the removed fusion/visual groups are absent.
    assert "visual_attn" not in groups and "fusion" not in groups
    assert "misc" not in groups  # the catch-all should stay empty for known names
    assert all(len(v) > 0 for v in groups.values())
    assert sum(len(v) for v in groups.values()) == len(names)


def test_param_groups_by_depth_buckets():
    names = [
        ("layoutlmv3.embeddings.word_embeddings.weight", nn.Parameter(torch.randn(2))),
        ("layoutlmv3.encoder.layer.0.x.weight", nn.Parameter(torch.randn(2))),
        ("layoutlmv3.encoder.layer.6.x.weight", nn.Parameter(torch.randn(2))),
        ("layoutlmv3.encoder.layer.11.x.weight", nn.Parameter(torch.randn(2))),
        ("classifier.weight", nn.Parameter(torch.randn(2))),
    ]
    d = pg.param_groups_by_depth(_Stub(names), 12)
    assert set(d) == {"input", "early", "mid", "late", "head"}
    assert all(len(v) == 1 for v in d.values())


# ─── C4 — Fisher-weighted displacement (forgetting localizer) ─────────────────
def test_fisher_weighted_displacement_localizes_movement():
    w_in = nn.Parameter(torch.zeros(4))
    w_head = nn.Parameter(torch.zeros(3))
    model = _Stub([("emb.word_embeddings.weight", w_in), ("classifier.weight", w_head)])
    fisher_old = {"emb.word_embeddings.weight": torch.ones(4), "classifier.weight": torch.ones(3)}
    before = snapshot_params(model)  # all zeros
    after = {
        "emb.word_embeddings.weight": torch.zeros(4),  # input did not move
        "classifier.weight": torch.tensor([2.0, 2.0, 2.0, 9.0]),  # grew + old rows moved by 2
    }
    groups = {"input": [w_in], "head": [w_head]}
    disp = fisher_weighted_displacement(
        fisher_old=fisher_old,
        params_before=before,
        params_after=after,
        param_groups=groups,
        model=model,
        reduction="sum",
    )
    assert disp["input"] == pytest.approx(0.0)
    # 3 old rows × (2^2) × Fisher(1) = 12; the new grown row is cropped out.
    assert disp["head"] == pytest.approx(12.0)


def test_fisher_drop_sign():
    fd = fisher_drop({"a": 1.0, "b": 2.0}, {"a": 0.5, "b": 2.0})
    assert fd["a"] == pytest.approx(-0.5)  # importance halved
    assert fd["b"] == pytest.approx(0.0)


# ─── empirical Fisher: per-sample squaring, NOT batch-sum-then-square ─────────
class _TinyTokenClassifier(nn.Module):
    """Minimal token classifier (embedding → linear head) for Fisher tests."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, input_ids, **_kw):  # noqa: ANN001
        class _Out:
            pass

        out = _Out()
        out.logits = self.head(self.emb(input_ids))
        return out


def _fisher_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def test_empirical_fisher_is_mean_of_squared_per_sample_grads():
    """The empirical Fisher must average *individually-squared* per-document
    gradients: F = (1/N) Σ_s g_s². It must NOT square a batch-summed gradient
    (Σ_s g_s)², which injects cross-terms 2Σ_{i<j} g_i g_j and inflates the
    head/backbone importance ratio. We pin the implementation to the textbook
    definition by comparing it to a hand-computed per-document reference, run
    through a DataLoader with batch_size>1 so the within-batch per-sample loop
    is exercised (a batch-sum bug would only show up with batch_size>1).
    """
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(0)

    class _DS(Dataset):
        def __init__(self, n=4):
            self.x = [
                {
                    "input_ids": torch.randint(0, 20, (5,)),
                    "labels": torch.tensor([0, 1, 2, -100, 1]),
                }
                for _ in range(n)
            ]

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i]

    model = _TinyTokenClassifier()
    ds = _DS(4)

    # Reference: explicit sum of squared per-document gradients, /N.
    ref = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for i in range(len(ds)):
        ex = ds[i]
        model.zero_grad()
        logits = model(ex["input_ids"].unsqueeze(0)).logits
        lbl = ex["labels"].unsqueeze(0)
        valid = lbl != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        safe = lbl.clone()
        safe[~valid] = 0
        nll = -log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        nll[valid].sum().backward()
        for n, p in model.named_parameters():
            ref[n] += p.grad.detach() ** 2
    for n in ref:
        ref[n] /= len(ds)

    model.zero_grad()
    dl = DataLoader(ds, batch_size=2, collate_fn=_fisher_collate)  # >1 on purpose
    got = empirical_fisher_diagonal(model, dl, n_samples=len(ds), device="cpu")

    for n in ref:
        assert torch.allclose(got[n], ref[n], atol=1e-6), f"Fisher mismatch on {n}"
    # The head must NOT be pathologically larger than the backbone the way a
    # batch-summed (Σg)² estimator would make it: sanity-bound the ratio.
    head_mean = float(got["head.weight"].mean())
    emb_mean = float(got["emb.weight"].mean())
    assert head_mean >= emb_mean  # head is genuinely more important
    assert head_mean / max(emb_mean, 1e-12) < 1e4  # but not absurdly inflated


# ─── M2 — per-token CKA ───────────────────────────────────────────────────────
def test_select_vectors_token_level_picks_valid_first_subwords():
    feat = torch.randn(2, 5, 8)
    attn = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    labels = torch.tensor([[3, -100, 4, -100, -100], [5, 2, -100, -100, -100]])
    v = _select_vectors(feat, attn, labels, token_level=True)
    assert v.shape == (4, 8)  # 2 + 2 valid first-subword tokens


def test_select_vectors_doc_level_mean_pools():
    feat = torch.randn(2, 5, 8)
    v = _select_vectors(feat, None, None, token_level=False)
    assert v.shape == (2, 8)


def test_linear_cka_identity_and_orthogonal_invariance():
    X = torch.randn(64, 8)
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-4)
    Q = torch.linalg.qr(torch.randn(8, 8))[0]
    assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-3)


# ─── C3 — statistical-power floor ─────────────────────────────────────────────
def test_min_achievable_p():
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from doccl.pilot.analyze import min_achievable_p

    assert min_achievable_p(6, 3) == pytest.approx(2 / comb(9, 3))
    assert min_achievable_p(6, 3) > 0.0167  # underpowered at 6-vs-3 (the original design)
    assert min_achievable_p(5, 5) < 0.0167  # 5-vs-5 clears the bar (the corrected design)


# ─── M3 — DocCL depth/head targeting knobs (pure logic) ───────────────────────
def _doccl_resolver():
    from doccl.methods.doccl import DocCL

    inst = object.__new__(DocCL)  # bypass __init__ (needs a model)
    return inst


def test_doccl_depth_lambda_all_targets_head_and_late():
    inst = _doccl_resolver()
    lam = inst._resolve_depth_lambda("all", {})
    assert lam["head"] > lam["late"] > lam["mid"] > 0
    assert lam["early"] == 0.0 and lam["input"] == 0.0


@pytest.mark.parametrize("target,nonzero", [("head_only", "head"), ("late_only", "late")])
def test_doccl_depth_lambda_ablations_isolate_one_bucket(target, nonzero):
    inst = _doccl_resolver()
    lam = inst._resolve_depth_lambda(target, {})
    assert lam[nonzero] > 0
    assert all(v == 0.0 for k, v in lam.items() if k != nonzero)


def test_doccl_depth_lambda_uniform_is_flat():
    inst = _doccl_resolver()
    lam = inst._resolve_depth_lambda("uniform", {"lambda_uniform": 1.0})
    assert len(set(lam.values())) == 1


def test_doccl_depth_lambda_rejects_unknown():
    inst = _doccl_resolver()
    with pytest.raises(ValueError):
        inst._resolve_depth_lambda("fusion_only", {})


# ─── M5 — per-class F1 + label frequencies ────────────────────────────────────
def test_per_class_f1_and_frequencies():
    pytest.importorskip("seqeval")
    from doccl.eval.metrics import compute_per_class_f1, label_frequencies

    label_map = {0: "O", 1: "B-KEY", 2: "I-KEY", 3: "B-VALUE", 4: "I-VALUE"}
    preds = [1, 2, 0, 3, 4]
    gold = [1, 2, 0, 3, 4]
    per_class = compute_per_class_f1(preds, gold, label_map)
    assert "KEY" in per_class and "VALUE" in per_class
    assert per_class["KEY"]["f1"] == pytest.approx(100.0)

    freq = label_frequencies(gold + [-100, -100], label_map)
    assert freq["entity"]["KEY"] == 1 and freq["entity"]["VALUE"] == 1
    assert freq["bio"]["O"] == 1  # -100 ignored


# ─── C1 — real text-only masking (needs transformers; slow) ───────────────────
@pytest.mark.slow
def test_apply_mask_text_only_keeps_input_ids():
    pytest.importorskip("transformers")
    from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
    from doccl.types import ModalityMask

    model = LayoutLMv3Wrapper(num_labels=7)
    ids = torch.randint(5, 100, (1, 8))
    px = torch.randn(1, 3, 224, 224)
    bbox = torch.randint(0, 1000, (1, 8, 4))

    ii, pv, bb = model._apply_mask(ids, px, bbox, ModalityMask.TEXT_ONLY)
    assert torch.equal(ii, ids)  # text KEPT (the C1 fix)
    assert torch.count_nonzero(pv) == 0 and torch.count_nonzero(bb) == 0

    ii2, pv2, bb2 = model._apply_mask(ids, px, bbox, ModalityMask.IMAGE_LAYOUT)
    assert not torch.equal(ii2, ids)  # text masked to PAD
    assert torch.equal(bb2, bbox)  # layout kept
