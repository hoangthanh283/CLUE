"""Tests for the secondary backbones (LiLT, BROS) + per-backbone encoders.

Offline-safe tests (no network, no GPU): param-group classification invariants
and the subword box/label alignment logic exercised with a stub tokenizer. The
full wrapper round-trips (``from_pretrained`` → forward → ``expand_classifier`` →
``forward_with_prompts``) download weights, so they are marked ``integration`` and
run on the grid machine.

Run offline subset:  pytest tests/test_backbones.py -m "not integration"
"""

from __future__ import annotations

import pytest
import torch

from doccl.data import encoders
from doccl.data.encoders import BROSEncoder, LiLTEncoder, build_encoder
from doccl.models import param_grouping
from doccl.models.bert_family_wrapper import BERTWrapper
from doccl.models.bros_wrapper import BROSWrapper
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.models.lilt_wrapper import LiLTWrapper

# ─────────────────────────────── param grouping ────────────────────────────────

# Representative parameter names from LiLT / BROS that must NOT fall into ``misc``.
_LAYOUT_NAMES = [
    "lilt.layout_embeddings.x_position_embeddings.weight",
    "lilt.layout_embeddings.y_position_embeddings.weight",
    "lilt.layout_embeddings.box_linear_embeddings.weight",
    "bros.embeddings.bbox_projection.weight",
    "bros.embeddings.bbox_sinusoid_emb.x_pos_emb.inv_freq",
]

_NON_MISC_NAMES = _LAYOUT_NAMES + [
    "lilt.embeddings.word_embeddings.weight",  # text_word_embed
    "bros.embeddings.word_embeddings.weight",  # text_word_embed
    "lilt.encoder.layer.0.attention.self.query.weight",  # attn_qkv
    "bros.encoder.layer.3.intermediate.dense.weight",  # ffn
    "classifier.weight",  # classifier
]


@pytest.mark.parametrize("name", _LAYOUT_NAMES)
def test_layout_params_classified_as_layout(name):
    assert param_grouping.classify_param(name) == "layout_2d_pos_embed"


@pytest.mark.parametrize("name", _NON_MISC_NAMES)
def test_no_real_param_lands_in_misc(name):
    """The ``misc`` group must stay empty for stock LiLT/BROS parameter names."""
    assert param_grouping.classify_param(name) != "misc"


# ─────────────────────────── subword encoder alignment ─────────────────────────


class _StubEncoding:
    """Minimal BatchEncoding stand-in: dict ``items`` + ``word_ids``."""

    def __init__(self, input_ids, attention_mask, word_ids):
        self._d = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }
        self._word_ids = word_ids

    def items(self):
        return self._d.items()

    def word_ids(self, _batch_index=0):
        return self._word_ids


class _StubTokenizer:
    """Tokenizes ["a", "bb"] → [CLS, a, bb#1, bb#2, SEP] (bb splits into 2)."""

    is_fast = True

    def __call__(self, words, **kwargs):  # noqa: ARG002
        return _StubEncoding(
            input_ids=[101, 10, 20, 21, 102],
            attention_mask=[1, 1, 1, 1, 1],
            word_ids=[None, 0, 1, 1, None],
        )


class _StubAutoTokenizer:
    @staticmethod
    def from_pretrained(name, use_fast=True, **kwargs):  # noqa: ARG004
        # **kwargs absorbs add_prefix_space (set by the encoder for RoBERTa/XLM-R).
        return _StubTokenizer()


@pytest.fixture
def _stub_tokenizer(monkeypatch):
    monkeypatch.setattr(encoders, "AutoTokenizer", _StubAutoTokenizer)


def test_lilt_encoder_box_label_alignment(_stub_tokenizer):
    enc = LiLTEncoder("dummy")
    out = enc.encode(
        image=None,
        words=["a", "bb"],
        boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
        word_labels=[1, 3],
    )
    # Label on the first subword of each word; -100 on continuation + specials.
    assert out["labels"].tolist() == [-100, 1, 3, -100, -100]
    # Box on every subword of a word; [0,0,0,0] on specials. LiLT keeps [0,1000] int.
    assert out["bbox"].tolist() == [
        [0, 0, 0, 0],
        [0, 0, 10, 10],
        [20, 20, 30, 30],
        [20, 20, 30, 30],
        [0, 0, 0, 0],
    ]
    assert out["bbox"].dtype == torch.long


def test_bros_encoder_normalizes_boxes(_stub_tokenizer):
    enc = BROSEncoder("dummy")
    out = enc.encode(
        image=None,
        words=["a", "bb"],
        boxes=[[0, 0, 1000, 500], [200, 200, 300, 300]],
        word_labels=[1, 3],
    )
    assert out["bbox"].dtype == torch.float
    # BROS normalizes [0,1000] → [0,1].
    assert out["bbox"][1].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.5])
    assert out["labels"].tolist() == [-100, 1, 3, -100, -100]


def test_build_encoder_unknown_family_raises():
    with pytest.raises(ValueError, match="No encoder for model family"):
        build_encoder({"family": "not-a-backbone"})


# ───────────────────────── full wrapper round-trips (network) ───────────────────


def _synthetic_batch(wrapper, batch_size=2, seq_len=16):
    n = wrapper.model.config.num_labels
    # VALID boxes: x0<=x1, y0<=y1 (real KIE datasets always emit ordered corners).
    # LiLT derives width/height embeddings from x1-x0 / y1-y0 and indexes an embedding
    # table with them, so an unordered (negative) box raises IndexError. Sort each pair.
    xy = torch.randint(0, 1000, (batch_size, seq_len, 4))
    x = xy[..., [0, 2]].sort(dim=-1).values  # x0 <= x1
    y = xy[..., [1, 3]].sort(dim=-1).values  # y0 <= y1
    bbox = torch.stack([x[..., 0], y[..., 0], x[..., 1], y[..., 1]], dim=-1)
    if getattr(wrapper, "_inner_attr", "") == "bros":
        bbox = bbox.float() / 1000.0  # BROS expects normalized [0,1] floats
    out = {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "bbox": bbox,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, n, (batch_size, seq_len)),
    }
    # LayoutLMv3 (the only vision backbone here) also needs pixel_values.
    if getattr(wrapper, "_has_image", False) or "layoutlmv3" in type(wrapper).__name__.lower():
        out["pixel_values"] = torch.zeros(batch_size, 3, 224, 224)
    return out


def _make_wrapper(wrapper_cls):
    """Construct a wrapper; LayoutLMv3/BERT need an explicit model_name, LiLT/BROS default."""
    name = wrapper_cls.__name__.lower()
    if "layoutlmv3" in name:
        return wrapper_cls(model_name="microsoft/layoutlmv3-base", num_labels=7)
    if "bert" in name:
        return wrapper_cls(model_name="bert-base-uncased", num_labels=7)
    return wrapper_cls(num_labels=7)


@pytest.mark.integration
@pytest.mark.parametrize("wrapper_cls", [LayoutLMv3Wrapper, LiLTWrapper, BROSWrapper, BERTWrapper])
def test_wrapper_forward_expand_and_groups(wrapper_cls):
    model = _make_wrapper(wrapper_cls)
    model.label_to_id = {f"L{i}": i for i in range(7)}
    model.id_to_label = {i: f"L{i}" for i in range(7)}
    batch = _synthetic_batch(model)

    out = model(**batch)
    assert out.loss is not None
    assert out.logits.shape[-1] == 7

    model.expand_classifier(["B-NEW", "I-NEW"])
    assert model.model.config.num_labels == 9

    groups = model.param_groups
    assert groups and "misc" not in groups
    assert model.param_groups_by_depth


@pytest.mark.integration
@pytest.mark.parametrize("wrapper_cls", [LayoutLMv3Wrapper, LiLTWrapper, BROSWrapper, BERTWrapper])
def test_wrapper_prompt_injection_shapes(wrapper_cls):
    model = _make_wrapper(wrapper_cls)
    batch = _synthetic_batch(model)
    query = model.encode_query(batch)
    assert query.shape == (2, model.hidden_size)

    prompts = torch.randn(2, 4, model.hidden_size)
    fwp_kwargs = dict(
        input_ids=batch["input_ids"],
        bbox=batch["bbox"],
        prompt_embeds=prompts,
        attention_mask=batch["attention_mask"],
    )
    # LayoutLMv3's prompt forward requires the vision stream (pixel_values); the
    # vision-free backbones do not take it. Pass it only when the batch carries it.
    if "pixel_values" in batch:
        fwp_kwargs["pixel_values"] = batch["pixel_values"]
    logits = model.forward_with_prompts(**fwp_kwargs)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == model.model.config.num_labels


# ─────────────────────────── LexSlot across backbones ──────────────────────────
# LexSlot places slot memories at the head + late encoder layers via forward hooks.
# The head hook (classifier pre/post) is backbone-agnostic; the late repr-slot hook
# must handle every encoder's layer-output shape — notably LiLT's nested
# ((text, layout), ...) tuple. These integration tests run the full
# before_task/train_task(KD-teacher deepcopy)/after_task/evaluate lifecycle on real
# weights for all four backbones so the cross-backbone bug class is caught here.


class _TinyLexDataset(torch.utils.data.Dataset):
    """A few synthetic documents matching a wrapper's expected input streams."""

    def __init__(self, wrapper, n=4, seq_len=12):
        from tests.test_backbones import _synthetic_batch  # self-import for the helper

        b = _synthetic_batch(wrapper, batch_size=n, seq_len=seq_len)
        self.items = [{k: v[i] for k, v in b.items()} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _lex_loader(wrapper, n=4):
    return torch.utils.data.DataLoader(_TinyLexDataset(wrapper, n=n), batch_size=2)


@pytest.mark.integration
@pytest.mark.parametrize("wrapper_cls", [LayoutLMv3Wrapper, LiLTWrapper, BROSWrapper, BERTWrapper])
def test_lexslot_lifecycle_all_backbones(wrapper_cls):
    """LexSlot full lifecycle (head+late slots, KD-teacher deepcopy) on every backbone."""
    from doccl.methods.lexslot import LexSlot
    from doccl.types import TaskInfo

    model = _make_wrapper(wrapper_cls)
    labels = [f"L{i}" for i in range(7)]
    model.label_to_id = {l: i for i, l in enumerate(labels)}  # noqa: E741
    model.id_to_label = {i: l for i, l in enumerate(labels)}  # noqa: E741

    cfg = {
        "lr": 5e-5,
        "weight_decay": 0.01,
        "epochs": 1,
        "max_grad_norm": 1.0,
        "early_stopping": False,
        "slot_depth": "head_late",  # exercises BOTH head + late repr-slots
        "slot_sharing": "soft",
        "n_tasks": 2,  # each of the 2 tasks claims its own disjoint slot block
        "n_slots_head": 6,
        "n_slots_late": 4,
        "repr_rank": 2,
        "fisher_n_samples": 2,
        "buffer_size": 10,
        "replay_batch_size": 2,
        "target_depth": "all",
    }
    method = LexSlot(model, cfg)
    # Late repr-slots must have been placed (head_late -> non-empty on a 12-layer encoder).
    assert method._late_idx, f"{wrapper_cls.__name__}: no late layers targeted"

    # Two DIL tasks (fixed head): full before/train/after lifecycle on real weights.
    for tid in (0, 1):
        loader = _lex_loader(model)
        task = TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=labels)
        method.before_task(task, loader)
        metrics = method.train_task(task, loader, val_loader=None)
        method.after_task(task, loader)
        assert torch.isfinite(
            torch.tensor(float(metrics.loss))
        ), f"{wrapper_cls.__name__}: task-{tid} loss not finite"

    results = method.evaluate({0: _lex_loader(model, 2), 1: _lex_loader(model, 2)})
    assert set(results) == {0, 1}
    for tid, m in results.items():
        assert 0.0 <= m.f1 <= 100.0, f"{wrapper_cls.__name__}: task {tid} F1 out of range"
