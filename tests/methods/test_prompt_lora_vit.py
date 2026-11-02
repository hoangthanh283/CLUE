"""Prompt family (CODA-Prompt) and CL-LoRA on a tiny offline ViT — the image-grid path.

Regression for the 2026-10-22 image-grid crashes: ``forward_with_prompts`` was a stub on
``ViTWrapper`` (KeyError 'input_ids' in prompt_base) and PEFT's TOKEN_CLS forward injected
``input_ids=`` into ``ViTForImageClassification`` (TypeError). One train_task + evaluate
each, CPU only.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import ViTConfig, ViTForImageClassification

from doccl.methods.cl_lora import CLLoRA
from doccl.methods.coda_prompt import CODAPrompt
from doccl.models.vit_wrapper import ViTWrapper
from doccl.types import TaskInfo

N = 4


def _tiny_vit(tmp_path):
    cfg = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=N,
    )
    ViTForImageClassification(cfg).save_pretrained(tmp_path)
    m = ViTWrapper(model_name=str(tmp_path), num_labels=N)
    m.id_to_label = {i: f"c{i}" for i in range(N)}
    m.label_to_id = {v: k for k, v in m.id_to_label.items()}
    return m


def _loader(n=8):
    items = [
        {"pixel_values": torch.randn(3, 32, 32), "labels": torch.tensor(i % N)} for i in range(n)
    ]
    return DataLoader(items, batch_size=4)


def _task():
    return TaskInfo(task_id=0, task_name="t0", label_set=[f"c{i}" for i in range(N)], is_first=True)


def _run(method):
    method.device = torch.device("cpu")
    method.before_task(_task(), _loader())
    tm = method.train_task(_task(), _loader(), val_loader=None)
    assert tm.n_steps == 2
    res = method.evaluate({0: _loader()})
    assert 0.0 <= res[0].f1 <= 100.0
    return res


def test_coda_prompt_trains_and_evaluates_on_vit(tmp_path):
    model = _tiny_vit(tmp_path)
    method = CODAPrompt(model, {"epochs": 1, "n_components": 3, "prompt_length": 2})
    _run(method)
    # the prompt path must return CLS logits in token layout (B, 1, C)
    batch = next(iter(_loader()))
    q = method._query(batch)
    pe, _ = method._select_prompts(q, batch)
    assert method._prompt_logits(batch, pe).shape == (4, 1, N)


def test_cl_lora_trains_and_evaluates_on_vit(tmp_path):
    model = _tiny_vit(tmp_path)
    method = CLLoRA(model, {"epochs": 1, "lora_rank": 2})
    # the classifier must stay trainable under the generic PeftModel wrapping
    assert any(p.requires_grad for p in model.model.classifier.parameters())
    _run(method)


def test_mask_old_logits_hides_previous_classes_in_training_ce(tmp_path):
    """AGENT FIX 2026-11-01: with ``mask_old_logits`` the second task's CE never touches
    the first task's logit rows (official prompt/CL-LoRA CIL protocol)."""
    model = _tiny_vit(tmp_path)
    method = CODAPrompt(
        model, {"epochs": 1, "n_components": 3, "prompt_length": 2, "mask_old_logits": True}
    )
    method.device = torch.device("cpu")
    assert method._mask_old(torch.zeros(2, 1, N)).isfinite().all()  # task 0: nothing masked
    method.train_task(_task(), _loader(), val_loader=None)
    assert method.state.custom["n_old"] == N
    masked = method._mask_old(torch.zeros(2, 1, N + 2))
    assert torch.isinf(masked[..., :N]).all() and masked[..., N:].isfinite().all()


def test_current_sliced_masks_old_classes_only_on_o_tokens():
    """AGENT IMPL 2026-11-02: latent_replay._current_ce mirror slice."""
    from types import SimpleNamespace

    from doccl.methods.latent_replay import LatentReplay

    m = SimpleNamespace(
        config={"current_sliced": True},
        state=SimpleNamespace(custom={"n_old": 3}),
        _teacher=None,
        _complete_current_labels=lambda b: b["labels"],
    )
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 1] = 10.0  # O-token strongly predicts OLD class 1 -> must not be punished
    logits[0, 1, 1] = 10.0  # entity token (label 4) predicting class 1 -> full softmax, punished
    labels = torch.tensor([[0, 4]])
    out = SimpleNamespace(logits=logits, loss=torch.tensor(99.0))
    loss = LatentReplay._current_ce(m, out, {"labels": labels})
    full = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
    assert loss < full  # the O-token's old-class mass was removed from its CE
    assert loss > 4.0  # the entity token is still punished (≈10 nats / 2 tokens)


def test_backward_completion_relabels_confident_old_class_o_tokens():
    """AGENT IMPL 2026-11-02: latent_replay._complete_current_labels (CPFD-style gate)."""
    from types import SimpleNamespace

    from doccl.methods.latent_replay import LatentReplay

    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 6.0  # O token, teacher confident on old class 2 -> relabelled
    logits[0, 1, 2] = 0.1  # O token, low confidence -> stays O
    logits[0, 2, 2] = 6.0  # entity token (label 4) -> untouched
    teacher = lambda **kw: SimpleNamespace(logits=logits)  # noqa: E731
    m = SimpleNamespace(
        config={"current_pseudo_tau": 0.9},
        state=SimpleNamespace(custom={"n_old": 3}),
        _teacher=teacher,
        _amp_autocast=lambda: __import__("contextlib").nullcontext(),
    )
    out = LatentReplay._complete_current_labels(m, {"labels": torch.tensor([[0, 0, 4]])})
    assert out.tolist() == [[2, 0, 4]]
