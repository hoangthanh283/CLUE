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
