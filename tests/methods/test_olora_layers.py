"""O-LoRA / CL-LoRA ``layers_to_transform``: LoRA sites only on the requested encoder layers.

Tiny random LayoutLMv3 (no download, CPU). Pins the capacity-matched CL-LoRA control:
Q/K/V + every ``dense`` on layers >= 4 gives 6 sites per layer, like SBR-Recall's 48 at 12 layers.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification

from doccl.methods.cl_lora import CLLoRA

_TARGETS = ["query", "key", "value", "dense"]


class _Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = LayoutLMv3Config(
            hidden_size=32,
            num_hidden_layers=12,
            num_attention_heads=2,
            intermediate_size=64,
            input_size=32,
            num_labels=3,
            visual_embed=False,
        )
        self.model = LayoutLMv3ForTokenClassification(cfg)


def _lora_layers(cfg: dict) -> list[int]:
    m = _Wrap()
    CLLoRA(m, {"lr": 5e-5, "epochs": 1, "lora_rank": 4, "target_modules": _TARGETS, **cfg})
    names = [n for n, mod in m.model.named_modules() if hasattr(mod, "lora_A")]
    return [int(n.split(".layer.")[1].split(".")[0]) for n in names]


def test_layers_to_transform_restricts_sites():
    layers = _lora_layers({"layers_to_transform": list(range(4, 12))})
    assert len(layers) == 48 and set(layers) == set(range(4, 12))


def test_default_is_every_layer():
    assert len(_lora_layers({})) == 72
