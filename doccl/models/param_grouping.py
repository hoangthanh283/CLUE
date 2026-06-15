"""Shared parameter grouping for per-component Fisher / displacement analysis.

Defined once so the LayoutLMv3 backbone and the BERT baseline are diagnosed with
the *same* component vocabulary (review C2/m2/m4 asked for honest, populated
groups that match the single-stream architecture). There is deliberately **no**
``visual_attn`` or ``fusion`` group: in a single-stream encoder text and visual
tokens share the same Q/K/V projections, so those are not separable parameter
groups. ``misc`` is a diagnostic safety net that should be empty for stock
backbones (assert this in tests).
"""
from __future__ import annotations

import re

import torch.nn as nn

# Canonical reporting order for the component groups.
GROUP_ORDER = [
    "text_word_embed",
    "layout_2d_pos_embed",
    "image_patch_embed",
    "pos_1d_embed",
    "attn_qkv",
    "attn_out",
    "ffn",
    "layernorm",
    "rel_pos_bias",
    "cls_pooler",
    "classifier",
    "misc",
]


def classify_param(name: str) -> str:
    """Map a parameter name to its honest component group.

    The chain is mutually exclusive and ordered most-specific first. Works for
    both LayoutLMv3 (which has layout/patch/rel-pos groups) and BERT (which does
    not — those groups simply stay empty and are dropped).
    """
    if "classifier" in name:
        return "classifier"
    if "word_embeddings" in name:
        return "text_word_embed"
    if re.search(r"[xyhw]_position_embeddings", name):
        return "layout_2d_pos_embed"
    if "patch_embed" in name:
        return "image_patch_embed"
    if "rel_pos" in name:  # relative 1D/2D position attention biases
        return "rel_pos_bias"
    if "attention" in name and ("query" in name or "key" in name or "value" in name):
        return "attn_qkv"  # shared text+visual self-attention projections
    if "attention.output.dense" in name:
        return "attn_out"  # attention output projection (kept out of ffn)
    if "LayerNorm" in name or name.endswith(".norm.weight") or name.endswith(".norm.bias"):
        return "layernorm"
    if "intermediate.dense" in name or ("output.dense" in name and "attention" not in name):
        return "ffn"  # feed-forward block (intermediate + FFN output)
    if (
        "position_embeddings" in name
        or "token_type_embeddings" in name
        or "pos_embed" in name
    ):
        return "pos_1d_embed"  # 1D/absolute + token-type + vision pos embeds
    if "cls_token" in name or "pooler" in name:
        return "cls_pooler"
    return "misc"


def param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Named, fully-populated parameter groups for ``model`` (empties dropped)."""
    groups: dict[str, list[nn.Parameter]] = {k: [] for k in GROUP_ORDER}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups[classify_param(name)].append(p)
    return {k: v for k, v in groups.items() if v}


def param_groups_by_depth(model: nn.Module, num_layers: int) -> dict[str, list[nn.Parameter]]:
    """Encoder parameters bucketed by depth: input / early / mid / late / head.

    Layer indices are parsed from ``encoder.layer.<i>`` names and the
    third-boundaries scale with ``num_layers`` so this holds for non-12-layer
    backbones. Drives the depth-wise forgetting story and the depth/head-targeted
    DocCL method.
    """
    third = max(num_layers // 3, 1)
    groups: dict[str, list[nn.Parameter]] = {
        "input": [],
        "early": [],
        "mid": [],
        "late": [],
        "head": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            groups["head"].append(p)
            continue
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        if m is None:
            groups["input"].append(p)
            continue
        layer_idx = int(m.group(1))
        if layer_idx < third:
            groups["early"].append(p)
        elif layer_idx < 2 * third:
            groups["mid"].append(p)
        else:
            groups["late"].append(p)
    return {k: v for k, v in groups.items() if v}
