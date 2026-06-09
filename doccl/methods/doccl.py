"""DocCL — the three candidate proposed methods (A / B / C).

Per the measure-first design (CLAUDE.md), the proposed method is *selected at the
Week-4 pilot decision gate*, not pre-locked. We pre-implement all three sketched
candidates so the winner runs immediately once the per-component diagnosis is in:

    Candidate A — LAPP + H-LoRA          (fusion-dominant forgetting + layout matters)
    Candidate B — Layout-Protected EWC   (uniform forgetting but 2D position drifts)
    Candidate C — Modality-Routed Prompts (scenario-dependent per-modality patterns)

Each exposes a ``target_component`` knob ({text, visual, layout, fusion, uniform})
so the Table 6.2 ablation — "does concentrating the mechanism on the
diagnosed component beat uniform treatment?" — runs through one interface. After
the pilot, the winner is aliased to ``doccl`` in ``scripts/train.py``.

Each candidate is a focused, runnable mechanism that reuses existing infrastructure
(O-LoRA, EWC, the prompt base) rather than a fragile combination — the AAAI-Lite
scope. If a candidate is selected, §3.3.6 specifies it in full.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from doccl.methods.ewc import EWC
from doccl.methods.o_lora import OLoRA
from doccl.methods.prompt_base import PromptBasedMethod, PromptPool

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate A — Hierarchical LoRA (component-banked orthogonal LoRA)
# ─────────────────────────────────────────────────────────────────────────────
class DocCL_A(OLoRA):
    """Candidate A — H-LoRA: component-banked orthogonal LoRA.

    LoRA adapters are placed only on the modules of the targeted component bank(s)
    — text (attention Q/K/V), visual (patch projection), fusion (attention output
    projection where text and visual tokens mix in the single-stream encoder) —
    with O-LoRA's per-module orthogonality across tasks. ``target_component``
    selects the bank for the ablation; "uniform" enables all banks (the full
    method). The layout-aware prompt-pool (LAPP) front-end described for Candidate
    A in §3.3 is the prompt-coupled refinement reserved for the selected method;
    the implemented core is the component-banked H-LoRA the ablation exercises.

    Layout cannot be LoRA-addressed (2D position embeddings are lookup tables, not
    Linear layers) — layout protection is precisely Candidate B's mechanism.
    """

    name = "doccl_a"

    COMPONENT_TARGETS: dict[str, list[str]] = {
        "text": ["query", "key", "value"],
        "fusion": ["attention.output.dense"],
        "visual": ["patch_embed.proj"],
        "uniform": ["query", "key", "value", "attention.output.dense", "patch_embed.proj"],
    }

    def __init__(self, model, config):
        comp = config.get("target_component", "uniform")
        if comp == "layout":
            raise ValueError(
                "DocCL_A (H-LoRA) cannot target 'layout': 2D position embeddings are "
                "not Linear layers. Use Candidate B for layout-targeted protection."
            )
        if comp not in self.COMPONENT_TARGETS:
            raise ValueError(f"Unknown target_component {comp!r} for DocCL_A.")
        # Inject component-specific LoRA targets before O-LoRA wraps the backbone.
        config = {**config, "target_modules": list(self.COMPONENT_TARGETS[comp])}
        super().__init__(model, config)
        self.target_component = comp


# ─────────────────────────────────────────────────────────────────────────────
# Candidate B — Layout-Protected EWC (per-component-group penalty weighting)
# ─────────────────────────────────────────────────────────────────────────────
class DocCL_B(EWC):
    """Candidate B — Layout-Protected EWC: per-group EWC penalty weighting.

    Selective EWC variant: a high penalty ``lambda_high`` protects the targeted
    component group (default the 2D layout-position embeddings), a low penalty
    ``lambda_low`` applies elsewhere. Reuses EWC's Fisher snapshotting and the
    wrapper's ``param_groups`` to map parameters to components. ``target_component``
    chooses the protected group for the ablation.
    """

    name = "doccl_b"

    COMPONENT_TO_GROUP = {
        "layout": "layout_2d_pos_embed",
        "text": "text_attn",
        "visual": "image_patch_embed",
        "fusion": "fusion",
        "ffn": "ffn",
    }

    def __init__(self, model, config):
        super().__init__(model, config)
        self.target_component = config.get("target_component", "layout")
        self.lambda_high = config.get("lambda_high", 5000.0)
        self.lambda_low = config.get("lambda_low", 100.0)
        # The EWC training loop applies (self.lambda_ / 2) * penalty; neutralise it
        # so the absolute per-group lambdas below are the effective weights.
        self.lambda_ = 2.0
        self._high_param_names = self._collect_group_names(self.target_component)
        if not self._high_param_names:
            log.warning(
                "DocCL_B: component %r maps to an empty param group; the high-lambda "
                "set is empty (behaves like uniform low-lambda EWC).",
                self.target_component,
            )

    def _collect_group_names(self, component: str) -> set[str]:
        group_key = self.COMPONENT_TO_GROUP.get(component)
        params = self.model.param_groups.get(group_key, []) if group_key else []
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}
        return {id_to_name[id(p)] for p in params if id(p) in id_to_name}

    def _ewc_penalty(self) -> torch.Tensor:
        if not self.state.custom["fisher"]:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        params = dict(self.model.named_parameters())
        for name, fisher_val in self.state.custom["fisher"].items():
            if name not in params or name not in self.state.custom["theta_star"]:
                continue
            lam = self.lambda_high if name in self._high_param_names else self.lambda_low
            p = params[name]
            theta_star = self.state.custom["theta_star"][name]
            penalty = penalty + lam * (fisher_val * (p - theta_star) ** 2).sum()
        return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Candidate C — Modality-Routed Prompts
# ─────────────────────────────────────────────────────────────────────────────
class _Router(nn.Module):
    """MLP router producing softmax weights over the modality sub-pools."""

    def __init__(self, in_dim: int, n_pools: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_pools)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)  # (B, n_pools)


class DocCL_C(PromptBasedMethod):
    """Candidate C — Modality-Routed Prompts.

    Three prompt sub-pools (text / visual / layout); a router conditioned on the
    CLS query and the layout signature φ(boxes) produces softmax weights that mix
    the per-pool selected prompts. ``target_component`` forces a single sub-pool
    (one-hot routing) for the ablation; "uniform"/"fusion" use the learned router.
    """

    name = "doccl_c"
    MODALITIES = ["text", "visual", "layout"]

    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        n_prompts = config.get("n_prompts", 10)
        prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 5)
        self.lambda_key = config.get("lambda_key", 0.5)
        self.grid = config.get("layout_grid", 4)
        self.target_component = config.get("target_component", "uniform")
        for m in self.MODALITIES:
            self._register_prompt_module(
                f"pool_{m}", PromptPool(n_prompts, prompt_length, hidden_dim)
            )
        router_in = hidden_dim + self.grid * self.grid
        self._register_prompt_module("router", _Router(router_in, n_pools=len(self.MODALITIES)))

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout_sig = self.model.get_layout_signature(batch["bbox"], self.grid)  # (B, grid^2)
        weights = self.router(torch.cat([query, layout_sig], dim=-1))  # (B, n_pools)

        if self.target_component in self.MODALITIES:  # ablation: force one sub-pool
            forced = torch.zeros_like(weights)
            forced[:, self.MODALITIES.index(self.target_component)] = 1.0
            weights = forced

        prompts = []
        aux = torch.zeros((), device=query.device)
        for m in self.MODALITIES:
            sel, key_pull = getattr(self, f"pool_{m}").select(query, self.top_k)  # (B,K,Lp,D)
            B, K, Lp, D = sel.shape
            prompts.append(sel.reshape(B, K * Lp, D))
            aux = aux + key_pull
        stacked = torch.stack(prompts, dim=1)  # (B, n_pools, K*Lp, D)
        combined = (weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)  # (B, K*Lp, D)
        return combined, self.lambda_key * aux / len(self.MODALITIES)
