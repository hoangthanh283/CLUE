"""DocCL — the selected depth/head-targeted method (plus legacy candidates).

The corrected per-component diagnosis (review C2/M1/M3) shows forgetting
concentrates in the **classifier head and late encoder layers** of the
single-stream LayoutLMv3, not in a separable "fusion" or "visual" component (those
are not measurable on a single-stream encoder). The selected method is therefore
``DocCL`` below: a depth/head-targeted consolidation that spends its stability
budget where forgetting actually lives. It is the registry's ``doccl``.

``DocCL_A/B/C`` are the earlier sketched candidates kept as ablation variants and
for the NeurIPS extension; the abandoned fusion-dominant / per-modality-visual
decision branches they served are removed from the thesis decision rule (M4).
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from doccl.eval.fisher import empirical_fisher_diagonal
from doccl.methods.buffer import ReservoirBuffer
from doccl.methods.ewc import EWC
from doccl.methods.naive import NaiveFineTune
from doccl.methods.o_lora import OLoRA
from doccl.methods.prompt_base import PromptBasedMethod, PromptPool
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)


# Candidate A — Hierarchical LoRA (component-banked orthogonal LoRA)
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


# Candidate B — Layout-Protected EWC (per-component-group penalty weighting)
class DocCL_B(EWC):
    """Candidate B — Layout-Protected EWC: per-group EWC penalty weighting.

    Selective EWC variant: a high penalty ``lambda_high`` protects the targeted
    component group (default the 2D layout-position embeddings), a low penalty
    ``lambda_low`` applies elsewhere. Reuses EWC's Fisher snapshotting and the
    wrapper's ``param_groups`` to map parameters to components. ``target_component``
    chooses the protected group for the ablation.
    """

    name = "doccl_b"

    # Maps an ablation target to a populated ``param_groups`` key. The single-stream
    # encoder has no separable ``fusion`` group (review C2), so the closest mixing
    # projection is the attention output (``attn_out``).
    COMPONENT_TO_GROUP = {
        "layout": "layout_2d_pos_embed",
        "text": "attn_qkv",
        "visual": "image_patch_embed",
        "fusion": "attn_out",
        "ffn": "ffn",
        "head": "classifier",
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
            # CIL head growth makes p / theta_star / fisher_val differ on the class
            # dimension; penalise only rows present in all three (the old classes with a
            # prior), as EWC._ewc_penalty does — else the broadcast subtraction crashes
            # with a shape mismatch at the first CIL task boundary.
            min_shape = tuple(
                min(a, b, c) for a, b, c in zip(p.shape, theta_star.shape, fisher_val.shape)
            )
            idx = tuple(slice(0, s) for s in min_shape)
            penalty = penalty + lam * (fisher_val[idx] * (p[idx] - theta_star[idx]) ** 2).sum()
        return penalty


# Candidate C — Modality-Routed Prompts
class _Router(nn.Module):
    """MLP router producing softmax weights over the modality sub-pools."""

    def __init__(self, in_dim: int, n_pools: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_pools))

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


# DocCL — the SELECTED method: depth/head-targeted consolidation
class DocCL(NaiveFineTune):
    """Depth/head-targeted continual learning — the method the diagnosis selects.

    Forgetting concentrates in the classifier head and late encoder layers (the
    CKA depth gradient + old-task-Fisher-weighted displacement), while the input
    encoders stay stable. DocCL spends its stability budget there, combining three
    mutually-reinforcing, individually-ablatable mechanisms:

      1. **Depth-scaled EWC** — a Fisher-weighted quadratic penalty whose per-bucket
         strength scales with where forgetting lives (head ≫ late > mid ≫ early ≈
         input ≈ 0), via ``model.param_groups_by_depth``.
      2. **Output distillation** — KL distillation of the previous task's output
         distribution from a frozen teacher, which protects the output-facing
         (head + late) representations functionally.
      3. **Head re-exposure replay** — a small reservoir buffer (replay was the
         strongest baseline) keeps the head grounded in earlier tasks.

    ``target_depth`` ∈ {``all`` (full), ``head_only``, ``late_only``, ``uniform``}
    drives the component-targeting ablation (Table 6.7): does concentrating the
    mechanism on the diagnosed locus beat treating the network uniformly?

    NOTE on the ``uniform`` baseline: it applies an *equal* per-bucket penalty
    (``lambda_uniform``, default 1.0) to *every* depth bucket, including the input
    and early encoders that the ``all`` schedule leaves at 0. It is therefore an
    equal-penalty-everywhere control, NOT a budget-matched reallocation: with the
    defaults its total weight Σ_g w_g = 5.0 exceeds the targeted schedule's 3.5.
    The ablation thus isolates *where* the budget is spent (a collapse under uniform
    despite a larger total budget implicates concentration, not magnitude); it is not
    a same-total-budget comparison. The thesis prose (§3.3.6, §6.2.3) is worded to
    match. To make it budget-matched instead, set ``lambda_uniform=0.7`` (=3.5/5).
    """

    name = "doccl"

    # Per-bucket penalty multipliers for the full method ("all").
    _DEPTH_LAMBDA = {"input": 0.0, "early": 0.0, "mid": 0.5, "late": 1.0, "head": 2.0}

    def __init__(self, model, config):
        super().__init__(model, config)
        self.lambda_ = config.get("lambda_", 2000.0)  # global EWC scale
        self.kd_alpha = config.get("kd_alpha", 1.0)
        self.temperature = config.get("temperature", 2.0)
        self.fisher_n_samples = config.get("fisher_n_samples", 200)
        self.replay_batch_size = config.get("replay_batch_size", 8)
        self.target_depth = config.get("target_depth", "all")
        self.use_replay = config.get("use_replay", True)
        self.state.custom["theta_star"] = {}  # name → snapshot tensor
        self.state.custom["fisher"] = {}  # name → accumulated Fisher diagonal
        self.state.custom["teacher"] = None
        if self.use_replay:
            self.state.buffer = ReservoirBuffer(capacity=config.get("buffer_size", 200))
        self._depth_lambda = self._resolve_depth_lambda(self.target_depth, config)

    def _resolve_depth_lambda(self, target: str, config: dict) -> dict[str, float]:
        base = dict(self._DEPTH_LAMBDA)
        if target == "all":
            return base
        if target == "uniform":
            u = config.get("lambda_uniform", 1.0)
            return {k: u for k in base}
        if target == "head_only":
            return {k: (base["head"] if k == "head" else 0.0) for k in base}
        if target == "late_only":
            return {k: (base["late"] if k == "late" else 0.0) for k in base}
        raise ValueError(
            f"Unknown target_depth {target!r}; expected all/uniform/head_only/late_only"
        )

    def _name_to_bucket(self) -> dict[str, str]:
        """name → depth bucket, rebuilt each call since the head grows across tasks."""
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}
        mapping: dict[str, str] = {}
        for bucket, params in self.model.param_groups_by_depth.items():
            for p in params:
                n = id_to_name.get(id(p))
                if n is not None:
                    mapping[n] = bucket
        return mapping

    def _depth_penalty(self) -> torch.Tensor:
        """Depth-scaled Fisher-weighted quadratic anchor to the previous task."""
        if not self.state.custom["fisher"]:
            return torch.zeros((), device=self.device)
        buckets = self._name_to_bucket()
        params = dict(self.model.named_parameters())
        penalty = torch.zeros((), device=self.device)
        for name, fisher_val in self.state.custom["fisher"].items():
            if name not in params or name not in self.state.custom["theta_star"]:
                continue
            lam = self._depth_lambda.get(buckets.get(name, "input"), 0.0)
            if lam == 0.0:
                continue
            p = params[name]
            theta_star = self.state.custom["theta_star"][name]
            # Classifier head grows across CIL boundaries — penalise only the rows
            # present in all three tensors (the old classes that have a prior).
            min_shape = tuple(
                min(a, b, c) for a, b, c in zip(p.shape, theta_star.shape, fisher_val.shape)
            )
            idx = tuple(slice(0, s) for s in min_shape)
            penalty = penalty + lam * (fisher_val[idx] * (p[idx] - theta_star[idx]) ** 2).sum()
        return penalty

    def _kd_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """KL distillation on the previously-known logits (head/late protection)."""
        T = self.temperature
        n_old = teacher_logits.shape[-1]
        student_old = student_logits[..., :n_old]
        student_log = F.log_softmax(student_old / T, dim=-1)
        teacher_prob = F.softmax(teacher_logits / T, dim=-1)
        # Per-token KL (sum over classes), then mean over valid tokens — NOT over
        # (token, class) pairs, which would deflate KD by n_old and weaken it as the
        # head grows. See LwF._kd_loss for the full rationale.
        per_token_kl = F.kl_div(student_log, teacher_prob, reduction="none").sum(-1)
        per_token_kl = per_token_kl * (T**2)
        return (per_token_kl * mask).sum() / mask.sum().clamp(min=1)

    def train_task(self, task: TaskInfo, train_loader, val_loader=None) -> TrainMetrics:
        from tqdm import tqdm

        self.model.train()
        teacher = self.state.custom.get("teacher")
        if teacher is not None:
            teacher.to(self.device)
            teacher.eval()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss, n_steps = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"DocCL T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()

                outputs = self.model(**batch)
                ce_loss = outputs.loss
                reg_loss = (self.lambda_ / 2) * self._depth_penalty()

                kd_loss = torch.zeros((), device=self.device)
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(**{k: v for k, v in batch.items() if k != "labels"})
                    mask = batch["labels"] != -100
                    kd_loss = self.kd_alpha * self._kd_loss(outputs.logits, t_out.logits, mask)

                replay_loss = torch.zeros((), device=self.device)
                if self.use_replay:
                    replay = self.state.buffer.sample(self.replay_batch_size)
                    if replay is not None:
                        replay = {k: v.to(self.device) for k, v in replay.items()}
                        replay.pop("_logits", None)
                        replay_loss = self.model(**replay).loss

                loss = ce_loss + reg_loss + kd_loss + replay_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()

                if self.use_replay:
                    self.state.buffer.add_batch(
                        {k: v for k, v in batch.items() if torch.is_tensor(v)}
                    )

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {
                        "ce": f"{ce_loss.item():.3f}",
                        "reg": f"{float(reg_loss):.3f}",
                        "kd": f"{float(kd_loss):.3f}",
                    }
                )
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def after_task(self, task: TaskInfo, train_loader) -> None:
        """Snapshot θ*, accumulate Fisher, and snapshot the teacher."""
        self.state.custom["theta_star"] = {
            name: p.detach().clone() for name, p in self.model.named_parameters() if p.requires_grad
        }
        new_fisher = empirical_fisher_diagonal(
            self.model, train_loader, n_samples=self.fisher_n_samples, device=self.device
        )
        gamma = self.config.get("ewc_gamma", 1.0)
        for name, f in new_fisher.items():
            old = self.state.custom["fisher"].get(name)
            if old is None or old.shape != f.shape:
                if old is not None:  # pad the (smaller, pre-expansion) old Fisher
                    padded = torch.zeros_like(f)
                    idx = tuple(slice(0, s) for s in old.shape)
                    padded[idx] = old
                    old = padded
                self.state.custom["fisher"][name] = f if old is None else gamma * old + f
            else:
                self.state.custom["fisher"][name] = gamma * old + f

        # Free the PREVIOUS teacher off the GPU before deepcopy'ing the new one: keeping
        # both on-device through copy.deepcopy is a transient ~2x-model VRAM spike at every
        # task boundary that OOMs a small card once the head has grown (DocCL already peaks
        # near the 6 GB ceiling). Mirrors the eviction cl_lora.after_task already does.
        old_teacher = self.state.custom.get("teacher")
        if old_teacher is not None:
            old_teacher.to("cpu")
            self.state.custom["teacher"] = None
            del old_teacher
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        teacher = copy.deepcopy(self.model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        if hasattr(teacher.model, "gradient_checkpointing_disable"):
            teacher.model.gradient_checkpointing_disable()
        self.state.custom["teacher"] = teacher
