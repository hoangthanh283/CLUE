"""Core types shared across the doccl package.

These dataclasses and protocols define the contracts between data, models,
methods, and evaluation. Keep this file minimal — it's imported everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import torch


class ScenarioType(str, Enum):
    """CL scenario taxonomy. Used in configs and W&B tags."""

    SINGLE = "single"  # single-task (sanity check, not CL)
    CIL = "class_incremental"
    DIL = "domain_incremental"
    TIL = "task_incremental"  # deferred to NeurIPS extension
    MIXED = "mixed"
    PILOT = "pilot"  # naive sequential for pilot study


class ModalityMask(str, Enum):
    """Which input streams are active during the forward pass.

    Used by the pilot-study conditions. Each mask zeroes the *inactive* input
    streams while keeping the architecture fixed (same layers, same shapes), so
    differences in forgetting are attributable to the modalities in play.

        TEXT_ONLY     C1: text present; image + layout zeroed (real text-only
                      LayoutLMv3 — keeps ``input_ids``).
        IMAGE_LAYOUT  C2: image + layout present; text zeroed (PAD).
        TEXT_LAYOUT   C3: text + layout present; image zeroed.
        FULL          C4: all three streams (standard LayoutLMv3).

    The external unimodal baseline (BERT-base) is a *separate* model
    (``doccl.models.bert_wrapper``), not a mask on LayoutLMv3.
    """

    TEXT_ONLY = "text_only"  # C1: text only (image + layout zeroed)
    IMAGE_LAYOUT = "image_layout"  # C2: LayoutLM no text
    TEXT_LAYOUT = "text_layout"  # C3: LayoutLM no image
    FULL = "full"  # C4: LayoutLM full


@dataclass
class TaskInfo:
    """Metadata about a single task in the CL sequence."""

    task_id: int
    task_name: str
    label_set: list[str]
    is_first: bool = False
    is_last: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    """Mutable state passed through the CL lifecycle.

    Methods extend this via the `custom` dict rather than subclassing.
    """

    buffer: Any = None  # replay buffer (ER, DER++)
    prompt_pool: Any = None  # prompt pool (L2P, DualPrompt, CODA-P, LAPP)
    lora_banks: dict[str, Any] = field(default_factory=dict)  # O-LoRA, H-LoRA
    importance_weights: dict[str, torch.Tensor] = field(default_factory=dict)  # EWC
    teacher_logits_cache: Any = None  # LwF / DER++
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainMetrics:
    """Per-task training metrics."""

    task_id: int
    loss: float
    n_steps: int
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Per-task evaluation metrics. Returned for each task at each evaluation point."""

    task_id: int
    f1: float
    precision: float = 0.0
    recall: float = 0.0
    n_samples: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class DocumentBatch(Protocol):
    """Protocol for batches yielded by document datasets.

    All datasets must yield batches with these keys (other keys allowed).
    """

    input_ids: torch.Tensor  # (B, L) long
    bbox: torch.Tensor  # (B, L, 4) long, normalized to [0, 1000]
    pixel_values: torch.Tensor  # (B, 3, 224, 224) float
    attention_mask: torch.Tensor  # (B, L) long
    labels: torch.Tensor  # (B, L) long, -100 for ignored
