"""Classifier head lifecycle management for continual learning."""

import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.models.layoutlm_models import BaseLayoutLMModel


class HeadManager:
    """Manages the classifier head for continual learning settings.

    Centralises all head operations:
    - class_il: single head that grows as new labels are introduced
    - task_il: per-task heads stored in memory and swapped at inference

    Args:
        model: The LayoutLM model whose classifier is managed.
        cl_setting: "class_il" or "task_il".
    """

    def __init__(self, model: BaseLayoutLMModel, cl_setting: str = "task_il") -> None:
        self.model = model
        self.cl_setting = cl_setting
        self._head_states: Dict[str, Dict[str, torch.Tensor]] = {}

    # ── model-level wrappers ──────────────────────────────────────────────

    def expand_classifier(self, num_new_labels: int) -> bool:
        """Grow the classifier for class-IL (preserves existing weights).

        Returns:
            True if the classifier was actually expanded (optimizer refresh needed).
        """
        if num_new_labels > self.model.num_labels:
            self.model.expand_classifier(num_new_labels)
            return True
        return False

    def reset_classifier(self, num_labels: int) -> None:
        """Replace the classifier with a fresh head of size num_labels."""
        self.model.reset_classifier(num_labels)

    # ── task head state management ────────────────────────────────────────

    def activate(self, task_name: str, num_labels: int) -> bool:
        """Load or create a task-specific head for task-IL.

        If a compatible saved state exists for task_name, loads it without
        touching the optimizer. Otherwise resets the classifier to a fresh head.

        Returns:
            True if the classifier was reset to a fresh head (optimizer refresh
            needed). False if an existing saved state was loaded.
        """
        state = self._head_states.get(task_name)
        if state is not None:
            w = state.get("weight")
            if w is not None and w.size(0) == num_labels:
                self.model.reset_classifier(num_labels)
                self.model.classifier.load_state_dict(copy.deepcopy(state))
                return False  # Loaded saved state — no optimizer refresh needed
        # No compatible saved state: create fresh head
        self.model.reset_classifier(num_labels)
        return True  # Fresh reset — optimizer refresh needed

    def save(self, task_name: str) -> None:
        """Cache the current classifier state under task_name."""
        self._head_states[task_name] = copy.deepcopy(
            self.model.classifier.state_dict()
        )

    def prepare_for_task(self, task_name: str, label_list: List[str]) -> bool:
        """Prepare the classifier head for a new task.

        Dispatches to activate() for task-IL or expand_classifier() for class-IL.

        Returns:
            True if the classifier changed and the optimizer needs refreshing.
        """
        if self.cl_setting == "task_il":
            return self.activate(task_name, len(label_list))
        else:  # class_il
            return self.expand_classifier(len(label_list))

    # ── persistence ───────────────────────────────────────────────────────

    def save_to_disk(
        self,
        heads_dir: Path,
        labels_by_task: Optional[Dict[str, List[str]]] = None,
    ) -> Dict:
        """Persist all task head states to disk as .pt files.

        Args:
            heads_dir: Directory to write <task_name>.pt files into.
            labels_by_task: Optional mapping from task name to label list
                            (used to populate heads_meta).

        Returns:
            heads_meta dict mapping task_name -> {num_labels, label_list}.
        """
        heads_dir.mkdir(parents=True, exist_ok=True)
        labels_by_task = labels_by_task or {}
        heads_meta: Dict = {}
        for head_name, state in self._head_states.items():
            torch.save(state, heads_dir / f"{head_name}.pt")
            w = state.get("weight")
            heads_meta[head_name] = {
                "num_labels": int(w.shape[0]) if w is not None else None,
                "label_list": labels_by_task.get(head_name),
            }
        return heads_meta
