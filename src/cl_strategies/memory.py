"""Episodic memory buffer utilities for ER/GEM/A-GEM.

This module implements a reservoir sampling-based memory buffer for continual learning
strategies. Reservoir sampling ensures uniform sampling probability over the data stream,
which is the standard approach for vanilla Experience Replay baselines.

References:
    - Vitter (1985) "Random sampling with a reservoir" (Algorithm R)
    - Rolnick et al. (2019) "Experience Replay for Continual Learning"
"""

import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class MemoryItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    labels: torch.Tensor
    task_id: int = 0  # Track which task this sample belongs to
    token_type_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    image: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None


@dataclass
class MemoryKey:
    """Key reference for disk-stored memory items."""
    key: str
    file_path: str

    def __post_init__(self):
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)


class MemoryBuffer:
    """Reservoir-sampling episodic memory buffer with disk storage.

    Implements vanilla reservoir sampling (Algorithm R) to maintain a fixed-size
    buffer with uniform sampling probability over the entire data stream.

    Algorithm:
        For the i-th item in the stream:
        1. If buffer not full: add item directly
        2. Otherwise: generate j = random(0, i-1)
           - If j < capacity: replace buffer[j] with item
           - Otherwise: discard item

    This ensures each item has probability min(capacity, n) / n of being in the buffer,
    regardless of when it was seen.

    The implementation uses disk storage to handle large buffers and document datasets
    with images, trading I/O overhead for memory savings.
    """

    def __init__(self, capacity: int, storage_dir: str = "./memory_storage"):
        self.capacity = int(capacity)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Replace items list with keys list
        self.keys: List[MemoryKey] = []
        self.n_seen = 0

        # Cache for task-specific keys to improve sampling efficiency
        self._task_key_cache: Dict[int, List[MemoryKey]] = {}
        self._cache_valid = True

    def __len__(self) -> int:
        return len(self.keys)

    def _generate_key_path(self) -> Tuple[str, str]:
        """Generate unique key and file path."""
        key = f"sample_{self.n_seen}_{uuid.uuid4().hex[:8]}"
        # Simple sharding: use last 2 chars of key for subdirectory
        subdir = key[-2:]
        file_path = str(self.storage_dir / subdir / f"{key}.pt")
        return key, file_path

    def _save_item(self, item: MemoryItem, file_path: str):
        """Save MemoryItem to disk."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(item, file_path)

    def _load_item(self, memory_key: MemoryKey) -> Optional[MemoryItem]:
        """Load MemoryItem from disk."""
        try:
            return torch.load(memory_key.file_path, map_location='cpu', weights_only=False)
        except (FileNotFoundError, Exception):
            return None

    def _delete_item(self, memory_key: MemoryKey):
        """Delete item file from disk."""
        try:
            Path(memory_key.file_path).unlink(missing_ok=True)
        except Exception:
            pass

    def _invalidate_cache(self):
        """Invalidate the task key cache."""
        self._cache_valid = False
        self._task_key_cache.clear()

    def _update_cache(self):
        """Update the task key cache if it's invalid."""
        if not self._cache_valid:
            self._task_key_cache.clear()
            for memory_key in self.keys:
                item = self._load_item(memory_key)
                if item is not None:
                    task_id = item.task_id
                    if task_id not in self._task_key_cache:
                        self._task_key_cache[task_id] = []
                    self._task_key_cache[task_id].append(memory_key)
            self._cache_valid = True

    def add_batch(self, batch: Dict[str, torch.Tensor], task_id: int = 0):
        """Add batch to memory with task_id tracking.

        Args:
            batch: Dictionary of tensors to store
            task_id: ID of the task these samples belong to (for GEM task-aware sampling)
        """
        bsz = batch["input_ids"].size(0)
        for i in range(bsz):
            self.n_seen += 1

            # Extract optional tensors safely
            token_type_ids = batch.get("token_type_ids")
            position_ids = batch.get("position_ids")
            image = batch.get("image")
            pixel_values = batch.get("pixel_values")

            # Create item with task_id
            item = MemoryItem(
                input_ids=batch["input_ids"][i].detach().cpu(),
                attention_mask=batch["attention_mask"][i].detach().cpu(),
                bbox=batch["bbox"][i].detach().cpu(),
                labels=batch["labels"][i].detach().cpu(),
                task_id=task_id,
                token_type_ids=token_type_ids[i].detach().cpu() if token_type_ids is not None else None,
                position_ids=position_ids[i].detach().cpu() if position_ids is not None else None,
                image=image[i].detach().cpu() if image is not None else None,
                pixel_values=pixel_values[i].detach().cpu() if pixel_values is not None else None,
            )

            # Reservoir sampling: decide BEFORE saving to disk to avoid orphaned files.
            # Bug fix: previously, items were always saved to disk first and then
            # discarded by reservoir sampling without deleting the file, causing
            # thousands of orphaned .pt files that consumed disk space and caused
            # SIGKILL when disk fills up during long training runs.
            if len(self.keys) < self.capacity:
                # Buffer not full: always keep this item
                key, file_path = self._generate_key_path()
                self._save_item(item, file_path)
                memory_key = MemoryKey(key=key, file_path=file_path)
                self.keys.append(memory_key)
                self._invalidate_cache()
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    # This item wins the lottery: save to disk and replace slot j
                    key, file_path = self._generate_key_path()
                    self._save_item(item, file_path)
                    memory_key = MemoryKey(key=key, file_path=file_path)
                    self._delete_item(self.keys[j])
                    self.keys[j] = memory_key
                    self._invalidate_cache()
                # else: item is discarded — no disk write, no orphaned file

    def sample(
        self, batch_size: int, device: torch.device, task_id: Optional[int] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample from memory buffer, optionally filtered by task_id.

        Args:
            batch_size: Number of samples to return
            device: Device to place tensors on
            task_id: If provided, only sample from this task. If None, sample from all tasks.

        Returns:
            Dictionary of collated tensors, or None if no samples available
        """
        if len(self.keys) == 0:
            return None

        # Filter keys by task_id if specified
        if task_id is not None:
            # Use cached task keys for efficiency
            self._update_cache()
            task_keys = self._task_key_cache.get(task_id, [])

            if len(task_keys) == 0:
                return None

            selected_keys = random.sample(task_keys, min(batch_size, len(task_keys)))
        else:
            # Sample from all tasks
            batch_size = min(batch_size, len(self.keys))
            selected_keys = random.sample(self.keys, batch_size)

        # Load items from disk on-demand
        samples = []
        for memory_key in selected_keys:
            item = self._load_item(memory_key)
            if item is not None:
                samples.append(item)

        if not samples:
            return None

        # Collate samples (same as before)
        collated: Dict[str, torch.Tensor] = {}
        keys = [
            "input_ids",
            "attention_mask",
            "bbox",
            "labels",
            "token_type_ids",
            "position_ids",
            "image",
            "pixel_values",
        ]
        for k in keys:
            vals = [getattr(it, k) for it in samples if getattr(it, k) is not None]
            if len(vals) == 0:
                continue
            collated[k] = torch.stack(vals, dim=0).to(device)
        return collated

    def cleanup(self):
        """Clean up all stored files."""
        for memory_key in self.keys:
            self._delete_item(memory_key)
        self.keys.clear()
        self._task_key_cache.clear()
        self._cache_valid = True

    def get_task_counts(self) -> Dict[int, int]:
        """Get count of samples per task.

        Returns:
            Dictionary mapping task_id to count of samples
        """
        task_counts = {}
        for memory_key in self.keys:
            item = self._load_item(memory_key)
            if item is not None:
                task_id = item.task_id
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
        return task_counts

    def inspect_sample(self, index: int = 0) -> Optional[Dict]:
        """Debug helper: inspect a stored sample."""
        if index >= len(self.keys):
            return None

        memory_key = self.keys[index]
        item = self._load_item(memory_key)
        if item is None:
            return None

        return {
            'key': memory_key.key,
            'file_path': memory_key.file_path,
            'task_id': item.task_id,
            'input_ids_shape': item.input_ids.shape,
            'attention_mask_shape': item.attention_mask.shape,
            'bbox_shape': item.bbox.shape,
            'labels_shape': item.labels.shape,
            'has_token_type_ids': item.token_type_ids is not None,
            'has_image': item.image is not None,
            'has_pixel_values': item.pixel_values is not None,
        }
