"""
Episodic memory buffer utilities for ER/GEM/A-GEM.
"""

import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class MemoryItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    labels: torch.Tensor
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
    """Reservoir-sampling episodic memory buffer with disk storage."""

    def __init__(self, capacity: int, storage_dir: str = "./memory_storage"):
        self.capacity = int(capacity)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Replace items list with keys list
        self.keys: List[MemoryKey] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.keys)

    def _generate_key_path(self) -> tuple[str, str]:
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
            return torch.load(memory_key.file_path, map_location='cpu')
        except (FileNotFoundError, Exception):
            return None

    def _delete_item(self, memory_key: MemoryKey):
        """Delete item file from disk."""
        try:
            Path(memory_key.file_path).unlink(missing_ok=True)
        except Exception:
            pass

    def add_batch(self, batch: Dict[str, torch.Tensor]):
        bsz = batch["input_ids"].size(0)
        for i in range(bsz):
            self.n_seen += 1
            
            # Create item as before
            item = MemoryItem(
                input_ids=batch["input_ids"][i].detach().cpu(),
                attention_mask=batch["attention_mask"][i].detach().cpu(),
                bbox=batch["bbox"][i].detach().cpu(),
                labels=batch["labels"][i].detach().cpu(),
                token_type_ids=batch.get("token_type_ids", None)[i].detach().cpu()
                if batch.get("token_type_ids", None) is not None
                else None,
                position_ids=batch.get("position_ids", None)[i].detach().cpu()
                if batch.get("position_ids", None) is not None
                else None,
                image=batch.get("image", None)[i].detach().cpu()
                if batch.get("image", None) is not None
                else None,
                pixel_values=batch.get("pixel_values", None)[i].detach().cpu()
                if batch.get("pixel_values", None) is not None
                else None,
            )
            
            # Generate key and save to disk
            key, file_path = self._generate_key_path()
            self._save_item(item, file_path)
            memory_key = MemoryKey(key=key, file_path=file_path)
            
            # Reservoir sampling with keys instead of items
            if len(self.keys) < self.capacity:
                self.keys.append(memory_key)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    # Delete old item from disk
                    self._delete_item(self.keys[j])
                    self.keys[j] = memory_key

    def sample(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.keys) == 0:
            return None
        
        batch_size = min(batch_size, len(self.keys))
        # Randomly sample keys instead of items
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
        collated: Dict[str, List[torch.Tensor]] = {}
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
            'input_ids_shape': item.input_ids.shape,
            'attention_mask_shape': item.attention_mask.shape,
            'bbox_shape': item.bbox.shape,
            'labels_shape': item.labels.shape,
            'has_token_type_ids': item.token_type_ids is not None,
            'has_image': item.image is not None,
            'has_pixel_values': item.pixel_values is not None,
        }
