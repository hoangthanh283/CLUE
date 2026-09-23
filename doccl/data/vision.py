"""Image-classification datasets for the image-CL scope test (Split CIFAR-100, Split ImageNet-R).

Each item is ``{"pixel_values": (3, 224, 224) float, "labels": 0-d long}`` where the label
is the *head* index (position of the native class in the fixed class order), so the growing
classifier of the CIL loop lines up without a remapper.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

VISION_ROOT = os.environ.get("DOCCL_VISION_ROOT", "data/vision")
# google/vit-base-patch16-224-in21k normalises to mean=std=0.5.
_NORM = transforms.Normalize([0.5] * 3, [0.5] * 3)
_TRAIN_TF = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), _NORM]
)
_EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), _NORM])


class VisionCILDataset(Dataset):
    """Subset of a torchvision dataset restricted to ``class_ids`` (native ids)."""

    def __init__(
        self,
        base: Dataset,
        class_ids: Sequence[int],
        head_index: dict[int, int],
        train: bool,
        allowed_idx: Sequence[int] | None = None,
    ):
        targets = np.asarray(base.targets)
        keep = set(int(c) for c in class_ids)
        pool = range(len(targets)) if allowed_idx is None else allowed_idx
        self.idx = [i for i in pool if int(targets[i]) in keep]
        self.base = base
        self.head_index = head_index
        self.tf = _TRAIN_TF if train else _EVAL_TF

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        img, y = self.base[self.idx[i]]
        return {"pixel_values": self.tf(img), "labels": torch.tensor(self.head_index[int(y)])}


def cifar100_bases() -> tuple[Dataset, Dataset, None, int]:
    root = os.path.join(VISION_ROOT, "cifar100")
    train = datasets.CIFAR100(root, train=True, download=True)
    test = datasets.CIFAR100(root, train=False, download=True)
    return train, test, None, 100


def imagenet_r_bases(split_seed: int = 0, train_frac: float = 0.8):
    """ImageNet-R has no official split; per-class 80/20 with a fixed seed (L2P convention).

    Expects the extracted archive at ``$DOCCL_VISION_ROOT/imagenet-r/<wnid>/*.jpg``.
    """
    root = os.path.join(VISION_ROOT, "imagenet-r")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ImageNet-R not found at {root}; download imagenet-r.tar from "
            "https://github.com/hendrycks/imagenet-r and extract it there."
        )
    base = datasets.ImageFolder(root)
    targets = np.asarray(base.targets)
    rng = np.random.RandomState(split_seed)
    train_idx, test_idx = [], []
    for c in range(len(base.classes)):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        cut = int(round(train_frac * len(idx)))
        train_idx += idx[:cut].tolist()
        test_idx += idx[cut:].tolist()
    return base, base, (train_idx, test_idx), len(base.classes)
