"""Centered Kernel Alignment (CKA) for representation similarity analysis.

Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited",
ICML 2019, arXiv:1905.00414

Used in pilot study to measure representational drift across CL task boundaries.
Linear CKA is invariant to orthogonal transformations and isotropic scaling,
making it appropriate for comparing activations across training checkpoints.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _center(X: Tensor) -> Tensor:
    """Center columns of X by subtracting per-column mean."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: Tensor, Y: Tensor) -> float:
    """Linear CKA between two activation matrices.

    Args:
        X: (N, d_x) — activations of the first model on N samples
        Y: (N, d_y) — activations of the second model on N samples

    Returns:
        scalar in [0, 1] — 1 means perfectly aligned subspaces (up to orthogonal
        transformation), 0 means orthogonal subspaces.

    Formula:
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Equivalently (using the kernel trick with linear kernels K = XX^T, L = YY^T):
        CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same N. Got {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for CKA")

    Xc = _center(X.float())
    Yc = _center(Y.float())

    # ||Y^T X||_F^2 = sum of squared entries of Y^T X
    cross = torch.norm(Yc.T @ Xc, p="fro") ** 2
    # ||X^T X||_F = Frobenius norm of X^T X (= sum of singular value squares)
    self_x = torch.norm(Xc.T @ Xc, p="fro")
    self_y = torch.norm(Yc.T @ Yc, p="fro")

    denom = self_x * self_y
    if denom < 1e-12:
        return 0.0
    return float(cross / denom)


def cka_per_layer(
    activations_a: dict[str, Tensor],
    activations_b: dict[str, Tensor],
) -> dict[str, float]:
    """Compute CKA per layer between two models on the same input batch.

    Args:
        activations_a: {layer_name: (N, d) tensor}
        activations_b: {layer_name: (N, d') tensor}

    Returns:
        {layer_name: cka_score}
    """
    results = {}
    common_keys = set(activations_a.keys()) & set(activations_b.keys())
    for k in sorted(common_keys):
        results[k] = linear_cka(activations_a[k], activations_b[k])
    return results


def collect_activations(
    model: torch.nn.Module,
    dataloader,
    layer_names: list[str],
    max_samples: int = 500,
    device: str | torch.device = "cuda",
) -> dict[str, Tensor]:
    """Run forward passes and collect mean-pooled activations from named layers.

    Uses forward hooks to capture intermediate outputs. Mean-pools over sequence
    length to get one (d,)-dim vector per sample.

    Args:
        model: model to inspect (in eval mode)
        dataloader: provides input batches
        layer_names: list of module names matching `model.named_modules()`
        max_samples: stop after collecting this many samples total
        device: where to run forward passes

    Returns:
        {layer_name: (N, d) tensor of activations}
    """
    model.eval()
    captured: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    hooks = []

    name_to_module = dict(model.named_modules())
    for name in layer_names:
        if name not in name_to_module:
            raise KeyError(f"Layer {name!r} not found in model")

        def _make_hook(layer_name: str):
            def hook(_module, _input, output):
                # Output may be tuple (some HF layers return (hidden, attn))
                feat = output[0] if isinstance(output, tuple) else output
                # Pool over sequence length: (B, L, D) -> (B, D)
                if feat.dim() == 3:
                    feat = feat.mean(dim=1)
                captured[layer_name].append(feat.detach().cpu())
            return hook

        hooks.append(name_to_module[name].register_forward_hook(_make_hook(name)))

    n_collected = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                model(**{k: v for k, v in batch.items() if k != "labels"})
                n_collected += batch["input_ids"].shape[0]
                if n_collected >= max_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    # Concatenate and truncate
    out = {}
    for name, chunks in captured.items():
        if chunks:
            out[name] = torch.cat(chunks, dim=0)[:max_samples]
    return out
