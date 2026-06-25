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


def _select_vectors(
    feat: Tensor,
    attention_mask: Tensor | None,
    labels: Tensor | None,
    token_level: bool,
) -> Tensor:
    """Reduce a captured ``(B, L, D)`` activation to a ``(M, D)`` matrix.

    Token-level (default): return one vector per **valid token** — the positions
    the token classifier actually uses (``attention_mask==1`` and, when labels
    are available, ``labels != -100``, i.e. real first-subword tokens). This
    preserves the per-token geometry that mean-pooling destroys (review M2).
    Sequences longer than the text length (encoder outputs that append visual
    patch tokens) are sliced to the text span before masking; activations with
    no text alignment (e.g. the patch embedding) keep all positions.

    Document-level (``token_level=False``): mean-pool over the sequence, the old
    behaviour, kept as a secondary robustness measure.
    """
    if feat.dim() != 3:
        return feat  # already (B, D) — e.g. pooler output
    B, L, D = feat.shape
    if not token_level:
        return feat.mean(dim=1)  # (B, D)

    valid: Tensor | None = None
    if attention_mask is not None:
        a_len = attention_mask.shape[1]
        if L >= a_len:
            f = feat[:, :a_len, :]  # text span (drop appended visual patches)
            valid = attention_mask.bool()
            if labels is not None and labels.shape[1] == a_len:
                valid = valid & (labels != -100)
        else:
            # Activation shorter than text (e.g. patch_embed): keep all positions.
            return feat.reshape(B * L, D)
    else:
        return feat.reshape(B * L, D)
    return f[valid]  # (M_valid, D)


def collect_activations(
    model: torch.nn.Module,
    dataloader,
    layer_names: list[str],
    max_samples: int = 2000,
    device: str | torch.device = "cuda",
    token_level: bool = True,
    modality_mask=None,
) -> dict[str, Tensor]:
    """Run forward passes and collect activations from named layers.

    By default returns one vector **per valid token** (``token_level=True``),
    counting ``max_samples`` in tokens; pass ``token_level=False`` for the legacy
    mean-pooled (one-vector-per-document) behaviour. ``modality_mask`` is
    forwarded to the model so the probe respects the pilot condition.

    Args:
        model: model to inspect (in eval mode)
        dataloader: provides input batches
        layer_names: module names matching ``model.named_modules()``
        max_samples: stop after this many tokens (token-level) or documents
        device: where to run forward passes
        token_level: per-token (True) vs mean-pooled document vectors (False)
        modality_mask: optional ``ModalityMask`` forwarded to ``model.forward``

    Returns:
        {layer_name: (N, d) tensor of activations}
    """
    model.eval()
    latest: dict[str, Tensor] = {}
    captured: dict[str, list[Tensor]] = {name: [] for name in layer_names}
    hooks = []

    name_to_module = dict(model.named_modules())
    for name in layer_names:
        if name not in name_to_module:
            raise KeyError(f"Layer {name!r} not found in model")

        def _make_hook(layer_name: str):
            def hook(_module, _input, output):
                # Unwrap nested tuples to the first tensor: HF encoder layers return
                # ``(hidden_states, ...)``; LiLT's two-stream layer nests further as
                # ``((text_hidden, layout_hidden), ...)`` — the text stream (the first
                # tensor reached) is the one CKA compares.
                feat = output
                while isinstance(feat, (tuple, list)):
                    feat = feat[0]
                latest[layer_name] = feat.detach()

            return hook

        hooks.append(name_to_module[name].register_forward_hook(_make_hook(name)))

    extra = {} if modality_mask is None else {"modality_mask": modality_mask}
    n_collected = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                attn = batch.get("attention_mask")
                labels = batch.get("labels")
                latest.clear()
                model(**{k: v for k, v in batch.items() if k != "labels"}, **extra)
                for name in layer_names:
                    feat = latest.get(name)
                    if feat is None:
                        continue
                    vecs = _select_vectors(feat, attn, labels, token_level)
                    captured[name].append(vecs.detach().cpu())
                ref = captured[layer_names[0]]
                n_collected = (
                    sum(c.shape[0] for c in ref)
                    if token_level
                    else n_collected + batch["input_ids"].shape[0]
                )
                if n_collected >= max_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    out = {}
    for name, chunks in captured.items():
        if chunks:
            out[name] = torch.cat(chunks, dim=0)[:max_samples]
    return out
