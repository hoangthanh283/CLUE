# doccl/methods/cuber.py
"""CUBER — whole-network gradient-subspace transfer (Lin et al., NeurIPS 2022, arXiv
2211.00789), the mandatory baseline for HGT. Same steering mechanism as HGT
(orthogonal-to-old-subspace + alpha * in-subspace) but applied to EVERY tracked nn.Linear's
weight gradient against that layer's stored INPUT subspace — i.e. whole-network, not
head-only. The HGT-vs-CUBER comparison isolates the value of localizing to the head
(cheaper, frozen-backbone-compatible, backbone-depth-independent).

Faithful to CUBER's mechanism at the level this study compares: per-layer input-subspace
projection with a positive-transfer component (alpha). We track every nn.Linear inside the
backbone + head, capture each one's input activations on scored tokens to build its second
moment, and hook each weight's gradient.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from doccl.methods.grad_subspace import feature_moment, steer_gradient, top_eigvecs
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CUBER"]


class CUBER(NaiveFineTune):
    name = "cuber"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.transfer_alpha = float(config.get("transfer_alpha", 1.0))
        self.subspace_k = int(config.get("subspace_k", 32))
        self.subspace_n_batches = int(config.get("subspace_n_batches", 16))
        # name -> list of per-task input subspaces (each (in_features, k)).
        self._layer_subspaces: dict[str, list[torch.Tensor]] = {}
        self._steer_enabled = False
        self._linear_names = self._collect_linears()
        self._register_grad_hooks()

    def _collect_linears(self) -> list[str]:
        return [n for n, mod in self.model.model.named_modules() if isinstance(mod, nn.Linear)]

    def _register_grad_hooks(self) -> None:
        named = dict(self.model.model.named_modules())
        for name in self._linear_names:
            w = named[name].weight
            w.register_hook(lambda g, nm=name: self._steer_for_layer(nm, g))

    def _stacked_basis(self, name: str, device, dtype) -> torch.Tensor:
        subs = self._layer_subspaces.get(name)
        if not subs:
            return torch.zeros(0, 0, device=device, dtype=dtype)
        stacked = torch.cat([u.to(device=device, dtype=dtype) for u in subs], dim=1)
        q, _ = torch.linalg.qr(stacked)
        return q

    def _steer_for_layer(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        if not self._steer_enabled or name not in self._layer_subspaces:
            return grad
        basis = self._stacked_basis(name, grad.device, grad.dtype)
        return steer_gradient(grad, basis, self.transfer_alpha)

    def before_task(self, task: TaskInfo, train_loader) -> None:
        self._steer_enabled = task.task_id > 0 and bool(self._layer_subspaces)

    def after_task(self, task: TaskInfo, train_loader) -> None:
        self._accumulate_layer_subspaces(train_loader)
        log.info(
            "cuber: task %d done; %d layers tracked (k=%d, alpha=%.2f)",
            task.task_id,
            len(self._layer_subspaces),
            self.subspace_k,
            self.transfer_alpha,
        )

    @torch.no_grad()
    def _accumulate_layer_subspaces(self, loader) -> None:
        """Capture each tracked Linear's input activations -> per-layer second moment -> eigvecs."""
        named = dict(self.model.model.named_modules())
        moments: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(nm):
            def hook(_mod, inp, _out):
                x = inp[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (tokens, in_features)
                m = feature_moment(x)
                moments[nm] = moments.get(nm, torch.zeros_like(m)) + m

            return hook

        for name in self._linear_names:
            handles.append(named[name].register_forward_hook(make_hook(name)))
        self.model.eval()
        for bi, batch in enumerate(loader):
            if bi >= self.subspace_n_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            self.model(**{k: v for k, v in batch.items() if k != "labels"})
        for h in handles:
            h.remove()
        for name, m in moments.items():
            self._layer_subspaces.setdefault(name, []).append(top_eigvecs(m.cpu(), self.subspace_k))
