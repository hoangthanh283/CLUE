"""CoLaR-MiB: CoLaR memory + shared softmax + MiB-style unbiased cross-entropy (AGENT IMPL
2026-11-11, D14b control).

MiB (Cermelli et al., CVPR 2020, arXiv:2002.00718, Sec. 3) trains step t with an unbiased CE in
which a pixel labelled background matches the event "background OR any old class":
``p~(b) = p(b) + sum_{c in old} p(c)``; every other label keeps the ordinary CE. Here:

- current-session token labelled O: ``-log(p(O) + sum_{c from earlier sessions} p(c))``;
- replayed token labelled O, document stored when the head had ``n_cap`` labels:
  ``-log(p(O) + sum_{c >= n_cap} p(c))`` (classes introduced after storage), our symmetric
  extension to the memory (MiB has no replay);
- tokens with an entity label: ordinary CE over the full shared softmax.

Everything else is CoLaR (memory, freeze map, replay batch, inference = argmax of the shared
softmax). Deliberately left out: MiB's unbiased distillation term and its classifier
initialisation of new classes from the background logit. With one session both
events reduce to {O}, i.e. plain CE.
"""

from __future__ import annotations

import torch

from doccl.methods.colar import CoLaR

__all__ = ["CoLaRMiB", "unbiased_ce"]


def unbiased_ce(logits: torch.Tensor, labels: torch.Tensor, merge: torch.Tensor) -> torch.Tensor:
    """Mean token NLL. O tokens (label 0) match the event ``merge`` (bool, broadcastable to
    logits, class 0 included); other tokens use plain CE; -100 is ignored."""
    logits = logits.float()
    lse = logits.logsumexp(-1)
    bg = logits.masked_fill(~merge, float("-inf")).logsumexp(-1)
    tgt = logits.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    nll = lse - torch.where(labels == 0, bg, tgt)
    valid = labels != -100
    return nll[valid].mean() if valid.any() else logits.sum() * 0.0


class CoLaRMiB(CoLaR):
    name = "colar_mib"

    def _current_ce(self, cur_out, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = cur_out.logits
        n_old = max(int(self.state.custom.get("n_old", 0)), 1)  # head ids < n_old: O + old
        merge = torch.arange(logits.shape[-1], device=logits.device) < n_old
        return unbiased_ce(logits, batch["labels"], merge.view(1, 1, -1))

    def _stack_replay(self, docs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        replay = super()._stack_replay(docs)
        replay["n_labels"] = torch.stack([d["n_labels"] for d in docs])  # head size at storage
        return replay

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        cls = torch.arange(out.logits.shape[-1], device=out.logits.device)
        n_cap = replay["n_labels"].to(cls.device).view(-1, 1, 1)
        merge = (cls.view(1, 1, -1) >= n_cap) | (cls.view(1, 1, -1) == 0)
        out.loss = unbiased_ce(out.logits, replay["labels"].to(cls.device), merge)
        return out
