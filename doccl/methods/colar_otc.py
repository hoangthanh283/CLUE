"""CoLaR-OTC — mass-constrained label completion under background shift (AGENT IMPL 2026-11-26).

Hypothesis (docs/OT_COMPLETION_PREREG_2026-11.md): pseudo-label repairs of the shifted
background failed on the *amount* of mass they assigned (the entropy rule wrote almost
nothing into the memory, BiLC flooded the current batch), not on the teacher's ranking. A
background token is a partial-label instance with a known candidate set {O} ∪ {classes its
annotator could not see}; a transport plan with class-frequency column marginals fixes the
assigned mass by construction (SoLar's objective, Wang et al. NeurIPS 2022, on session masks).

This file holds the three assignment rules that Gate 0 compares at equal information, and the
opt-in probe that dumps what they need. No rule changes training yet.

  sinkhorn_assign   transport plan, rows = tokens (mass 1), columns = candidate classes
  quota_assign      non-OT control with the SAME class prior: per-class top-k
  threshold_assign  non-OT control without a prior: argmax with a confidence gate (BiLC rule)

``mass_probe: true`` (Gate 0 only): keeps ``labels_full`` beside the stored documents and
writes, per session boundary, the teacher scores of every O-labelled token with its true
label. The true labels are read by the dump alone, never by a loss.

``current_oracle: true`` (Gate 1a only, not deployable): the current-side loss reads the true
label of every class already in the head. With ``refresh: oracle`` and
``refresh_before_task: true`` on the stored side this is the two-sided upper bound of any
completion rule under one shared softmax.
"""

from __future__ import annotations

import logging
import pathlib

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from doccl.methods.colar_r import CoLaRRefresh, oracle_refresh
from doccl.types import TaskInfo

log = logging.getLogger(__name__)


def sinkhorn_assign(
    logp: torch.Tensor,
    col_mass: torch.Tensor,
    lam: float = 10.0,
    n_iter: int = 50,
    col_relax: float = 1.0,
) -> torch.Tensor:
    """Entropic transport plan between N tokens (mass 1 each) and K candidate columns.

    logp (N, K) log-scores, column 0 = background; col_mass (K,) target mass per column
    (same total as N for the balanced problem). Kernel = exp(lam * logp), as in SoLar; SoLar's
    lam = 3 leaves the plan too smooth for a hard label (argmax keeps a third of the mass in
    tests/methods/test_colar_otc.py), so the default is 10.
    ``col_relax`` in (0, 1] is the exponent of the column update: 1 enforces the column
    marginal exactly, < 1 is the KL-relaxed (unbalanced) update that lets a misspecified
    class prior bend. The last update is on the rows, so every row sums to 1 and
    ``plan.argmax(-1)`` is the completed label.
    """
    kernel = lam * logp.double()
    log_mass = col_mass.double().clamp_min(0).log()  # zero-mass column -> -inf -> gets nothing
    g = torch.zeros(kernel.shape[1], dtype=torch.float64)
    for _ in range(n_iter):
        f = -torch.logsumexp(kernel + g[None], dim=1)
        g = col_relax * (log_mass - torch.logsumexp(kernel + f[:, None], dim=0))
        g = torch.nan_to_num(g, nan=float("-inf"), posinf=0.0)  # unreachable column
    f = -torch.logsumexp(kernel + g[None], dim=1)
    return (kernel + f[:, None] + g[None]).exp().float()


def quota_assign(probs: torch.Tensor, col_mass: torch.Tensor) -> torch.Tensor:
    """Per-class top-k with the class prior and no coupling: class c (c >= 1) claims its
    round(col_mass[c]) highest-scoring tokens; a token claimed twice takes the higher score;
    unclaimed tokens are background (0)."""
    n = probs.shape[0]
    claimed = torch.full_like(probs, -1.0)
    for c in range(1, probs.shape[1]):
        k = min(int(col_mass[c].round().item()), n)
        if k > 0:
            idx = probs[:, c].topk(k).indices
            claimed[idx, c] = probs[idx, c]
    best, cls = claimed.max(-1)
    return torch.where(best >= 0, cls, torch.zeros_like(cls))


def threshold_assign(probs: torch.Tensor, tau: float) -> torch.Tensor:
    """Argmax with a confidence gate: the rule of the backward completion in latent_replay."""
    conf, cls = probs.max(-1)
    return torch.where((cls != 0) & (conf >= tau), cls, torch.zeros_like(cls))


class CoLaROTC(CoLaRRefresh):
    name = "colar_otc"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.mass_probe = bool(config.get("mass_probe", False))
        self._head_sizes: list[int] = []  # head size after each session
        self._class_freq: dict[int, float] = {}  # class -> share of valid tokens when annotated
        if self.mass_probe:
            # ponytail: reuse the true-label plumbing of the oracle modes so capture keeps
            # labels_full; only _dump reads it.
            self._TRUE_LABEL_MODES = (*CoLaRRefresh._TRUE_LABEL_MODES, self.refresh)

    # ------------------------------------------------------------------ probe
    def _dump(self, name: str, payload: dict) -> None:
        out = pathlib.Path(getattr(self, "out_dir", None) or ".")
        out.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out / name)
        log.info("colar_otc: wrote %s", out / name)

    @torch.no_grad()
    def _probe_current(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """One pass over the session's training set with the model as it stands before the
        session (= the teacher): label counts for the frequency table, and the old-class
        scores of every O-labelled token with its true label."""
        n_old = int(self.state.custom.get("n_old", 0))
        n_cur = self._head_size()
        counts = torch.zeros(n_cur, dtype=torch.long)
        probs, truth = [], []
        was_training = self.model.training
        self.model.eval()
        for batch in train_loader:
            labels = batch["labels"]
            valid = labels != -100
            counts += torch.bincount(labels[valid], minlength=n_cur)[:n_cur]
            if n_old <= 0:
                continue
            feed = {
                k: v.to(self.device)
                for k, v in batch.items()
                if torch.is_tensor(v) and k not in ("labels", "labels_full")
            }
            logits = self.model(**feed).logits[..., :n_old].float()
            o_tok = labels == 0
            probs.append(logits.softmax(-1).cpu()[o_tok].half())
            truth.append(batch["labels_full"][o_tok].to(torch.int16))
        if was_training:
            self.model.train()
        n_valid = int(counts.sum())
        n_prev = self._head_sizes[-1] if self._head_sizes else 1
        for c in range(n_prev, n_cur):  # the classes this session annotates
            self._class_freq[c] = counts[c].item() / max(n_valid, 1)
        self._dump(
            f"mass_probe_cur_t{task.task_id}.pt",
            {
                "task_id": task.task_id,
                "n_old": n_old,
                "n_cur": n_cur,
                "n_valid": n_valid,
                "counts": counts,
                "class_freq": dict(self._class_freq),
                "head_sizes": list(self._head_sizes),
                "probs": torch.cat(probs) if probs else None,  # (N_O, n_old) teacher scores
                "labels_full": torch.cat(truth) if truth else None,
            },
        )

    def _probe_memory(self, task: TaskInfo) -> None:
        """Scores of the just-trained model on the O-labelled tokens of every document banked
        before this session, with their true labels (stored-side completion target: the
        classes this session introduced)."""
        n_cur = self._head_size()
        docs = [d for d in self.store if int(d["n_labels"]) < n_cur and "labels_full" in d]
        if not docs:
            return
        probs, truth, n_valid = [], [], 0
        for d, p in zip(docs, self._memory_probs(docs), strict=True):
            o_tok = d["labels"] == 0
            n_valid += int((d["labels"] != -100).sum())
            probs.append(p[o_tok].half())
            truth.append(d["labels_full"][o_tok].to(torch.int16))
        self._dump(
            f"mass_probe_mem_t{task.task_id}.pt",
            {
                "task_id": task.task_id,
                "n_prev": self._head_sizes[-2] if len(self._head_sizes) > 1 else 1,
                "n_cur": n_cur,
                "n_valid": n_valid,
                "n_docs": len(docs),
                "class_freq": dict(self._class_freq),
                "probs": torch.cat(probs),  # (N_O, n_cur) current-model scores
                "labels_full": torch.cat(truth),
            },
        )

    def _current_ce(self, cur_out, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.config.get("current_oracle", False):
            return super()._current_ce(cur_out, batch)
        logits = cur_out.logits.float()
        labels = oracle_refresh(batch["labels"], batch["labels_full"].long(), logits.shape[-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        if self.mass_probe:
            self._probe_current(task, train_loader)

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        self._head_sizes.append(self._head_size())
        if self.mass_probe:
            self._probe_memory(task)
