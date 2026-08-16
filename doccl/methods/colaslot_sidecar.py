"""CoLaR with gradient-separated, aged LexSlot sidecars."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.colaslot import CoLaSlot
from doccl.types import TaskInfo, TrainMetrics


class CoLaSlotSidecar(CoLaSlot):
    """Keep CoLaR exact; learn slots separately and expose only aged owners."""

    name = "colaslot_sidecar"

    def __init__(self, model, config):
        super().__init__(model, config)
        self._sidecar_read = False
        self._stage_eval = False

    def _slot_parameters(self):
        return [p for module in self._all_slot_modules() for p in module.parameters()]

    @staticmethod
    def _apply_read_policy(slot_module, read: bool, newest: int, stage_eval: bool) -> None:
        gate = slot_module._infer_gate
        if gate is None:
            return
        owners = torch.as_tensor(slot_module.slot_owner, device=gate.device)
        if not read:
            gate.zero_()
        elif stage_eval:
            gate[:, owners == newest] = 0

    def _install_infer_gate(self, module, args, kwargs) -> None:
        super()._install_infer_gate(module, args, kwargs)
        newest = len(self._task_sigs) - 1
        for slot_module in self._all_slot_modules():
            self._apply_read_policy(slot_module, self._sidecar_read, newest, self._stage_eval)

    def _sidecar_replay_loss(self, replay):
        return torch.zeros((), device=self.device)

    def _backward_slot_loss(self, batch, replay=None, replay_scale: float = 1.0):
        slot_ids = {id(p) for p in self._slot_parameters()}
        frozen = [p for p in self.model.parameters() if p.requires_grad and id(p) not in slot_ids]
        for parameter in frozen:
            parameter.requires_grad_(False)
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self._sidecar_read = True
        try:
            with torch.random.fork_rng(devices=rng_devices):
                loss = self.model(**batch).loss
                if replay is not None:
                    loss = loss + replay_scale * self._sidecar_replay_loss(replay)
                loss.backward()
                return loss.detach()
        finally:
            self._sidecar_read = False
            for parameter in frozen:
                parameter.requires_grad_(True)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        slot_params = self._slot_parameters()
        slot_ids = {id(p) for p in slot_params}
        base_params = [p for p in self.trainable_parameters() if id(p) not in slot_ids]

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"SC T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                ce_loss = self.model(**batch).loss
                replay_loss = torch.zeros((), device=self.device)
                replay = self._sample_replay()
                if replay is not None:
                    replay_loss = self._replay_forward(replay).loss
                base_loss = ce_loss + self._replay_loss_scale(task.task_id) * replay_loss
                base_loss.backward()
                torch.nn.utils.clip_grad_norm_(base_params, max_grad_norm)

                slot_loss = self._backward_slot_loss(
                    batch, replay, self._replay_loss_scale(task.task_id)
                )
                torch.nn.utils.clip_grad_norm_(slot_params, max_grad_norm)
                optimizer.step()
                self._post_optimizer_step()

                total_loss += float((base_loss + slot_loss).item())
                n_steps += 1
                pbar.set_postfix(
                    {
                        "ce": f"{ce_loss.item():.3f}",
                        "replay": f"{replay_loss.item():.3f}",
                        "slot": f"{slot_loss.item():.3f}",
                    }
                )
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    @contextmanager
    def diagnostic_forward_context(self):
        self._sidecar_read = True
        self._stage_eval = True
        try:
            yield
        finally:
            self._stage_eval = False
            self._sidecar_read = False

    def evaluate(self, eval_loaders):
        with self.diagnostic_forward_context():
            return super().evaluate(eval_loaders)
