"""LexMem v5 — relational reconstruction memory (graph-as-memory).

Design: docs/design/LEXMEM_V5_GRAPH_RECONSTRUCTION.md. Tracks CLUE-yc4.

The predecessor line (LexSlot / LexMem KV-slot / Lexical Ledger) stored *answers*
and was falsified: a stored logit is stale the moment the head drifts. v5 inverts
this — the memory stores the *ability to reconstruct training signal*. Each task's
per-class token-feature distribution is kept as a Gaussian node in a growing graph;
edges encode drift-immune relations between class-nodes. At each later session we
*reconstruct* (feature, label) pairs by message-passing over a node and its
neighbours, then consolidate them into the head. This synthesises the balanced
all-task gradient that only real replay provided (Finding 3 of the analysis paper)
without storing documents.

Stage 1 (this file): fixed (computed, non-learned) edges + Gaussian nodes, and the
decisive edge-ablation — ``edges_enabled=True`` (message-passing) vs ``False`` (a
bank of identical Gaussians, i.e. FeCAM-style independent-node reconstruction). Same
nodes, same storage; the only difference is whether relations are used. Stage 2 (a
free-parameter growing GNN) is gated behind this ablation clearing its bar and is not
implemented here.

The v3b spine is reused wholesale: the diagnosed freeze map keeps the trunk stable
(so stored Gaussians stay approximately valid) and online EWC regularises the plastic
bucket. v5's departure from v3b: the classifier head is made **plastic again** and is
consolidated on reconstructed past-task features — the head is where forgetting lives,
so it is where reconstruction must be replayed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812 — canonical torch alias (repo-wide)
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.lexmem import LexMem
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

_MIN_TOKENS_PER_CLASS = 16  # below this a class Gaussian is too noisy to store
_COV_EPS = 1e-4  # diagonal floor for numerical stability when sampling


@dataclass
class ClassNode:
    """One (task, class) node: the sparse trace we store instead of documents."""

    task_id: int
    label: int
    mu: torch.Tensor  # (D,) centroid in the encoder space at learning time
    cov_diag: torch.Tensor  # (D,) diagonal covariance (low-rank omitted in Stage 1)
    cov_lowrank: torch.Tensor  # (R, D) top-R covariance directions (R may be 0)
    count: int  # tokens that formed this class (for background/edge weighting)
    doc_labels: frozenset[int] = field(default_factory=frozenset)  # co-occurring labels

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.mu, self.cov_diag, self.cov_lowrank))


class RelationalMemory:
    """A growing, append-only graph of class Gaussians with computed edges.

    Nodes are frozen once written (a new task only *adds* nodes and edges *to*
    existing nodes) — this is the invariant that stops the memory itself from
    catastrophically forgetting. Edges are recomputed additively; existing edge
    values are never rewritten.
    """

    def __init__(self, hidden_dim: int, cov_rank: int, n_hops: int, edge_temp: float):
        self.hidden_dim = hidden_dim
        self.cov_rank = cov_rank
        self.n_hops = n_hops
        self.edge_temp = edge_temp
        self.nodes: list[ClassNode] = []
        # edges[i][j] = relation weight between node i and node j (symmetric)
        self.edges: dict[int, dict[int, float]] = {}

    def add_task_nodes(self, new_nodes: list[ClassNode]) -> None:
        """Append this task's nodes and wire edges to all prior nodes (append-only)."""
        start = len(self.nodes)
        self.nodes.extend(new_nodes)
        for i in range(start, len(self.nodes)):
            self.edges.setdefault(i, {})
            for j in range(len(self.nodes)):
                if i == j:
                    continue
                w = self._edge_weight(self.nodes[i], self.nodes[j])
                self.edges[i][j] = w
                self.edges.setdefault(j, {})[i] = w

    def _edge_weight(self, a: ClassNode, b: ClassNode) -> float:
        """Drift-immune relation: centroid cosine + label co-occurrence.

        Both terms are (approximately) invariant to the encoder's coordinate frame:
        cosine is scale/rotation-robust relative to absolute position, and
        co-occurrence is a pure input-structural statistic. Same-node-label pairs
        across tasks (the DIL unified label space) get a co-occurrence boost.
        """
        cos = float(F.cosine_similarity(a.mu, b.mu, dim=0).clamp(-1, 1))
        co = 1.0 if (a.label == b.label or a.label in b.doc_labels) else 0.0
        return 0.5 * (cos + 1.0) + 0.5 * co  # in [0, 1.5]

    def _neighbor_ids(self, node_id: int) -> list[int]:
        return sorted(self.edges.get(node_id, {}), key=lambda j: -self.edges[node_id][j])

    def reconstruct(
        self,
        node_ids: list[int],
        per_node: int,
        edges_enabled: bool,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Regenerate (feature, label) pairs for the given nodes.

        edges_enabled=True  → message-passing: a node's sampling distribution is a
                              neighbour-weighted blend of its own and neighbours'
                              Gaussians (competence reconstituted from relations).
        edges_enabled=False → bank: each node sampled from its own Gaussian only
                              (independent-node reconstruction — the ablation baseline).
        """
        feats: list[torch.Tensor] = []
        labels: list[int] = []
        for nid in node_ids:
            node = self.nodes[nid]
            mu, cov = node.mu.to(device), node.cov_diag.to(device)
            if edges_enabled:
                mu, cov = self._message_passed_gaussian(nid, device)
            samples = self._sample_gaussian(mu, cov, per_node, generator, device)
            feats.append(samples)
            labels.extend([node.label] * per_node)
        if not feats:
            return (
                torch.empty(0, self.hidden_dim, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        return torch.cat(feats), torch.tensor(labels, dtype=torch.long, device=device)

    def _message_passed_gaussian(
        self, node_id: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K-hop neighbour-weighted mean/cov (Stage 1 fixed-edge message passing).

        The self node dominates (weight 1.0); neighbours pull the reconstruction
        toward related structure proportional to edge weight, decayed by hop.
        """
        node = self.nodes[node_id]
        acc_mu = node.mu.to(device).clone()
        acc_cov = node.cov_diag.to(device).clone()
        total_w = 1.0
        frontier = {node_id: 1.0}
        seen = {node_id}
        for hop in range(self.n_hops):
            decay = 0.5 ** (hop + 1)
            nxt: dict[int, float] = {}
            for src, src_w in frontier.items():
                for j in self._neighbor_ids(src):
                    if j in seen:
                        continue
                    w = src_w * self.edges[src][j] * decay
                    nxt[j] = max(nxt.get(j, 0.0), w)
            for j, w in nxt.items():
                seen.add(j)
                acc_mu = acc_mu + w * self.nodes[j].mu.to(device)
                acc_cov = acc_cov + w * self.nodes[j].cov_diag.to(device)
                total_w += w
            frontier = nxt
            if not frontier:
                break
        return acc_mu / total_w, acc_cov / total_w

    def _sample_gaussian(
        self,
        mu: torch.Tensor,
        cov_diag: torch.Tensor,
        n: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        std = (cov_diag.clamp_min(_COV_EPS)).sqrt()
        eps = torch.randn(n, mu.shape[0], generator=generator, device=device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)

    def node_ids_for_tasks(self, task_ids: set[int]) -> list[int]:
        return [i for i, nd in enumerate(self.nodes) if nd.task_id in task_ids]

    def storage_bytes(self) -> int:
        node_b = sum(nd.storage_bytes() for nd in self.nodes)
        edge_b = sum(len(v) for v in self.edges.values()) * 4  # float32 weights
        return node_b + edge_b


class LexMemV5(LexMem):
    """Relational reconstruction memory. See module docstring / design doc."""

    def __init__(self, model, config):
        # Disable the KV-slot memory path entirely; v5's memory is the graph.
        # freeze_late_n must be >= 0 so the trunk freeze map still applies (the
        # base __init__ guards against mem_enabled=false with freeze_late_n=-1).
        config = dict(config)
        config["mem_enabled"] = False
        if int(config.get("freeze_late_n", -1)) < 0:
            config["freeze_late_n"] = 4
        super().__init__(model, config)

        self.edges_enabled = bool(config.get("edges_enabled", True))
        self.cov_rank = int(config.get("cov_rank", 0))
        self.n_hops = int(config.get("n_hops", 2))
        self.edge_temp = float(config.get("edge_temp", 1.0))
        self.recon_per_class = int(config.get("recon_per_class", 64))
        self.recon_weight = float(config.get("recon_weight", 1.0))
        self.head_lr = float(config.get("head_lr", config.get("lr", 5e-5)))
        # Deterministic reconstruction sampling (seeded off the run seed if present).
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(config.get("seed", 0)) * 100003 + 5)

        self.graph = RelationalMemory(self.hidden_dim, self.cov_rank, self.n_hops, self.edge_temp)

    def after_task(self, task: TaskInfo, train_loader) -> None:
        """Freeze the trunk after task 0 (but keep the head plastic), then write
        this task's class-Gaussian nodes into the growing graph."""
        if task.task_id == 0:
            self._apply_freeze_map_keep_head()
            if self.drift_probe_batches > 0:
                self._snapshot_probe(train_loader)
        # EWC on the plastic trunk bucket (inherited machinery).
        if self.ewc_lambda > 0:
            self._accumulate_fisher(task, train_loader)
        # Write nodes for this task (append-only; prior nodes/edges untouched).
        new_nodes = self._build_task_nodes(task, train_loader)
        self.graph.add_task_nodes(new_nodes)
        log.info(
            "lexmem_v5: task %d — added %d class nodes; graph now %d nodes, %d bytes",
            task.task_id,
            len(new_nodes),
            len(self.graph.nodes),
            self.graph.storage_bytes(),
        )

    def _apply_freeze_map_keep_head(self) -> None:
        """v5 freeze map: freeze the diagnosed trunk locus but keep the head PLASTIC.

        The base map freezes the head (its role goes to the KV memory). v5 replays
        reconstructed features *into* the head, so the head must stay trainable — it
        is the consolidation target.
        """
        self._apply_freeze_map()  # freezes late-n trunk layers AND the head
        for p in self.model.model.classifier.parameters():
            p.requires_grad = True
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("lexmem_v5: freeze map applied, head kept plastic; trainable=%d", n_train)

    def _build_task_nodes(self, task: TaskInfo, train_loader) -> list[ClassNode]:
        """Per-class Gaussian (μ, diag cov) from masked classifier-input features."""
        feats, labels = self._collect_feats_labeled(train_loader)
        nodes: list[ClassNode] = []
        if feats.numel() == 0:
            log.warning("lexmem_v5: task %d produced no features", task.task_id)
            return nodes
        present = torch.unique(labels)
        doc_labels = frozenset(int(x) for x in present.tolist())
        for lab in present.tolist():
            sel = feats[labels == lab]
            if sel.shape[0] < _MIN_TOKENS_PER_CLASS:
                continue
            sel = sel.float()
            mu = sel.mean(0)
            cov_diag = sel.var(0, unbiased=False)
            nodes.append(
                ClassNode(
                    task_id=task.task_id,
                    label=int(lab),
                    mu=mu.detach().cpu(),
                    cov_diag=cov_diag.detach().cpu(),
                    cov_lowrank=torch.empty(0, self.hidden_dim),
                    count=int(sel.shape[0]),
                    doc_labels=doc_labels,
                )
            )
        return nodes

    def _collect_feats_labeled(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked classifier-input features paired with their token labels.

        Mirrors ``_collect_feats`` but keeps labels so per-class stats can be built.
        Ignored tokens (label -100) are dropped.
        """
        feat_chunks: list[torch.Tensor] = []
        lab_chunks: list[torch.Tensor] = []
        n = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="v5 node-feats", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
                if self._cur_feats is None:
                    continue
                f = self._cur_feats
                labels = batch.get("labels")
                mask = batch.get("attention_mask")
                if mask is not None and mask.shape[:2] == f.shape[:2]:
                    keep = mask.bool()
                else:
                    keep = torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)
                if labels is not None and labels.shape[:2] == f.shape[:2]:
                    keep = keep & (labels != -100)
                    lab_flat = labels[keep]
                else:
                    lab_flat = torch.zeros(int(keep.sum()), dtype=torch.long, device=f.device)
                feat_chunks.append(f[keep].detach().half().cpu())
                lab_chunks.append(lab_flat.detach().cpu())
                n += int(keep.sum())
                if n >= 2 * self.key_sample_cap:
                    break
        if was_training:
            self.model.train()
        if not feat_chunks:
            return torch.empty(0, self.hidden_dim), torch.empty(0, dtype=torch.long)
        feats = torch.cat(feat_chunks)
        labs = torch.cat(lab_chunks)
        if feats.shape[0] > self.key_sample_cap:
            idx = torch.randperm(feats.shape[0])[: self.key_sample_cap]
            feats, labs = feats[idx], labs[idx]
        return feats.to(self.device), labs.to(self.device)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        if task.task_id == 0:
            return super(LexMem, self).train_task(task, train_loader, val_loader)
        return self._train_reconstruct(task, train_loader, val_loader)

    def _train_reconstruct(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None
    ) -> TrainMetrics:
        """Tasks >= 1: train head (+ plastic trunk) on new-task data plus features
        reconstructed from the graph for every past task (balanced consolidation)."""
        self.model.train()
        # Head (consolidation target) + plastic trunk bucket. Snapshot exactly the
        # modules that train so the early-stopper restores the right best-weights.
        head = self.model.model.classifier
        head_params = [p for p in head.parameters() if p.requires_grad]
        backbone = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("model.classifier")
        ]
        groups: list[dict] = [{"params": head_params, "lr": self.head_lr, "weight_decay": 0.0}]
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": float(self.config.get("lr", 5e-5)),
                    "weight_decay": float(self.config.get("weight_decay", 0.01)),
                }
            )
        params = head_params + backbone
        opt_cls = torch.optim.AdamW if self.mem_optimizer == "adamw" else torch.optim.SGD
        optimizer = opt_cls(groups)
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self._amp_setup()

        # NOTE: v5 deliberately does NOT use best-val-F1 restoration. The validation
        # signal available here is CURRENT-task F1, which the reconstruction-
        # consolidation loss (targeting PAST tasks) barely moves. Restoring the
        # best-current-F1 epoch therefore reverts exactly the recon updates that make
        # edges-on differ from edges-off — collapsing the whole method to a no-op and
        # producing byte-identical arms (the bug this replaces). We instead keep the
        # fully consolidated final state and use a fixed epoch budget. Patience-based
        # *stopping* on a retention signal is future work (needs all-seen-task loaders).
        past_ids = self.graph.node_ids_for_tasks(set(range(task.task_id)))
        total_loss, n_steps = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"v5 T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                with self._amp_autocast():
                    out = self.model(**batch)
                    loss = out.loss
                    recon_loss = self._reconstruction_loss(past_ids)
                    loss = loss + self.recon_weight * recon_loss
                    if self.ewc_lambda > 0 and self._fisher:
                        loss = loss + self._ewc_penalty()
                self._amp_backward_step(loss, optimizer, params, max_grad_norm)
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "recon": f"{float(recon_loss):.3f}"}
                )
            self._log_weight_diagnostics()
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def _reconstruction_loss(self, past_ids: list[int]) -> torch.Tensor:
        """Cross-entropy of the current head on reconstructed past-task features.

        Feeds regenerated features directly through the classifier head (bypassing
        the frozen trunk — the features already live in head-input space), so the
        head receives balanced gradients from every past task each step. This is the
        synthesised-replay signal; edges_enabled toggles the ablation.
        """
        if not past_ids:
            return torch.zeros((), device=self.device)
        feats, labels = self.graph.reconstruct(
            past_ids,
            per_node=self.recon_per_class,
            edges_enabled=self.edges_enabled,
            generator=self._gen,
            device=self.device,
        )
        if feats.numel() == 0:
            return torch.zeros((), device=self.device)
        logits = self.model.model.classifier(feats)
        return F.cross_entropy(logits, labels)
