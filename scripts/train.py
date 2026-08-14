"""Main training entry point for CL experiments.

Usage:
    python scripts/train.py method=naive scenario=single_funsd
    python scripts/train.py method=ewc scenario=cil_cord seed=42
    python scripts/train.py method=der_pp scenario=cil_cord training.epochs=15
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

import wandb
from doccl.data.encoders import build_encoder
from doccl.data.scenarios import get_scenario
from doccl.eval.metrics import CLMetricsTracker, compute_per_class_f1
from doccl.methods.aglr_replay import AGLRReplay
from doccl.methods.cl_lora import CLLoRA
from doccl.methods.coda_prompt import CODAPrompt
from doccl.methods.colar import CoLaR
from doccl.methods.colar_bal import CoLaRBal
from doccl.methods.colar_cb import CoLaRCB
from doccl.methods.colar_knn import CoLaRKNN
from doccl.methods.colar_meta import CoLaRMeta
from doccl.methods.colar_wsvd import CoLaRWSVD
from doccl.methods.colaslot import CoLaSlot
from doccl.methods.colaslot_ra import CoLaSlotRA
from doccl.methods.colaslot_rf import CoLaSlotRF
from doccl.methods.colaslot_ro import CoLaSlotRO
from doccl.methods.coreset_memory import CoresetMemory
from doccl.methods.cpfd import CPFD
from doccl.methods.cuber import CUBER
from doccl.methods.der import DERpp
from doccl.methods.doc_merge import DocMerge
from doccl.methods.doccl import DocCL, DocCL_A, DocCL_B, DocCL_C
from doccl.methods.dualprompt import DualPrompt
from doccl.methods.er import ER
from doccl.methods.er_cflat import ERCFlat
from doccl.methods.ewc import EWC
from doccl.methods.fisher_mask import FisherMaskFreeze
from doccl.methods.gauss_replay import GaussReplay
from doccl.methods.hgt import HGT
from doccl.methods.hybrid_routed_prompt import HybridRoutedPrompt
from doccl.methods.is3 import IS3
from doccl.methods.l2p import L2P
from doccl.methods.larm import LARM
from doccl.methods.latent_replay import LatentReplay
from doccl.methods.lca import LCA
from doccl.methods.lexmem import LexMem
from doccl.methods.lexmem_v5 import LexMemV5
from doccl.methods.lexslot import LexSlot
from doccl.methods.lexslot_fm import LexSlotFM
from doccl.methods.lwf import LwF
from doccl.methods.magmax import MagMax
from doccl.methods.marginal_methods import LogitAdjust, MarginalAnchor
from doccl.methods.naive import JointMultiTask, NaiveFineTune
from doccl.methods.nullspace_analytic import NullSpaceAnalyticCL
from doccl.methods.o_lora import OLoRA
from doccl.methods.proxy_latent_replay import ProxyLatentReplay
from doccl.methods.sd_lora import SDLoRA
from doccl.methods.spectral_memory import SpectralMemory
from doccl.models.bert_family_wrapper import BERTWrapper
from doccl.models.bros_wrapper import BROSWrapper
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.models.lilt_wrapper import LiLTWrapper
from doccl.utils.tb_logger import TBLogger

log = logging.getLogger(__name__)


# Maps ``cfg.model.family`` → backbone wrapper class. LayoutLMv3 is the primary;
# LiLT/BROS are the vision-free secondaries for the generalization study. A new
# backbone is registered here + a ``configs/model/<name>.yaml`` carrying ``family``.
MODEL_REGISTRY = {
    "layoutlmv3": LayoutLMv3Wrapper,
    "lilt": LiLTWrapper,
    "bros": BROSWrapper,
    # External text-only comparator (unimodal): the genuine BERT baseline.
    "bert": BERTWrapper,
}


# Per-task dataset for each scenario (mirrors analyze_results.SCENARIO_TASK_DATASETS
# and doccl/data/scenarios.py). Static fallback used only when the scenario carries no
# per-task ``native_dataset`` metadata AND its length matches the task count.
SCENARIO_TASK_DATASETS: dict[str, list[str]] = {
    "dil": ["funsd", "sroie", "cord"],
    "dil_receipts": ["sroie", "cord", "wildreceipt"],
    "cil_cord": ["cord", "cord", "cord", "cord", "cord"],
    "mixed": ["funsd", "funsd", "sroie", "cord", "cord", "funsd"],
    # Cross-lingual DIL: all 5 language tasks share the single XFUND baseline (schema
    # is constant across languages), mirroring how cil_cord reuses one CORD baseline.
    "dil_xlingual": ["xfund", "xfund", "xfund", "xfund", "xfund"],
    "cil_wildreceipt": ["wildreceipt", "wildreceipt", "wildreceipt", "wildreceipt"],
}

# Single dataset per task for CIL scenarios that carry no per-task metadata; used to
# size the FWT baseline to *any* task count (e.g. the long-horizon variants).
_CIL_SINGLE_DATASET = {
    "cil_cord": "cord",
    "cil_wildreceipt": "wildreceipt",
    "cil_funsd": "funsd",
}


def _resolve_task_datasets(scenario) -> list[str] | None:
    """Per-task single-task-baseline dataset, robust to ordering + task count.

    1) per-task ``native_dataset`` metadata (DIL family — survives reordering);
    2) the static map when its length matches the task count (mixed, base CIL);
    3) a single CIL dataset repeated to the task count (long-horizon variants).
    """
    metas = [t.metadata.get("native_dataset") for t in scenario.tasks]
    if all(m is not None for m in metas):
        return metas
    n = len(scenario.tasks)
    static = SCENARIO_TASK_DATASETS.get(scenario.name)
    if static is not None and len(static) == n:
        return static
    single = _CIL_SINGLE_DATASET.get(scenario.name)
    if single is not None:
        return [single] * n
    return None


def load_fwt_baselines(scenario, baseline_csv: Path) -> list[float] | None:
    """Build the per-task baseline vector b_i for FWT from the single-task CSV.

    Returns a list aligned to the scenario's tasks (b_i = from-scratch single-task F1
    on task i's dataset), or None if the scenario is unmapped or the CSV/datasets are
    missing — in which case FWT is left as NaN (honestly unavailable) for this run.
    """
    datasets = _resolve_task_datasets(scenario)
    if datasets is None or not baseline_csv.exists():
        return None
    import csv

    means: dict[str, float] = {}
    try:
        with baseline_csv.open() as fh:
            for row in csv.DictReader(fh):
                means[row["dataset"]] = float(row["single_task_f1_mean"])
    except (OSError, KeyError, ValueError):
        return None
    if not all(d in means for d in datasets):
        return None  # missing a needed dataset baseline → don't fabricate FWT
    return [means[d] for d in datasets]


METHOD_REGISTRY = {
    "naive": NaiveFineTune,
    "joint": JointMultiTask,
    # RCA kill-tests #4/#5 (docs/RCA_KILLTESTS_PREREG_2026-07.md): marginal-anchored
    # training objectives storing only per-task gold label marginals (9 floats/task).
    "marginal_kl": MarginalAnchor,
    "logit_adjust": LogitAdjust,
    "ewc": EWC,
    "lwf": LwF,
    "er": ER,
    "der_pp": DERpp,
    # 2025 currency baseline: C-Flat++ (flat-minima/SAM) bolted onto ER.
    "er_cflat": ERCFlat,
    "o_lora": OLoRA,
    # 2025 currency baseline: CL-LoRA (dual-adapter LoRA), successor to O-LoRA.
    "cl_lora": CLLoRA,
    "l2p": L2P,
    "lca": LCA,
    # 2023 class-incremental NER baseline: CPFD (pseudo-labels + attention distillation, EMNLP'23).
    "cpfd": CPFD,
    # 2024 token-level incremental NER baseline: IS3 (two-shift: E2O + O2E, ACL'24).
    "is3": IS3,
    # 2024 model-merging baseline: MagMax (max-magnitude task-vector selection),
    # merge-only control (no post-merge classifier realignment, unlike ``lca``).
    "magmax": MagMax,
    # 2025 LoRA baseline: SD-LoRA (decoupled magnitude/direction, rehearsal-free, ICLR'25).
    "sd_lora": SDLoRA,
    "dualprompt": DualPrompt,
    "coda_prompt": CODAPrompt,
    # Hybrid dense+sparse task router over a task-pinned prompt pool (feasibility
    # prototype). ``method.router`` ∈ {dense, sparse, hybrid} toggles the routing
    # ablation; writes results/<run>/routing.json with the per-task routing hit-rate.
    "hrp": HybridRoutedPrompt,
    "hgt": HGT,
    "cuber": CUBER,
    "doc_merge": DocMerge,
    # Proposed method: depth/head-targeted DocCL, derived from the corrected
    # diagnosis (forgetting concentrates in the classifier head + late layers).
    # ``method.target_depth`` ∈ {all, head_only, late_only, uniform} drives the
    # component-targeting ablation (Table 6.7).
    "doccl": DocCL,
    # LexSlot: standalone lexically-gated slot memories (no DocCL base; plain CE on a
    # partially-frozen backbone, no replay/KD/Fisher). ``method.slot_depth`` ∈
    # {head_only, head_late, head_late_mid, uniform} and ``method.slot_sharing`` ∈
    # {soft, hard, off} are the two ablation axes.
    "lexslot": LexSlot,
    # LexMem: sparse lexical-memory head (SMF-style, arXiv 2510.15103 adapted to doc
    # IE). Task 0 full FT then frozen base; tasks >= 1 train only top-t TF-IDF-selected
    # memory-value rows. ``method.select`` ∈ {tfidf, tf, random} is the ablation axis.
    "lexmem": LexMem,
    # LexMem v2: diagnosis-guided freeze map (head + late-N frozen = the forgetting
    # locus; early/mid trainable = plasticity) + feature-space memory values.
    "lexmem_v2": LexMem,
    # LexMem control: identical freeze map, NO memory — isolates the freeze map's
    # own contribution (reviewer control for the v2 comparison).
    "lexmem_ctrl": LexMem,
    # LexMem v3: v2 + online EWC restricted to the plastic early/mid bucket
    # (counters the pressure-redirection drift the v2 pilot measured, CKA 0.19).
    "lexmem_v3": LexMem,
    # Gaussian head-replay on the v3b drift-controlled trunk (exemplar-free,
    # stores per-(class x task) feature Gaussians) — analysis-program experiment #2.
    "gauss_replay": GaussReplay,
    # Experiment #6: parameter-granularity migration test — per-tensor top-p
    # Fisher-important entries hard-frozen, interleaved remainder plastic.
    "fisher_mask": FisherMaskFreeze,
    # Falsification-chain level 6: frozen-trunk activation replay at encoder layer k
    # (Pellegrini-style latent replay). Tests the migration law's untested cell —
    # real past-task gradients into the plastic upper layers on never-stale features.
    # ``method.docs_per_task=0`` is the freeze-only control arm.
    "latent_replay": LatentReplay,
    # SLR (Spectral Latent Replay): latent_replay with the raw activation bank replaced
    # by a per-task rank-r spectral summary (SVD factors) — memory indexed by the
    # forgetting subspace. The proposed method; Gate 0 = beat aglr_replay at lower bytes.
    "spectral_memory": SpectralMemory,
    # AGLR-CL port (arXiv 2505.08524): per-(task x class) Gaussians in layer-k space +
    # attention-salience filter — memory indexed BY CLASS. The Gate-0 comparator for SLR.
    "aglr_replay": AGLRReplay,
    # CoresetMemory: stores REAL layer-k activation centroids (k-means) per (task,class),
    # not a fitted distribution — the one feature-replay variant not falsified at Gate 0.
    "coreset_memory": CoresetMemory,
    # PLaR: whole-doc latent replay from a pseudo-labeled PUBLIC proxy pool (WildReceipt) —
    # zero private storage; consistency-law-prescribed successor to the falsified variants.
    "proxy_latent_replay": ProxyLatentReplay,
    # CoLaR: latent_replay with per-DOCUMENT rank-r SVD storage — d50 coverage at ~d5 bytes
    # (whole-doc binding preserved; per-doc matrices ARE low-rank though the pooled space isn't).
    "colar": CoLaR,
    # CoLaR + isolated additive slots: tests whether task-specific capacity can break
    # the compressed-replay retention frontier before adding lexical routing.
    "colaslot": CoLaSlot,
    "colaslot_r": CoLaSlot,
    "colaslot_ra": CoLaSlotRA,
    "colaslot_rf": CoLaSlotRF,
    "colaslot_ro": CoLaSlotRO,
    # CoLaR-Bal: CoLaR + soft-target replay (dark knowledge) to protect sparse classes (SROIE)
    "colar_bal": CoLaRBal,
    # CoLaR-CB: same bytes; balances old-task mass and sparse-label replay gradients.
    "colar_cb": CoLaRCB,
    # CoLaR-WSVD: same CoLaR bytes, entity-weighted compression error allocation.
    "colar_wsvd": CoLaRWSVD,
    # CoLaR-kNN: read-side memory — CoLaR's store doubles as a drift-free labeled datastore
    # (layers <k frozen), blended with the head as a kNN readout at eval. Training untouched.
    "colar_knn": CoLaRKNN,
    # CoLaR-Meta: memory in weight DYNAMICS — Benna-Fusi multi-timescale consolidation on
    # the plastic bucket, composed with replay. Falsification-risk track (EWC precedent).
    "colar_meta": CoLaRMeta,
    # LARM: CoLaR replay fused with a lexically-routed additive feature-rewrite memory (replay
    # writes the memory values, OCR-cosine routing reads them). Targets BWT×FWT Pareto-dominance.
    "larm": LARM,
    # Null-Space Analytic CL: hierarchical lexical memory + gradient projection
    # + analytic head update. Stores (feature, position, label) tuples per
    # (task, class); projects backbone gradients to prevent feature drift.
    "nullspace_analytic": NullSpaceAnalyticCL,
    # LexMem v3b: v3 with lambda retuned (1000 -> 300; CKA 0.999 was over-stiff,
    # SROIE 51) + hard exclusion of prior-task slots from selection (Jaccard
    # 0.31 slot overwrite drove task-1 forgetting 51 -> 12).
    "lexmem_v3b": LexMem,
    # LexMem v5: relational reconstruction memory (graph-as-memory). Stores per-class
    # feature Gaussians as growing-graph nodes; reconstructs (feature,label) training
    # signal via message-passing and consolidates into the head. edges_enabled toggles
    # the Stage-1 ablation (message-passing vs FeCAM-style independent-node bank).
    "lexmem_v5": LexMemV5,
    # LexSlot-FM: Functional Memory LexSlot (dual-stream; frozen encoder + per-task
    # frozen functional heads, lexically blended with a plastic base head).
    "lexslot_fm": LexSlotFM,
    # Legacy sketched candidates, kept as ablation variants / NeurIPS extension.
    "doccl_a": DocCL_A,
    "doccl_b": DocCL_B,
    "doccl_c": DocCL_C,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_hparams(method_cfg) -> dict:
    """Flatten a method's Hydra config to a plain JSON-safe dict of its real values.

    Excludes the ``name`` key (already recorded). Captures every method knob as it
    was actually resolved (defaults included) so a run's exact configuration —
    cflat_lambda, use_replay, lambda_ortho, buffer_size, kd_alpha, target_depth,
    rho, etc. — is permanently recoverable from metrics.json.
    """
    from omegaconf import OmegaConf

    try:
        raw = OmegaConf.to_container(method_cfg, resolve=True)
    except Exception:
        raw = dict(method_cfg)
    out = {}
    for k, v in (raw or {}).items():
        if k == "name":
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)  # nested/sequence -> stringified, never lost
    return out


def save_run_metrics(
    out_dir: Path,
    cfg: DictConfig,
    tracker: CLMetricsTracker,
    method,
    task_times: list[float],
) -> None:
    """Persist a structured per-run ``metrics.json`` for offline ingestion.

    Decouples ``scripts/analyze_results.py`` from W&B: on an offline Vast.ai box
    the LaTeX result tables (6.1/6.3) and forgetting curves are built directly
    from ``results/<run>/{matrix.npy,metrics.json}``. Records the CL metrics, the
    accuracy matrix, per-task wall time, parameter counts, and peak GPU memory
    (the raw signals behind the computational-overhead table).
    """
    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    metrics = {
        "method": cfg.method.name,
        "scenario": cfg.scenario.name,
        "seed": int(cfg.seed),
        # Backbone family so the result pipeline can distinguish e.g. a text-only
        # BERT "naive" run from the LayoutLMv3 "naive" run (same method string).
        "model_family": cfg.model.get("family", "layoutlmv3"),
        "target_component": cfg.method.get("target_component"),
        "target_depth": cfg.method.get("target_depth"),
        **tracker.to_dict(),
        "wall_time_per_task_s": [float(t) for t in task_times],
        "total_wall_time_s": float(sum(task_times)),
        "mean_time_per_task_s": float(sum(task_times) / max(len(task_times), 1)),
        "total_params": int(method.total_param_count()),
        "trainable_params": int(method.trainable_param_count()),
        "peak_gpu_mem_mb": peak_mem_mb,
        # Replay/episodic memory footprint in bytes for methods that expose it (latent/
        # spectral/AGLR replay) — the x-axis of the SLR accuracy-vs-memory Pareto curve.
        # None for methods without a replay bank. See doccl/methods/{spectral_memory,
        # aglr_replay,latent_replay}.py :: memory_bytes().
        "replay_memory_bytes": (
            int(method.memory_bytes()) if hasattr(method, "memory_bytes") else None
        ),
        # Full method hyper-parameters as actually resolved at run time. Records the
        # real value of every knob (e.g. cflat_lambda, use_replay, lambda_ortho,
        # buffer_size) so a run's configuration is always recoverable from its
        # metrics.json — closing the silent-default trap where e.g. er_cflat ran with
        # cflat_lambda=0.0 (plain SAM) but nothing recorded it.
        "method_hparams": _resolve_hparams(cfg.method),
        "training_hparams": {
            k: cfg.training.get(k)
            for k in ("batch_size", "epochs", "gradient_checkpointing", "amp", "fp16", "lr")
            if cfg.training.get(k) is not None
        },
    }
    if hasattr(method, "diagnostic_metrics"):
        metrics["method_diagnostics"] = method.diagnostic_metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics to %s", out_dir / "metrics.json")


def save_per_class_f1(out_dir: Path, model, eval_loaders: dict, device) -> None:
    """Per-entity-type F1 on the final model for each seen task (review M5).

    DIL-degeneracy evidence: surfaces whether an aggregate F1 is carried by a
    dominant class (e.g. VALUE) while sparse classes (KEY, present only in
    FUNSD/SROIE) are effectively unlearned. Uses the standard model forward, so
    call only for standard-forward methods.
    """
    model.eval()
    id_to_label = getattr(model, "id_to_label", {})
    out: dict[str, dict] = {}
    with torch.no_grad():
        for tid, loader in eval_loaders.items():
            preds, golds = [], []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                p = logits.argmax(-1)
                lbl = batch["labels"]
                m = lbl != -100
                preds.extend(p[m].cpu().tolist())
                golds.extend(lbl[m].cpu().tolist())
            out[str(tid)] = compute_per_class_f1(preds, golds, id_to_label)
    with open(out_dir / "per_class_f1.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved per-class F1 to %s", out_dir / "per_class_f1.json")


def log_boundary_diagnostics(
    tb,
    model,
    fisher_old: dict,
    params_before: dict,
    boundary: str,
    step: int,
) -> None:
    """Log the forgetting localiser at a task boundary (reuses eval/fisher).

    Writes, for the just-finished task transition, the old-task Fisher-weighted
    parameter displacement (per component group AND per depth bucket) — i.e. *where*
    the previous task's knowledge was overwritten. This is the forgetting-specific
    localiser the pilot computes; here it runs for every CL method. (Per-layer CKA is
    available via doccl.eval.cka for a dedicated representation-drift study.)
    """
    from doccl.eval.fisher import fisher_weighted_displacement, snapshot_params

    params_after = snapshot_params(model)
    disp_group = fisher_weighted_displacement(
        fisher_old, params_before, params_after, model.param_groups, model
    )
    disp_depth = fisher_weighted_displacement(
        fisher_old, params_before, params_after, model.param_groups_by_depth, model
    )
    tb.log_group_scalars(disp_group, f"displacement/{boundary}", step)
    tb.log_group_scalars(disp_depth, f"displacement_by_depth/{boundary}", step)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Include the ablation knob in the run name so ablation runs (same method,
    # different target) get distinct result dirs and are identifiable. DocCL uses
    # target_depth; the legacy candidates use target_component.
    run_name = f"{cfg.scenario.name}_{cfg.method.name}_seed{cfg.seed}"
    # Non-default backbones (BERT text-only, LiLT, BROS) suffix the run so their
    # result dirs/metrics don't collide with the LayoutLMv3 run of the same method.
    model_family = cfg.model.get("family", "layoutlmv3")
    if model_family != "layoutlmv3":
        run_name += f"_{model_family}"
    target_component = cfg.method.get("target_component")
    target_depth = cfg.method.get("target_depth")
    if cfg.method.name == "doccl" and target_depth not in (None, "all"):
        # "all" is the canonical full method (no suffix); ablations get one.
        run_name += f"_{target_depth}"
    elif cfg.method.name == "lexslot":
        # LexSlot has two ablation axes (slot_depth x slot_sharing). The canonical full
        # method is slot_depth=head_late + slot_sharing=soft (no suffix); any deviation
        # gets a suffix so ablation runs get distinct result dirs and never collide.
        slot_depth = cfg.method.get("slot_depth", "head_late")
        slot_sharing = cfg.method.get("slot_sharing", "soft")
        if slot_depth != "head_late":
            run_name += f"_{slot_depth}"
        if slot_sharing != "soft":
            run_name += f"_{slot_sharing}"
    elif cfg.method.name == "lexslot_fm":
        # LexSlot-FM has a gate_mode axis. The canonical default is "layout"; any
        # other gate mode gets a suffix so ablation runs are distinct.
        gate_mode = cfg.method.get("gate_mode", "layout")
        if gate_mode != "layout":
            run_name += f"_{gate_mode}"
    elif cfg.method.name == "lexmem_v5":
        # LexMem v5's ablation axis is edges_enabled (message-passing vs the
        # independent-node bank). The canonical full method is edges-on (no suffix);
        # the edges-off bank gets a "_bank" suffix so the two arms never collide.
        if not cfg.method.get("edges_enabled", True):
            run_name += "_bank"
    elif cfg.method.name == "fisher_mask":
        # Granularity arm: canonical p=0.8 (no suffix); other fractions get _p<N>.
        top_p = cfg.method.get("mask_top_p", 0.8)
        if top_p != 0.8:
            run_name += f"_p{int(round(top_p * 100))}"
    elif cfg.method.name == "latent_replay":
        # Latent replay's axes are split depth and buffer size. Canonical operating
        # point is k=8 / 5 docs per task (no suffix); other depths get "_k<N>" and
        # docs_per_task=0 is the freeze-only control arm ("_ctrl").
        split_k = cfg.method.get("split_layer_k", 8)
        docs = cfg.method.get("docs_per_task", 5)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if docs == 0:
            run_name += "_ctrl"
        elif docs != 5:
            run_name += f"_d{docs}"
        if cfg.method.get("doc_selection", "random") == "kcenter":
            run_name += "_kc"
    elif cfg.method.name in (
        "colar",
        "colaslot",
        "colaslot_r",
        "colaslot_ra",
        "colaslot_rf",
        "colaslot_ro",
        "colar_bal",
        "colar_cb",
        "colar_wsvd",
        "colar_knn",
    ):
        # CoLaR axes: split depth, docs, per-doc SVD rank, selection. Canonical k=8/d=5/r=64.
        # colar_knn adds the readout blend weight (canonical lambda=0.3, no suffix).
        split_k = cfg.method.get("split_layer_k", 8)
        docs = cfg.method.get("docs_per_task", 5)
        rank = cfg.method.get("rank_r", 64)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if docs != 5:
            run_name += f"_d{docs}"
        if rank != 64:
            run_name += f"_r{rank}"
        if cfg.method.get("doc_selection", "random") == "kcenter":
            run_name += "_kc"
        if cfg.method.name == "colar_wsvd":
            ew = cfg.method.get("svd_entity_weight", 4.0)
            if ew != 4.0:
                run_name += f"_ew{int(round(ew * 10))}"
        if cfg.method.name == "colar_cb":
            power = cfg.method.get("replay_balance_power", 0.5)
            if power != 0.5:
                run_name += f"_p{int(round(power * 100))}"
        if cfg.method.name == "colar_knn":
            lam = cfg.method.get("knn_lambda", 0.3)
            if lam != 0.3:
                run_name += f"_lam{int(round(lam * 100))}"
            if cfg.method.get("knn_class_balance", False):
                run_name += "_cb"
    elif cfg.method.name == "colar_meta":
        # CoLaR-Meta axes: CoLaR's k/d/rank + consolidation depth m (canonical m=3;
        # m=1 is the no-op control) and diffusion strength eta.
        split_k = cfg.method.get("split_layer_k", 8)
        docs = cfg.method.get("docs_per_task", 5)
        rank = cfg.method.get("rank_r", 64)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if docs != 5:
            run_name += f"_d{docs}"
        if rank != 64:
            run_name += f"_r{rank}"
        m_levels = cfg.method.get("meta_m", 3)
        if m_levels != 3:
            run_name += f"_m{m_levels}"
        eta = cfg.method.get("meta_eta", 0.01)
        if eta != 0.01:
            run_name += f"_eta{eta}"
    elif cfg.method.name == "larm":
        # LARM axes: CoLaR's k/d/rank + memory rank + the ablation switches. Canonical
        # k=8/d=5/rank_r=64/mem_rank=16/route/add (no suffix).
        split_k = cfg.method.get("split_layer_k", 8)
        docs = cfg.method.get("docs_per_task", 5)
        rank = cfg.method.get("rank_r", 64)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if docs != 5:
            run_name += f"_d{docs}"
        if rank != 64:
            run_name += f"_r{rank}"
        if cfg.method.get("mem_rank", 16) != 16:
            run_name += f"_m{cfg.method.get('mem_rank')}"
        if not cfg.method.get("route_replay", True):
            run_name += "_noroute"
        if not cfg.method.get("add_not_replace", True):
            run_name += "_replace"
    elif cfg.method.name == "spectral_memory":
        # SLR axes: split depth and spectral rank. Canonical is k=8 / rank=16 (no suffix);
        # the Gate-1 Pareto sweep varies rank ("_r<N>"), rank=0 is the no-replay control.
        split_k = cfg.method.get("split_layer_k", 8)
        rank_r = cfg.method.get("rank_r", 16)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if rank_r == 0:
            run_name += "_ctrl"
        elif rank_r != 16:
            run_name += f"_r{rank_r}"
    elif cfg.method.name == "aglr_replay":
        # AGLR-CL axes: split depth and salience-keep fraction. Canonical is k=8 /
        # keep=0.5 (no suffix); other keeps get "_keep<NN>".
        split_k = cfg.method.get("split_layer_k", 8)
        keep = cfg.method.get("attn_keep", 0.5)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if keep != 0.5:
            run_name += f"_keep{int(round(keep * 100))}"
    elif cfg.method.name == "coreset_memory":
        # Coreset axes: split depth and centroids-per-class. Canonical k=8 / m=8 (no suffix).
        split_k = cfg.method.get("split_layer_k", 8)
        mpc = cfg.method.get("centroids_per_class", 8)
        if split_k != 8:
            run_name += f"_k{split_k}"
        if mpc != 8:
            run_name += f"_m{mpc}"
    elif cfg.method.name == "proxy_latent_replay":
        # PLaR axes: split depth, proxy docs per task, confidence floor, pool. Canonical
        # k=8/d=5/tau=0/wildreceipt (no suffix).
        split_k = cfg.method.get("split_layer_k", 8)
        if split_k != 8:
            run_name += f"_k{split_k}"
        docs = cfg.method.get("docs_per_task", 5)
        tau = cfg.method.get("pseudo_conf_tau", 0.0)
        pool = cfg.method.get("proxy_dataset", "wildreceipt")
        if docs != 5:
            run_name += f"_d{docs}"
        if tau > 0:
            run_name += f"_tau{int(round(tau * 100))}"
        if cfg.method.get("soft_labels", False):
            run_name += "_soft"
        if cfg.method.get("task_masked_labels", False):
            run_name += "_tmask"
        if pool != "wildreceipt":
            run_name += f"_{pool}"
    elif cfg.method.name == "nullspace_analytic":
        # Axes: projection_mode, samples_per_class, soft_lambda, freeze_backbone.
        # Canonical: surgery/spc50/lambda0.01/no-freeze (no suffix).
        mode = cfg.method.get("projection_mode", "surgery")
        spc = cfg.method.get("samples_per_class", 50)
        lam = cfg.method.get("soft_lambda", 0.01)
        freeze = cfg.method.get("freeze_backbone_after_t0", False)
        if mode != "surgery":
            run_name += f"_{mode}"
        if spc != 50:
            run_name += f"_spc{spc}"
        if lam != 0.01:
            run_name += f"_lam{int(round(lam * 100))}"
        if freeze:
            run_name += "_freeze"
    elif target_component is not None:
        run_name += f"_{target_component}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.scenario.name, cfg.method.name, f"seed{cfg.seed}"],
        mode=cfg.wandb.get("mode", "online"),
    )

    # The encoder (per-backbone tokenization) is set as the process default before
    # the datasets are built so every loader tokenizes for the active backbone.
    scenario = get_scenario(
        cfg.scenario.name,
        encoder=build_encoder(cfg.model),
        **(cfg.scenario.get("kwargs") or {}),
    )
    log.info("Scenario %s: %d tasks", scenario.name, len(scenario.tasks))

    # Initial num_labels = first task's label set size
    n_init_labels = len(scenario.tasks[0].label_set)
    model_cls = MODEL_REGISTRY[cfg.model.get("family", "layoutlmv3")]
    model = model_cls(
        model_name=cfg.model.name,
        num_labels=n_init_labels,
    )
    # Pre-populate label maps for first task
    model.label_to_id = {label: i for i, label in enumerate(scenario.tasks[0].label_set)}
    model.id_to_label = {i: label for label, i in model.label_to_id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(
        "Model: %s. Total params: %d, Trainable: %d",
        cfg.model.name,
        model.total_param_count(),
        model.trainable_param_count(),
    )

    method_cls = METHOD_REGISTRY[cfg.method.name]
    method = method_cls(model, OmegaConf.to_container(cfg.method, resolve=True))

    # Opt-in mixed precision (bf16 on Ampere+, else fp16). Methods that wire the AMP
    # helpers (NaiveFineTune → naive/joint/doccl) honor this; others run full precision.
    method.amp_enabled = bool(cfg.training.get("amp", False) or cfg.training.get("fp16", False))
    if method.amp_enabled:
        log.info("Mixed precision enabled (cfg.training.amp/fp16).")

    # Activation checkpointing (after any PEFT wrapping) — fits small-VRAM GPUs.
    if cfg.training.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
        log.info("Gradient checkpointing enabled (lower memory, ~20-30%% slower).")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Seed the tracker with single-task baselines b_i so it can compute a real per-run
    # FWT (alongside AA/BWT/AF) once the zero-shot upper-triangular term is recorded
    # below. None if baselines aren't available yet → FWT stays NaN (honest).
    fwt_baselines = load_fwt_baselines(scenario, Path("results/table_single_task_baselines.csv"))
    if fwt_baselines is not None:
        log.info(
            "Loaded FWT baselines b_i for %s: %s",
            cfg.scenario.name,
            [round(b, 2) for b in fwt_baselines],
        )
    tracker = CLMetricsTracker(num_tasks=len(scenario.tasks), baseline_perf=fwt_baselines)
    eval_loaders_seen: dict[int, DataLoader] = {}
    out_dir = Path(cfg.output_dir) / run.name
    # Let methods that emit side artifacts (e.g. hrp's routing.json) know the run dir.
    method.out_dir = str(out_dir)

    # TensorBoard: live forgetting diagnostic alongside W&B. Writes per-run event
    # files under results/<run>/tb so `tensorboard --logdir results` aggregates all runs.
    tb = TBLogger(out_dir / "tb", enabled=bool(cfg.get("tensorboard", {}).get("enabled", True)))
    task_names = [t.task_name for t in scenario.tasks]
    last_step = len(scenario.tasks) - 1

    # Deep forgetting diagnostics (gradient/weight histograms + Fisher-weighted
    # displacement + per-layer CKA at task boundaries). Heavier than the always-on
    # retention scalars, so off by default — enable for a dedicated analysis run
    # via tensorboard.diagnostics=true. Attaching tb to the method turns on the
    # shared per-epoch weight-histogram hook (doccl.methods.base).
    tb_diag_on = bool(tb.enabled and cfg.get("tensorboard", {}).get("diagnostics", False))
    if tb_diag_on:
        method.tb_diag = tb
        log.info("TB deep diagnostics enabled (weight/grad histograms + Fisher/CKA).")

    # Special path for Joint: concatenate all train datasets and treat as 1 task
    if cfg.method.name == "joint":
        log.info("Joint training mode: concatenating all train datasets")
        # Expand classifier to cover ALL labels across all tasks
        all_labels: list[str] = []
        for t in scenario.tasks:
            for label in t.label_set:
                if label not in model.label_to_id and label not in all_labels:
                    all_labels.append(label)
        if all_labels:
            model.expand_classifier(all_labels)
            model = model.to(device)

        # Use the full-label joint pool when the scenario provides one (CIL scenarios mask
        # out-of-session entities per task, so concatenating train_datasets would feed the
        # same document with conflicting labels and collapse training). Falls back to the
        # per-task datasets for scenarios whose tasks are disjoint documents (e.g. DIL).
        joint_sources = scenario.joint_train_datasets or scenario.train_datasets
        joint_train = ConcatDataset(joint_sources)
        joint_loader = DataLoader(
            joint_train,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        # Early-stopping val signal for Joint: the pooled eval set across ALL tasks
        # (Joint trains on all data, so its convergence is measured over all tasks).
        # Without this, train_task got val_loader=None and trained the full
        # method.epochs budget — 100 forced epochs over the huge pooled dataset, the
        # single biggest time sink in the grid. With it, Joint early-stops on
        # plateaued pooled val-F1 exactly like every other method.
        joint_val = ConcatDataset(list(scenario.eval_datasets))
        joint_val_loader = DataLoader(
            joint_val,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        # Train once on joint data
        synthetic_task = scenario.tasks[0]
        t0 = time.perf_counter()
        method.train_task(synthetic_task, joint_loader, val_loader=joint_val_loader)
        joint_time = time.perf_counter() - t0

        # Evaluate on each task's eval set
        for tid, eval_ds in enumerate(scenario.eval_datasets):
            eval_loaders_seen[tid] = DataLoader(
                eval_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
        results = method.evaluate(eval_loaders_seen)
        # Fill last row of matrix (since Joint = oracle = T-1 step)
        for tid in range(len(scenario.tasks)):
            tracker.matrix[len(scenario.tasks) - 1, tid] = results[tid].f1

        wandb.log({f"final/eval/task_{tid}/f1": r.f1 for tid, r in results.items()})
        wandb.log({"final/AA": tracker.average_accuracy()})
        log.info("Joint final AA: %.2f", tracker.average_accuracy())

        tb.log_scalars(
            {
                "final/AA": tracker.average_accuracy(),
                **{f"final/eval/task_{tid}/f1": r.f1 for tid, r in results.items()},
            },
            step=last_step,
        )
        tb.log_forgetting_matrix(tracker.matrix, step=last_step, task_names=task_names)
        tb.close()

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "matrix.npy", tracker.matrix)
        save_run_metrics(out_dir, cfg, tracker, method, [joint_time])

        wandb.finish()
        return

    # Standard CL loop
    task_times: list[float] = []
    for task_idx, task in enumerate(scenario.tasks):
        log.info("=== Task %d/%d: %s ===", task_idx + 1, len(scenario.tasks), task.task_name)

        # Expand classifier for new labels in this task
        new_labels = [label for label in task.label_set if label not in model.label_to_id]
        if new_labels:
            log.info("Expanding classifier with %d new labels", len(new_labels))
            model.expand_classifier(new_labels)
            model = model.to(device)

        train_loader = DataLoader(
            scenario.train_datasets[task_idx],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            scenario.eval_datasets[task_idx],
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        # Zero-shot (forward-transfer) term: evaluate the model on THIS task BEFORE
        # training it — i.e. R[task_idx-1, task_idx], the upper-triangular entry FWT
        # needs. The classifier head was just expanded (new rows ~ N(0, 0.02)), so this
        # is a genuine zero-shot reading carried over from the previous task. We do this
        # before before_task/train_task so no current-task gradient has touched the model.
        if task_idx > 0:
            zs = method.evaluate({task_idx: eval_loader})
            tracker.matrix[task_idx - 1, task_idx] = zs[task_idx].f1
            log.info("Zero-shot on task %d (for FWT): F1=%.2f", task_idx, zs[task_idx].f1)

        eval_loaders_seen[task_idx] = eval_loader

        # Deep diagnostic (opt-in): before training task t, snapshot θ^{t-1} and the
        # OLD task's Fisher so the post-task Fisher-weighted displacement localises
        # where the previous task's knowledge moves. Skipped on the first task (no
        # "previous" task yet) and when diagnostics are off (keeps the grid fast).
        diag_params_before = None
        diag_fisher_old = None
        if tb_diag_on and task_idx > 0:
            from doccl.eval.fisher import empirical_fisher_diagonal, snapshot_params

            prev_loader = eval_loaders_seen[task_idx - 1]
            diag_params_before = snapshot_params(model)
            diag_fisher_old = empirical_fisher_diagonal(
                model, prev_loader, n_samples=200, device=device
            )

        # Lifecycle
        method.before_task(task, train_loader)
        t0 = time.perf_counter()
        # Pass the current task's eval loader as the validation signal for val-F1
        # early stopping (train until current-task F1 plateaus, then restore best).
        train_metrics = method.train_task(task, train_loader, val_loader=eval_loader)
        task_times.append(time.perf_counter() - t0)
        method.after_task(task, train_loader)

        if tb_diag_on and diag_params_before is not None:
            try:
                log_boundary_diagnostics(
                    tb,
                    model,
                    diag_fisher_old,
                    diag_params_before,
                    f"{task_idx-1}_to_{task_idx}",
                    task_idx,
                )
            except Exception as e:  # never fail a run over a diagnostic
                log.warning("boundary diagnostics skipped: %s", e)

        # Evaluate on all seen tasks
        results = method.evaluate(eval_loaders_seen)
        tracker.update(task_idx, {tid: {"f1": r.f1} for tid, r in results.items()})

        wandb.log(
            {
                "task_idx": task_idx,
                "train/loss": train_metrics.loss,
                **{f"eval/task_{tid}/f1": r.f1 for tid, r in results.items()},
            }
        )
        # TensorBoard: the live forgetting diagnostic — retention curves + the matrix
        # heatmap re-rendered each task so it fills in as forgetting accrues.
        tb.log_scalars(
            {
                "train/loss": train_metrics.loss,
                **{f"eval/task_{tid}/f1": r.f1 for tid, r in results.items()},
            },
            step=task_idx,
        )
        tb.log_retention(tracker.matrix, task_idx)
        tb.log_forgetting_matrix(tracker.matrix, step=task_idx, task_names=task_names)
        log.info(
            "After task %d: %s",
            task_idx,
            {tid: f"F1={r.f1:.2f}" for tid, r in results.items()},
        )

    summary = tracker.summary()
    log.info("Final: AA=%.2f BWT=%.2f AF=%.2f", summary["AA"], summary["BWT"], summary["AF"])
    wandb.log({"final/" + k: v for k, v in summary.items()})
    tb.log_scalars({f"final/{k}": v for k, v in summary.items()}, step=last_step)
    tb.log_scalars(
        {f"forgetting/task_{i}": v for i, v in tracker.per_task_forgetting().items()},
        step=last_step,
    )
    tb.close()

    # Save tracker matrix + structured metrics for offline ingestion
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "matrix.npy", tracker.matrix)
    log.info("Saved matrix to %s", out_dir)
    save_run_metrics(out_dir, cfg, tracker, method, task_times)

    # Per-class F1 on the final model — DIL-degeneracy evidence (review M5).
    # Standard-forward methods only (prompt/LoRA methods have a custom forward).
    # er_cflat uses ER's standard model forward (no PEFT/prompts) → eligible.
    # cl_lora is PEFT-wrapped (custom forward) → excluded, like o_lora.
    _std_forward = {
        "naive",
        "joint",
        "marginal_kl",
        "logit_adjust",
        "ewc",
        "lwf",
        "er",
        "der_pp",
        "er_cflat",
        "doccl",
        "lca",
        "magmax",
        "sd_lora",
        "is3",
        "cpfd",
        "hgt",
        "cuber",
        "lexslot",
        "lexslot_fm",
        "lexmem",
        "lexmem_v2",
        "lexmem_ctrl",
        "lexmem_v3",
        "lexmem_v3b",
        "lexmem_v5",
        "latent_replay",
        "spectral_memory",
        "aglr_replay",
        "coreset_memory",
        "proxy_latent_replay",
        "colar",
        "colaslot",
        "colaslot_r",
        "colaslot_ra",
        "colaslot_rf",
        "colaslot_ro",
        "colar_bal",
        "colar_cb",
        "colar_wsvd",
        # colar_knn: per-class F1 here reflects the PARAMETRIC head only (save_per_class_f1
        # runs its own model forward, bypassing method.evaluate's kNN blend) — known gap.
        "colar_knn",
        "colar_meta",
        "larm",
        "nullspace_analytic",
        "fisher_mask",
    }  # noqa: N806
    if cfg.method.name in _std_forward:
        try:
            save_per_class_f1(out_dir, model, eval_loaders_seen, device)
        except Exception as e:  # never fail a run over a diagnostic
            log.warning("per-class F1 skipped: %s", e)

    wandb.finish()


if __name__ == "__main__":
    main()
