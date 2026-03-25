"""
Typed configuration dataclasses for cl4ie.

Replaces nested Dict[str, Any] config with typed dataclasses for the three
most-accessed config sections: training, model, and cl_strategy.

Usage:
    from src.utils import load_config
    from src.config import ExperimentConfig

    cfg = ExperimentConfig.from_dict(load_config("configs/layoutlmv3_class_il.yaml"))
    print(cfg.training.learning_rate, cfg.cl_strategy.name, cfg.model.model_type)
"""

import copy
import dataclasses
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelInnerConfig:
    """Maps to model.config.* in YAML."""
    num_labels: int
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1

    def __post_init__(self):
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInnerConfig":
        return cls(
            num_labels=int(d["num_labels"]),
            hidden_dropout_prob=float(d.get("hidden_dropout_prob", 0.1)),
            attention_probs_dropout_prob=float(d.get("attention_probs_dropout_prob", 0.1)),
            classifier_dropout=float(d.get("classifier_dropout", 0.1)),
        )


@dataclasses.dataclass
class ModelConfig:
    """Maps to model.* in YAML."""
    name: str
    model_type: str
    pretrained_model_name: str
    config: ModelInnerConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=d["name"],
            model_type=d["model_type"],
            pretrained_model_name=d["pretrained_model_name"],
            config=ModelInnerConfig.from_dict(d.get("config", {})),
        )


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """Maps to training.* in YAML."""
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw"
    scheduler: str = "linear"
    # AGENT FIX: Changed default from 4 to 0.
    # Root cause: on this 7.7 GB RAM system, all 5 task datasets are pre-loaded into
    # Python heap (~4 GB). When PyTorch's DataLoader forks num_workers=4 child
    # processes, each child inherits the full heap via copy-on-write, multiplying
    # RAM consumption by ~5 and triggering the OOM killer (confirmed in journalctl).
    # Setting num_workers=0 eliminates the fork-based RAM multiplication; data
    # loading is handled on the main thread (slightly slower but OOM-safe).
    num_workers: int = 0
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 10
    log_steps: int = 100
    metric_for_best_model: str = "f1"
    save_best_model: bool = True
    load_best_model_at_end: bool = True
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    eval_steps: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            batch_size=int(d["batch_size"]),
            learning_rate=float(d["learning_rate"]),
            num_epochs=int(d["num_epochs"]),
            weight_decay=float(d.get("weight_decay", 0.01)),
            gradient_accumulation_steps=int(d.get("gradient_accumulation_steps", 1)),
            optimizer=str(d.get("optimizer", "adamw")),
            scheduler=str(d.get("scheduler", "linear")),
            num_workers=int(d.get("num_workers", 0)),
            warmup_ratio=float(d.get("warmup_ratio", 0.1)),
            early_stopping_patience=int(d.get("early_stopping_patience", 10)),
            log_steps=int(d.get("log_steps", 100)),
            metric_for_best_model=str(d.get("metric_for_best_model", "f1")),
            save_best_model=bool(d.get("save_best_model", True)),
            load_best_model_at_end=bool(d.get("load_best_model_at_end", True)),
            persistent_workers=bool(d["persistent_workers"]) if d.get("persistent_workers") is not None else None,
            prefetch_factor=int(d["prefetch_factor"]) if d.get("prefetch_factor") is not None else None,
            eval_steps=int(d["eval_steps"]) if d.get("eval_steps") is not None else None,
        )


# ---------------------------------------------------------------------------
# CL strategy configs
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StrategyConfig:
    """Base strategy config. Maps to cl_strategy.* in YAML."""
    name: str = "sequential"

    @classmethod
    def from_dict(cls, d: Dict[str, Any], unified_label_space: bool = False) -> "StrategyConfig":
        """Dispatch factory: reads 'name' and returns the correct subclass."""
        name = (d.get("name") or "sequential").lower()
        target_cls = _STRATEGY_CONFIG_MAP.get(name, StrategyConfig)
        # Delegate to subclass from_dict so type-casting (float(), int()) is applied.
        if target_cls is not StrategyConfig and target_cls is not cls:
            return target_cls.from_dict(d, unified_label_space=unified_label_space)
        known = {f.name for f in dataclasses.fields(target_cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if target_cls is LwFConfig:
            kwargs["unified_label_space"] = unified_label_space
        return target_cls(**kwargs)


@dataclasses.dataclass
class ERConfig(StrategyConfig):
    """Experience Replay config."""
    name: str = "er"
    memory_size: int = 2000
    replay_batch_size: int = 32
    replay_weight: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **_) -> "ERConfig":
        return cls(
            name=d.get("name", "er"),
            memory_size=int(d.get("memory_size", 2000)),
            replay_batch_size=int(d.get("replay_batch_size", 32)),
            replay_weight=float(d.get("replay_weight", 1.0)),
        )


@dataclasses.dataclass
class EWCConfig(StrategyConfig):
    """Elastic Weight Consolidation config."""
    name: str = "ewc"
    ewc_lambda: float = 0.4
    fisher_cache_dir: str = "ewc_cache"
    n_fisher_samples: Optional[int] = None
    ewc_chunk_size: int = 1_000_000
    store_fishers_on_cpu: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **_) -> "EWCConfig":
        return cls(
            name=d.get("name", "ewc"),
            ewc_lambda=float(d.get("ewc_lambda", 0.4)),
            fisher_cache_dir=str(d.get("fisher_cache_dir", "ewc_cache")),
            n_fisher_samples=int(d["n_fisher_samples"]) if d.get("n_fisher_samples") is not None else None,
            ewc_chunk_size=int(d.get("ewc_chunk_size", 1_000_000)),
            store_fishers_on_cpu=bool(d.get("store_fishers_on_cpu", True)),
        )


@dataclasses.dataclass
class GEMConfig(StrategyConfig):
    """Gradient Episodic Memory config."""
    name: str = "gem"
    memory_size: int = 500
    samples_per_task: int = 5
    max_tasks: int = 10
    qp_tolerance: float = 1e-3
    qp_regularization: float = 1e-6
    margin: float = 0.5
    clear_cache_every: int = 5

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **_) -> "GEMConfig":
        return cls(
            name=d.get("name", "gem"),
            memory_size=int(d.get("memory_size", 500)),
            samples_per_task=int(d.get("samples_per_task", 5)),
            max_tasks=int(d.get("max_tasks", 10)),
            qp_tolerance=float(d.get("qp_tolerance", 1e-3)),
            qp_regularization=float(d.get("qp_regularization", 1e-6)),
            margin=float(d.get("margin", 0.5)),
            clear_cache_every=int(d.get("clear_cache_every", 5)),
        )


@dataclasses.dataclass
class AGEMConfig(StrategyConfig):
    """Averaged Gradient Episodic Memory config."""
    name: str = "agem"
    memory_size: int = 1000
    replay_batch_size: int = 4
    constraint_threshold: float = -1e-6
    clear_cache_every: int = 5
    use_balanced_sampling: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **_) -> "AGEMConfig":
        return cls(
            name=d.get("name", "agem"),
            memory_size=int(d.get("memory_size", 1000)),
            replay_batch_size=int(d.get("replay_batch_size", 4)),
            constraint_threshold=float(d.get("constraint_threshold", -1e-6)),
            clear_cache_every=int(d.get("clear_cache_every", 5)),
            use_balanced_sampling=bool(d.get("use_balanced_sampling", False)),
        )


@dataclasses.dataclass
class LwFConfig(StrategyConfig):
    """Learning without Forgetting config."""
    name: str = "lwf"
    lwf_alpha: float = 0.5
    lwf_temperature: float = 2.0
    unified_label_space: bool = False

    def __post_init__(self):
        if not self.unified_label_space:
            raise ValueError(
                "LwF requires unified label space! "
                "Set 'label_space.unified: true' in config. "
                "This ensures teacher and student have same output dimensions."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], unified_label_space: bool = False) -> "LwFConfig":
        return cls(
            name=d.get("name", "lwf"),
            lwf_alpha=float(d.get("lwf_alpha", 0.5)),
            lwf_temperature=float(d.get("lwf_temperature", 2.0)),
            unified_label_space=unified_label_space,
        )


_STRATEGY_CONFIG_MAP: Dict[str, type] = {
    "sequential": StrategyConfig,
    "none": StrategyConfig,
    "joint": StrategyConfig,
    "er": ERConfig,
    "experience_replay": ERConfig,
    "ewc": EWCConfig,
    "gem": GEMConfig,
    "agem": AGEMConfig,
    "a-gem": AGEMConfig,
    "lwf": LwFConfig,
}


# ---------------------------------------------------------------------------
# Root experiment config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ExperimentConfig:
    """Root experiment config. Typed fields for model, training, cl_strategy.
    dataset, tasks, data_processing, wandb etc. remain as raw dicts (Phase 1 scope).
    """
    experiment_name: str
    output_dir: str
    cl_setting: str
    model: ModelConfig
    training: TrainingConfig
    cl_strategy: StrategyConfig
    label_space: Dict[str, Any] = dataclasses.field(default_factory=dict)
    dataset: Dict[str, Any] = dataclasses.field(default_factory=dict)
    data_processing: Dict[str, Any] = dataclasses.field(default_factory=dict)
    tasks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    evaluation: Dict[str, Any] = dataclasses.field(default_factory=dict)
    output: Dict[str, Any] = dataclasses.field(default_factory=dict)
    wandb: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Build ExperimentConfig from a raw YAML-loaded dict."""
        unified = bool(d.get("label_space", {}).get("unified", False))
        strategy_dict = copy.deepcopy(d.get("cl_strategy", {}))
        cl_strategy = StrategyConfig.from_dict(strategy_dict, unified_label_space=unified)

        return cls(
            experiment_name=d["experiment_name"],
            output_dir=str(d.get("output_dir", "")),
            cl_setting=str(d.get("cl_setting", "task_il")).lower(),
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig.from_dict(d["training"]),
            cl_strategy=cl_strategy,
            label_space=copy.deepcopy(d.get("label_space", {})),
            dataset=copy.deepcopy(d.get("dataset", {})),
            data_processing=copy.deepcopy(d.get("data_processing", {})),
            tasks=copy.deepcopy(d.get("tasks", [])),
            evaluation=copy.deepcopy(d.get("evaluation", {})),
            output=copy.deepcopy(d.get("output", {})),
            wandb=copy.deepcopy(d.get("wandb", {})),
        )

    def with_output_dir(self, output_dir: str) -> "ExperimentConfig":
        """Return a new ExperimentConfig with output_dir set."""
        return dataclasses.replace(self, output_dir=output_dir)

    def replace_model_num_labels(self, num_labels: int) -> "ExperimentConfig":
        """Return a new ExperimentConfig with model.config.num_labels replaced."""
        new_inner = dataclasses.replace(self.model.config, num_labels=num_labels)
        new_model = dataclasses.replace(self.model, config=new_inner)
        return dataclasses.replace(self, model=new_model)

    def with_task_overrides(self, overrides: Dict[str, Any]) -> "ExperimentConfig":
        """Return a new ExperimentConfig with per-task overrides applied.

        Handles typed fields (training, model) by merging override dicts with
        existing dataclass values. Raw dict fields (dataset, label_space, etc.)
        are shallow-merged.
        """
        if not overrides:
            return self

        updates: Dict[str, Any] = {}

        # Merge training overrides if present
        if "training" in overrides:
            merged_training = {**dataclasses.asdict(self.training), **overrides["training"]}
            updates["training"] = TrainingConfig.from_dict(merged_training)

        # Merge model overrides if present (rare but supported)
        if "model" in overrides:
            merged_model = _deep_merge_dicts(dataclasses.asdict(self.model), overrides["model"])
            updates["model"] = ModelConfig.from_dict(merged_model)

        # Raw dict fields: shallow merge
        for key in ("dataset", "label_space", "data_processing", "evaluation", "output", "wandb"):
            if key in overrides:
                current = getattr(self, key)
                if isinstance(overrides[key], dict) and isinstance(current, dict):
                    updates[key] = {**current, **overrides[key]}
                else:
                    updates[key] = overrides[key]

        # Scalar fields (e.g., cl_setting, experiment_name)
        for key in ("experiment_name", "output_dir", "cl_setting"):
            if key in overrides:
                updates[key] = overrides[key]

        return dataclasses.replace(self, **updates)


def _deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base dict (used for model config overrides)."""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
