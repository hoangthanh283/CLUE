"""CL scenario builders.

Three scenarios for AAAI submission:
    - CIL: class-incremental within a single dataset (CORD primarily, FUNSD warm-up)
    - DIL: domain-incremental across datasets with unified schema
    - Mixed: interleaving class-IL and domain shifts

Plus utility scenarios:
    - SINGLE_*: single-task baseline (sanity check)
    - PILOT: naive sequential FUNSD → CORD → SROIE for pilot study

TIL is deferred to NeurIPS extension.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from doccl.data.cord import CORDDataset
from doccl.data.dil_remapping import DIL_LabelRemapper, DIL_UNIFIED_LABELS
from doccl.data.funsd import FUNSDDataset
from doccl.data.sroie import SROIEDataset
from doccl.types import ScenarioType, TaskInfo


@dataclass
class CLScenario:
    """A continual learning scenario = sequence of (TaskInfo, train_ds, eval_ds)."""

    name: str
    scenario_type: ScenarioType
    tasks: list[TaskInfo]
    train_datasets: list[Dataset]
    eval_datasets: list[Dataset]


# ─── CIL scenarios ─────────────────────────────────────────────────────────────


def build_cil_funsd() -> CLScenario:
    """FUNSD class-incremental warm-up: 4 entity types → 2 sessions × 2 entities.

    Session 0: HEADER, QUESTION
    Session 1: ANSWER (OTHER stays as 'O')
    """
    splits = [
        ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"],
        ["B-ANSWER", "I-ANSWER"],
    ]
    train_dss = [FUNSDDataset(split="train", label_filter=lbls) for lbls in splits]
    eval_dss = [FUNSDDataset(split="test", label_filter=lbls) for lbls in splits]

    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"funsd_cil_t{i}",
            label_set=splits[i],
            is_first=(i == 0),
            is_last=(i == len(splits) - 1),
        )
        for i in range(len(splits))
    ]
    return CLScenario("cil_funsd", ScenarioType.CIL, tasks, train_dss, eval_dss)


def build_cil_cord(num_sessions: int = 5) -> CLScenario:
    """CORD class-incremental: 30 fine classes → num_sessions × (30/num_sessions) classes.

    Default: 5 sessions × 6 classes.

    Partitioning strategy: respect super-class structure where possible —
    Session 0: menu.* core (6 classes)
    Session 1: menu.sub.* + menu.vatyn (6 classes)
    Session 2: sub_total.* (6 classes)
    Session 3: total.* (6 classes)
    Session 4: void_menu.* + sub.* (6 classes)
    """
    if num_sessions != 5:
        raise NotImplementedError(
            "Only 5-session CIL-CORD supported in v1. Custom splits via direct API."
        )

    cls_per_session = [
        ["menu.cnt", "menu.discountprice", "menu.itemsubtotal", "menu.nm", "menu.num", "menu.price"],
        ["menu.unitprice", "menu.vatyn", "menu.sub.cnt", "menu.sub.nm", "menu.sub.price", "menu.sub.unitprice"],
        ["sub_total.discount_price", "sub_total.etc", "sub_total.othersvc_price",
         "sub_total.service_price", "sub_total.subtotal_price", "sub_total.tax_price"],
        ["total.cashprice", "total.changeprice", "total.creditcardprice",
         "total.emoneyprice", "total.menuqty_cnt", "total.menutype_cnt"],
        ["total.total_etc", "total.total_price", "void_menu.nm", "void_menu.price", "sub.nm", "sub.cnt"],
    ]

    bio_splits = [[f"B-{c}" for c in s] + [f"I-{c}" for c in s] for s in cls_per_session]

    train_dss = [
        CORDDataset(split="train", granularity="fine", label_filter=lbls)
        for lbls in bio_splits
    ]
    eval_dss = [
        CORDDataset(split="test", granularity="fine", label_filter=lbls)
        for lbls in bio_splits
    ]

    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"cord_cil_t{i}",
            label_set=bio_splits[i],
            is_first=(i == 0),
            is_last=(i == len(bio_splits) - 1),
            metadata={"super_class": ["menu", "menu", "sub_total", "total", "mixed"][i]},
        )
        for i in range(len(bio_splits))
    ]
    return CLScenario("cil_cord", ScenarioType.CIL, tasks, train_dss, eval_dss)


# ─── DIL scenario ──────────────────────────────────────────────────────────────


def build_dil() -> CLScenario:
    """Domain-incremental: FUNSD → SROIE → CORD-superclass with unified schema.

    Unified label space (4 classes + O = 9 BIO tags):
        HEADER  — section headers
        KEY     — field labels (questions in forms, fixed labels in receipts)
        VALUE   — field values (answers, prices, dates, addresses)
        OTHER   — auxiliary entities (CORD void_menu, sub)

    See docs/dil_schema_mapping.md for full mapping rationale.
    """
    # Build underlying datasets
    funsd_train = FUNSDDataset("train")
    funsd_test = FUNSDDataset("test")
    sroie_train = SROIEDataset("train")
    sroie_test = SROIEDataset("test")
    cord_super_train = CORDDataset("train", granularity="super")
    cord_super_test = CORDDataset("test", granularity="super")

    # Wrap each with DIL_LabelRemapper
    train_dss: list[Dataset] = [
        DIL_LabelRemapper(funsd_train, "funsd", funsd_train.id_to_label),
        DIL_LabelRemapper(sroie_train, "sroie", sroie_train.id_to_label),
        DIL_LabelRemapper(cord_super_train, "cord", cord_super_train.id_to_label),
    ]
    eval_dss: list[Dataset] = [
        DIL_LabelRemapper(funsd_test, "funsd", funsd_test.id_to_label),
        DIL_LabelRemapper(sroie_test, "sroie", sroie_test.id_to_label),
        DIL_LabelRemapper(cord_super_test, "cord", cord_super_test.id_to_label),
    ]

    tasks = [
        TaskInfo(
            task_id=0,
            task_name="dil_funsd",
            label_set=DIL_UNIFIED_LABELS,
            is_first=True,
            is_last=False,
            metadata={"native_dataset": "funsd"},
        ),
        TaskInfo(
            task_id=1,
            task_name="dil_sroie",
            label_set=DIL_UNIFIED_LABELS,
            is_first=False,
            is_last=False,
            metadata={"native_dataset": "sroie"},
        ),
        TaskInfo(
            task_id=2,
            task_name="dil_cord",
            label_set=DIL_UNIFIED_LABELS,
            is_first=False,
            is_last=True,
            metadata={"native_dataset": "cord"},
        ),
    ]
    return CLScenario("dil", ScenarioType.DIL, tasks, train_dss, eval_dss)


# ─── Mixed scenario ────────────────────────────────────────────────────────────


def build_mixed() -> CLScenario:
    """6-session mixed scenario interleaving class-IL within and domain shifts.

    Sessions:
        0: FUNSD HEADER+QUESTION       (class-IL within forms)
        1: FUNSD ANSWER                 (class-IL within forms)
        2: SROIE all 4 fields           (domain shift: forms → receipts)
        3: CORD super menu+sub_total    (class-IL within receipts, super-class space)
        4: CORD super total+void+sub    (class-IL within receipts)
        5: FUNSD revisit (full)         (domain return — tests cross-session retention)

    Uses native label spaces per session (no remapping). The classifier expands
    monotonically as new labels arrive, so all label spaces accumulate in the
    output head.
    """
    sessions = []

    # Session 0-1: FUNSD class-IL
    funsd_s0_labels = ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"]
    funsd_s1_labels = ["B-ANSWER", "I-ANSWER"]
    sessions.append({
        "name": "mixed_funsd_s0",
        "labels": funsd_s0_labels,
        "train": FUNSDDataset("train", label_filter=funsd_s0_labels),
        "test": FUNSDDataset("test", label_filter=funsd_s0_labels),
        "metadata": {"phase": "funsd-CIL-1of2"},
    })
    sessions.append({
        "name": "mixed_funsd_s1",
        "labels": funsd_s1_labels,
        "train": FUNSDDataset("train", label_filter=funsd_s1_labels),
        "test": FUNSDDataset("test", label_filter=funsd_s1_labels),
        "metadata": {"phase": "funsd-CIL-2of2"},
    })

    # Session 2: SROIE (domain shift — forms to receipts)
    sroie_labels = SROIEDataset.LABEL_NAMES[1:]  # exclude 'O'
    sessions.append({
        "name": "mixed_sroie",
        "labels": sroie_labels,
        "train": SROIEDataset("train"),
        "test": SROIEDataset("test"),
        "metadata": {"phase": "domain-shift-1"},
    })

    # Session 3-4: CORD class-IL on super classes
    cord_s0_super = ["B-menu", "I-menu", "B-sub_total", "I-sub_total"]
    cord_s1_super = ["B-total", "I-total", "B-void_menu", "I-void_menu", "B-sub", "I-sub"]
    sessions.append({
        "name": "mixed_cord_s0",
        "labels": cord_s0_super,
        "train": CORDDataset("train", "super", label_filter=cord_s0_super),
        "test": CORDDataset("test", "super", label_filter=cord_s0_super),
        "metadata": {"phase": "cord-CIL-1of2"},
    })
    sessions.append({
        "name": "mixed_cord_s1",
        "labels": cord_s1_super,
        "train": CORDDataset("train", "super", label_filter=cord_s1_super),
        "test": CORDDataset("test", "super", label_filter=cord_s1_super),
        "metadata": {"phase": "cord-CIL-2of2"},
    })

    # Session 5: FUNSD revisit (tests retention against domain interference)
    funsd_all = FUNSDDataset.LABEL_NAMES[1:]  # already added in S0/S1, won't re-expand
    sessions.append({
        "name": "mixed_funsd_revisit",
        "labels": funsd_all,
        "train": FUNSDDataset("train"),
        "test": FUNSDDataset("test"),
        "metadata": {"phase": "domain-return"},
    })

    train_dss = [s["train"] for s in sessions]
    eval_dss = [s["test"] for s in sessions]
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=s["name"],
            label_set=s["labels"],
            is_first=(i == 0),
            is_last=(i == len(sessions) - 1),
            metadata=s["metadata"],
        )
        for i, s in enumerate(sessions)
    ]
    return CLScenario("mixed", ScenarioType.MIXED, tasks, train_dss, eval_dss)


# ─── Utility scenarios ─────────────────────────────────────────────────────────


def build_single(dataset_name: str) -> CLScenario:
    """Single-task baseline (no CL, sanity check)."""
    if dataset_name == "funsd":
        train, test = FUNSDDataset("train"), FUNSDDataset("test")
        labels = FUNSDDataset.LABEL_NAMES
    elif dataset_name == "cord":
        train, test = CORDDataset("train", "fine"), CORDDataset("test", "fine")
        labels = CORDDataset.LABEL_NAMES_FINE
    elif dataset_name == "sroie":
        train, test = SROIEDataset("train"), SROIEDataset("test")
        labels = SROIEDataset.LABEL_NAMES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    task = TaskInfo(
        task_id=0,
        task_name=f"{dataset_name}_single",
        label_set=labels,
        is_first=True,
        is_last=True,
    )
    return CLScenario(
        f"single_{dataset_name}", ScenarioType.SINGLE, [task], [train], [test]
    )


def build_pilot(order: list[int] | None = None) -> CLScenario:
    """Pilot study scenario: naive sequential FUNSD → CORD → SROIE.

    Used to characterize per-component forgetting in pilot study (Phase 2).
    Each task uses the FULL label set of its respective dataset.

    Args:
        order: optional permutation of [0,1,2] over (FUNSD, CORD, SROIE). Defaults
            to [0,1,2]. Pass e.g. [2,1,0] for the alternate task order used in the
            §6.1.3 stability check (does the dominant-component finding survive a
            different ordering?).
    """
    order = order or [0, 1, 2]
    train_all = [FUNSDDataset("train"), CORDDataset("train", "fine"), SROIEDataset("train")]
    eval_all = [FUNSDDataset("test"), CORDDataset("test", "fine"), SROIEDataset("test")]
    label_all = [
        FUNSDDataset.LABEL_NAMES,
        CORDDataset.LABEL_NAMES_FINE,
        SROIEDataset.LABEL_NAMES,
    ]
    names_all = ["funsd", "cord", "sroie"]

    train_dss = [train_all[i] for i in order]
    eval_dss = [eval_all[i] for i in order]
    label_sets = [label_all[i] for i in order]
    names = [names_all[i] for i in order]

    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"pilot_{names[i]}",
            label_set=label_sets[i],
            is_first=(i == 0),
            is_last=(i == len(order) - 1),
        )
        for i in range(len(order))
    ]
    return CLScenario("pilot", ScenarioType.PILOT, tasks, train_dss, eval_dss)


# ─── Registry ──────────────────────────────────────────────────────────────────

SCENARIO_REGISTRY = {
    "single_funsd": lambda: build_single("funsd"),
    "single_cord": lambda: build_single("cord"),
    "single_sroie": lambda: build_single("sroie"),
    "cil_funsd": build_cil_funsd,
    "cil_cord": build_cil_cord,
    "dil": build_dil,
    "mixed": build_mixed,
    "pilot": build_pilot,
}


def get_scenario(name: str, **kwargs) -> CLScenario:
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario {name}. Available: {list(SCENARIO_REGISTRY)}")
    builder = SCENARIO_REGISTRY[name]
    return builder(**kwargs) if kwargs else builder()
