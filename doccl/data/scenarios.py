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

from doccl.data.cil_remapping import CIL_LabelRemapper
from doccl.data.cord import CORDDataset
from doccl.data.encoders import KIEEncoder, set_default_encoder
from doccl.data.dil_remapping import DIL_LabelRemapper, DIL_UNIFIED_LABELS
from doccl.data.funsd import FUNSDDataset
from doccl.data.receipt_remapping import RECEIPT_UNIFIED_LABELS, Receipt_LabelRemapper
from doccl.data.sroie import SROIEDataset
from doccl.data.wildreceipt import WildReceiptDataset
from doccl.data.xfund import XFUNDDataset
from doccl.types import ScenarioType, TaskInfo

# Default language order for the cross-lingual DIL scenario: 4 Latin-script + Chinese,
# so the sequence stresses both vocabulary drift and a script change (de→es→fr→it→zh).
XLINGUAL_DEFAULT_LANGS = ["de", "es", "fr", "it", "zh"]


def _cil_head_snapshots(
    session_label_sets: list[list[str]],
) -> tuple[list[dict[str, int]], list[list[str]]]:
    """Compute the cumulative model-head label map seen at each CIL session.

    Returns ``(snapshots, label_sets_with_O)`` where:
      - ``snapshots[i]`` is the head ``label -> index`` map in effect when session ``i``
        is trained: ``O`` at 0 followed by every label introduced in sessions ``0..i``
        (deduplicated, append order). This matches the head grown by
        ``expand_classifier`` once ``O`` is in the task-0 label set.
      - ``label_sets_with_O[i]`` is session ``i``'s own label set prefixed with ``O``
        (what ``TaskInfo.label_set`` should carry so the head is sized correctly).
    """
    cumulative: dict[str, int] = {"O": 0}
    snapshots: list[dict[str, int]] = []
    label_sets_with_O: list[list[str]] = []
    for labels in session_label_sets:
        for label in labels:
            if label not in cumulative:
                cumulative[label] = len(cumulative)
        snapshots.append(dict(cumulative))  # snapshot AFTER adding this session's labels
        label_sets_with_O.append(["O"] + list(labels))
    return snapshots, label_sets_with_O


@dataclass
class CLScenario:
    """A continual learning scenario = sequence of (TaskInfo, train_ds, eval_ds)."""

    name: str
    scenario_type: ScenarioType
    tasks: list[TaskInfo]
    train_datasets: list[Dataset]
    eval_datasets: list[Dataset]
    # Full-label training pool for the Joint upper bound. The per-task ``train_datasets``
    # above are CIL-masked (out-of-session entities relabelled to O), so concatenating them
    # feeds the same document multiple times with CONFLICTING labels — which collapses Joint
    # training. ``joint_train_datasets`` instead holds each underlying document ONCE with ALL
    # its labels, in the same head-index space as ``tasks``' cumulative label set. When None,
    # the Joint path falls back to ``train_datasets`` (correct for scenarios whose tasks are
    # disjoint documents, e.g. DIL).
    joint_train_datasets: list[Dataset] | None = None




def build_cil_funsd() -> CLScenario:
    """FUNSD class-incremental warm-up: 4 entity types → 2 sessions × 2 entities.

    Session 0: HEADER, QUESTION
    Session 1: ANSWER (OTHER stays as 'O')
    """
    splits = [
        ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"],
        ["B-ANSWER", "I-ANSWER"],
    ]
    snapshots, label_sets = _cil_head_snapshots(splits)

    def _wrap(split: str) -> list[Dataset]:
        wrapped = []
        for i, lbls in enumerate(splits):
            base = FUNSDDataset(split=split, label_filter=lbls)
            wrapped.append(CIL_LabelRemapper(base, base.id_to_label, snapshots[i]))
        return wrapped

    train_dss = _wrap("train")
    eval_dss = _wrap("test")

    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"funsd_cil_t{i}",
            label_set=label_sets[i],
            is_first=(i == 0),
            is_last=(i == len(splits) - 1),
        )
        for i in range(len(splits))
    ]
    return CLScenario("cil_funsd", ScenarioType.CIL, tasks, train_dss, eval_dss)


# Canonical 5×6 CORD partition (validated). Super-class-respecting; flattening it
# in this order and re-chunking yields the longer-horizon variants.
_CORD_CANONICAL_SESSIONS = [
    ["menu.cnt", "menu.discountprice", "menu.itemsubtotal", "menu.nm", "menu.num", "menu.price"],
    [
        "menu.unitprice",
        "menu.vatyn",
        "menu.sub.cnt",
        "menu.sub.nm",
        "menu.sub.price",
        "menu.sub.unitprice",
    ],
    [
        "sub_total.discount_price",
        "sub_total.etc",
        "sub_total.othersvc_price",
        "sub_total.service_price",
        "sub_total.subtotal_price",
        "sub_total.tax_price",
    ],
    [
        "total.cashprice",
        "total.changeprice",
        "total.creditcardprice",
        "total.emoneyprice",
        "total.menuqty_cnt",
        "total.menutype_cnt",
    ],
    [
        "total.total_etc",
        "total.total_price",
        "void_menu.nm",
        "void_menu.price",
        "sub.nm",
        "sub.cnt",
    ],
]


def _cord_class_sessions(num_sessions: int, order: list[int] | None) -> list[list[str]]:
    """Partition CORD's 30 fine classes into ``num_sessions`` ordered sessions.

    ``num_sessions=5`` returns the canonical super-class-respecting 5×6 partition
    (validated). Other counts re-chunk the flattened 30-class list (canonical order)
    into equal groups. ``order`` permutes the resulting session sequence.
    """
    if num_sessions == 5:
        cls_per_session = [list(s) for s in _CORD_CANONICAL_SESSIONS]
    else:
        flat = [c for session in _CORD_CANONICAL_SESSIONS for c in session]  # 30, canonical order
        if len(flat) % num_sessions != 0:
            raise ValueError(f"num_sessions must divide {len(flat)} evenly (got {num_sessions}).")
        per = len(flat) // num_sessions
        cls_per_session = [flat[i : i + per] for i in range(0, len(flat), per)]

    if order is not None:
        if sorted(order) != list(range(num_sessions)):
            raise ValueError(f"order must be a permutation of 0..{num_sessions - 1}, got {order}.")
        cls_per_session = [cls_per_session[i] for i in order]
    return cls_per_session


def build_cil_cord(num_sessions: int = 5, order: list[int] | None = None) -> CLScenario:
    """CORD class-incremental: 30 fine classes → ``num_sessions`` sessions.

    ``num_sessions=5`` uses the canonical super-class-respecting 5×6 partition
    (validated, unchanged). Other counts re-chunk the flattened 30-class list (in
    canonical order) into equal groups — enabling longer horizons such as 10×3 for
    the long-sequence study. ``order`` permutes the session sequence (a permutation
    of ``0..num_sessions-1``) for the ordering-robustness study.
    """
    cls_per_session = _cord_class_sessions(num_sessions, order)
    bio_splits = [[f"B-{c}" for c in s] + [f"I-{c}" for c in s] for s in cls_per_session]

    # Cumulative head label maps per session + each session's label_set prefixed with O.
    # The underlying datasets emit native CORD ids; CIL_LabelRemapper translates those
    # into the head-index space so targets line up with the (growing) classifier head.
    snapshots, label_sets = _cil_head_snapshots(bio_splits)

    def _wrap(split: str) -> list[Dataset]:
        wrapped = []
        for i, lbls in enumerate(bio_splits):
            base = CORDDataset(split=split, granularity="fine", label_filter=lbls)
            wrapped.append(CIL_LabelRemapper(base, base.id_to_label, snapshots[i]))
        return wrapped

    train_dss = _wrap("train")
    eval_dss = _wrap("test")

    # Joint upper-bound pool: ONE full-label CORD dataset (every receipt once with ALL its
    # entities) remapped into the cumulative 61-tag head space (snapshots[-1]). This avoids
    # the per-session masking conflict that collapses Joint when train_datasets are concatenated.
    full_head = snapshots[-1]
    full_train = CORDDataset(split="train", granularity="fine")
    joint_train = [CIL_LabelRemapper(full_train, full_train.id_to_label, full_head)]

    # Super-class metadata only applies to the canonical, in-order 5-session layout.
    super_meta = (
        ["menu", "menu", "sub_total", "total", "mixed"]
        if num_sessions == 5 and order is None
        else None
    )
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"cord_cil_t{i}",
            label_set=label_sets[i],
            is_first=(i == 0),
            is_last=(i == len(bio_splits) - 1),
            metadata={"super_class": super_meta[i]} if super_meta else {},
        )
        for i in range(len(bio_splits))
    ]
    return CLScenario(
        "cil_cord",
        ScenarioType.CIL,
        tasks,
        train_dss,
        eval_dss,
        joint_train_datasets=joint_train,
    )


def build_cil_wildreceipt(num_sessions: int = 4) -> CLScenario:
    """WildReceipt class-incremental: 24 entity classes → num_sessions sessions.

    Mirrors build_cil_cord. WildReceipt's BIO LABEL_NAMES are derived from the upstream
    ClassLabel at load time, so we instantiate the train split once to read the class
    list, partition the 24 entity classes (each contributing B-/I-) across sessions,
    and grow the head session by session via CIL_LabelRemapper.
    """
    # Read the (data-derived) full BIO label set once. After this, WildReceiptDataset
    # LABEL_NAMES/NUM_LABELS class attributes are populated.
    full_train = WildReceiptDataset(split="train")
    entity_classes = [
        n[2:] for n in WildReceiptDataset.LABEL_NAMES if n.startswith("B-")
    ]  # 24 class names (key/value pairs), upstream order

    # Partition entity classes as evenly as possible into num_sessions.
    per = -(-len(entity_classes) // num_sessions)  # ceil
    class_sessions = [entity_classes[i : i + per] for i in range(0, len(entity_classes), per)]
    bio_splits = [[f"B-{c}" for c in s] + [f"I-{c}" for c in s] for s in class_sessions]

    snapshots, label_sets = _cil_head_snapshots(bio_splits)

    def _wrap(split: str) -> list[Dataset]:
        wrapped = []
        for i, lbls in enumerate(bio_splits):
            base = WildReceiptDataset(split=split, label_filter=lbls)
            wrapped.append(CIL_LabelRemapper(base, base.id_to_label, snapshots[i]))
        return wrapped

    train_dss = _wrap("train")
    eval_dss = _wrap("test")

    # Joint pool: one full-label WildReceipt set remapped into the final cumulative head.
    full_head = snapshots[-1]
    joint_train = [CIL_LabelRemapper(full_train, full_train.id_to_label, full_head)]

    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"wildreceipt_cil_t{i}",
            label_set=label_sets[i],
            is_first=(i == 0),
            is_last=(i == len(bio_splits) - 1),
        )
        for i in range(len(bio_splits))
    ]
    return CLScenario(
        "cil_wildreceipt",
        ScenarioType.CIL,
        tasks,
        train_dss,
        eval_dss,
        joint_train_datasets=joint_train,
    )




def build_dil(order: list[int] | None = None) -> CLScenario:
    """Domain-incremental: FUNSD → SROIE → CORD-superclass with unified schema.

    Unified label space (4 classes + O = 9 BIO tags):
        HEADER  — section headers
        KEY     — field labels (questions in forms, fixed labels in receipts)
        VALUE   — field values (answers, prices, dates, addresses)
        OTHER   — auxiliary entities (CORD void_menu, sub)

    ``order`` permutes the 3-domain sequence (default [0,1,2] = funsd→sroie→cord)
    for the ordering-robustness study. See docs/dil_schema_mapping.md.
    """
    # Build underlying datasets, then assemble in the requested order.
    domains = [
        ("funsd", FUNSDDataset("train"), FUNSDDataset("test")),
        ("sroie", SROIEDataset("train"), SROIEDataset("test")),
        (
            "cord",
            CORDDataset("train", granularity="super"),
            CORDDataset("test", granularity="super"),
        ),
    ]
    order = order or list(range(len(domains)))
    if sorted(order) != list(range(len(domains))):
        raise ValueError(f"order must be a permutation of 0..{len(domains) - 1}, got {order}.")
    domains = [domains[i] for i in order]

    train_dss: list[Dataset] = [
        DIL_LabelRemapper(tr, name, tr.id_to_label) for name, tr, _ in domains
    ]
    eval_dss: list[Dataset] = [
        DIL_LabelRemapper(te, name, te.id_to_label) for name, _, te in domains
    ]
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"dil_{name}",
            label_set=DIL_UNIFIED_LABELS,
            is_first=(i == 0),
            is_last=(i == len(domains) - 1),
            metadata={"native_dataset": name},
        )
        for i, (name, _, _) in enumerate(domains)
    ]
    return CLScenario("dil", ScenarioType.DIL, tasks, train_dss, eval_dss)


def build_dil_receipts(order: list[int] | None = None) -> CLScenario:
    """Receipt-domain DIL: SROIE → CORD-super → WildReceipt, unified receipt schema.

    All three are receipts, so they share a richer, far less lossy unified space
    than the heterogeneous form↔receipt ``dil`` (6 classes + O = 13 fixed BIO tags;
    see ``receipt_remapping``). The domain shift is *which fields each receipt type
    exposes*, with a constant head — a clean pure domain-IL. ``order`` permutes the
    sequence (default [0,1,2]). Tasks are disjoint documents, so
    ``joint_train_datasets=None`` (the Joint path concatenates correctly).
    """
    domains = [
        ("sroie", SROIEDataset("train"), SROIEDataset("test")),
        (
            "cord",
            CORDDataset("train", granularity="super"),
            CORDDataset("test", granularity="super"),
        ),
        ("wildreceipt", WildReceiptDataset("train"), WildReceiptDataset("test")),
    ]
    order = order or list(range(len(domains)))
    if sorted(order) != list(range(len(domains))):
        raise ValueError(f"order must be a permutation of 0..{len(domains) - 1}, got {order}.")
    domains = [domains[i] for i in order]

    train_dss: list[Dataset] = [
        Receipt_LabelRemapper(tr, name, tr.id_to_label) for name, tr, _ in domains
    ]
    eval_dss: list[Dataset] = [
        Receipt_LabelRemapper(te, name, te.id_to_label) for name, _, te in domains
    ]
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"dilrcpt_{name}",
            label_set=RECEIPT_UNIFIED_LABELS,
            is_first=(i == 0),
            is_last=(i == len(domains) - 1),
            metadata={"native_dataset": name},
        )
        for i, (name, _, _) in enumerate(domains)
    ]
    return CLScenario("dil_receipts", ScenarioType.DIL, tasks, train_dss, eval_dss)


def build_dil_xlingual(langs: list[str] | None = None) -> CLScenario:
    """Cross-lingual domain-incremental: XFUND language sequence, unified schema.

    The domain shift here is *language* (default de→es→fr→it→zh), NOT schema: every
    XFUND language shares FUNSD's HEADER/QUESTION/ANSWER tags, mapped into the fixed
    DIL unified label space. Because the label space is constant across tasks, the head
    never grows and forgetting is pure representation drift — the cleanest test of
    whether the output-side forgetting finding holds under language shift.

    Tasks are disjoint documents (different languages), so joint_train_datasets=None
    (the Joint path concatenates train_datasets correctly, as in build_dil).
    """
    langs = langs or XLINGUAL_DEFAULT_LANGS

    train_dss: list[Dataset] = []
    eval_dss: list[Dataset] = []
    tasks: list[TaskInfo] = []
    for i, lang in enumerate(langs):
        tr = XFUNDDataset("train", lang=lang)
        te = XFUNDDataset("test", lang=lang)
        train_dss.append(DIL_LabelRemapper(tr, "xfund", tr.id_to_label))
        eval_dss.append(DIL_LabelRemapper(te, "xfund", te.id_to_label))
        tasks.append(
            TaskInfo(
                task_id=i,
                task_name=f"dil_xfund_{lang}",
                label_set=DIL_UNIFIED_LABELS,
                is_first=(i == 0),
                is_last=(i == len(langs) - 1),
                metadata={"native_dataset": "xfund", "lang": lang},
            )
        )
    return CLScenario("dil_xlingual", ScenarioType.DIL, tasks, train_dss, eval_dss)




def build_mixed() -> CLScenario:
    """6-session mixed scenario interleaving class-IL within and domain shifts.

    Sessions:
        0: FUNSD HEADER+QUESTION       (class-IL within forms)
        1: FUNSD ANSWER                 (class-IL within forms)
        2: SROIE all 4 fields           (domain shift: forms → receipts)
        3: CORD super menu+sub_total    (class-IL within receipts, super-class space)
        4: CORD super total+void+sub    (class-IL within receipts)
        5: FUNSD revisit (full)         (domain return — tests cross-session retention)

    Each session's underlying dataset emits its own native label ids; a
    CIL_LabelRemapper translates them into the cumulative head-index space so the
    classifier head (grown monotonically by expand_classifier) and the targets stay
    aligned. The FUNSD revisit (session 5) reuses labels already added in sessions
    0-1, so the cumulative map does not re-expand for it.
    """
    sessions = []

    # Session 0-1: FUNSD class-IL
    funsd_s0_labels = ["B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"]
    funsd_s1_labels = ["B-ANSWER", "I-ANSWER"]
    sessions.append(
        {
            "name": "mixed_funsd_s0",
            "labels": funsd_s0_labels,
            "train": FUNSDDataset("train", label_filter=funsd_s0_labels),
            "test": FUNSDDataset("test", label_filter=funsd_s0_labels),
            "metadata": {"phase": "funsd-CIL-1of2"},
        }
    )
    sessions.append(
        {
            "name": "mixed_funsd_s1",
            "labels": funsd_s1_labels,
            "train": FUNSDDataset("train", label_filter=funsd_s1_labels),
            "test": FUNSDDataset("test", label_filter=funsd_s1_labels),
            "metadata": {"phase": "funsd-CIL-2of2"},
        }
    )

    # Session 2: SROIE (domain shift — forms to receipts)
    sroie_labels = SROIEDataset.LABEL_NAMES[1:]  # exclude 'O'
    sessions.append(
        {
            "name": "mixed_sroie",
            "labels": sroie_labels,
            "train": SROIEDataset("train"),
            "test": SROIEDataset("test"),
            "metadata": {"phase": "domain-shift-1"},
        }
    )

    # Session 3-4: CORD class-IL on super classes
    cord_s0_super = ["B-menu", "I-menu", "B-sub_total", "I-sub_total"]
    cord_s1_super = ["B-total", "I-total", "B-void_menu", "I-void_menu", "B-sub", "I-sub"]
    sessions.append(
        {
            "name": "mixed_cord_s0",
            "labels": cord_s0_super,
            "train": CORDDataset("train", "super", label_filter=cord_s0_super),
            "test": CORDDataset("test", "super", label_filter=cord_s0_super),
            "metadata": {"phase": "cord-CIL-1of2"},
        }
    )
    sessions.append(
        {
            "name": "mixed_cord_s1",
            "labels": cord_s1_super,
            "train": CORDDataset("train", "super", label_filter=cord_s1_super),
            "test": CORDDataset("test", "super", label_filter=cord_s1_super),
            "metadata": {"phase": "cord-CIL-2of2"},
        }
    )

    # Session 5: FUNSD revisit (tests retention against domain interference)
    funsd_all = FUNSDDataset.LABEL_NAMES[1:]  # already added in S0/S1, won't re-expand
    sessions.append(
        {
            "name": "mixed_funsd_revisit",
            "labels": funsd_all,
            "train": FUNSDDataset("train"),
            "test": FUNSDDataset("test"),
            "metadata": {"phase": "domain-return"},
        }
    )

    # Cumulative head snapshots over the per-session label sets, then wrap each
    # session's native dataset so its ids land in the head-index space for that session.
    snapshots, label_sets = _cil_head_snapshots([s["labels"] for s in sessions])
    train_dss = [
        CIL_LabelRemapper(s["train"], s["train"].id_to_label, snapshots[i])
        for i, s in enumerate(sessions)
    ]

    # Joint upper-bound pool: each DISTINCT underlying dataset once with ALL its labels,
    # remapped into the full cumulative head (snapshots[-1]). FUNSD (sessions 0/1/revisit)
    # and CORD-super (sessions 3/4) each appear masked multiple times in train_dss, which
    # would conflict if concatenated; here each is included exactly once, full-label.
    full_head = snapshots[-1]
    funsd_full = FUNSDDataset("train")
    sroie_full = SROIEDataset("train")
    cord_full = CORDDataset("train", "super")
    joint_train = [
        CIL_LabelRemapper(funsd_full, funsd_full.id_to_label, full_head),
        CIL_LabelRemapper(sroie_full, sroie_full.id_to_label, full_head),
        CIL_LabelRemapper(cord_full, cord_full.id_to_label, full_head),
    ]
    eval_dss = [
        CIL_LabelRemapper(s["test"], s["test"].id_to_label, snapshots[i])
        for i, s in enumerate(sessions)
    ]
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=s["name"],
            label_set=label_sets[i],
            is_first=(i == 0),
            is_last=(i == len(sessions) - 1),
            metadata=s["metadata"],
        )
        for i, s in enumerate(sessions)
    ]
    return CLScenario(
        "mixed",
        ScenarioType.MIXED,
        tasks,
        train_dss,
        eval_dss,
        joint_train_datasets=joint_train,
    )




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
    elif dataset_name == "xfund":
        # Single-task XFUND baseline uses one representative language (fr) for the FWT
        # b_i reference of the cross-lingual scenario (all langs share the schema).
        train, test = XFUNDDataset("train", lang="fr"), XFUNDDataset("test", lang="fr")
        labels = XFUNDDataset.LABEL_NAMES
    elif dataset_name == "wildreceipt":
        train, test = WildReceiptDataset("train"), WildReceiptDataset("test")
        labels = WildReceiptDataset.LABEL_NAMES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    task = TaskInfo(
        task_id=0,
        task_name=f"{dataset_name}_single",
        label_set=labels,
        is_first=True,
        is_last=True,
    )
    return CLScenario(f"single_{dataset_name}", ScenarioType.SINGLE, [task], [train], [test])


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



SCENARIO_REGISTRY = {
    "single_funsd": lambda: build_single("funsd"),
    "single_cord": lambda: build_single("cord"),
    "single_sroie": lambda: build_single("sroie"),
    "single_xfund": lambda: build_single("xfund"),
    "single_wildreceipt": lambda: build_single("wildreceipt"),
    "cil_funsd": build_cil_funsd,
    "cil_cord": build_cil_cord,
    "cil_wildreceipt": build_cil_wildreceipt,
    # Long-horizon variants (Study 2): distinct registry names so run dirs/W&B names
    # don't collide with the base scenarios. kwargs (num_sessions/order) overridable.
    "cil_cord_long": lambda **kw: build_cil_cord(num_sessions=kw.pop("num_sessions", 10), **kw),
    "cil_wildreceipt_long": lambda **kw: build_cil_wildreceipt(
        num_sessions=kw.pop("num_sessions", 8), **kw
    ),
    "dil": build_dil,
    "dil_receipts": build_dil_receipts,
    "dil_xlingual": build_dil_xlingual,
    "mixed": build_mixed,
    "pilot": build_pilot,
}


def get_scenario(name: str, *, encoder: KIEEncoder | None = None, **kwargs) -> CLScenario:
    """Build a scenario. ``encoder`` (per-backbone tokenization) is set as the
    process-wide default before the datasets are constructed, so every loader in
    this scenario tokenizes for the active backbone. Defaults to LayoutLMv3."""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario {name}. Available: {list(SCENARIO_REGISTRY)}")
    if encoder is not None:
        set_default_encoder(encoder)
    builder = SCENARIO_REGISTRY[name]
    return builder(**kwargs) if kwargs else builder()
