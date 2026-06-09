# DIL Schema Mapping — FUNSD → SROIE → CORD

The Domain-Incremental Learning (DIL) scenario requires that all three datasets
share a unified label space. This document records the mapping decision and
rationale.

## Unified label space (9 BIO tags)

| ID | Label | Semantics |
|----|----|----|
| 0 | O | non-entity |
| 1 | B-HEADER / 2: I-HEADER | Section/document headers |
| 3 | B-KEY / 4: I-KEY | Field labels (form question, receipt label) |
| 5 | B-VALUE / 6: I-VALUE | Field values (form answer, receipt total/date/address) |
| 7 | B-OTHER / 8: I-OTHER | Auxiliary entities not fitting the above (CORD void/sub) |

## Mapping per source dataset

### FUNSD (4 native classes → mapping)

```
HEADER   → HEADER
QUESTION → KEY      (form questions function as field labels)
ANSWER   → VALUE
OTHER    → O        (FUNSD's OTHER is essentially noise)
```

**Rationale:** FUNSD is closest to the unified schema by design — its
QUESTION/ANSWER pairing is the canonical KEY/VALUE relationship.

### SROIE (4 native classes → mapping)

```
COMPANY → KEY       (the entity-name field is conceptually a "key")
ADDRESS → VALUE
DATE    → VALUE
TOTAL   → VALUE
```

**Rationale:** SROIE has no explicit KEY/VALUE distinction in its label set —
all 4 fields are extracted entities. We treat COMPANY (the receipt's owner) as
KEY-like (it identifies what the receipt is about) and the other three as
VALUE-like (numeric/positional information).

This is a defensible-but-somewhat-arbitrary choice. An alternative would be to
map all 4 to VALUE, leaving FUNSD as the only source of KEY annotations.
**We chose the asymmetric mapping** because it preserves discriminative gradient
signal across all 9 tags during training — important for CL where each task
should exercise the full classifier head.

### CORD super-class (5 native super-classes → mapping)

```
menu       → VALUE   (menu items are the receipt's primary content/values)
sub_total  → VALUE   (computed totals)
total      → VALUE   (final totals)
void_menu  → OTHER   (void/cancelled lines — auxiliary)
sub        → OTHER   (sub-totals/breakdowns — auxiliary)
```

**Rationale:** All productive content in CORD is a VALUE (numeric or itemized).
Void and sub categories represent auxiliary annotations that don't fit
KEY/VALUE; mapping to OTHER preserves them rather than collapsing to O.

## Sequence ordering for DIL

Sessions in order:

1. **FUNSD**: easiest CL domain (smallest vocab, clearest KEY/VALUE)
2. **SROIE**: moderate domain shift (forms → receipts, but still 4 fields)
3. **CORD**: largest domain shift (super-class space, no KEY tags)

Why this order: gradually increasing distance from the "canonical" KEY/VALUE
schema. Reverse order (CORD → SROIE → FUNSD) is also defensible and could be
investigated as a robustness check.

## Known limitations

- The KEY tag is sparse on SROIE (only COMPANY) and absent on CORD.  This
  creates a slight class imbalance in DIL evaluation — the model effectively
  sees KEY only on FUNSD examples.  Future work: consider extending CORD's
  super-class space to explicitly tag receipt headers as KEY.
- The OTHER tag is sparse on FUNSD (mapped to O) and SROIE (no entities mapped).
  This is acceptable — OTHER is a "catch-all for the rare CORD-specific tags"
  rather than a class with consistent semantics.

## Where this is implemented

- `doccl/data/dil_remapping.py` — `DIL_LabelRemapper`, mappings, unified label list
- `doccl/data/scenarios.py::build_dil` — wires up the three remapped datasets

## Validation

Run after dataset loaders are functional:

```bash
python -c "
from doccl.data.scenarios import build_dil
s = build_dil()
for i, t in enumerate(s.tasks):
    print(f'Task {i}: {t.task_name} | label_set={t.label_set}')
    train_ds = s.train_datasets[i]
    sample = train_ds[0]
    print(f'  sample labels: {sorted(set(sample[\"labels\"].tolist())) }')
"
```

Expected: all three tasks share the same `label_set` (the 9 unified tags),
and per-sample label distributions reflect the per-dataset mapping.
