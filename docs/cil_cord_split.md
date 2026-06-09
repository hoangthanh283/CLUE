# CIL-CORD: 5-session split rationale

The CIL-CORD scenario partitions CORD's 30 fine-grained classes into 5 sessions
of 6 classes each. This document explains the partitioning choice.

## The 30 fine-grained classes (BIO-aware → 61 total tags including O)

CORD-v2's annotation schema groups entities into 5 super-classes:
- **menu**: 12 fine classes (menu items + sub-items + attributes)
- **sub_total**: 6 fine classes (running totals)
- **total**: 8 fine classes (final amounts)
- **void_menu**: 2 fine classes (cancelled lines)
- **sub**: 2 fine classes (sub-categories)

(See `doccl/data/cord.py::CORD_FINE_LABELS` for the canonical list.)

## Partitioning strategy

We split by super-class boundaries where possible, with two exceptions to
balance to 6 classes per session:

| Session | Classes (6 each) | Super-class context |
|---|---|---|
| 0 | menu.{cnt, discountprice, itemsubtotal, nm, num, price} | core menu fields |
| 1 | menu.unitprice, menu.vatyn, menu.sub.{cnt, nm, price, unitprice} | extended menu + sub-items |
| 2 | sub_total.{discount_price, etc, othersvc_price, service_price, subtotal_price, tax_price} | full sub_total class |
| 3 | total.{cashprice, changeprice, creditcardprice, emoneyprice, menuqty_cnt, menutype_cnt} | total payment methods |
| 4 | total.{total_etc, total_price}, void_menu.{nm, price}, sub.{nm, cnt} | residuals + auxiliary |

## Why this choice

**Pros:**
- **Curriculum**: easier "menu" classes first, more ambiguous "void/sub" last
- **Coherence**: each session has internal semantic coherence (won't have
  random mixing of unrelated classes), making the CIL signal cleaner
- **Balance**: roughly equal training set size per session (~600-800 receipts)

**Cons:**
- Session 1 mixes core menu attributes with sub-item attributes (potential
  intra-session ambiguity)
- Session 4 mixes 3 super-classes for balance (could affect interpretation)

## Alternative partitioning

If session 1 or 4 prove problematic, alternatives:

1. **Pure super-class split**: 5 sessions = 5 super-classes, but uneven sizes
   (12, 6, 8, 2, 2). The 2-class sessions would likely overfit.
2. **Random partition**: simpler but loses curriculum ordering and coherence.
3. **3-session split**: 30/3 = 10 classes per session — fewer but harder
   transitions. Could be a robustness check.

## Tracking forgetting per super-class

Even though sessions don't strictly align with super-classes, we tag each
session with a `super_class` metadata field
(`task.metadata["super_class"]`) for analysis. This lets us answer
"does forgetting concentrate within or across super-classes?"

```python
from doccl.data.scenarios import build_cil_cord
s = build_cil_cord()
for t in s.tasks:
    print(t.task_id, t.task_name, t.metadata.get("super_class"))
```

## Where this is implemented

- `doccl/data/scenarios.py::build_cil_cord` — partitioning logic
- `doccl/data/cord.py::CORD_FINE_LABELS` + `SUPERCLASS_MAP` — class definitions
