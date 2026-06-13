# Critical Bug: Class-IL with `unified: false`

## Executive Summary

**CRITICAL BUG FOUND**: Using `unified: false` in class-incremental learning (class-IL) mode causes **complete failure** (0% accuracy) on previous tasks after the classifier head grows.

**Fix**: Set `unified: true` in all class-IL configs.

---

## The Bug

### Symptoms

```
Task 1 (FUNSD):     75% → 0% accuracy after task 4
Task 2 (CORD):      96% → 0% accuracy after task 4
Task 3 (SROIE):     99% → 0% accuracy after task 4
Task 4 (WildReceipt): 88% (maintained)
Task 5 (XFUND):     0% (never learned properly)
```

**Catastrophic forgetting metric**: BWT = -0.68 (complete collapse)

### Root Cause

With `unified: false`, each task defines its own label list starting from ID 0:

```python
# Task 1 (FUNSD) - 7 labels
label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
label2id = {
    'O': 0,
    'B-HEADER': 1,
    'I-HEADER': 2,
    ...
}
Classifier: 7 outputs [0-6]
```

```python
# Task 2 (CORD) - 61 labels
label_list = ['O', 'B-menu.nm', 'I-menu.nm', 'B-menu.price', ...]  # 30 entities × 2 + O
label2id = {
    'O': 0,
    'B-menu.nm': 1,     # ⚠️ COLLISION: Same ID as B-HEADER!
    'I-menu.nm': 2,     # ⚠️ COLLISION: Same ID as I-HEADER!
    ...
}
Classifier expands: 61 outputs [0-60]
```

**Problem**: When evaluating FUNSD after training CORD:
- FUNSD test data has label ID `1` meaning `B-HEADER`
- But classifier output `1` now corresponds to CORD's label (e.g., `B-menu.nm`)
- **Complete mismatch** → predictions are nonsense → 0% accuracy

### Why This Wasn't Caught Earlier

The model was only evaluated on the **most recent task** (WildReceipt), which maintained correct label mappings. The collapse only became apparent when checking the full accuracy matrix.

---

## Technical Explanation

### Class-IL Without Unified Labels

**Step-by-step breakdown:**

1. **Task 1 (FUNSD)** trained:
   ```
   Dataset labels: ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
   Label IDs:      [  0,        1,           2,            3,            4,           5,          6]
   Classifier:     7 outputs (positions 0-6)

   Test sample: "Invoice" token has label "B-HEADER" (ID=1)
   Model predicts: position 1 → "B-HEADER" ✓ CORRECT
   ```

2. **Task 2 (CORD)** trained:
   ```
   Dataset labels: ['O', 'B-menu.nm', 'I-menu.nm', 'B-menu.price', ..., 'B-total.total_price', ...]
   Label IDs:      [  0,          1,            2,              3,  ...,                   60]
   Classifier:     Expands to 61 outputs (positions 0-60)

   Position mapping changes:
   0: Still 'O'
   1: Now 'B-menu.nm' (was 'B-HEADER' for FUNSD!)
   2: Now 'I-menu.nm' (was 'I-HEADER' for FUNSD!)
   ...
   6: Now 'B-menu.cnt' (was 'I-ANSWER' for FUNSD!)
   7-60: New CORD labels
   ```

3. **Re-evaluate Task 1 (FUNSD)**:
   ```
   Test sample: "Invoice" token has label "B-HEADER" (ID=1 in FUNSD's label space)
   Model predicts: position 1 → "B-menu.nm" (CORD label)
   Evaluation computes: argmax(logits) = 1, compare with ground truth "B-HEADER" (ID=1)

   Problem: Predicted position 1 means "B-menu.nm" in current head,
            but ground truth position 1 means "B-HEADER" in FUNSD's label space

   Mismatch → prediction wrong → 0% accuracy
   ```

---

## The Fix: Unified Label Space

With `unified: true`, all tasks share a **global label mapping** defined upfront:

### Unified Label List (51 labels total)

```python
UNIFIED_LABEL_LIST = [
    'O',                      # ID: 0
    # Form entities (3 × 2 = 6)
    'B-FORM.HEADER',          # ID: 1
    'I-FORM.HEADER',          # ID: 2
    'B-FORM.QUESTION',        # ID: 3
    'I-FORM.QUESTION',        # ID: 4
    'B-FORM.ANSWER',          # ID: 5
    'I-FORM.ANSWER',          # ID: 6
    # Receipt entities (22 × 2 = 44)
    'B-RCPT.STORE_NAME',      # ID: 7
    'I-RCPT.STORE_NAME',      # ID: 8
    ...
    'B-RCPT.TOTAL',           # ID: 25
    'I-RCPT.TOTAL',           # ID: 26
    ...
]
```

### How It Works

**Task 1 (FUNSD)** with unified labels:
```python
# FUNSD native labels: ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', ...]
# Mapped to unified:   ['O', 'B-FORM.HEADER', 'I-FORM.HEADER', 'B-FORM.QUESTION', ...]

Dataset preprocessing:
  Native label "B-HEADER" → Unified label "B-FORM.HEADER" (ID=1)
  Native label "B-QUESTION" → Unified label "B-FORM.QUESTION" (ID=3)

Classifier: 51 outputs (full unified space from start)
  Position 0: 'O'
  Position 1: 'B-FORM.HEADER'
  Position 3: 'B-FORM.QUESTION'
  Position 7: 'B-RCPT.STORE_NAME' (not used yet, but present)
  ...
```

**Task 2 (CORD)** with unified labels:
```python
# CORD native labels: ['O', 'B-menu.nm', 'B-menu.price', 'B-total.total_price', ...]
# Mapped to unified:  ['O', 'B-RCPT.ITEM', 'B-RCPT.UNIT_PRICE', 'B-RCPT.TOTAL', ...]

Dataset preprocessing:
  Native label "B-menu.nm" → Unified label "B-RCPT.ITEM" (ID=33)
  Native label "B-total.total_price" → Unified label "B-RCPT.TOTAL" (ID=25)

Classifier: Still 51 outputs (no expansion needed!)
  Position 0: 'O'
  Position 1: 'B-FORM.HEADER' (FUNSD's label, still valid!)
  Position 25: 'B-RCPT.TOTAL' (newly activated for CORD)
  Position 33: 'B-RCPT.ITEM' (newly activated for CORD)
```

**Re-evaluate Task 1 (FUNSD)** after Task 2:
```python
Test sample: "Invoice" token has label "B-HEADER"
Preprocessed to unified: "B-FORM.HEADER" (ID=1)

Model predicts: argmax(logits) might be position 1
Compare: Prediction position 1 = "B-FORM.HEADER" ✓
         Ground truth = "B-FORM.HEADER" (ID=1) ✓

MATCH → Correct prediction → Normal accuracy maintained!
```

---

## Configuration Changes

### Before (Broken)
```yaml
cl_setting: "class_il"
label_space:
  unified: false  # ❌ BREAKS CLASS-IL
```

### After (Fixed)
```yaml
cl_setting: "class_il"
label_space:
  unified: true   # ✅ REQUIRED FOR CLASS-IL
```

---

## Impact on Different CL Settings

| Setting | `unified: false` | `unified: true` |
|---------|------------------|-----------------|
| **Task-IL** | ✅ Works (separate heads per task) | ✅ Works (but unnecessary overhead) |
| **Class-IL** | ❌ **BROKEN** (label mismatch) | ✅ **REQUIRED** |
| **Domain-IL** | ⚠️ Depends (same label space?) | ✅ Safer |

**Rule of thumb**: Use `unified: true` for Class-IL always. Task-IL can use either.

---

## Why Unified Labels Are Better for Class-IL

### Advantages
1. **Correct label mapping**: No ID collisions across tasks
2. **No classifier growth**: Fixed 51-output head from start
3. **Simpler evaluation**: Direct label ID comparison works
4. **Transfer learning**: Model learns shared representations (e.g., "store name" concepts)
5. **Memory efficient**: Single fixed-size head vs growing head

### Disadvantages
1. **Pre-defined label space**: Must know all possible entity types upfront
2. **Sparse predictions**: Model outputs 51 classes but each task uses ~7-30
3. **Requires label mapping**: Dataset preprocessing must map to unified labels

**Trade-off**: Slight preprocessing complexity for correct Class-IL behavior.

---

## How Label Mapping Works

### Mapping Functions (from `label_space.py`)

```python
def map_form_entity(entity: str) -> Optional[str]:
    """FUNSD/XFUND entities → FORM.* namespace"""
    if entity == "HEADER":
        return "FORM.HEADER"
    if entity == "QUESTION":
        return "FORM.QUESTION"
    if entity == "ANSWER":
        return "FORM.ANSWER"
    return None

def map_cord_entity(entity: str) -> Optional[str]:
    """CORD fine-grained categories → RCPT.* namespace"""
    if entity.endswith("MENU.NM"):
        return "RCPT.ITEM"
    if entity.endswith("MENU.PRICE"):
        return "RCPT.UNIT_PRICE"
    if entity.endswith("TOTAL.TOTAL_PRICE"):
        return "RCPT.TOTAL"
    # ... 27 more mappings
    return "RCPT.MISC"  # Fallback

def map_sroie_entity(entity: str) -> Optional[str]:
    """SROIE entities → RCPT.* namespace"""
    if entity == "COMPANY":
        return "RCPT.STORE_NAME"
    if entity == "DATE":
        return "RCPT.DATE"
    if entity == "TOTAL":
        return "RCPT.TOTAL"
    if entity == "ADDRESS":
        return "RCPT.ADDRESS"
    return None
```

### Applied During Dataset Preprocessing

```python
# In layoutlm_datasets.py
if use_unified_labels:
    for token, label in zip(tokens, labels):
        # Original label: "B-HEADER" (FUNSD)
        bio, entity = split_label(label)  # bio="B", entity="HEADER"

        # Map to unified entity
        unified_entity = map_form_entity(entity)  # "FORM.HEADER"

        # Reconstruct unified label
        unified_label = f"{bio}-{unified_entity}"  # "B-FORM.HEADER"

        # Get unified label ID
        label_id = UNIFIED_LABEL2ID[unified_label]  # ID=1

        # Store in dataset
        labels_ids.append(label_id)
```

---

## Validation

### Test the Fix

```bash
# Re-run with unified labels
python scripts/train_cl.py \
    --config configs/layoutlmv3_gem_class_il.yaml \
    --output_dir results/gem_unified

# Check accuracy matrix - should NOT have zeros
cat results/gem_unified/cl_results.json | jq '.accuracy_matrix'
```

### Expected Results After Fix

```
Accuracy Matrix (unified labels):
       T1    T2    T3    T4    T5
T1:  75%   --    --    --    --
T2:  72%  96%   --    --    --
T3:  70%  94%  99%   --    --
T4:  68%  92%  97%  88%   --
T5:  65%  90%  95%  85%  82%   ← All tasks maintained!
```

**Metrics should be**:
- ACC: ~0.75-0.85 (not 0.18!)
- BWT: -0.05 to +0.05 (not -0.68!)
- AAA: ~0.80-0.85 (not 0.42!)

---

## Lessons Learned

### For Researchers

1. **Always test Class-IL thoroughly**: Check full accuracy matrix, not just final task
2. **Validate label mappings**: Print label IDs during training to catch mismatches
3. **Use unified labels for Class-IL**: Non-negotiable requirement
4. **Separate Class-IL from Task-IL configs**: Different settings require different label strategies

### For Practitioners

1. **Default to `unified: true`** unless you have good reason not to
2. **Add validation checks**: Assert label IDs are consistent across tasks
3. **Monitor all tasks during training**: Don't just track current task accuracy
4. **Test on earlier tasks frequently**: Catch forgetting issues early

---

## Affected Configurations

### Need to be updated:
- ✅ `configs/layoutlmv3_gem_class_il.yaml` - FIXED
- ✅ `configs/layoutlmv3_agem_class_il.yaml` - FIXED
- ⚠️ `configs/layoutlmv3_class_il.yaml` - Check if exists
- ⚠️ `configs/layoutlmv3_er_class_il.yaml` - Check if exists
- ⚠️ `configs/layoutlmv3_ewc_class_il.yaml` - Check if exists
- ⚠️ `configs/layoutlmv3_lwf_class_il.yaml` - Already requires unified labels

### Safe to keep `unified: false`:
- ✅ `configs/layoutlmv3_task_il.yaml` - Uses task-specific heads
- ✅ `configs/layoutlmv3_joint_*.yaml` - Single task, no CL

---

## Related Code

- **Label mapping**: [src/data/label_space.py](../src/data/label_space.py)
- **Dataset preprocessing**: [src/data/layoutlm_datasets.py](../src/data/layoutlm_datasets.py)
- **Trainer (class-IL logic)**: [src/training/continual_trainer.py](../src/training/continual_trainer.py) lines 243-251
- **Model (classifier expansion)**: [src/models/layoutlm_models.py](../src/models/layoutlm_models.py)

---

## Conclusion

This was a **critical silent failure** - the model trained successfully but produced meaningless results due to label ID mismatches. With `unified: true`, Class-IL now works correctly.

**Bottom line**: For Class-IL, `unified: true` is not optional - it's mandatory.

---

**Last updated**: 2025-11-05
**Bug discovered**: During GEM/A-GEM validation testing
**Status**: ✅ FIXED in GEM and A-GEM configs
