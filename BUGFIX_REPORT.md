# Bug-Fix Report — CL Method Audit (2026-06-24)

Multi-round adversarial code audit (every method file + all evaluation/data/loop
infrastructure) triggered by suspect baseline results. Two independent review passes
plus by-hand verification of each finding. This documents what was fixed, **which
results must be re-run**, and what is disclose-only (not a bug).

## CONFIRMED BUGS FIXED (results affected → MUST RE-RUN)

| # | Bug | File | Effect | Results to re-run |
|---|-----|------|--------|-------------------|
| 1 | O-LoRA ortho penalty computed `A_curr.T @ A_past` (768×768, input-feature co-activation) instead of `A_curr @ A_past.T` (r×r, row-subspace orthogonality). Penalty never vanished for orthogonal adapters → the defining O-LoRA constraint was **never enforced**. | `o_lora.py:_ortho_loss` | O-LoRA results invalid | **all `o_lora`** (cil_cord, dil, mixed, dil_xlingual, cil_wildreceipt × 3 seeds) |
| 2 | CL-LoRA inherits the same penalty via `OLoRA._ortho_loss`. | `cl_lora.py` (inherited) | CL-LoRA results invalid | **all `cl_lora`** |
| 3 | DER++ CE-replay batch popped `_logits` but not `_logit_width`; wrapper.forward has no `**kwargs` → `TypeError` at first non-empty-buffer step → **crash on re-run**. | `der.py` | DER++ crashes if re-run; old results from a pre-fix build | **all `der_pp`** if regenerating |
| 4 | DocCL `after_task` deepcopy'd the new teacher with the OLD teacher still on GPU → ~2× model VRAM spike each task boundary → OOM on 6 GB. | `doccl.py:after_task` | OOM crashes (not wrong numbers) on small GPU | none invalidated; prevents crashes |
| 5 | LwF same teacher-eviction OOM. | `lwf.py:after_task` | OOM on small GPU | none invalidated |
| 6 | `DocCL_B._ewc_penalty` lacked the CIL `min_shape` slice → `RuntimeError` at first CIL head growth. | `doccl.py:DocCL_B` | `doccl_b` ablation crashes on any CIL run | `doccl_b` (legacy ablation only) |

**Infrastructure fix (systemic):** `train.py` now records `method_hparams` +
`training_hparams` in every `metrics.json`, so a run's real config (e.g. `cflat_lambda`)
is always recoverable. This closes the silent-default trap below.

## C-Flat++ LABELING (action needed, not a code bug)

Every `er_cflat` run used the config default `cflat_lambda=0.0` — i.e. **plain ER+SAM**,
NOT C-Flat++ (the curvature term was off; no script ever overrode it, on any GPU).
Decision: **keep current results labeled "ER+SAM"** (a valid Foret-2021 flat-minima
baseline), and run genuine **C-Flat++** separately via the new grid knob
`CFLAT_LAMBDA=0.1` (≥16 GB GPU; produces distinct `_curv` run-names). Thesis tables
must relabel `ER+C-Flat++` → `ER+SAM` for the current numbers.

## DISCLOSE-ONLY (verified NOT bugs — no code change)

- **EWC** is correct Online-EWC (Schwarz 2018). CIL AA≈0.5 is the genuine
  over-regularization from `ewc_gamma=1.0` Fisher accumulation over 5 tasks
  (Huszár 2019), not a defect. Disclose in thesis; `ewc_gamma` now recorded.
- **Metrics (AA/BWT/AF/FWT), CKA, CIL label-remapping, the CL loop order, eval-mode/
  no_grad** — all audited CLEAN. No result corruption from infrastructure.
- **AF** uses the Lopez-Paz (= −BWT) convention, not Chaudhry's max-checkpoint. Note it.
- **Early-stopping** uses the test split as the val signal — community-standard, applied
  identically to all methods (rankings unaffected); inflates only absolute F1. CORD has
  an unused `validation` split that could be wired later.
- **CORD / WildReceipt BIO** edge cases (empty-leading-word; adjacent same-class merge)
  are LOW-risk and unlikely on the curated datasets.
- **SAM** clips g₀ before the ascent step — a self-consistent variant, doesn't
  invalidate the comparison.

## SEPARATE PRE-EXISTING ISSUE (not from this audit's scope, no results affected)

- **LiLT backbone** forward crashes with `padding_idx=-1` (`base_wrapper.py` + LiLT HF
  model). LiLT has **0 runs in any result** (LayoutLMv3 is the only backbone used), so no
  thesis number is affected. Must be fixed before adding LiLT as the secondary backbone
  (the Q1 recommendation). Tracked separately.

## RE-RUN PRIORITY (next GPU session)

1. `o_lora` + `cl_lora` — all scenarios × 3 seeds (orthogonality now real).
2. `der_pp` — all scenarios × 3 seeds (crash fixed; clean numbers).
3. Optionally `er_cflat` with `CFLAT_LAMBDA=0.1` for genuine C-Flat++ rows.

DocCL / EWC / LwF / ER / prompt / naive / joint results remain **valid** (their fixes
were OOM-prevention or unused ablations, not result-changing).
