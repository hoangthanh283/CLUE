# Legacy-protocol forensics: dagger cells (2026-11-11)

What every legacy (dagger) fallback in `scripts/rca/make_clean_tables.py` (dicts `C`, `M`, `D`,
list `BASE`) actually ran with. Evidence was read from logs and JSON only; no torch, no datasets.
The extractor is a read-only script over the run's `.log` (the Hydra config dump; every log has
exactly one dump, so there were no retries), `metrics.json` (`method_hparams` / `training_hparams`)
and the tensorboard event filename (start time and host). None of the runs has a `.hydra/` dir.

## Summary of differences from the clean protocol

1. **Early stopping.** Every legacy cell except Joint stopped on the current session's **test
   split** (`val_frac` unset, so train.py used the eval loader). The legacy Joint runs held out
   10% of the pooled full-label training set (`val_frac: 0.1`).
2. **Epoch caps.**
   - 100 for Naive, Joint, ER, DER++, LwF, CPFD and IS3 on CIL-CORD and Mixed.
   - 30 for CODA-Prompt and CL-LoRA.
   - 10 for every CoLaR-family row and LCA.
   - The June DIL grid rows (Naive, Joint, ER, LwF, DER++, CODA-Prompt, CL-LoRA) saved no
     config. Their caps come from the grid script: 100, and 30 for CODA-Prompt.
   - The clean protocol uses 10 everywhere and 100 for Joint.
3. **Precision.** The rescored CIL-CORD/Mixed rows (`results/rescore`, Nov 3 to 10) set
   `training.amp=true`, which on this RTX 2060 means fp16 with a GradScaler. That includes the
   CoLaR rows and the Mixed CoLaR-SBR row. Three exceptions ran in fp32 (from code history, with
   matching throughput):
   - IS3 and LCA in every run: their training loops never enable autocast.
   - Seed 42 of ER, DER++, LwF, CPFD, CL-LoRA and CODA-Prompt: those runs started before AMP was
     wired into their loops (commit 963ffa7, 11-04 20:36).

   The gate runs (`results/gates/cil_protocol_2026-11`: CIL-CORD CoLaR-SBR, no-memory SBR,
   SBR-Recall, +LexSlot, DIL SBR) and the October DIL CoLaR/CPFD/IS3/LCA runs are fp32. The June
   DIL grid rows: precision unknown.
4. **Memory and scoring.**
   - Legacy ER and DER++ keep a 200-sample reservoir. The clean ER row is ER-250 (50 documents per
     session).
   - CoLaR rows keep 50 documents per session.
   - All CIL-CORD and Mixed legacy runs use session-restricted scoring
     (`cil_restrict_eval: true`), as in the clean protocol. DIL has a fixed label space, so the
     restriction does not apply there.
   - No legacy run has `full_label_f1`, so the full-label column is clean-only.

## Evidence legend

- **log**: Hydra config dump at the top of the run's `.log`.
- **mj**: `metrics.json` `method_hparams` / `training_hparams`.
- **script**: the launching script (value not confirmed by a run artifact).
- **code**: train.py or method code at the run's start time, dated by git history and by the
  start time in the log or tensorboard filename.
- **Host**: from the tensorboard filename. `thanh` is the local RTX 2060 box. `6fa3a881ff3e` is a
  rented container.

Precision codes:
- **fp32**: `training.amp=false`.
- **fp16**: `training.amp=true` and the method loop is wired (base.py chooses fp16 + GradScaler
  on Turing, since 38648af, 10-05).
- **fp32\***: `training.amp=true` but the loop never enabled autocast, so the run was effectively
  fp32 (code). The log still prints "Mixed precision enabled" because train.py logs the flag, not
  the loop.

## CIL-CORD (dict `C`, and `BASE` cil rows; root, run-name prefix)

| Table label | Source | Seeds | Epoch cap | Precision | Early stop | Memory | Scoring |
|---|---|---|---|---|---|---|---|
| Naive | `results/rescore/cil_cord_naive_seed{s}` | 42, 7, 123 | 100 (log) | fp16 (log + code) | test split (log: no val_frac) | none | restricted (log) |
| No memory + SBR (`nobank`) | `gates/.../cil_cord_colar_sbr_noreplay_seed{s}` | 42 | 10 (log) | fp32 (log) | test split (log) | 0 docs (log) | restricted (log) |
| CoLaR | `results/rescore/cil_cord_colar_seed{s}` (dir `_k4_d50_r128`) | 42, 7 | 10 (log) | fp16 (log + code) | test split (log) | 50 docs/session, rank 128, layer 4 (log) | restricted (log) |
| CoLaR-SBR | `gates/.../cil_cord_colar_sbr_seed{s}` | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) | restricted (log) |
| CL-LoRA (2x2 cell and `BASE`) | `results/rescore/cil_cord_cl_lora_seed{s}` | 42, 7 | 30 (log) | s42 fp32\*, s7 fp16 (code) | test split (log) | none | restricted (log) |
| SBR-Recall | `gates/.../cil_cord_colar_sbr_recall_r64_seed{s}` | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 0 docs (log) | restricted (log) |
| Joint training (reference) | `results/rescore/cil_cord_joint_seed{s}` | 42, 7, 123 | 100 (log) | fp16 (log + code) | held-out 10% of pooled full-label train (log `val_frac: 0.1`; all three started after the leak fix d22f176, 11-03 16:27) | n/a | restricted (log) |
| ER | `results/rescore/cil_cord_er_seed{s}` | 42, 7, 123 | 100 (log) | s42 fp32\*, s7/s123 fp16 (code) | test split (log) | 200-sample reservoir, total (log `buffer_size: 200`) | restricted (log) |
| + LexSlot head slots | `gates/.../cil_cord_colar_sbr_slot_seed{s}` | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) | restricted (log) |
| LwF | `results/rescore/cil_cord_lwf_seed{s}` | 42, 7, 123 (s123 AA = 0.0, dropped as a collapse) | 100 (log) | s42 fp32\*, s7/s123 fp16 (code) | test split (log) | none | restricted (log) |
| DER++ | `results/rescore/cil_cord_der_pp_seed{s}` | 42, 7, 123 | 100 (log) | s42 fp32\*, s7/s123 fp16 (code) | test split (log) | 200-sample reservoir (log) | restricted (log) |
| CODA-Prompt | `results/rescore/cil_cord_coda_prompt_seed{s}` | 42, 7 | 30 (log) | s42 fp32\*, s7 fp16 (code) | test split (log) | none | restricted (log) |
| CPFD (2023) | `results/rescore/cil_cord_cpfd_seed{s}` | 42, 7 (s7 AA = 0.0, kept: CPFD is not drop_zero) | 100 (log) | s42 fp32\* (started 20:14, 22 min before 963ffa7; throughput 1.56 vs 2.00 it/s for s7), s7 fp16 | test split (log) | none | restricted (log) |
| IS3 (2024) | `results/rescore/cil_cord_is3_seed{s}` | 42, 7 | 100 (log) | fp32\* both (is3.train_task never calls `_amp_setup`; same throughput s42/s7) | test split (log) | none | restricted (log) |
| LCA (2026) | `results/rescore/cil_cord_lca_seed{s}` | 42, 7 | 10 (log) | fp32\* both (lca.train_task has no autocast) | test split (log) | Gaussian class stats (no documents) | restricted (log) |
| CoLaR-SBR, BROS (`sbr_bros`) | clean only | none | n/a | n/a | n/a | n/a | n/a |

## Mixed (dict `M`, and `BASE` mix rows)

All of these are `results/rescore/mixed_<m>_seed{s}`, except LexSlot and Recall, which are under
`gates/...`. Every row uses restricted scoring (log).

| Table label | Seeds | Epoch cap | Precision | Early stop | Memory |
|---|---|---|---|---|---|
| Naive | 42, 7 | 100 (log) | fp16 | test split (log) | none |
| CoLaR (dir `_k4_d50_r128`) | 42, 7 | 10 (log) | fp16 | test split (log) | 50 docs/session (log) |
| CoLaR-SBR | 42, 7 | 10 (log) | **fp16** (log `amp: true`; latent_replay loop wired) | test split (log) | 50 docs/session (log) |
| Joint training (reference) | 42, 7 | 100 (log) | fp16 | held-out 10% (log) | n/a |
| ER | 42, 7 | 100 (log) | fp16 (both started after 963ffa7) | test split (log) | 200-sample reservoir (log) |
| + LexSlot (gate) | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) |
| SBR-Recall (gate, `r64c`) | 42 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) |
| LwF | 42, 7 (s7 AA = 0.0, dropped) | 100 | fp16 | test split | none |
| DER++ | 42, 7 | 100 | fp16 | test split | 200-sample reservoir |
| CODA-Prompt | 42, 7 | 30 | fp16 | test split | none |
| CL-LoRA | 42, 7 | 30 | fp16 | test split | none |
| CPFD | 42, 7 | 100 | fp16 | test split | none |
| IS3 | 42, 7 (s42 AA = 0.0, dropped) | 100 | fp32\* | test split | none |
| LCA | 42, 7 | 10 | fp32\* | test split | Gaussian stats |

All values in the last seven rows come from the log; precision comes from code.

## DIL (dict `D`, and `BASE` dil rows)

The DIL label space is fixed, so the session restriction does not apply (c430b87 restricts only
sessions with a partial label set).

| Table label | Source | Seeds | Epoch cap | Precision | Early stop | Memory |
|---|---|---|---|---|---|---|
| Naive | `results/dil_naive_seed{s}` (June grid, host `thanh`) | 42, 7, 123 | **unknown** (no saved config); script default 100 | **unknown** (script default AMP=1, bf16 via DOCCL_AMP) | test split (code: `val_frac` did not exist before 11-01) | none |
| Joint training (reference) | `results/dil_joint_seed{s}` (June, `thanh`) | 42, 7, 123 | unknown; script 100 | unknown | pooled eval sets = test (code at the time) | n/a |
| ER | `results/dil_er_seed{s}` (June, no tb, host unknown) | 42, 7, 123 | unknown; script 100 | unknown (script AMP=1; ER loop itself unwired until 11-04) | test split (code) | script/config default 200-sample reservoir |
| LwF | `results/dil_lwf_seed{s}` (June, no tb) | 42, 7, 123 | unknown; script 100 | unknown | test split (code) | none |
| DER++ | `results/dil_der_pp_seed{s}` (June, no tb) | 42, 7, 123 | unknown; script 100 | unknown | test split (code) | script default 200 |
| CODA-Prompt | `results/dil_coda_prompt_seed{s}` (June, `6fa3a881ff3e`) | 42, 7, 123 | unknown; script 30 (PROMPT_EPOCHS_CAP) | unknown | test split (code) | none |
| CL-LoRA | `results/dil_cl_lora_seed{s}` (June, `6fa3a881ff3e`) | 42, 7, 123 | unknown; script 100 | unknown | test split (code) | none |
| CoLaR | `results/dil_colar_e10_seed{s}_k4_d50_r128` (10-17/18, `thanh`) | 42, 7, 123 | 10 (mj) | fp32 (mj `amp: false`) | test split (code) | 50 docs/session (mj) |
| CoLaR-SBR | `gates/.../dil_colar_sbr2_seed{s}` | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) |
| + LexSlot | `gates/.../dil_colar_sbr_slot_seed{s}` | 42, 7, 123 | 10 (log) | fp32 (log) | test split (log) | 50 docs/session (log) |
| CPFD | `results/dil_cpfd_seed{s}` (10-22, `thanh`) | 42, 7, 123 | 10 (mj) | fp32 (mj) | test split (code) | none |
| IS3 | `results/dil_is3_seed{s}` (10-22) | 42 | 10 (mj) | fp32 (mj) | test split (code) | none |
| LCA | `results/dil_lca_seed{s}` (10-21) | 42, 7, 123 | 10 (mj) | fp32 (mj) | test split (code) | Gaussian stats |

June-grid batch size: unknown. The script default was 16, against 2 for every later run (log/mj).

## Notes the paper must not contradict

- The protocol hazard says "fp16 makes LwF and IS3 collapse to exactly 0". That holds for LwF:
  CIL-CORD s123 and Mixed s7 are fp16 zeros. It does **not** hold for IS3: by code, every IS3 run
  trained in fp32, including the Mixed s42 zero. Attribute IS3 zeros to the method, not to
  precision. CPFD CIL-CORD s7 = 0.0 is an fp16 run.
- The legacy ER row (200-sample reservoir, cap 100) is not the same configuration as the clean
  ER-250 row.
- The Mixed CoLaR-SBR legacy cell is fp16. The CIL-CORD CoLaR-SBR legacy cell is fp32, from the
  gate runs, not the rescore runs.
