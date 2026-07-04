# STATE

## SESSION HANDOFF (2026-07-04 close)

- **Done this session:** v5 graph-as-memory gate run to completion → FALSIFIED
  (both arms AA 63/SROIE 13, byte-identical). Found+fixed the val-restore no-op bug.
  Then the KEY pivot (user): v5's mechanism = solved literature (FeCAM/FeTrIL/PASS) →
  **reframed the whole program as a diagnostic+falsification paper, v5 = tombstone.**
  Docs `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` + STATE updated & pushed (7514cf3).
- **Quality gates:** 287 fast tests green, ruff clean on v5+train.py.
- **NEXT SESSION = WRITE THE PAPER, not more methods.** Lead = migration finding +
  convergent negative result. Concrete next steps (cheapest first, per program review):
  (1) head-refit probe oracle already done (AA 55.3 — representation forgetting is REAL);
  (2) exp #4 migration robustness (2 freeze maps × 2 task orders + cil_cord_long) — the
  one remaining experiment; (3) relabel thesis "LexSlot 87.3" row (it's the DocCL hybrid).
- **KNOWN BLOCKER:** `bd` is in a degraded re-clone-warning state (stray dolt server
  killed but local DB still complaining). Paper-reframe follow-up issue NOT filed in bd —
  captured here + in FINDINGS doc instead. Fix bd (or `bd export`/re-clone carefully)
  before relying on it next session. Don't block on it.

## FRAMING LOCKED: diagnostic+falsification paper, NOT a method paper (2026-07-04)

- **Decision (user):** v5 (feature-Gaussian graph replay) is NOT a contribution.
  Its core mechanism = PASS/FeTrIL/FeCAM (solved 2021-2023) → un-submittable as novelty.
  v5 enters the paper ONLY as the **terminal tombstone** of the falsification chain:
  even the field's strongest buffer-free tool (+ a novel relational variant) can't beat
  representation drift (edges-on AA 63.2, SROIE→13.0).
- **The paper's actual novelty:** (1) the diagnosis — forgetting is head-localized,
  architecture-general, and the locus MIGRATES under naive protection; (2) a rigorous
  convergent negative result: FIVE buffer-free families fail for one identified reason
  (representation drift), so "only replay grounds the head." NOT any method.
- **Venue:** CoLLAs/TMLR/ACL-Findings for the negative result; AAAI/ICML main-track only
  if the LEAD is the migration finding, never a method.
- **Docs updated:** `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` (v5 = falsification level 4;
  "is NOT a method paper" section). v5 code stays (evidence), no further method work on it.
- **v5 gate COMPLETE (both arms):** edges-on = edges-off = AA 63.17, SROIE→13.0,
  BYTE-IDENTICAL (ΔAA exactly 0.00) despite differing recon-loss streams (1.42 vs 0.013).
  → do NOT claim an edges ablation; the head-side signal is swamped by trunk drift (recon
  grad reaches head but doesn't move the evaluated model). Uninformative ablation is itself
  consistent with Finding 3 (bottleneck = trunk representation drift, not head). v5 = tombstone.
  Runs: results/dil_lexmem_v5_seed42{,_bank}. Earlier byte-identical pair (pre-fix, from
  the val-restore bug) archived at results/_invalid_v5_restore_bug/.

## Analysis program: experiments 1-3 COMPLETE (2026-07-03)

**#1 Head-refit oracle (LOW branch):** pooled probe on naive trunk AA 55.3
(FUNSD per-task probe caps at 41.9). Representation forgetting is REAL in
doc-IE — Davari's "representations survive" does NOT transfer. All head-only
methods on a naive trunk ceiling-bounded at ~55. results/head_refit_oracle.json

**#2 Gaussian head-replay (FALSIFIED, 4th chain level):** AA 63.6 ≈ v3b 66.1±3.0.
Failure signature IDENTICAL across head mechanisms (slots vs replay: SROIE
~67→~17 despite balanced head gradients) => mid-task bottleneck is per-task
REPRESENTATION drift, not head misalignment. results/dil_gauss_replay_seed42

**#3 Retention-information curve (dil seed42):**
| stored | AA |
| naive (nothing) | 41.3 |
| input counts (ledger) | 33.8 |
| naive-trunk probe ceiling | 55.3 |
| ER 1 doc/task | 57.6 |
| feature Gaussians | 63.6 |
| slot memory (v3b, 3 seeds) | 66.1 ± 3.0 |
| **ER 5 docs/task** | **82.1** |
| ER 10 docs/task | 84.8 |
| ER 50/task | 85.6 | ER-200 87.9 | joint 88.7 |
=> FIVE raw documents per task beat every buffer-free memory we built by 16 AA.
Sharpened claim: raw inputs are the only storage that survives representation
drift (encoder re-renders current features); all derived quantities go stale.
Runs: results/retention_curve/er_buf{3,15,30,150}.

**Remaining: #4 migration robustness** (2 freeze maps x 2 task orders +
cil_cord_long) — needs dil task-order scenario variants. Then thesis writing.
v3b 3-seed: AA 66.1±3.0. gauss_replay code committed (bec95b4).

## LexSlot-FM: FM refit bug FIXED — AA +5.25/+5.68 on both gate modes (2026-07-03)

- **Bug found + fixed:** `_refit_old_fm_slots` ran `self.model.eval()` → forward hook
  used `_replacement_blend` (gate-routed, uniform blend) instead of `_additive_blend`
  (single-slot, `_cur_fm_idx`). Gradients dispersed to all FM slots + base head, never
  reaching only the target slot. Fixed by switching to `self.model.train()` so
  `_additive_blend` fires with `_cur_fm_idx` set to the old slot.
- **Results comparison (seed=7, dil FUNSD→SROIE→CORD):**

  | Variant | AA | T0 | T1 | T2 |
  |---------|-------|-------|-------|-------|
  | Frozen encoder (s=42, no refit) | 45.31 | 20.50 | 46.79 | 68.64 |
  | Layout+EWC (no refit) | 44.09 | 31.12 | 27.59 | 73.56 |
  | Layout+EWC+refit (**FIXED**) | **48.67** | **40.78** | 20.35 | 84.87 |
  | TF-IDF+EWC (no refit) | 39.05 | 19.53 | 8.80 | 88.81 |
  | TF-IDF+EWC+refit (**FIXED**) | **44.73** | **26.10** | **19.45** | 88.64 |

  **Key findings:**
  1. FM refit (train-mode fix) improves both gates — Layout AA +5.25, TF-IDF AA +5.68
  2. T0 jumps +11.09 (Layout) and +6.57 (TF-IDF) — FM_0 refit now actually works
  3. T2 stays high on both gates: 84.87 (Layout) / 88.64 (TF-IDF)
  4. T1 (SROIE) still degraded by encoder drift — 50-sample FM_1 replay buffer insufficient
  5. **Best AA so far: 48.67** (Layout gate + EWC λ=1000 + FM refit 3ep/50samples)
- **Remaining issue:** T1 collapses from 70.98 (post-training) to 20.35 (after CORD training).
  FM_1 refit helps (+4.82 vs buggy before fix) but the 50-sample buffer can't fully recover
  SROIE features after the encoder shifts during CORD training. Possible fixes: larger replay
  buffer (200+ samples), more refit epochs, or adaptive refit learning rate.
- **Files changed:** `scripts/train.py` (added `gate_mode` suffix to lexslot_fm run names);
  `doccl/methods/lexslot_fm.py` (train mode in refit). bd: **CLUE-85g**.
- **Blocked on:** user decision for next direction (increase refit budget, conditional gate,
  more seeds, or pivot).

## Ledger gate falsified standalone; strategy fork OPEN (2026-07-02 23:30)

- Ledger gate v2 (relational context keys): CORD OWN 76.1 (!) but cumulative
  FUNSD 21 / SROIE 23 << 60 bar; key-collision interference -18 on CORD.
  Memory-as-retention falsified at all 3 levels (parametric / +drift-ctrl /
  input-keyed). Script: scripts/lexical_ledger_gate.py (commit 12dfba2);
  results/ledger_gate{_v1_unigram,}.json.
- **USER DECISION PENDING** (asked, away): (a) analysis-paper + v3b spine
  [recommended], (b) push v3b vs replay, (c) xlingual ledger gate.
- Meanwhile: v3b seeds 7+123 running overnight (serves all forks).


## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Older history
Superseded/completed sections (June + early-July: LexMem pilot, baseline porting,
backbone-agnostic audit, HGT/DocMERGE/LCA/HRP branches, bug sagas) moved to
`STATE_ARCHIVE.md` (2026-07-04) to keep this ledger lean. All DONE/FALSIFIED — kept for provenance.
