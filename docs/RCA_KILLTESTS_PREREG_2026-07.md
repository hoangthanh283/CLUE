# Pre-registered kill-tests for the readout-marginal-snap root cause

**Status:** PRE-REGISTERED 2026-07-17, before any kill-test run exists. Follows the RCA
(`docs/RCA_FORGETTING_BASELINES_2026-07.md`, root cause = readout-marginal snap) and the
judged method brainstorm (top-5 shortlist). All runs: dil (funsd→sroie→cord), seed 42,
LayoutLMv3, standard recipe (batch 2, grad-ckpt, val-F1 early stop), queued behind the
read-side chain. Adjudicate against these rules as written.

**Design note (deviation from the brainstorm's "BBSE" label):** BBSE assumes a FIXED
classifier with a stable channel P(pred|gold) — violated here by construction (the head's
change IS the pathology). The correct analog for a prior-shift-in-the-head is
**marginal-matching reweighting** of the final softmax using stored per-task gold label
marginals (9 floats/task — a legitimate, near-zero-byte memory object). Three variants,
decreasing information:

| variant | uses | task-ID needed |
|---|---|---|
| (a) marginal-match oracle | old task's stored gold marginal q_old; iterative per-class scaling until predicted marginal = q_old | yes (diagnostic ceiling) |
| (b) prior-ratio one-step | w(y) = q_old(y)/q_last(y), single reweight | yes |
| (c) per-doc EM (Saerens–Latinne) | q_last as source prior; EM re-estimates each document's own marginal from its ~512 tokens | **no** |

If the snap hypothesis is right, (b) already recovers most of what (a) does; (c) tests
deployability (a document is its own calibration batch).

## Decision rules

1. **Readout suite (a/b/c) on retrained naive** (final-boundary logits, ALL 3 tasks pooled
   — matching the RCA's AA convention — plus old-tasks-only, FULL mask):
   *(amended 2026-07-17 post red-team, before any run executed)*
   - Smoke gate: retrained naive final FULL AA must be 37.9 ± 2 (else stop — retrain drift).
   - **SUPPORT** requires BOTH: (i) pooled AA of the best variant recovers > 15 pts
     (37.9 → ≥ ~53; head-refit ceiling 55.3 is the reference), AND (ii) AA_old =
     mean(funsd, sroie) recovers > 15 pts from its own uncorrected value. Report both.
   - **Non-regression guard:** cord F1 must not drop > 2 pts under any variant.
   - **KILL the readout-repair family** if best variant's pooled AA recovers < 5 pts —
     split by the pre-registered **saturation check**: mean uncorrected KEY/HEADER
     probability mass on old-task tokens ≥ 1e-3 → "correction mechanism ineffective on
     live logits" (family dead); < 1e-3 → "information destroyed pre-correction" (needs a
     training-time intervention; family verdict deferred, mechanism untested).
   - Per-class prediction: VALUE and O recover strongly; KEY/HEADER recover only where the
     final model still assigns them non-trivial probability mass — CORD-era logits may have
     annihilated them (H2 terminal case). Report KEY/HEADER recovered-F1 explicitly.
   - Numerics guard: prior_ratio uses a Laplace-smoothed denominator (α=1e-4; CORD's gold
     marginal has exact-zero KEY/HEADER mass) and its max weight is reported; > 1e4 flags
     the variant's KEY/HEADER numbers as numerically fragile — read them only alongside
     per_doc_em's.
   - (c) vs (a): (c) landing within 5 AA pts of (a) = task-ID-free correction is viable
     (method-chapter candidate). (c) failing while (a)/(b) work = information bottleneck,
     not mechanism failure.
2. **Frozen-trunk naive** (`dil_naive_frozen_seed42_rca.json`): head causality **CONFIRMED**
   if KEY/HEADER extinction (final F1 < 5), O-collapse (O-row acc < 5%), and snap-cos
   (within 0.02) match full naive at the final boundary, with final AA within ±5.
   Materially less extinction → trunk contributes; hedge the root-cause statement.
3. **marginal_kl (training-time KL anchor, stores 9 floats/task):** the KL target is the
   mixture of PRIOR-task marginals only (current task excluded — its marginal is what CE
   already pulls toward; amended post red-team, which caught the current-task-inclusive
   implementation as a bug before any run). The AA 45–55 window is a SANITY CHECK
   (like the smoke gate), NOT part of the verdict — outside it, investigate before
   adjudicating. The adjudicable rule is the dissociation: recovery concentrated on
   O/VALUE confusions (H4′ = marginal-fixable) while KEY/HEADER stay < 5 (H2 =
   zero-re-exercise extinction is not a marginal problem). KEY/HEADER > 20 would REFUTE
   that part of H2's terminal-case reading.
4. **logit_adjust (balanced-softmax / cumulative-prior adjustment, buffer-free):**
   - SURVIVAL bar: KEY or HEADER final F1 > 20 (H2's own survival threshold) → genuine
     buffer-free method signal.
   - MANDATORY acquisition check: at-learning FULL F1 per task within 5 pts of naive's
     diag. [CORRECTION 2026-07-17 eve, at adjudication: the reference hard-coded here
     pre-run (88.5/84.0/97.6, taken from matrix.npy of a different naive run) was stale —
     Tier B naive's actual at-learning diagonal is 87.8/82.5/96.1. The logit_adjust guard
     passes under either reference; recorded per pre-registration discipline.] An
     EWC-style acquisition regression (sroie 41.9) voids any retention gain.
   - Training-time adjustment uses logits + τ(log q_task − log q_cum) inside the CE only
     (predict with raw logits ⇒ posterior under the cumulative prior); τ = 1.0 first, no
     tuning before the verdict.
5. All n=1 seed — verdicts provisional; only multi-seed if a verdict becomes load-bearing.

## Novelty framing (red-team-corrected, before results)

- `logit_adjust` = known mechanism (Menon et al. ICLR'21 logit adjustment / Ren et al.
  NeurIPS'20 balanced softmax — cited in the code docstring), tested in a new fixed-head
  DIL regime. NOT a novel mechanism claim.
- `per_doc_em` = Saerens–Latinne (2002) EM at document granularity; the delta is the
  calibration-unit size (~512 tokens/doc, task-ID-free), not the algorithm. Closest
  test-time precedent: TTLSA (arXiv:2211.15646).
- **Pre-written fallback framing for per-doc EM** (so the headline is honest regardless of
  numbers): it is a zero-byte, task-ID-free result for the RECOVERABLE subset
  (marginal-snap-caused, non-extinguished classes). H2's terminal case (KEY/HEADER at the
  final boundary) is an explicit non-goal — it requires re-exercise, not recalibration.
  A terminal-case method would be replay-adjacent, not this.
- BiC/WA/IL2M inapplicability to fixed-head DIL: verified clean (all require a
  new-class-introduction event this scenario lacks).

## Runs & artifacts

| test | run | output |
|---|---|---|
| suite train | `rca_readout_fixes.py --stage train` | `results/rca/killtests/naive_logits_task{0,1,2}.npz`, `train_meta.json` |
| suite correct | `rca_readout_fixes.py --stage correct` (CPU) | `results/rca/killtests/readout_fixes.json` + `.md` |
| frozen trunk | `rca_baselines.py --method naive --freeze-trunk` | `results/rca/dil_naive_frozen_seed42_rca.json` |
| KL anchor | `rca_baselines.py --method marginal_kl` | `results/rca/dil_marginal_kl_seed42_rca.json` |
| logit adjust | `rca_baselines.py --method logit_adjust` | `results/rca/dil_logit_adjust_seed42_rca.json` |

Chain: `scripts/run_rca_killtests.sh` (detached, pgrep-guarded behind the read-side chain,
resume-safe). ~3.5 GPU-h total. `rca_synthesize.py` picks the new JSONs up automatically.
