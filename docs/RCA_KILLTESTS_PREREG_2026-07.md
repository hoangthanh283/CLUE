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

1. **Readout suite (a/b/c) on retrained naive** (final-boundary logits, funsd+sroie, FULL):
   - Smoke gate: retrained naive final FULL AA must be 37.9 ± 2 (else stop — retrain drift).
   - **SUPPORT** if best variant recovers > 15 AA pts (37.9 → ≥ ~53; head-refit ceiling 55.3
     is the reference — recovery at/near it means the snap is ~the whole readout deficit).
   - **KILL the readout-repair family** if best variant recovers < 5 pts.
   - Per-class prediction: VALUE and O recover strongly; KEY/HEADER recover only where the
     final model still assigns them non-trivial probability mass — CORD-era logits may have
     annihilated them (H2 terminal case). Report KEY/HEADER recovered-F1 explicitly.
   - (c) vs (a): (c) landing within 5 AA pts of (a) = task-ID-free correction is viable
     (method-chapter candidate). (c) failing while (a)/(b) work = information bottleneck,
     not mechanism failure.
2. **Frozen-trunk naive** (`dil_naive_frozen_seed42_rca.json`): head causality **CONFIRMED**
   if KEY/HEADER extinction (final F1 < 5), O-collapse (O-row acc < 5%), and snap-cos
   (within 0.02) match full naive at the final boundary, with final AA within ±5.
   Materially less extinction → trunk contributes; hedge the root-cause statement.
3. **marginal_kl (training-time KL anchor, stores 9 floats/task):** expected AA 45–55.
   The pre-registered dissociation: recovery concentrated on O/VALUE confusions (H4′ =
   marginal-fixable) while KEY/HEADER stay < 5 (H2 = zero-re-exercise extinction is not a
   marginal problem). KEY/HEADER > 20 would REFUTE that part of H2's terminal-case reading.
4. **logit_adjust (balanced-softmax / cumulative-prior adjustment, buffer-free):**
   - SURVIVAL bar: KEY or HEADER final F1 > 20 (H2's own survival threshold) → genuine
     buffer-free method signal.
   - MANDATORY acquisition check: at-learning FULL F1 per task within 5 pts of naive's
     (88.5/84.0/97.6 diag) — an EWC-style acquisition regression (sroie 41.9) voids any
     retention gain.
   - Training-time adjustment uses logits + τ(log q_task − log q_cum) inside the CE only
     (predict with raw logits ⇒ posterior under the cumulative prior); τ = 1.0 first, no
     tuning before the verdict.
5. All n=1 seed — verdicts provisional; only multi-seed if a verdict becomes load-bearing.

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
