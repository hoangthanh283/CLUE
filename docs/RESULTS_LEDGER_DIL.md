# Results Ledger — dil (FUNSD → SROIE → CORD), LayoutLMv3 primary

Compiled 2026-07-11 from `results/dil_*/metrics.json` on disk (authoritative) plus session
notes for deleted artifacts (marked †). Final-row = retention after the last task
[FUNSD, SROIE, CORD]. Multi-seed rows report mean±sd over seeds {42, 7, 123}.

**Budget caveat — do not compare across budgets.** "grid" = the thesis grid operating point
(val-F1 early stopping, 10-epoch cap). "5ep" = this week's capped diagnostic budget (seed 42
only). "1ep" = smoke reference. PLaR/falsification conclusions were drawn within-budget.

## 1. Oracles & raw-document replay (grid budget) — the upper bounds

| method | AA | BWT | final-row (s42) | note |
|---|---|---|---|---|
| joint (oracle) | 88.7±1.1 | 0.0 | [87.9, 83.4, 97.7] | pooled training, upper bound |
| ER (raw docs) | 87.8±1.4 | −2.8 | [84.4, 76.6, 97.7] | classic experience replay |
| ER + C-Flat | 88.4±0.2 | −1.8 | [87.6, 81.0, 97.2] | 2025 currency baseline |
| DER++ | 88.0±0.9 | −3.4 | [85.6, 81.5, 97.6] | logits+raw replay |

## 2. Classical / PEFT / prompt baselines (grid budget)

| method | AA | BWT | final-row (s42) | verdict |
|---|---|---|---|---|
| naive | 41.3±1.5 | −73.1 | [18.3, 4.3, 97.6] | the floor |
| EWC | 41.3±1.1 | −2.8* | [79.0, 4.2, 37.3] | *underfits late tasks; ≈ naive AA |
| LwF | 43.6±1.1 | −69.1 | [25.2, 4.8, 97.6] | regularization fails |
| O-LoRA | 40.8±0.3 | −56.8 | [30.1, 3.6, 89.3] | ≈ naive |
| CL-LoRA | 41.1±0.3 | −58.8 | [26.3, 4.8, 91.4] | ≈ naive |
| L2P | 30.4±0.4 | −11.5 | [13.3, 2.0, 77.3] | prompt family caps ~30 |
| DualPrompt | 29.9±0.6 | −11.0 | [14.0, 1.7, 74.9] | 〃 |
| CODA-Prompt | 32.5±1.2 | −10.5 | [15.0, 2.8, 82.5] | 〃 |

## 3. The program's proposed-then-falsified line (grid budget; evidence chain)

| method | AA | BWT | final-row (s42) | status |
|---|---|---|---|---|
| DocCL | 84.7±1.7 | −4.6 | [82.5, 76.0, 95.3] | works but carries a buffer |
| LexSlot (hybrid, `_off`) | 87.3±1.0 | −2.3 | [87.2, 80.7, 97.3] | thesis row; gate ineffective |
| LexSlot standalone | 42.2 | −67.2 | [25.1, 4.8, 96.8] | ≈ naive → dropped |
| LexSlot-FM | 42.4±8.2 | — | [20.5, 46.8, 68.6] | capacity-limited |
| LexMem v1 / v2 | 41.1 / 38.8 | — | — | ≈ naive |
| LexMem v3 / v3b | 60.8 / 66.1±3.6 | −24.5 | [84.7, 23.3, 89.9] | mid-task collapse |
| LexMem v5 (graph) | 63.2 | −29.3 | [86.6, **13.0**, 89.9] | falsified (both arms byte-identical) |
| gauss_replay (head Gaussians) | 63.6 | −27.2 | [84.3, 16.9, 89.7] | mid-task collapse |
| fisher_mask (exp 6, best p95) | 54.7 | −46.3 | [62.4, 5.8, 96.0] | law bends, doesn't break (45.0–54.7 over p) |

## 4. Latent replay — the family that works (exp 4/5 + this week's controls)

| variant | budget | AA | BWT | final-row | memory |
|---|---|---|---|---|---|
| **k4, 50 docs** | grid | **87.3** | −2.2 | [88.8, 76.7, 96.6] | ≈160 MB latents, no raw docs — the headline |
| k4, 5 docs | grid | 78.2±2.8 | −16 | [72.6, 57.6, 97.9] | s42/7/123 = 76.0/77.2/81.4 |
| k2, 5 docs | grid | 79.7 | −13.9 | [73.1, 68.5, 97.4] | |
| k4, 1 doc | grid | 61.2 | −40.7 | [62.5, 25.0, 96.3] | |
| k8, 0 docs (ctrl) | grid | 37.3 | −74.1 | [15.2, 2.4, 94.4] | freeze map alone ≈ naive |
| k8, 5 docs | **1ep** | 64.9 | −17.5 | [58.9, 49.3, 86.6] | ⚠ overwrote the converged k8/d5 dir (value lost) |
| k8, 4 docs | **5ep** | 63.8 | −33.4 | [60.8, 37.1, 93.6] | 13.4 MB — the consistency-control anchor |

## 5. THIS WEEK — buffer-free feature replay: FALSIFIED (all k=8, 5ep, seed 42)

| method (replay content) | AA | BWT | final-row | memory | vs anchor (63.8) |
|---|---|---|---|---|---|
| SpectralMemory "SLR" (synth, rank-16 subspace) | 41.9 | −66.3 | [29.3, **3.2**, 93.1] | 0.45 MB | −21.9 |
| AGLR-CL port (synth, full-d class Gaussians) | 39.4 | −69.8 | [22.4, **3.0**, 92.9] | 0.39 MB | −24.4 |
| CoresetMemory (REAL k-means centroids, 4 carriers)† | 36.7 | −74.0 | [14.5, **2.9**, 92.7] | 0.48 MB | −27.1 |
| CoresetMemory (real, 50 carriers, 16/class) | 39.7 | −69.3 | [22.8, **3.1**, 93.2] | 4.05 MB | −24.1 |

† artifact dir deleted (numbers from run log / STATE.md); regenerate for the paper table.

**The consistency law (the mechanism, single-variable):** 4 whole real docs = 63.8 vs 4
decoupled carriers + real centroid features = 36.7 → **+27 AA from (feature, position, label)
co-occurrence alone**. Not synthesis (real centroids fail), not rank (full-d fails; pooled
features are near-full-rank: r16=43%, r256=92% — but per-doc IS low-rank: r64=88%, r128=95%),
not carrier diversity (50 ≈ 4).

## 6. THIS WEEK — PLaR (proxy latent replay; ZERO private bytes; k=8, seed 42)

| variant | budget | AA | BWT | final-row | public mem |
|---|---|---|---|---|---|
| d5, hard pseudo-labels | 5ep | 45.8 | −59.6 | [40.2, 4.1, 93.1] | 16.7 MB |
| d50, hard | 5ep | 57.2 | −42.3 | [**73.3**, 5.1, 93.3] | 167 MB |
| d50, soft (dark knowledge) | 5ep | **58.9** | −40.7 | [**76.1**, 7.0, 93.5] | 167 MB |
| d50, soft | converged | 50.3 | −54.2 | [51.8, 5.5, 93.5] | convergence *hurts* proxy replay |

Read: FUNSD grounds *above* matched private replay (76.1 vs 60.8) with zero private storage;
SROIE never grounds (~5) at any count/format — settled diagnosis = **feature-region coverage**
(WildReceipt covers FUNSD's latent region under the task-0-tuned frozen trunk, not SROIE's).
Next: coverage probe → coverage-targeted proxy retrieval (see STATE.md).

## 7. dil_xlingual (7-language XFUND; grid budget; secondary)

| method | AA | | method | AA |
|---|---|---|---|---|
| DER++ | 80.3±0.2 | | EWC | 54.3±13 |
| ER | 79.5±1.1 | | O-LoRA | 47.8±2.5 |
| joint | 78.5±4.1 | | naive (BERT) | 44.7±1.6 |
| DocCL | 75.1±0.7 | | prompts (L2P/Dual/CODA) | 14–23 |
| LwF | 74.2±0.9 | | | |
| naive | 72.9±0.7 | | | |

## Cross-backbone spot-checks on disk (dil)
naive: LiLT 40.2±0.5, BERT 39.4±0.3 (≈ LayoutLMv3 41.3 — floor is architecture-general).
ER: LiLT 85.6±0.5. DER++: LiLT 84.3±0.3. EWC-LiLT 48.4±6.0.

## Provenance / hygiene
- Other scenarios (cil_cord, cil_funsd, …): regenerate via `analyze_results.py --source local`.
- ⚠ `dil_latent_replay_seed42` now holds the 1-epoch smoke (64.9); the pre-2026-07-10
  converged k8/d5 value was overwritten and must be re-run for the paper table.
- All 2026-07-10/11 session runs are seed-42 single-seed at capped budgets — re-run at the
  grid budget × 3 seeds (Vast.ai) before publication use.
