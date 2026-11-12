# Review log (Reviewer role, append-only)

## R1 (2026-11-11): audit of the rewritten paper

Snapshot audited: `HEAD = e3ca7b9` (P1 consistency pass included; P1 already fixed the "ER, matched
memory" label, "level with", "within three points" and "between naive and CoLaR" wordings, so those
are not re-listed). Tables were regenerated at 07:18; my independent recomputation (own script, raw
`metrics.json`, `statistics.stdev` = ddof 1) matches every cell of the current `vals.tex`,
`tab_2x2.tex` and `tab_main_clean.tex` to 0.05. Body ends on page 7, 0 undefined references in
main and supp.

### Verdict: **BLOCK**

One CRITICAL is a code bug with a 3-line fix. The other is pending data: every headline cell is
still legacy protocol. Five HIGH items are text or label errors that one Paper round can fix.

### CRITICAL

**CR1. Generator fallback breaks hard rule 6 and will soon change headline numbers without
warning.** `scripts/rca/make_clean_tables.py:39-50` (`pick`) and
`scripts/rca/make_retention_fig.py:31-32` take the first source with 3 seeds. Failing that, they
take the *first non-empty* source, and the clean source is always listed first. So a clean cell
with only 1 or 2 seeds beats a legacy cell with 2 seeds, or with 1. The docstring says the
opposite ("falls back to the legacy run until all three seeds exist"). Concretely:
- the v2 queue's next runs `cil_cord_colar_sbr_noreplay_seed42`, `cil_cord_colar_seed42` and
  `cil_cord_cl_lora_seed42` will each replace a legacy 2×2 cell (20.4; 28.8/31.9; 18.7/18.6) with a
  one-seed clean value that has no dagger. Through `\nNoBank`, `\nColar` and `\nClLora` that value
  also lands in the intro, the teaser bars and §3.3. The 2×2 then mixes the two protocols.
- it is already live in the full-label column: CoLaR-SBR prints `88.8` (one clean seed, s42) with
  no dagger, beside a dagger AA from different runs in the same row.

Fix: in `pick`, return a clean source only when `len(vs)==3`. Otherwise return the first
*legacy* non-empty source. For a column with no legacy source (full-label), print per-seed values
with an explicit n, or "--". Apply the same rule in `matrices()`.

**CR2. C5/C4 not yet resolved in the data. All headline cells are legacy protocol.** The Setup
(main.tex:93) describes the clean protocol, but every AA cell in `tab_2x2`, `tab_main_clean`,
Table 1 and the abstract/intro/teaser macros carries † (early stopping on the test split). The
baselines are also fp16 with a 100-epoch cap (ER, DER++, LwF, CPFD, IS3, naive, joint) or a
30-epoch cap (CL-LoRA, CODA), against fp32 at 10 epochs for the SBR/Recall/LexSlot rows. Clean
seeds on disk: one (CoLaR-SBR s42). Fix: the queue, not prose. At observed wall times
(1.5-2.0 h per CIL-CORD run; joint 0.6 h), the CIL-CORD 2×2 cells and references in v2 order are
about 30 h of work and the whole v2 queue about 100 h. That fits before the Nov 17 07:00 freeze
only if nothing fails. Regenerate after each cell completes. Any column still † at the freeze must
be stated as such in the caption.

### HIGH

**H1. False claim: "every method sits at 16–24 AA" under legacy scoring** (main.tex:138). Legacy
grid `results/cil_cord_<m>_seed*/metrics.json`, LayoutLMv3:

| Method | Legacy CIL-CORD AA |
|---|---|
| naive | 19.0 |
| LwF | 18.8 |
| DER++ | 18.2 |
| ER | 16.0 |
| CL-LoRA | 15.2 |
| LCA | 8.2 |
| IS3 | 4.4 |
| CPFD (2 seeds) | 1.4 |
| EWC | **0.5** (also in supp `tab:grid`) |
| CODA-Prompt | **0.0** |

Fix: "naive, LwF, ER and DER++ sit at 16–19 against 33 for joint training, and several
baselines collapse below that".

**H2. The 2×2 isolation row is not a single-variable contrast, yet the text says it is.** The
claims at issue:
- main.tex:151: "the same isolation with a blocked readout reaches 92.8".
- `tab:twobytwo` caption: "per-session low-rank adapters ... CL-LoRA with a shared softmax, SBR-Recall with the blocked readout".
- teaser caption: "per-session isolation is on par with it ... but only when the readout is blocked".

What the two cells actually ran:

| | CL-LoRA cell (`doccl/methods/cl_lora.py`, `configs/method/cl_lora.yaml`) | SBR-Recall cell |
|---|---|---|
| Adapters | O-LoRA orthogonal subspaces, rank 8, Q/K/V only | rank-64 Q/K/V+FFN deltas |
| Inference | all task adapters applied together, no per-session routing | one delta per block |
| Extra loss | LwF-style KD (kd_alpha 1.0) | none |
| Base | not frozen | frozen after session 1 |
| Precision, epoch cap | fp16, 30 | fp32, 10 |

Same sentence, separate literal error: "each session reaches 93 F1 when it is learned". The
CL-LoRA diagonals are s42 94.6/86.5/87.8/86.3/93.3 and s7 92.6/89.5/80.9/84.4/92.9.

Fix:
- either run Recall's own adapters with a single shared softmax block as the true control,
- or relabel the row "low-rank adapters, no memory", drop "the same isolation", and soften the teaser claim;
- and replace "93" with "81 to 95".

**H3. Unscoped universals that remain** (the C3 pattern):
- Abstract: background conflict is "an effect that no memory can repair". The evidence is one
  seed of oracle label refresh (61.2, `gates/.../cil_cord_colar_r_oracle_seed42`) and two fp16
  seeds of loss restriction (42.0). Scope it: "that no memory we tried repairs".
- main.tex:65: "any method that keeps updating a shared network without old-task supervision
  loses the old readout". Tested: EWC, LwF and five families, on DIL only. Scope it as main.tex:132
  already does.
- main.tex:67: "Each remedy is necessary and neither is sufficient". Row 3 of the paper's own 2×2
  contradicts this: isolation replaces the memory and reaches 92.8 with no memory. Fix: "each
  cause needs its own remedy; neither remedy is sufficient alone".

**H4. The full-label F1 companion metric (I2) is promised but essentially absent.** Contribution 1
("we pair it with a mask-free full-label F1") and the Scoring paragraph say it is reported. In
`tab:main` the column is "--" everywhere except CoLaR-SBR = 88.8, which is one clean seed from a
different protocol than its row's AA. Joint and naive, the two numbers that carry the "scoring
artefact" claim, have no full-label value. Fix: fill the column from the clean runs (naive, joint,
CoLaR, CoLaR-SBR, Recall) under the CR1 rule. If they are not ready by the freeze, change
contribution 1 to "we add".

**H5. Wrong collapse footnote, and an inconsistent zero-drop rule.** In `tab_main_clean`, IS3
CIL-CORD carries `c` ("one seed collapsed ... and is excluded"), but nothing was excluded: the
values are 5.60 and 7.67, and no s123 run exists. Cause: `make_clean_tables.py:176`
(`mark = "c" if dz else ""`) marks every drop-zero row whether or not a value was dropped. CPFD,
by contrast, keeps its zeros: CIL s7 0.00, Mixed s42 0.27, and DIL s42 0.00, which is fp32. That
contradicts the "fp16-only collapse" framing for CPFD. Fix: set `c` only when a value was actually
removed, and state the CPFD treatment (kept, since it collapses in fp32 too).

### MEDIUM

- **M1. The dagger hides three different legacy setups.** The `tab:main` caption says only
  "evaluation-split early stopping; fp16 for baselines". In fact:
  - the SBR, no-memory SBR, Recall, LexSlot and DIL-SBR cells are fp32 with a 10-epoch cap (gate runs);
  - the naive, joint, ER, DER++, LwF, CPFD and IS3 cells are fp16 with a 100-epoch cap, and CODA and CL-LoRA fp16 with a 30-epoch cap;
  - the DIL baselines come from the older grid.

  The 2×2 therefore compares fp16 CoLaR (28.8/31.9) with fp32 CoLaR-SBR. An fp32 CoLaR s42
  reading exists (34.06, `gates/cil_protocol_2026-11/cil_cord_colar_seed42_k4_d50_r128`). "SBR-Recall
  on par with joint" compares fp32 Recall with fp16 joint (fp32 joint s42 = 93.05). The C4 budget
  mismatch is still undisclosed in main. Fix: one caption sentence per dagger group (or the fp32
  cells). This disappears once CR2 lands.
- **M2. The cosine is attributed to the wrong methods.** "They fail in one way ... (cosine
  0.997–0.998 against 0.60)" (main.tex:132) is attributed to the five families. The source
  (`STATE.md:1056`, `results/rca/c_marginal_snap.csv`) measured naive and LwF. Fix: attribute
  correctly, or cite a measurement on the five families.
- **M3. The rank-8 Recall ablation quotes the run with memory.** "rank 8 ... gives 80.5" is
  `cil_cord_colar_sbr_recall_seed42` (docs_per_task 50). The memory-free rank-8 run, which matches
  the headline Recall setting, is 82.5 (`..._recall_d0_seed42`; diagonals 64.9/67.7 on sessions 3
  and 4). Fix: quote 82.5, or state the memory.
- **M4. Baseline rows never read clean runs.** `BASE` rows use `src(None, …)`, so the clean CL-LoRA
  and LwF runs the v2 queue produces will reach the 2×2 (`C['cllora']`) but not the `tab:main`
  CL-LoRA or LwF rows. The two tables would then disagree on the same cell. Fix: give the BASE
  rows clean names `cil_cord_<key>`.
- **M5. The supplement setup paragraph is stale** (supp.tex:55). It still says "batch size 8 ...
  bf16 autocast ... early stopping on task val F1, cap 100 (never binding)". That contradicts main
  Setup (batch 2; † = early stopping on the evaluation split; fp16 for the re-scored baselines) and
  the senior-review correction. Fix: rewrite it per dagger group.
- **M6. The modality-mask BWT triple cannot be reproduced as aggregated** ("BWT −81, −88, −79",
  main.tex:110). From `results/pilot/*.json`:
  - full: −89.9/−90.0/−89.5 → −89.8;
  - text+layout: −86.5/−86.8/−88.2 → −87.2;
  - text-only: −79.4/−78.7 plus s123, a dead run (matrix all 0.0) → −79.1 over two seeds.

  `STATE.md:261` already flags −81.3 as an unreconciled aggregation that included the `_ord210`
  runs. The dead text-only s123 is also inside the n=3 of supp `tab:depthprofile` ("text only,
  9505×"). Fix: report −89.8/−87.2/−79.1 (two seeds) and drop s123 from the depth table, or say so.
- **M7. The calibration bound is one fp16 run.** "changes AA by at most 0.6" comes from the
  fp16 rescore of SBR s42 (85.2 → 85.8, `STATE.md:22`). Add "(seed 42)".

### LOW

- L1. Table 1 (`tab:dil`) is typed by hand, which breaks rule 6. Three cells disagree with the
  files by more than 0.05: BROS LwF 35.2 (file 35.15), BROS DER++ 87.6 (87.55), BERT ER s.d. 1.0
  (0.95). Generate it instead.
- L2. supp.tex:112, CPFD on DIL: "reported as 6.2±7.2" should be 6.1±7.3 (0.00/4.25/14.19). "Final
  row zero on every domain for seeds 42 and 7" is false for s7 (final row 3.0/9.7/0.0).
- L3. main.tex:102: the collapse numbers 89.1→3.3, 83.9→2.1 and 88.9→6.6 trace to the pilot stream
  FUNSD→CORD→SROIE (`results/pilot/{c4_full,cl_lilt,cr_bros}`). The next sentence points to Table 1
  (DIL, FUNSD→SROIE→CORD), where the same quantity is 89.0→4.0, 80.2→3.2 and 88.5→8.6. Name the
  stream.
- L4. main.tex:95, "so an error is charged once". A false positive on a receipt that has no
  entity of its class's session is never charged, because that receipt is absent from that
  session's test split. Say "at most once".
- L5. The LexSlot row stores raw token ids (`store_input_ids: true`,
  `configs/method/colar_sbr_slot.yaml`). Disclose it in the Ablations paragraph. (CoLaR and SBR do
  not store them; checked.)
- L6. `macro()` writes `\n<Name>{0}` for empty cells (`\nSbrBros`, `\nRecallFull`, `\nJointFull`).
  None is used now, but any later use would print 0. Write `--` instead.
- L7. Limitations, "trail raw replay by 5–9 points on BERT and LiLT". The means are 5.3 and 8.8,
  but per seed BERT s42 trails by only 0.3 (76.4 vs 76.7). Say "on average".
- L8. supp `tab:ledger`: the CoLaR-R row (oracle 26.3/64.4) uses legacy scoring but is not marked,
  while main cites the restricted-scoring oracle 61.2.
- L9. The full-label hook swallows every exception (train.py:413). A failed evaluation leaves only
  a log warning and a "--" cell. That is acceptable, but grep each clean log for "full-label eval
  skipped".

### Part A: status of the senior review items

| Item | Status | Evidence / what is missing |
|---|---|---|
| C1 (p<0.01 on Fisher importance) | RESOLVED | main:110 drops the test; "weighting favours the head by construction"; supp:197 reports 3/12 rejections on importance and disowns it. |
| C1b (no-power generality test) | PARTLY | supp:197 calls the permutation test uninformative; `tab:depthprofile` reports the head-excluded profile (I reproduced it with `pilot_depth_profile.py`; it matches). No effect sizes. The "same profile on all four backbones" claim is visual only. The text-only row includes a dead seed (M6). |
| C2 (replay in the consolidation probe) | RESOLVED | Removed from main; supp:199 discloses the 200-document buffer and ER 87.9. |
| C3 (universal claims) | PARTLY | Scoped at main:132 and in contribution 2; still unscoped in the abstract ("no memory can repair"), main:65 ("any method") and main:67 ("each remedy is necessary") (H3). Recall on DIL is not run; argued only (main:195). |
| C4 (confounded Table 3) | PARTLY | fp16 disclosed, collapsed zeros footnoted (the IS3 footnote is wrong, H5), ER memory size now in the caption, n<3 printed per seed. Epoch-cap mismatch undisclosed, three legacy setups behind one dagger (M1). Fix pending in queue (CR2). |
| C5 (test-split model selection) | NOT RESOLVED (data) | Setup describes the clean protocol and the dagger correctly. Every headline cell is still legacy (CR2). supp:55 still says "val F1" (M5). |
| C6 (controls, prior art) | PARTLY | (iii) RESOLVED: Kim et al. 2022, EASE, InfLoRA, ICE and SpanKL cited at main:81 (keys present in the .bbl). (i) ER+SBR not run. (ii) No isolation baseline with task inference; the CL-LoRA cell is not one (H2). |
| I1 ("at the oracle") | PARTLY | "Oracle" and "bound" are gone from main and supp; Limitations: "a reference, not a bound". Joint not re-tuned. |
| I2 (restricted-scoring argument + full-label F1) | PARTLY | Argument given at main:95 (needs "at most once", L4). Full-label column empty except one seed (H4). |
| I3 (two Independent numbers) | RESOLVED | The Independent column is gone from main; one value, 88.8, in supp `tab:grid`. |
| I4 (SBR on DIL below CoLaR) | RESOLVED | main:206 states 86.2 vs 87.8 and its only two differences. |
| I5 (Recall cost and scope) | PARTLY | Costs: 54 MB (checked: 7.08M params × 4 × 2 B), about 3.7× (44/12 layer passes); order dependence, Mixed n=1 and the fixed-label limit are stated. Order permutation and Mixed seeds 7/123 not yet run (both in queue). |
| I6 (privacy, storage) | RESOLVED | "no raw pixels or token ids ... no formal privacy claim"; 2.7× is stated against the latent memory. CoLaR/SBR do not store ids (checked); LexSlot does (L5). |
| I7 (framing, method figure, loss) | RESOLVED | New title; teaser shows the 2×2 and an SBR schematic; eq. (1) to (3) give the loss and inference. |
| I8 (sample size, CIs) | PARTLY | Test-set sizes in Limitations; no confidence intervals. |
| Minor: legacy columns | RESOLVED | Moved to supp and labelled. |
| Minor: in-house names | RESOLVED | LexSlot defined inline; HGT, CUBER and LexMem only in the supp ledger. |
| Minor: "registered" | RESOLVED in main | main:132 "pass bar fixed before the run"; supp:105/167 "pre-registered" (acceptable). |
| Minor: "exact retention" | RESOLVED | "every curve is flat". |
| Minor: n=2 mean±s.d. | RESOLVED | Per-seed values printed. |
| Minor: code availability | PARTLY | supp:103 "will ship with the code"; no statement in main. |

### Part B: numbers

**Mismatches with the generated files (>0.05):** none at HEAD. (At 2f0a5e9, DER++ CIL-CORD showed
21.3/18.8; s123 = 21.07 had landed at 03:10, and the 07:18 regeneration fixed it to 20.4±1.4.)
Recomputed cells, per seed (42/7/123):

| Cell | Per seed | Mean (± s.d.) |
|---|---|---|
| CIL naive | 18.97/19.08/18.60 | 18.9±0.3 |
| CIL SBR (gate fp32) | 87.62/86.27/90.15 | 88.0±2.0 |
| CIL Recall (gate fp32) | 92.69/93.37/92.44 | 92.8±0.5 |
| CIL joint (fp16) | 92.84/90.48/92.43 | 91.9±1.3 |
| CIL ER (200 documents) | 26.91/22.05/24.01 | 24.3±2.4 |
| CIL LexSlot | 90.15/89.72/89.23 | 89.7±0.5 |
| CIL DER++ | 21.27/18.76/21.07 | 20.4±1.4 |
| CIL CoLaR | 28.83/31.87 | 2 seeds |
| CIL no-memory SBR | 20.36 | 1 seed |
| CIL CL-LoRA | 18.67/18.59 | 2 seeds |
| Mixed SBR (fp16) | 89.77/89.62 | 2 seeds |
| Mixed joint | 86.94/88.46 | 2 seeds |
| Mixed CoLaR | 75.56/74.55 | 2 seeds |
| Mixed ER | 75.58/73.33 | 2 seeds |
| Mixed LexSlot | 88.48/89.99/90.32 | 89.6±1.0 |
| Mixed Recall + memory | 85.94 | 1 seed |
| DIL naive | 40.06/43.03/40.85 | 41.3±1.5 |
| DIL CoLaR | 87.58/88.30/87.38 | 87.8±0.5 |
| DIL SBR (v2 rule) | 86.76/84.77/87.08 | 86.2±1.3 |
| DIL joint | 89.68/89.03/87.49 | 88.7±1.1 |
| DIL ER | 86.21/88.73/88.62 | 87.9±1.4 |
| DIL LexSlot | 87.24/87.25/86.62 | 87.0±0.4 |

All baseline rows also match: LwF and IS3 after the zero drop, CPFD, CODA-Prompt, LCA, CL-LoRA.
Registry mapping checked:
- no `_bert`/`_bros`/`_lilt` leak (DIL CoLaR uses the exact suffix `_k4_d50_r128`; legacy `""` suffixes are exact);
- the clean `*` globs cannot reach the BROS directories (`…_sbr_bros_seed*_bros`) or the order runs (`cil_cord_o2_*`);
- the Mixed SBR cell is v2-rule code (the rescore ran 11-06, after commit 608461c); an fp32 v2 s42 reading of 89.73 exists in the gate directory and is unused.

Mapping issues are the ones listed above: ER is 200 documents (now captioned), CL-LoRA is not a
matched isolation control (H2), and fp16 and fp32 cells are mixed under one dagger (M1).

**Prose literals (main.tex), traced:**

| Literal | Location | Source | Status |
|---|---|---|---|
| 33 (legacy joint) | 65, 95, 138 | legacy grid 33.43/33.57/32.94 = 33.3 | OK |
| 16–24 | 138 | legacy grid | **false (H1)** |
| 89.1→3.3, 83.9→2.1, 88.9→6.6 | 102 | `results/pilot` | OK; wrong stream implied (L3) |
| CKA 1.00/0.78/0.34/0.25/0.16 | 110 | ledger F1 | traced, not recomputed |
| BWT −81/−88/−79 | 110 | pilot JSONs | **not reproducible as aggregated (M6)** |
| "two to four orders of magnitude" | 110 | `tab:depthprofile` (111× to 9505×) | OK |
| 41.1/0.0, 38.8/−76.8, 0.19 | 112 | ledger F2 (one seed) | traced |
| 21.6/21.9, 43.9, 39.8, 42.2, 66.0, 88 | 132 | supp ledger / ledger F5 | OK |
| cosine 0.997–0.998 / 0.60 | 132 | `STATE.md:1056` (naive and LwF) | **misattributed (M2)** |
| ≤−0.1 AA, mass about 10⁻⁶ | 132 | ledger F5a | traced |
| 2.5, 17, −11.9/−2.6/+3.5/+5.2 | 134 | recomputed from supp `tab:grid` | OK |
| 61.2 (seed 42) | 140 | gate oracle 61.17 | OK |
| 42.0 (two seeds) | 140 | rescore sliced 44.17/39.85 (fp16) | OK |
| "about half" | 140 | 0.46–0.51 depending on which seed-42 runs are used | OK |
| 93 (CL-LoRA per session) | 151 | diagonals 80.9–94.6 | **false (H2)** |
| 87.3, 41.9, 39.4, 36.7–39.7, 63.8, 36.7, 27 | 165 | ledger F6/F6a | traced |
| ViT 87.5/65.0, 80.0/55.1, 88.3/72.3 | 165 | ledger F9/F9a | traced |
| 60 MB, 2.7×, 163 MB, 80.6 at 32 MB | 167 | ledger F6b | traced |
| 44.7, 85.9 (Mixed Recall) | 195 | gate r64b 44.66, r64c 85.94 | OK (r64 = 62.07 is the bug run, ledger F17b) |
| 14.8 | 212 | gate `sbr_frozen` 14.77 | OK |
| ≤0.6 (calibration) | 212 | `STATE.md:22`, one fp16 seed | M7 |
| 80.5 (rank 8) | 212 | run with 50-document memory | **M3** |
| "near 65" | 212 | 61.5/68.0 diagonals | OK |
| 96 MB | 214 | `replay_memory_bytes` 100,672,250 B = 96.0 MiB | OK |
| 54 MB | 214 | 7.08M params × 4 × 2 B = 54.0 MiB | OK |
| 3.7× | 214 | (4 + 8×5)/12 = 3.67 | OK |
| 5–9 | 218 | DIL CoLaR vs ER: BERT 5.3, LiLT 8.8, BROS 0.5 | OK on means (L7) |
| 50–347, 149/50, 800/100/100, 626/347 | Setup | dataset sizes | OK |

**Comparative sentences, checked per seed:**

| Sentence | Per-seed check | Verdict |
|---|---|---|
| Recall "on par with joint" | −0.15/+2.89/+0.01 (fp32 vs fp16) | OK |
| CoLaR-SBR "within a few points of joint" | −5.2/−4.2/−2.3 | OK; "2 to 5 points" would be more precise |
| CoLaR-SBR "reaches" joint on Mixed | +2.8/+1.2 | OK |
| CoLaR "matches raw-exemplar replay" on DIL | +1.4/−0.4/−1.2 (but 150 vs 200 documents) | OK |
| "at or below CoLaR (30.4)" | every baseline seed ≤ 26.9 | OK |
| Table 1 "within one point of joint on average" | means | OK |
| LexSlot "within two points ... narrows the spread" | means and s.d. | OK |
| Retention figure | per-seed shapes all 5×5 | OK |

**Clean-protocol runs** (`results/clean_2026-11/`):

| Run | Settings confirmed in log + `metrics.json` | AA | BWT | Full-label F1 |
|---|---|---|---|---|
| `cil_cord_smoke_naive_e1_seed42` | amp false, val_frac 0.1, epochs 1 (smoke) | 18.05 | −76.09 | 12.55 |
| `cil_cord_colar_sbr_seed42` | amp false, fp16 false, val_frac 0.1, epochs 10 (stopped at 5/5/5/6/6), docs 50, k 4, r 128 | 87.76 | −2.41 | 88.83 |

For the SBR s42 run, the batches per epoch (360/172/247/326/351) equal 90% of the session pools
(800/382/549/723/780) at batch 2. That confirms the held-out split comes from training data. Its
legacy counterpart was 87.62, so held-out early stopping moves AA by +0.14 on this seed.
`cil_cord_colar_sbr_seed7` is still running (no `metrics.json`).

### Part C: code review (`git diff f82c3d1^..HEAD -- scripts/ doccl/ tests/ configs/`)

- Full-label hook (`train.py:391-416`, `scenarios.py`): **correct.**
  - It runs after the session loop, outside `restrict_preds`, so `model._allowed` is None and no restriction is active.
  - The test set is remapped with `snapshots[-1]`, the final head map, so it is right for the permuted orders o2/o3 too.
  - `ColarSBR.evaluate` does not use `tid` for decisions. It overwrites `eval_dump[last]`, but `eval_dump.pt` is saved before the hook runs, so nothing is lost.
  - It is wired for `cil_cord*` only; on Mixed and DIL it is a no-op, as documented. The joint path is covered as well.
  - Remaining risk: exceptions are swallowed (L9).
- `split_val`: **no test leakage.** The validation set is drawn only from the session's training
  set, through `random_split` with `Generator(seed + task_idx)`, so the split is identical across
  methods for a given scenario and seed. Joint uses `seed` on the pooled set, a different but
  legitimate split. Early stopping runs under the same session restriction as evaluation.
- Generators: CR1 (fallback order), H5 (unconditional `c` mark), M4 (BASE rows without clean
  sources), L6 (empty macros = 0). Retention figure: inherits CR1; matrix shapes are uniform
  (5×5), so the averaging is valid.
- `pilot_depth_profile.py`: reproduces the supp table exactly (n=3 per condition). The text-only
  condition includes the dead s123 run (M6).
- Queue drivers v1/v2: the resume check `ls $P/${NAME}*/metrics.json` is prefix-safe for the
  names in use. The BROS run directories carry a `_bros` suffix and cannot collide with the
  LayoutLMv3 globs. The v2 CL-LoRA clean cell uses the config defaults, which inherits H2: it is
  still not a matched control for SBR-Recall.

## R1b (2026-11-11): bibliography audit

Scope: all 52 keys cited in `paper_cvpr/main.tex`, `supp.tex` and `fig/*.tex` at HEAD 94c8068.
`refs.bib` has not changed since baff53b. Keys that are in `refs.bib` but never cited (e.g.
`le2025wavecre`, `zhang2025clcrossvqa`, `ge2025dmole`) are not in the built list and were not
audited.

Method: for each record I checked title, author list and order, venue, year and pages against the
following sources (in this order of preference):
- **DOI record:** the publisher-deposited Crossref metadata, at `https://doi.org/<DOI>` (38 entries);
- **arXiv:** the arXiv API record (author order, plus the venue note in the comments field);
- **PMLR:** the proceedings page (`zhai2023investigating`);
- **ICLR/OpenReview pages:** for `lca2026`, `ramasesh2021anatomy` and `yoon2018den`.

LoRA (ICLR 2022), OGD (AISTATS 2020) and Learning without Forgetting (LwF, ECCV 2016) have their
venue confirmed by arXiv and Google Research pages, not by opening the proceedings page.

**Verdict for the bibliography: no non-existent reference (no CRITICAL).** One entry is WRONG
(venue), four need MINOR corrections, and 47 are OK.

### Entries that need a change

| Key | Status | What is wrong | Correct record | Source |
|---|---|---|---|---|
| `gencner` | **WRONG (venue)** | Cited as an arXiv preprint, but it is published (arXiv comment: "Accepted by IJCNN 2025") | Yang, Yawen and Ma, Fukun and Meng, Shiao and Liu, Aiwei and Wen, Lijie. "GenCNER: A Generative Framework for Continual Named Entity Recognition". 2025 International Joint Conference on Neural Networks (IJCNN), 2025, pp. 1–9, doi:10.1109/IJCNN64981.2025.11227587 (arXiv:2510.11444) | https://doi.org/10.1109/IJCNN64981.2025.11227587 ; https://arxiv.org/abs/2510.11444 |
| `lca2026` | MINOR (author name) | Second author is given as "Vargas, Danilo" | Tran, Tung and Vasconcellos Vargas, Danilo and Than, Khoat. "LCA: Local Classifier Alignment for Continual Learning". ICLR 2026 (arXiv:2603.09888) | https://arxiv.org/abs/2603.09888 ; https://iclr.cc/virtual/2026/poster/10011599 |
| `kumar2023protoner` | MINOR (year, series) | The publisher record dates the volume 2024 and places it in a series; the bib entry has 2023 and no series | Kumar, Ritesh and Goyal, Saurabh and Verma, Ashish and Isahagian, Vatche. "ProtoNER: Few Shot Incremental Learning for Named Entity Recognition Using Prototypical Networks". Business Process Management Workshops (BPM 2023), Lecture Notes in Business Information Processing, Springer, 2024, pp. 70–82, doi:10.1007/978-3-031-50974-2_6 | https://doi.org/10.1007/978-3-031-50974-2_6 ; https://arxiv.org/abs/2310.02372 |
| `gdibench` | MINOR (authors truncated) | The first 10 of 16 authors are correct and in order, then "and others". No venue found: it is still arXiv only | Li, Siqi; Shen, Yufan; Chen, Xiangnan; Chen, Jiayi; Ju, Hengwei; Duan, Haodong; Mao, Song; Zhou, Hongbin; Zhang, Bo; Fu, Bin; Cai, Pinlong; Wen, Licheng; Shi, Botian; Liu, Yong; Cai, Xinyu; Qiao, Yu. arXiv:2505.00063, 2025 | https://arxiv.org/abs/2505.00063 |
| `zhang2023slca` | MINOR (venue string) | booktitle is just "ICCV" | Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19091–19101, doi:10.1109/ICCV51070.2023.01754 | https://doi.org/10.1109/ICCV51070.2023.01754 |

Optional page and volume completions, none of which is wrong as written:
- `wang2024survey`: TPAMI 46(8):5362–5383, doi:10.1109/TPAMI.2024.3367329;
- `hong2022bros`: AAAI 36(10):10767–10775;
- `huang2022layoutlmv3`: 4083–4091;
- `he2025cllora`: 30534–30544;
- `liang2024inflora`: 23638–23647;
- `bian2024cflat`: NeurIPS 37, 7608–7630.

### Entries verified OK (47)

| Group | Keys |
|---|---|
| Regularisation and replay | aljundi2018mas, buzzega2020der, kirkpatrick2017ewc, li2017lwf (ECCV 2016), lopezpaz2017gem, rebuffi2017icarl, rolnick2019er, zenke2017si |
| Optimisation geometry | farajtabar2020ogd, saha2021gpm, bian2024cflat |
| Pre-trained-model, prompt and adapter methods | wang2022l2p, wang2022dualprompt, smith2023coda, wang2023olora, he2025cllora, mcdonnell2023ranpac, zhou2024ease, liang2024inflora |
| Architectural isolation | mallya2018packnet, yoon2018den |
| Surveys and theory | masana2022class (TPAMI 2023, 45(5)), wang2024survey, kim2022theoretical |
| Exemplar-free methods | zhu2021pass, petit2023fetril, goswami2023fecam |
| Latent and generative replay | hayes2020remind, pellegrini2020latent, vandeven2020brain (Nat. Commun. 11:4069) |
| Forgetting analysis and tools | ramasesh2021anatomy, zhai2023investigating (CPAL, PMLR 234:202–227), kornblith2019similarity, mccloskey1989catastrophic, hu2022lora |
| Document encoders | huang2022layoutlmv3, wang2022lilt, hong2022bros |
| Continual NER | zhang2023cpfd, zheng2022cfner (EMNLP 2022, 3602–3615), zhang2023spankl (AAAI 37(11):13993–14001), qiu2024is3 (Findings ACL 2024, 777–791), liu2023ice (Findings ACL 2023, 2241–2257) |
| Continual relation extraction | han2020emar, cui2021rpcre, zhao2022crl |
| Document continual learning | minouei2022table (Applied Sciences 12(18):8969, doi:10.3390/app12188969) |

### Sentence checks: does the cited paper say what the sentence attributes to it?

| # | Sentence (main.tex Related Work) | Verdict | Reason (source) |
|---|---|---|---|
| 1 | Current wording: "ICE trains per-session classifiers on a frozen encoder ... neither replays stored examples" | SUPPORTED | ICE stands for "Individual Classifiers with Frozen Feature Extractor": it trains one classifier per session, in the setting where no old-class samples are stored (ACL Anthology 2023.findings-acl.141). |
| 1b | Earlier wording: "neither trains the trunk" | NOT SUPPORTED for SpanKL | SpanKL fine-tunes a shared bert-large-cased contextual encoder with AdamW (arXiv:2302.12200, Sec. on implementation). The current wording, "per-type span heads with distillation; neither replays stored examples", is SUPPORTED: SpanKL uses a span layer per task, per-type sigmoid binary cross-entropy, and Bernoulli-KL distillation, with no rehearsal. |
| 2 | "Document-level CL is nascent: ... ProtoNER addresses few-shot incremental NER" | SUPPORTED, but undersold | ProtoNER does key-value/NER extraction on visually rich documents with LayoutLMv2/v3/LiLT-type models and adds classes from about 30 samples, using prototypical networks and a hybrid loss, without retaining the original training data (arXiv:2310.02372). It is the closest prior work to this paper's setting (class-incremental KIE on LayoutLM-family encoders). Suggest: "ProtoNER adds new KIE classes to a LayoutLM-family model from few labelled documents with prototypical networks". |
| 3 | "REMIND quantises later-layer features, Latent Replay banks activations at a chosen layer ... **Each stores one summary per token or class** and targets image labels" | **NOT SUPPORTED, HIGH** | REMIND stores, *per training image*, product-quantised codes of a *mid-level* CNN feature map (an m×m×s integer array) and replays them through the plastic upper layers (arXiv:1910.02509, Sec. 3). Latent Replay stores per-pattern "activation volumes" at an intermediate layer (arXiv:1912.01100). Neither keeps one summary per class or token: both keep one compressed or raw entry per example. That is CoLaR's design too, so the stated distinction from the closest prior work is false. What does differ: a document has one label per token bound to a position, and per-document truncated SVD replaces product quantisation. Fix the sentence to say that, and write "mid-level", not "later-layer". |
| 4 | "generative replay samples a per-class model" | NOT SUPPORTED | Brain-inspired replay uses a *single* conditional VAE whose latent prior is a Gaussian mixture with one mode per class, and it replays hidden-level representations (Nat. Commun. 11:4069, PMC7426273). Fix: "samples hidden representations from one class-conditional generator". |
| 5 | "Zhai et al. report the same [forgetting at the layers nearest the output] for multimodal LLMs" | NOT SUPPORTED | Zhai et al. evaluate forgetting with EMT, treating the multimodal LLM as an image classifier, and find degradation and hallucination as fine-tuning proceeds, "even when the image encoder remains frozen". There is no layer-wise localisation (arXiv:2309.10313; PMLR 234). Fix: "and Zhai et al. find forgetting in multimodal-LLM fine-tuning even with the image encoder frozen", and drop "the same". |
| 6 | "Kim et al. decompose class-incremental learning into within-task prediction and task-identity prediction, and tie the latter to OOD detection" | SUPPORTED | Abstract: "decomposed into ... Within-task Prediction (WP) and Task-id Prediction (TP)"; "TP is correlated with OOD detection" (NeurIPS 2022 proceedings). |
| 7 | "recent adapter methods train one module per task and infer which to use at test time (EASE, InfLoRA)" | **NOT SUPPORTED** | EASE trains one adapter per task but concatenates all of them for "joint decision-making across multiple subspaces": there is no task inference (arXiv:2403.12030). InfLoRA reparameterises the pre-trained weights within an interference-free subspace, so a single model runs at test time with no module selection (arXiv:2404.00228). Fix: drop "and infer which to use", or cite methods that do infer task identity. HiDe-Prompt (NeurIPS 2023) and MoE-Adapters (CVPR 2024) are candidates; both are named from memory and must be verified before adding. |
| 8 | "GDI-Bench freezes layers heuristically" | NOT SUPPORTED as worded | GDI-Model's Layer-wise Adaptive Freeze-Tuning (LW-AFT) freezes about 99% of individual *parameters*, not layers. Which ones stay trainable is set by update magnitude after a 10% expert fine-tune, with a per-layer budget (arXiv:2505.00063v2, Sec. 4). It is also a supervised fine-tuning study of a multimodal LLM, not a continual-learning method. Fix: "GDI-Bench mitigates forgetting in multimodal-LLM fine-tuning by freezing most parameters, chosen by their update magnitude". |
| 9 | "continual table detection uses plain ER" | SUPPORTED | Abstract: "We employed an effective technique called experience replay" (doi:10.3390/app12188969). |
| 10 | "Ramasesh et al. localise forgetting in image classifiers to the layers nearest the output" | SUPPORTED | "deeper layers are disproportionately the source of forgetting"; "higher layers refer to layers closest to the output" (arXiv:2007.07400). |
| 11 | "LCA: merge models and realign the classifier" | SUPPORTED | It merges parameter-efficient fine-tuning (PEFT) parameters, then realigns the classifier on samples drawn from per-class Gaussians (arXiv:2603.09888). |
| 12 | "CPFD, CFNER: distillation and pseudo-labels" | SUPPORTED for CPFD; UNCLEAR for CFNER | CPFD uses pooled-feature distillation and confidence-based pseudo-labelling. CFNER's abstract describes causal-effect distillation from Other-class tokens plus curriculum learning for label noise; pseudo-labels are not named in the abstract (arXiv:2210.03980). |
| 13 | "GenCNER: span generation" | SUPPORTED loosely | It casts continual NER as "entity triplet sequence generation" with pseudo-labelling and KD. "Entity-triplet generation" would be more exact. |
| 14 | "EMAR, RP-CRE, CRL: episodic memory with prototypes ... all sentence-level and text-only" | SUPPORTED | Memory replay with prototypes or memory-based nearest-class-mean (NCM) classification. All are relation-extraction work on sentences. |

## R2 (2026-11-11): confirmation audit

Snapshot: `HEAD = 348b00f`, `main.pdf` built 07:55 (after `main.tex` 07:55:09), `supp.pdf` 07:44,
generated tables 07:33. Recomputation: own script, raw `metrics.json`, `statistics.stdev`
(ddof 1). No model, dataset, torch or pytest touched.

### Verdict: **BLOCK** (one HIGH, text only, one Paper round)

CR2 (legacy headline cells) is waived for draft uploads and the abstract clause "an effect that
no memory can repair" is with the user; neither is counted. What blocks is **N1**: the 2×2's
third row, its caption, §4.3 and the teaser describe the CL-LoRA cell as per-session adapters
on a trained base. The code runs **one** LoRA adapter shared by all sessions, on a **frozen**
pre-trained encoder. (R1's H2 table had the same error, "all task adapters applied together;
base not frozen", and the Paper agent copied it. I correct it here.) Every R1 CRITICAL and HIGH
item is otherwise fixed, and all numbers match.

### Part A: status of the R1 and R1b items

| Item | Status | Evidence (current text / code) |
|---|---|---|
| CR1 | CONFIRMED FIXED | `choose()` (make_clean_tables.py:44-54) returns a clean source only when `len(vs) == 3`, else the first non-empty *legacy* source; `pick()` and `matrices()` (make_retention_fig.py:22-34) both call it. Full-label: `pick_clean_only` + `fmt(show_n=True)` prints per-seed values with `^{n=k}` when n<3, mean±s.d. only at n=3. Residual: L-j (unused `\v*Full` macros). |
| CR2 | WAIVED (data) | Every AA cell still †; 2 of 3 clean CoLaR-SBR seeds on disk, s123 running. |
| H1 | CONFIRMED FIXED | main:128 "naive training, LwF, ER and DER++ sit at 16--19 AA against 33 for joint training, and several baselines collapse below that band". Per seed ER s7 = 14.4 (L-e). |
| H2 | PARTLY; see **N1** | "81 to 95" fixed (diagonals 80.9 to 94.6); row relabelled; teaser softened ("a different adapter method"); main:141 adds "rather than isolating the readout alone". But the new caption text is factually wrong about the CL-LoRA cell (N1). |
| H3 (body) | CONFIRMED FIXED | main:65 "on a fixed label space every method we tested that keeps updating a shared network without old-task supervision loses the old readout"; main:67 "Neither remedy is sufficient alone and each cause needs its own remedy". Echo of the OPEN abstract clause at main:65, "Memory ... is blind to the second" (L-o). |
| H4 | CONFIRMED FIXED (wording) | C1 "we add a mask-free full-label F1, reported for clean-protocol runs as they complete"; Scoring "with the number of seeds marked"; cell `88.8^{n=1}`. Stale: clean s7 (91.19) landed at 07:49, after the 07:33 generation (L-m). Joint and naive full-label are still "--". |
| H5 | CONFIRMED FIXED | `drops.append(len(kept) < len(vs))`, so the `c` mark appears only when a value was removed; IS3 is now unmarked (`5.6/7.7`); LwF keeps `c`. Footnote: "LwF only ... CPFD and IS3 keep their zero seeds". |
| M1 | CONFIRMED FIXED | Setup:93 lists the early-stopping, epoch-cap and precision groups and the 2×2's precision split; supp `tab:legacyprotocol`. Residual: L-f (the epoch-cap list does not hold on DIL). |
| M2 | CONFIRMED FIXED | main:122 "Where we measured the mechanism, on naive training and LwF". |
| M3 | CONFIRMED FIXED | main:204 "rank 8 ... gives 82.5 (seed 42) ... stuck at 65 and 68" (`recall_d0_seed42`: 82.50, diag 64.9/67.7). |
| M4 | CONFIRMED FIXED | BASE rows: `src(f"cil_cord_{key}", (RESC, ...))`, so clean names match the queue (`cil_cord_lwf`, `cil_cord_cl_lora`). |
| M5 | PARTLY | supp Hyperparameters fixed ("differ between groups ... `tab:legacyprotocol` lists them"). Still stale: supp `tab:legacy` caption "CoLaR, CoLaR-R and the baselines use 10 training epochs per task with early stopping". The June-grid naive/LwF/ER/DER++/CODA/CL-LoRA/joint rows saved no `epochs` (script default 100/30); only CPFD/IS3/LCA show 10. This contradicts the supplement's own `tab:legacyprotocol`. |
| M6 | CONFIRMED FIXED | main:110 "BWT −89.8 and −87.2 over three seeds, −79.1 over the two live text-only seeds"; `tab:depthprofile` text-only n=2. Re-derived exactly (Part B.5). |
| M7 | CONFIRMED FIXED | main:204 "at most 0.6 (seed 42)". |
| L1 | PARTLY | Main Table 1 is now generated (`tab_dil_grid.tex`) and matches the files. Regression: supp `tab:grid` is still typed by hand with the old values (BROS LwF 35.2, BROS DER++ 87.6, BERT ER ±1.0), so it disagrees with main Table 1 (35.1, 87.5, 0.9) on the same runs (L-a). |
| L2 | CONFIRMED FIXED | supp:115 "reported as 6.1±7.3"; "zero on every domain for seed 42 and 3.0/9.7/0.0 for seed 7". |
| L3 | CONFIRMED FIXED | main:102 "on the stream FUNSD→CORD→SROIE"; "Table 1 gives the DIL picture". |
| L4 | CONFIRMED FIXED | main:95 "charged at most once"; C1 the same. |
| L5 | CONFIRMED FIXED | main:204 "Unlike CoLaR, LexSlot stores the token ids of its documents." |
| L6 | CONFIRMED FIXED | `macro()` writes `--` for empty cells. |
| L7 | CONFIRMED FIXED | main:210 "by 5--9 points on average". |
| L8 | CONFIRMED FIXED | `tab:ledger` ‡ "legacy scoring". |
| L9 | CONFIRMED (monitor) | Both clean logs print "Full-label final F1 (unmasked test set)"; no "skipped". |
| R1b `gencner` (WRONG) | CONFIRMED FIXED | .bbl: "2025 International Joint Conference on Neural Networks (IJCNN), pages 1--9, 2025. arXiv:2510.11444". |
| R1b `lca2026` | CONFIRMED FIXED | "Danilo Vasconcellos~Vargas". |
| R1b `kumar2023protoner` | CONFIRMED FIXED | year 2024; the bib has series and volume 492, and the bst does not print the series. |
| R1b `gdibench` | CONFIRMED FIXED | all 16 authors, arXiv:2505.00063. |
| R1b `zhang2023slca` | CONFIRMED FIXED | "Proceedings of the IEEE/CVF ICCV, pages 19091--19101". |
| R1b #1b SpanKL | CONFIRMED FIXED | "SpanKL uses per-type span heads with distillation; neither replays stored examples". |
| R1b #3 REMIND/Latent Replay (HIGH) | CONFIRMED FIXED | "REMIND stores product-quantised mid-level CNN feature maps for each training image ... Like CoLaR, REMIND and Latent Replay keep one entry per example, but each of their examples carries a single label". Consistent with R1b's facts. |
| R1b #4 generative replay | CONFIRMED FIXED | "samples hidden representations from one class-conditional generator". |
| R1b #5 Zhai et al. | CONFIRMED FIXED | "find that multimodal LLMs forget under fine-tuning even when the image encoder is frozen; our drift analysis agrees with the former". |
| R1b #7 EASE/InfLoRA | CONFIRMED FIXED | "need no task identity at test time, either by combining the features of all modules (EASE) or by folding them into the pre-trained weights (InfLoRA)". Consistent with R1b's facts. |
| R1b #8 GDI-Bench | CONFIRMED FIXED | "freezing most parameters, chosen by their update magnitude". |
| R1b #2 ProtoNER (undersold) | FIXED, but a new claim was added (**N5**) | "ProtoNER, the closest work to ours, adds new key-information classes to a LayoutLM-family model from a few labelled documents with prototypical networks". New: "none addresses the background class under label growth". Checked against the paper; it is not supported. |
| R1b #12 CFNER, #13 GenCNER | CONFIRMED FIXED | "causal-effect distillation with curriculum learning (CFNER) or entity-triplet generation". |

### New findings

#### HIGH

**N1. The CL-LoRA cell of the 2×2 is misdescribed.** The code runs neither per-session adapters
nor a trained base.

Where the claim appears:
- `tab_2x2.tex` row label (generator line 199): "Per-session low-rank adapters, no memory".
- `tab:twobytwo` caption: "rank-8 orthogonal adapters on the attention projections, all applied together at inference, distillation loss, base trained".
- main:141: "The third row replaces the memory by per-session low-rank adapters: CL-LoRA, with a shared softmax, collapses".
- teaser bar-group label "per-session adapters", under the CL-LoRA bar too.

Evidence:
- `doccl/methods/o_lora.py:54-66` builds one `LoraConfig` and calls `get_peft_model` once. No
  `add_adapter` / `set_adapter` exists anywhere in `doccl/` or `scripts/train.py`. The one
  `default` adapter is trained in every session. `_ortho_loss` penalises its current `A`
  against **snapshots of itself** taken at earlier session ends (`after_task`).
- PEFT freezes every non-LoRA parameter except the head (`modules_to_save`). The legacy runs
  confirm it: `results/rescore/cil_cord_cl_lora_seed{42,7}` `trainable_params = 489,277`.
  That is exactly 36 rank-8 Q/K/V sites × 12,288 (= 442,368) plus the 61-way head (46,909);
  `total_params` exceeds naive's by exactly 442,368. So the encoder is the frozen pre-trained
  checkpoint from session 1 onward.
- The paper contradicts itself: Related Work lists CL-LoRA among "methods over a frozen
  backbone".
- The module docstring ("per-task LoRA on Q/K/V") does not match the code.

Fix (text, generator label; Paper owns both):
- Row label: "Low-rank adapters, no memory" (or "Low-rank adaptation of a frozen trunk, no memory").
- Caption, CL-LoRA clause: "CL-LoRA (our port: one rank-8 adapter on Q/K/V of the frozen
  pre-trained encoder, shared by all sessions and trained with distillation and an
  orthogonality penalty against its own earlier states; fp16 or fp32 by seed, 30-epoch cap)
  with a shared softmax".
- main:141: "CL-LoRA, one low-rank adapter shared by all sessions with a shared softmax, collapses".
- Teaser label: "low-rank adapters".

Keep "SBR-Recall (one rank-64 delta per session ...)" as is: it is correct (Part C).

#### MEDIUM

- **N2. CL-LoRA / O-LoRA ports are not per-task (fidelity; could change numbers).** Same code
  as N1. Published O-LoRA/CL-LoRA keep one frozen adapter per task; this port retrains one
  adapter and pushes it away from its own past states. That penalty actively erases the old
  adapter. This affects:
  - every CL-LoRA cell (tab:main CIL/Mixed/DIL; supp `tab:prompt_lora` O-LoRA and CL-LoRA rows; image tables);
  - the supp sentence "LoRA adapters acquire but forget as much as naive training does, because the adapter does not protect the head" (the single overwritten adapter is an equally likely cause).

  Fix:
  - disclose in Setup or supp ("our port keeps a single adapter across sessions");
  - drop the causal clause;
  - long term, a per-task port.

  The queued capacity-matched run inherits this (Part C).
- **N3. The Limitations sentence will become false on the next regeneration.** main:210: "Cells
  marked † ... are not compared against clean-protocol cells in the text."
  - When `cil_cord_colar_sbr_seed123` finishes (running since 07:49), `choose()` switches `\vSbr` to clean.
  - The abstract ("attain `\nSbr` against `\nJoint`"), intro, §4.3, §6 and the 2×2 will then compare a clean cell with † cells.
  - Fix now: "A cell never combines the two protocols; † cells and clean cells are compared only where the text says so."
- **N4. "EWC and LwF stay at the naive level" (tab:dil caption, main:122) fails per backbone and per seed.**
  - EWC over naive: LiLT +8.2 (s42 52.32 vs 40.68), BERT +7.2, BROS +4.7.
  - LwF on BROS: −5.1.
  - Fix: "stay within about 8 points of naive and far below replay".
- **N5. The new ProtoNER claim is not supported.** main:85: "none addresses the background class
  under label growth". ProtoNER (arXiv:2310.02372, Sec. 3.3 Training procedure) discards the
  "Other"-class prototypes before every incremental phase and annotates its incremental samples
  for all n+1 classes. Its results discussion explains learning from few samples by
  sub-clusters that future-class words, "labeled as 'Other' during the base training", form
  under the "Other" class. Fix: "ProtoNER avoids the conflict by annotating its few incremental
  documents for every class; none of these works separates the causes of forgetting".
- **M5 residual** (see Part A): supp `tab:legacy` caption, "the baselines use 10 training epochs".

#### LOW

- L-a: supp `tab:grid` hand-typed values disagree with main Table 1 (35.2 / 35.1, 87.6 / 87.5, ±1.0 / ±0.9). Generate it, or copy the generated values.
- L-b: supp `tab:grid` caption, "on CIL-CORD every method, replay included, sits at or below the naive level". False: LwF BROS 23.1 and BERT 24.3 against naive 19.2 and 19.3.
- L-c: supp IS3 prose uses population s.d.: "4.4±1.7" (ddof 1: 2.2), "30.1±1.0" (ddof 1: 1.2; `tab:legacy` prints 30.10±1.18).
- L-d: memory units are mixed.
  - "60 MB" is decimal (DIL `replay_memory_bytes` 60,403,200 = 57.6 MiB).
  - "96 MB" and "54 MB" are MiB (100,672,250 B; 56.6e6 B).
  - Use one unit.
- L-e: "16--19" per seed: ER s7 = 14.43. Write "14--19" or "on average".
- L-f: Setup epoch caps ("100 for ... CPFD and IS3, 30 for ... CL-LoRA") hold only on CIL-CORD and Mixed. DIL CPFD and IS3 are 10; DIL CL-LoRA has script default 100. Add "on CIL-CORD and Mixed".
- L-g: The modality-mask sentence omits the no-text condition, BWT −66.7 over three seeds (it forgets least). Add it.
- L-h: The prose uses `\nColar` (30.4, two seeds) and `\nNoBank` (20.4, one seed) with no n (intro, §4.2, §6). The tables show seeds; the sentences do not.
- L-i: `pick()` drops `v <= 0.5`, not exact zeros as the docstring and footnote say. It is harmless today (only LwF; its dropped values are 0.00). Use `v != 0.0` or fix the docstring.
- L-j: `\vSbrFull`, `\vRecallFull` and `\vJointFull` print an n<3 mean with no n mark. They are unused; do not use them in prose.
- L-k: The v3 driver has no lock. Relaunching it while a run is in progress (no `metrics.json` yet) starts a second GPU job. Add `flock` the next time a driver is written (not while this one runs).
- L-l: `docs/LEGACY_PROTOCOL_TABLE_2026-11.md:99` says IS3 Mixed s42 was "dropped". The generator keeps it, and the doc's own note at line 129 says so.
- L-m: Generated tables are stale. The clean CoLaR-SBR s7 run landed at 07:49; after regeneration the full-label cell reads `88.8/91.2^{n=2}`.
- L-n: "LwF does not collapse in fp32" rests on one fp32 run (CIL-CORD s42, 18.45). Say "in its fp32 run".
- L-o: main:65 "Memory ... is blind to the second" is the body form of the OPEN abstract clause. Resolve the two together.
- L-p: main:198 "every method with a shared softmax stays at or below CoLaR". Joint training also uses a shared softmax. Write "every continual method".

### Part B: numbers

**B.1, mismatches > 0.05: none.** Every cell of `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex` and
`tab_dil_grid.tex` reproduces from the run files (rescore runs are one directory deeper; CoLaR
directories carry `_k4_d50_r128`). Per seed (42/7/123), new or changed cells only (the rest as
in R1):
- CIL-CORD LwF: 18.45/18.51/0.00 → `18.4/18.5^{†,c}`.
- Mixed LwF: 47.75/0.00 → `47.7^{†,c}`.
- CIL-CORD IS3: 5.60/7.67 (no mark).
- Mixed IS3: 0.00/40.59.
- CPFD: CIL-CORD 2.72/0.00; Mixed 0.27/15.09; DIL 0.00/4.25/14.19.
- CL-LoRA: CIL-CORD 18.67/18.59.
- Table 1 LiLT: naive 40.14±0.47, EWC 48.38±5.87, ER 85.65±0.48, DER++ 84.35±0.31, joint 83.79±1.18; LwF has a run directory with no `metrics.json` → "--".
- Table 1 BROS: LwF 35.15±3.10, DER++ 87.55±1.08.
- Table 1 BERT: ER 77.06±0.95.

No cell lacks a source. The selection rule in code is exactly the stated rule, for AA (`pick`)
and for the retention figure (`matrices`). The full-label column can show mean±s.d. only when
three clean seeds exist.

**B.2, clean-protocol runs** (`results/clean_2026-11/`; log config dump + `metrics.json`):

| Run | amp / fp16 | val_frac | epochs cap (stopped at) | cil_restrict_eval | full_label_f1 | AA | BWT | final row of `matrix.npy` |
|---|---|---|---|---|---|---|---|---|
| `cil_cord_smoke_naive_e1_seed42` | false / false | 0.1 | 1 (smoke) | true | 12.55 | 18.05 | −76.09 | 0/0/0/0/90.26 |
| `cil_cord_colar_sbr_seed42` | false / false | 0.1 | 10 (5/5/5/6/6) | true | 88.83 | 87.76 | −2.41 | 91.41/82.78/88.14/83.06/93.40 |
| `cil_cord_colar_sbr_seed7` | false / false | 0.1 | 10 (8/9/9/6/3) | true | 91.19 | 91.32 | −1.61 | 92.07/91.24/92.04/88.52/92.71 |

- Batches per epoch (360/172/247/326/351 at batch 2) equal 90% of the session pools in both seeds.
- `replay_memory_bytes` = 100,672,250 in both seeds.
- s7 has no DONE line in `queue.log` (orphaned by the v1→v2→v3 swap, logged as SWAP); its files are complete.
- `cil_cord_colar_sbr_seed123` is running (v3 pid 3338470).
- The log's "~130.1 MB fp16" bank line overcounts: it counts the newest batch before SVD. `replay_memory_bytes` is the right figure.

**B.3, prose literals.** Every literal in `main.tex` traces to a result file, the supplement, the
evidence ledger or `STATE.md`. New since R1, recomputed: 81--95, 16--19 (L-e), −89.8/−87.2/−79.1,
82.5 with 64.9/67.7, 42.0 (44.17/39.85), 61.2 (61.17), 14.8 (14.77). As in R1, the CKA profile
(1.00/0.78/0.34/0.25/0.16), the freezing numbers (41.1/0.0, 38.8/−76.8, 0.19), the ladder, the
ViT numbers and 163 MB / 80.6 are traced to the ledger and `STATE.md`, not recomputed.

**B.4, comparatives per seed** (only changed or new ones; R1's list still holds):

| Sentence | Check | Verdict |
|---|---|---|
| "plain replay comes close to joint training on every backbone" | ER − joint: −0.9/+1.9/+0.3/−0.2 | OK |
| "EWC and LwF stay at the naive level" | up to +8.2 (mean) and +12 (seed) | **N4** |
| "naive, LwF, ER, DER++ sit at 16--19" | ER s7 14.4 | L-e |
| "Only the cell with both approaches joint" | −5.2/−4.2/−2.3 | OK |
| "sessions reach 81 to 95" | 80.9 to 94.6 | OK |
| "every method with a shared softmax stays at or below CoLaR" | every baseline seed ≤ 26.9 < 28.8 | OK, but joint counts too (L-p) |
| LexSlot "within two points ... narrows the spread" | +1.7 / −0.1 / +0.8; s.d. 0.46 vs 1.97, 0.36 vs 1.25 | OK |
| "Perfect memory labels recover about half" | (61.2−28.8)/(92.8−28.8) = 0.51 | OK |

**B.5, legacy-settings table and depth profile.**
- Every row of supp `tab:legacyprotocol` agrees with `docs/LEGACY_PROTOCOL_TABLE_2026-11.md`:
  caps, precision, memory, seeds, the LwF ‡ and the first-grid * footnote including batch 16.
- `tab:depthprofile`, re-derived from `results/pilot/c*_seed*.json` (dead = final AA < 1.0; dead
  runs: `c1_text_seed123` and the three unused `c1_bert` runs), matches every cell:
  - full 273×, 0.03/0.70/0.24/0.03;
  - text+layout 361×;
  - text-only n=2, 801×, 0.03/0.34/0.51/0.12;
  - no text 247×;
  - LiLT 111×;
  - BROS 738×;
  - BERT 1608×.
- BWT: full −89.8 (−89.51/−89.90/−90.03), text+layout −87.2, text-only −79.1 (−79.41/−78.73), no text −66.7 (L-g). The `_ord210` runs form their own group (−75.2) and carry no displacement records.

### Part C: code review (`git diff ac76a6c..HEAD -- scripts/ doccl/ tests/ configs/`)

**`doccl/methods/o_lora.py` (`layers_to_transform`): inert when unset.** With the key absent,
`layers_to_transform=None` and `layers_pattern=None` are the `LoraConfig` defaults, so earlier
CL-LoRA and O-LoRA results are unaffected (cannot change a paper number).

Sites of the capacity-matched command (v3 `CLR64`: rank 64, alpha 128, targets query/key/value/dense, layers 4..11) against SBR-Recall:
- SBR-Recall (`colar_sbr_recall.py:75-81`) wraps every `nn.Linear` whose last name component is in the targets, in `_encoder_layers()[4:]`. That is 6 per layer (Q, K, V, attention.output.dense, intermediate.dense, output.dense) × 8 layers = 48.
- PEFT with `layers_pattern="layer"` and `[4..11]` hits the same 48. `tests/methods/test_olora_layers.py` pins 48 vs 72; I did not run it (rule 1).
- Rank and alpha match, so per-session capacity matches: 7.08M parameters, one delta's worth.

Differences that remain, and that the caption must state if the run is reported:
1. **one adapter shared by all sessions** vs one delta per session (Recall stores 4×);
2. **base = frozen pre-trained checkpoint**, layers 0..3 never adapted, vs Recall's base
   fine-tuned on session 1 and then frozen;
3. distillation toward the previous model (`kd_alpha` 1.0, T 2), absent in Recall;
4. orthogonality penalty (λ 0.5) against the adapter's own earlier states, absent in Recall;
5. LoRA dropout 0.1 vs none;
6. one inference pass with the current adapter vs one pass per delta;
7. PEFT-wrapped shared-softmax head vs SBR head plus per-session background logits.

Items 1 and 2 are N1/N2: the run is a sites-, rank- and alpha-matched control, **not** a
per-session-adapter control. It could change the 2×2 only as an added row.

**v3 driver (`clean_protocol_v3_2026-11.sh`).**
- Order:
  1. CoLaR-SBR ×3;
  2. Recall ×3;
  3. no-memory SBR ×3;
  4. CoLaR ×3;
  5. joint ×3 (cap 100);
  6. naive ×3;
  7. CL-LoRA ×3;
  8. CL-LoRA-r64 ×3;
  9. ER-250 s42;
  10. Recall o2/o3;
  11. BROS ×3;
  12. ER-250 s7/s123;
  13. LwF ×3;
  14. Mixed ×12;
  15. DIL ×15;
  16. Mixed Recall-c s7/s123.
- Names are prefixes of the run directories: CoLaR runs get `_k4_d50_r128` and BROS runs `_bros`. `cil_cord_cl_lora_seed*` and `cil_cord_cl_lora_r64_seed*` cannot match each other, in the driver's `ls` or in the generator's globs.
- Jobs run sequentially. A failure writes FAIL and moves on, and is retried on relaunch.
- There is no lock against a second launch (L-k).
- At the observed 1.7 to 2.3 h per CIL-CORD run, the joint/naive full-label values arrive around 20 h out and CL-LoRA-r64 around 40 h out. The full queue (about 50 runs) does not fit before the Nov 17 07:00 freeze; Mixed and DIL clean cells will likely stay †.

**`scripts/rca/make_clean_tables.py`.**
- Selection rule correct.
- Drop-zero applies to legacy sources only (`not c`).
- The `c` mark appears only when something was dropped.
- BASE rows read clean names.
- The DIL grid writer reads `results/dil_<m>_seed<s>{"", _lilt, _bros, _bert}` exactly. No suffix leak: LayoutLMv3 uses the empty suffix, so it cannot reach `_bert` and the like.
- Findings: L-i, L-j. Neither changes a printed number today.

**`pilot_depth_profile.py` / `pilot_displacement_fig.py`.**
- Same dead rule: the figure imports `dead_reason`, and `load_pilot_results`' narrower all-zero rule is a subset.
- Excluded runs are printed, not silently dropped.
- The figure filters by `condition`, which would also admit `_ord210` runs, but those have no displacement records.
- Output matches the table.

The other diffs (`analyze_results.py`, `make_image_table.py`, `build_falsification_figures.py`)
change labels and an axis limit only.

### Part D: submission hygiene

| Check | Result |
|---|---|
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10, so the body is pages 1 to 7. |
| Undefined references / citations | PASS. 0 in `main.log` and `supp.log`; 52 cited keys, all in `refs_precompiled.bbl`. |
| Overfull boxes | PASS. 0 in both logs. |
| `---` in paper inputs | PASS. 0 in `main.tex`, `supp.tex`, `fig/*.tex`, `tab_*.tex`, `vals.tex`. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and `supp.pdf`. |
| Anonymity, main.pdf | PASS. "Anonymous CVPR submission"; no names, institution, paths, script names or dates. "6 GB GPU" in the Table 1 caption is benign. |
| Anonymity, supp.pdf | PASS. Exposes in-house instance names (LexMem v3b/v5, HGT/CUBER, in the ledger only), public HF dataset IDs and "6 GB GPU" / "24 GB card". `.tex` comments carry repository paths and dates (`supp.tex` 110-114 and 153-155), so do not ship the source. |
| Every float referenced | PASS. Main: teaser, `fig:conservation`, `tab:dil`, `tab:twobytwo`, `fig:pareto`, `tab:main`, `fig:retention`. Supp: all 14 labels referenced. |
| Pointers main → supp | PASS. Sec. A (setup), D (legacy settings; displacement; `tab:grid`), C (ledger; rebalancing nulls; `tab:image`), B (`tab:image_supp`) each hold what the pointer says. Supp → main: "Sec. 3" (Setup), "Sec. 4.1" (drift), "Sec. 5.1" (CoLaR/ViT) all correct. |
| PDFs current | PASS for text. Tables lag the disk by one clean seed (L-m). |

### OPEN (not counted)
- Abstract clause "an effect that no memory can repair" (user decision; its body echo is L-o).
- CR2: † headline cells, pending data.

## R3 (2026-11-11): confirmation of P4

Scope: `git diff cfccda1..37e7df4 -- paper_cvpr/ scripts/rca/make_clean_tables.py
scripts/rca/make_retention_fig.py`, the PDFs rebuilt at 08:16, and the v4 driver (5be5c26).
Read-only. No model or dataset touched.

### Verdict: **PASS for draft upload**

N1 is fixed everywhere. No CRITICAL or HIGH finding remains besides the waived CR2 and the
user-pending abstract clause. Two MEDIUM items should go into the next Paper round:
- **N4 residual:** the new "within ten points / thirty points below" bounds fail per seed.
- **N6 (new):** the new Sec. 5.3 sentence contradicts the paper's own Related Work.

### Status table

| Item | Status | Current text / cell |
|---|---|---|
| **N1** generator label | CONFIRMED FIXED | `make_clean_tables.py`: `f"Low-rank adapters, no memory & ..."`; `tab_2x2.tex` row "Low-rank adapters, no memory". |
| **N1** `tab:twobytwo` caption | CONFIRMED FIXED | "its two cells differ in adapter design as well as in readout: our port of CL-LoRA (one rank-8 adapter on the attention projections of the frozen pre-trained encoder, shared by all sessions, trained with distillation and an orthogonality penalty against its own earlier states; fp16 or fp32 by seed, 30-epoch cap) with a shared softmax, and SBR-Recall (one rank-64 delta per session on the attention and feed-forward layers above k, base frozen after the first session, fp32, 10-epoch cap)". Matches the code. |
| **N1** Sec. 4.3 | CONFIRMED FIXED | "CL-LoRA, one low-rank adapter shared by all sessions, with a shared softmax, collapses ... SBR-Recall, one low-rank delta per session ... This row is not a controlled swap of the readout". |
| **N1** teaser | CONFIRMED FIXED | Bar label "low-rank adapters". Caption: "one low-rank delta per session with a blocked readout (SBR-Recall) is on par with it without a memory, while a single low-rank adapter shared by all sessions with a shared softmax (our CL-LoRA port) collapses". |
| **N1** leftovers | CONFIRMED FIXED | Every remaining "per-session" in main.tex and supp.tex refers to SBR, SBR-Recall, ICE, scoring or session pools; none refers to CL-LoRA. Cosmetic only: the v4 driver comment (line 40) still says "per-session isolation with a shared softmax". |
| **N2** | CONFIRMED FIXED | main:91: "Our CL-LoRA port trains a single rank-8 adapter on the attention projections of the frozen encoder, shared by all sessions, unlike the published one-adapter-per-task design, so its numbers describe this port." The supp Hyperparameters paragraph states the same for O-LoRA and CL-LoRA, with λ = 0.5 and T = 2, both matching the configs. The causal clause is gone: "Our single-adapter LoRA ports acquire but forget as much as naive training does." |
| **N3** | CONFIRMED FIXED | Limitations: "Every cell is marked by its protocol and never mixes the two, so a comparison across protocols is visible from the marks in the tables". |
| **N4** | **PARTLY (MEDIUM)** | See below. Affects the `tab:dil` caption and main:122 ("EWC and LwF stay within ten points of naive training and at least thirty points below joint training"), and the supp `tab:grid` caption ("regularisation and distillation stay within ten points of naive training"). |
| **N5** | CONFIRMED FIXED | "None of these document-level works separates the causes of forgetting." (The background claim is removed.) |
| M5 residual | PARTLY (now LOW) | The false "10 epochs" claim is gone. The new pointer misleads: "`tab:legacyprotocol` lists them for the rows shared with the main paper". `tab:legacy`'s CIL-CORD and Mixed columns are the earlier grid (`results/{cil_cord,mixed}_<m>_seed*`, legacy scoring), not the rescore runs behind the main paper's † cells; only its DIL column is shared. |
| L1 residual | CONFIRMED FIXED | supp `tab:grid`: BROS LwF `35.1±3.1`, DER++ `87.5±1.1`, BERT ER `77.1±0.9`, equal to `tab_dil_grid.tex`. The table is still typed by hand. |
| Units (L-d) | CONFIRMED FIXED for the file-backed numbers | 101 MB = 100,672,250 B; 57 MB = 4 × 7,077,888 × 2 B = 56.6e6; 60 MB = 60,403,200 B. Note: "163 MB", "about 32 MB" and the pareto labels (0.45/0.39/0.48 MB) come from the ledger and figure constants ("restored from logs"); their unit cannot be confirmed from a `metrics.json`. The logs print MB as bytes/1e6, so decimal is likely. |
| "16--19 AA on average" (L-e) | CONFIRMED FIXED | main:128. |
| IS3 s.d. (L-c) | CONFIRMED FIXED | "AA 4.4±2.2 ... 30.1±1.2". ddof 1 on 5.02/6.17/2.00 and 28.85/31.18/30.26. The run-set scoping is still open; see (a). |
| supp grid caption (L-b) | CONFIRMED FIXED | "on CIL-CORD every method stays below joint training under this scoring, and only LwF (on BROS and BERT) rises above naive training". Checked against every grid cell. Its "within ten points" clause is N4. |
| zero-drop rule (L-i) | CONFIRMED FIXED | `kept = [v for v in vs if v != 0.0] ...`. The output is unchanged (the LwF zeros are exact). |
| no-text condition (L-g) | CONFIRMED FIXED | "the model without text forgets least, −66.7 over three seeds" (−65.92/−66.15/−67.93). |
| L-n, L-p, L-f, L-o | CONFIRMED FIXED | "did not collapse in its fp32 run"; "every continual method with a shared softmax"; "On CIL-CORD and Mixed, epoch caps differ by method (the DIL caps are in the supplement)"; "no memory we tested resolved the second". |

**N4 per seed** (DIL run files; difference to naive of the same seed, and distance below joint
of the same seed):

| Backbone | Method | s42 | s7 | s123 |
|---|---|---|---|---|
| LayoutLMv3 | EWC | +0.1 / 49.5 | −0.7 / 46.7 | +0.3 / 46.3 |
| LayoutLMv3 | LwF | +2.5 / 47.1 | +1.6 / 44.4 | +3.0 / 43.7 |
| LiLT | EWC | **+11.6** / 30.6 | +1.8 / 43.5 | **+11.3** / 32.2 |
| BROS | EWC | +6.3 / 38.9 | +1.2 / 48.3 | +6.7 / 40.1 |
| BROS | LwF | −3.1 / 48.2 | −6.6 / 56.1 | −5.8 / 52.6 |
| BERT | EWC | +6.4 / 31.9 | **+10.3 / 27.2** | +4.9 / 32.8 |
| BERT | LwF | +1.0 / 37.2 | +0.8 / 36.7 | +1.5 / 36.2 |

The bounds hold on the means (largest gain over naive +8.2, LiLT EWC; smallest gap to joint
30.6, BERT EWC) but fail in three runs. Fix: write "on average" in all three places, or use
per-run bounds ("within 12 points of naive training and at least 27 below joint training in
every run").

### New findings

**MEDIUM N6. The new Sec. 5.3 sentence overclaims and contradicts Related Work.** main:187:
"It has no shared-softmax counterpart: per-session deltas need something to select them at test
time, and without blocks the only selector would be a task-identity oracle". Related Work
(main:81) says the opposite: "recent adapter methods add one module per task and need no task
identity at test time, either by combining the features of all modules (EASE) or by folding
them into the pre-trained weights (InfLoRA)". Kim et al. is cited there for task-identity
*prediction*, not an oracle. Fix: "A shared-softmax counterpart would have to combine or merge
the deltas (as EASE and InfLoRA do) or predict the session with a separate model; we did not
run one. SBR's per-session background logit is such a predictor, trained at no extra cost."

**LOW**
- (a) The supp IS3 sentence and `tab:main` use different run sets *and* different scorings.
  - Supp prose ("AA 4.4±2.2 on CIL-CORD ... 30.1±1.2", "diagonal ... on seed 123"; also `tab:legacy`): the earlier grid `results/{cil_cord,mixed}_is3_seed{42,7,123}`. Three seeds, 10-epoch cap, fp32, test-split early stopping, **legacy scoring** (no restriction; it predates c430b87).
  - `tab:main` IS3 cells (5.6/7.7, 0.0/40.6): `results/rescore/{cil_cord,mixed}_is3_seed{42,7}`. Two seeds, **100-epoch cap**, fp32 in effect, test-split early stopping, **session-restricted scoring** (log `cil_restrict_eval: true`).
  - The Mixed seed-42 runs differ (28.85 vs 0.00) although the precision is the same, so the zero is not a rescoring of the same model.
  - The supp sentence must say: "In the earlier grid (legacy scoring, 10-epoch cap, three seeds) IS3 ... gives AA 4.4±2.2 on CIL-CORD ... 30.1±1.2 on Mixed (`tab:legacy`); the session-restricted re-runs in the main comparison (100-epoch cap, two seeds) read 5.6/7.7 and 0.0/40.6."
  - The CPFD sentence in the same paragraph ("AA 2.7 and 0.0 on seeds 42 and 123"; "on Mixed it reads 6.0, 0.0 and 0.1") is also from the earlier grid and needs the same qualifier.
- (b) CoLaR-R, sliced. A row is needed in `tab:legacyprotocol`, because main:130 quotes a legacy number from it: "42.0 (two seeds)" = `results/rescore/cil_cord_colar_r_sliced_seed{42,7}` (44.17/39.85). The row: "CIL-CORD | cap 10 | fp16 (log `amp: true`, CoLaR loop wired) | 50 docs/session | 2 seeds" (test-split early stopping, restricted scoring).
  - The same sentence's oracle refresh, "61.2 (seed 42)" = `gates/.../cil_cord_colar_r_oracle_seed42_k4_d50_r128`, is fp32, cap 10, 50 docs, 1 seed, and is not listed either.
  - `docs/LEGACY_PROTOCOL_TABLE_2026-11.md` has neither.
  - `tab:legacy`'s own CoLaR-R row (25.38±2.38 / 61.30±1.91) is a third run set (`results/{cil_cord,mixed}_colar_r_sliced_seed*_k4_d50_r128`: fp32, cap 10, 3 seeds, legacy scoring). It is covered by fixing the M5 pointer.
- M5 pointer, suggested text: "Its CIL-CORD and Mixed columns are runs of the earlier grid, not the † cells of the main paper: CoLaR, CoLaR-R, LexSlot, CPFD, IS3 and LCA ran in fp32 with a 10-epoch cap; the other rows saved no configuration (grid default cap 100, 30 for CODA-Prompt). The DIL column is the main paper's."

### Regenerated tables

Only three things changed:
- the row label;
- the CoLaR-SBR full-label cell, `88.8/91.2^{n=2}` (clean s42 88.83, s7 91.19);
- `\vSbrFull` 90.0 (= 90.01; unused in prose).

Every other cell of `vals.tex`, `tab_2x2.tex` and `tab_main_clean.tex` is byte-identical to R2.
`retention_cilcord.pdf` was regenerated from the same inputs (CoLaR n=2 legacy, SBR n=3 gate,
Recall n=3 gate).

### Clean runs

There is no new finished run. `cil_cord_colar_sbr_seed123` is running (orphan of v3, pid 3395398,
30 min at 08:20, no `metrics.json`); v4 (pid 3470572) waits on it.

### Queue driver v4 (`clean_protocol_v4_2026-11.sh`)

- **Lock:** `exec 9>$P/.driver.lock; flock -n 9 || {... REFUSED ...; exit 1}` before the wait. Children get `9>&-`, so the lock dies with the driver. This closes L-k for v4-style drivers; v1 to v3 are dead.
- **Order:**
  1. CoLaR-SBR, Recall-r64, no-memory SBR, CoLaR, joint and naive on CIL-CORD;
  2. CL-LoRA;
  3. ER-250 s42;
  4. Recall o2/o3;
  5. Mixed SBR and joint;
  6. BROS;
  7. ER-250 s7/s123;
  8. LwF;
  9. the rest of Mixed;
  10. DIL;
  11. Mixed Recall-c.

  No `cl_lora_r64`.
- **Names:** unchanged from v3; every run name is a prefix of its run directory (`_k4_d50_r128`, `_bros`), and no two names prefix each other.
- **Failure:** same as v3: FAIL logged, the queue continues, and a relaunch retries.
- **Caveat:** the clean ER cell needs ER-250 s7/s123, which come after BROS, so the ER row will likely stay † at the freeze.

### Hygiene

- Body: "References" starts on page 8 of 10.
- `main.log` and `supp.log`: 0 undefined, 0 overfull.
- `---`: 0 in every paper input.
- Em-dash: 0 in both PDFs.
- Abstract: byte-identical to 0a896ef (sha1 `fa56d93a…` for both).
- `supp.pdf` is now 8 pages.

## R4 (2026-11-11): audit of the complete paper and appendix

Snapshot: `HEAD = c53cf54`; `main.pdf` 09:13 (after `main.tex` 09:12), `supp.pdf` 09:10 (after
`supp.tex` 09:10), generated `supp_*.tex` 09:09. Tracked paper files are clean against HEAD.
Recomputation: own scripts on raw `metrics.json` / `matrix.npy` / logs, `statistics.stdev`
(ddof 1). No model, dataset, torch, `doccl` import or pytest touched.

### Verdict: **PASS for draft upload**

No CRITICAL or HIGH finding remains besides the waived pending-data item (CR2) and the
user-pending abstract clause. Every generated appendix table reproduces from the run files
(no mismatch > 0.05), the supervision-plan table matches the code and the logs cell by cell,
and the new SBR-Recall "cannot drift" statement is correct in the case it is scoped to.
Six MEDIUM items should go into the next Paper round:
- two of them are descriptions that the paper's own appendix matrices contradict (R4-M1, R4-M4);
- one is a generator change needed before the clean CoLaR-SBR seed 123 lands (R4-M6).

### MEDIUM

**R4-M1. The retention descriptions are contradicted by the matrices and the figure they describe.**
- Where:
  - main `fig:retention` caption: "With a shared softmax every session decays as soon as the next arrives";
  - main "Retention per session" (new in c53cf54): "With a shared softmax each session decays as soon as the next arrives";
  - supp App. D.1: "With a shared softmax every column decays as soon as the next session arrives";
  - supp App. D.1, blocked readout: "its column loses a few points over the following sessions".
- Evidence. `supp_mat_colar.tex` (CoLaR rescore, seeds 42/7, the same matrices the figure plots):
  - column 3 goes 37.1 → **66.3** → 53.8;
  - column 4 goes 34.0 → **40.9**;
  - per seed, s42 column 3 is 22.5 → 60.6, and s7 column 1 is 95.4 → 37.1 → 55.3 (it rises again at session 3).
- Evidence. `supp_mat_sbr.tex`: column 2 goes 94.1 → 79.3 on the mean, and 94.4 → 72.8 on s42 (−21.6).
- Fix (text only):
  - shared softmax: "the first two sessions lose 30 to 50 points at the next boundary; later sessions are learned only to 19 to 37 F1 and fluctuate afterwards";
  - blocked readout: "loses between 1 and 15 points (session 2 most, 94.1 to 79.3)".
  The main paragraph's "slow loss on the earliest sessions" can stay.

**R4-M2. The Conclusion's SBR-Recall sentence is unscoped against the paper's own Mixed result.**
- Sentence: "When label growth supplies a routing key, per-session low-rank deltas (SBR-Recall)
  can replace the memory."
- Mixed has label growth and per-session background logits (the routing key, as Sec. 5.3
  defines it). Yet SBR-Recall without a memory scores 44.7 there (`mixed_colar_sbr_recall_r64b_seed42`).
- Sec. 5.3 itself gives a different failure: the all-background terms have no positive
  counterweight, not a missing key.
- Contribution 3 and Limitations are scoped correctly ("on a single-schema stream"; "does not
  replace memory ... across annotation schemas").
- Fix: "On a single-schema stream, where label growth supplies the routing key, per-session
  low-rank deltas (SBR-Recall) can replace the memory; across annotation schemas they still need it."
- The abstract clause "when label growth itself supplies the routing key" has the same gap. The
  abstract is user-agreed, so that part goes to the PM as NEEDS DECISION. I do not block on it.

**R4-M3. The Conclusion repeats the OPEN universal.** The sentence is "a better memory does not
repair a readout that supervises one background class with contradictory labels".
- It is the body form of the OPEN abstract clause ("no memory can repair"), and R2 L-o had it
  removed from the introduction.
- Evidence: one seed of true-label refresh (61.2), two fp16 seeds of replay slicing (42.0), and
  three seeds of both-sliced (62.9/69.2/49.4).
- Fix: "in our experiments even a memory with true labels did not repair ...". Resolve it together
  with the abstract clause.

**R4-M4. A typed stale-label count is wrong.** App. E: "A deployable variant that refreshes with
the model's own confident predictions rewrites none on seeds 42 and 7".
- `results/logs_vision/mixed_colar_r_pseudo_seed42_k4_d50_r128.log` (2026-10-15 13:14:55):
  "refreshed 200 docs, **84** tokens relabelled (169 candidates)". The counts are 0/0/0/84 on s42.
- Seed 7 is 0/0/0/0 and seed 123 is 7/0/0/36, as typed.
- Ledger row F10c and `STATE.md` (ticks of 2026-10-17 06:09 and 2026-10-16 11:46) carry the same error.
- Fix: "rewrites 0/0/0/84 tokens on seed 42, none on seed 7 and 7/0/0/36 on seed 123 (true-label
  refresh: 3120 to 3307 at the first refresh)". "Almost no" still holds: 84 tokens against
  6,076 oracle tokens on that seed (3120 + 2640 + 0 + 316), though 84 against 316 at that boundary. Correct the ledger
  (PM/Dev).

**R4-M5. The reason given for the all-background mode is false for SROIE and CORD.**
- Where: main Sec. 5.2 ("so none of its entities can occur in the document") and App. C, bullet
  "all-background" ("none of its entities can occur").
- Both are receipt schemas, and a CORD receipt has a total, a date and usually a company name. Those
  are SROIE entities in kind, annotated under CORD's tags.
- The rule is a convention, not a fact about the document: the document is annotated under its
  own schema, so no token carries one of b's tags, and every token is treated as a negative for b.
- Fix: "when b belongs to a different annotation schema: the document is annotated under its own
  schema, so we treat every token as a negative for b (a convention that uses the schema of each
  session at training time; SROIE and CORD share entity kinds such as totals and dates)". The
  Limitations sentence on the schema already covers the rest.

**R4-M6. `supp_slot.tex` will mix protocols at the next regeneration.**
- `write_slot` pairs `C["sbr"]` (the main-table source) with the gate LexSlot runs.
  `cil_cord_colar_sbr_seed123` is running in `clean_2026-11` (pid 3395401). When it lands,
  `choose()` switches `C["sbr"]` to clean, and the CIL-CORD row will difference clean CoLaR-SBR
  against gate LexSlot, seed by seed.
- Prose built on that row will then also compare across protocols:
  - App. D ("$\vSbr$ against $\vSlot$ on CIL-CORD");
  - main Ablations ("narrows the spread between seeds on CIL-CORD"). The clean seeds 87.8/91.3 are
    already wider than gate 87.6/86.3/90.1.
- Fix (Paper owns the generator):
  - in `write_slot`, read the CIL-CORD CoLaR-SBR side from `src(None, (GATE, "cil_cord_colar_sbr", ""))`,
    the same fp32 gate protocol as LexSlot, and label the column;
  - in the prose, replace `\vSbr` by a gate-only macro;
  - or switch the LexSlot cell only when clean LexSlot runs exist.

### LOW

- **R4-L1 (Part A.4, nuance).** "neither the base, nor its delta, nor its rows of the readout
  receive a gradient after its session: old blocks cannot drift, and routing by $e_b$ is unchanged."
  - Correct for gradients.
  - The old tag rows (and their bias entries) live in the one classifier tensor, so AdamW's
    decoupled weight decay still shrinks them. The bound is ≤ 0.5%: 9,665 steps after session 1
    on `recall_r64_seed42`, × lr·λ = 5e-7. The old $w_b$ have no gradient and are skipped.
  - Each old block's $e_b$ is unchanged, but the arg-max over blocks changes as new blocks
    compete. Old sessions' F1 moves by −0.8 to +1.3 in the r64 matrices.
  - Suggest: "each old block's $e_b$ is unchanged (up to weight decay); only competition from new
    blocks moves its tokens".
- **R4-L2.** $o_b=w_b^\top h(x)$ omits the bias of `nn.Linear(hidden, 1)`.
- **R4-L3.** Alg. 1 line "AdamW step on layers $k{+}1..12$, head, all $w_b$": in session 1 every
  layer trains (the freeze comes after). Add "(session 1: all layers)".
- **R4-L4.** App. E calls the control a "frozen trunk": the takeaway says "or on a frozen trunk",
  and a later sentence says "learned at 87.0 on the same frozen trunk". The run is `split_layer_k=11`:
  layer 12, the head and $w_b$ stay plastic (trainable 7,098,638 = one layer + 13×769 + 769).
  Write "an almost frozen trunk (only the last layer and the readout plastic)". The body, the
  conflict-table label ("layers 1..11 frozen") and App. E's own sentence are correct.
- **R4-L5 (labels_full, Paper agent's question).** See Part D.
- **R4-L6.** The deltas are counted at 2 B per parameter, but they are trained and evaluated in fp32.
  No run evaluated fp16-rounded deltas. Write "57 MB if stored in fp16 (113 MB in fp32 as run)",
  or evaluate once.
- **R4-L7.** `tab:legacyprotocol` lacks rows for the App. E gate controls, although `tab:conflict`'s
  caption points there:
  - CoLaR gate 34.1, replay-sliced gate 39.4, both-sliced ×3, + pseudo, bidirectional and the
    k=11 control are all fp32, cap 10, 50 docs (none for k=11). Verified from `training_hparams.amp`
    and `method_hparams`.
  - The rank-8 Recall runs are missing too.
  - Add one row.
- **R4-L8.** `supp_delta.tex` has no protocol mark, and the `tab:delta` and `tab:recallmix` captions
  say "fp32 gate runs" without †. Add †: these are test-split early-stopped runs.
- **R4-L9.** The main Setup now says only that legacy cells differ "in epoch cap (10 to 100) and
  precision". The first DIL grid's unconfirmed settings (batch size 16) and the per-row precision
  are only in App. G. That is acceptable with the pointer; optionally add "(App. G)" to the
  `tab:dil` caption.
- **R4-L10.** App. B, "so the restricted AA of the main comparison is not inflated by the coverage
  gap", rests on two seeds of one method. Scope it to CoLaR-SBR.
- **R4-L11.** Main Sec. 4.2 says "App. E collects these controls with their seeds and the number of
  stale labels the memory holds". The counts in App. E are Mixed (earlier grid), not CIL-CORD. Say
  so, or add the CIL-CORD oracle counts, which are in the logs:
  - s42: 101/760/1461/1402;
  - s7: 66/944/1366/1269;
  - s123: 74/943/1315/1110.
- **R4-L12.** App. C: "the receipt session reached 23.6". DIL has two receipt sessions; it is the
  SROIE one (`dil_colar_sbr_seed42` diagonal 86.7/23.6/97.9, AA 66.49).
- **R4-L13.** App. A: the held-out split is seeded by "run seed + session index"
  (`split_val(..., cfg.seed + task_idx)`; joint uses `cfg.seed`), not by the run seed alone.
- **R4-L14.** The generator has a regeneration race. `matrix.npy` is written about a minute before
  `metrics.json` (clean s42: 05:28 vs 05:29). A regeneration in that window moves a matrix (and
  `retention_cilcord.pdf`) to clean while its table cell stays legacy. Regenerate only after the
  DONE line.
- **R4-L15 (anonymity, supp.pdf).** What `supp.pdf` exposes:
  - the package path `doccl/data/scenarios.py`;
  - dated run paths `results/clean_2026-11` and `results/gates/cil_protocol_2026-11`;
  - script names (`scripts/rca/make_clean_tables.py`), run-directory names (`mixed_colar_sbr2_seed42`, ...)
    and `_CORD_CANONICAL_SESSIONS`.

  None of these identifies anyone by itself, but `doccl` could match a public repository. I
  recommend generic wording ("our scenario builder", "the generator script"), no dates and no
  package name. "RTX 2060 (6 GB)", "14 GB host memory" and "24 GB card" are acceptable.
  `main.pdf`: no paths, names, dates or package names.
- **R4-L16.** App. A, "two annotation schemas per form type", is unclear. Suggest: "three
  annotation schemas (FUNSD for forms; SROIE and CORD for receipts)".
- **R4-L17.** `alg:recall` is not `\cref`'d inside the supplement; main cites it as typed "Alg. 2",
  which is the correct number.

### Part A: the body as a finished paper

**A.1 Regressions against R1–R3.** None of substance.
- These are still fixed: H1 (16–19 on average); H2/N1 (CL-LoRA wording in the 2×2 caption, Sec. 4.3,
  and in the new Experiments sentence "those that keep a frozen encoder with a shared softmax (CODA-Prompt,
  our CL-LoRA port)"); H3 (intro); M2, M3 (82.5), M6, M7; N3, N4 (per-run 12/27), N6; L4, L-d, L-p.
- The new Conclusion reintroduces the H3/L-o pattern once (R4-M3) and drops the single-schema scope
  once (R4-M2).
- The Setup paragraph lost its per-group cap and precision list to App. G (R4-L9).

**A.2 New body claims since 37e7df4** (`git diff 37e7df4..HEAD -- paper_cvpr/main.tex`):

| Claim | Check | Status |
|---|---|---|
| Mixed: "two class-incremental FUNSD sessions, a shift to SROIE, two class-incremental sessions on CORD's super-classes and a FUNSD revisit with all its labels" | `build_mixed` sessions 0–5 | OK |
| † cells differ "in epoch cap (10 to 100) and precision" | `tab:legacyprotocol` | OK (L9) |
| "A cell switches to the clean protocol only when all three clean seeds exist, so no cell combines the two" | `choose()` | OK; `supp_slot` will break it (M6) |
| `tab:dil` caption "On average ... within ten ... thirty"; body "within 12 ... at least 27 ... in every run" | R3 per-seed table (max +11.6, min 27.2) | OK |
| 2×2 caption precision split | legacy-protocol rows; gate runs `amp=false` | OK |
| "Training in brief" | `colar_sbr.py`, `latent_replay.py` | OK; the replay term is a mean over the four documents, not a sum (trivial) |
| SBR-Recall "cannot drift" | code (A.4) | OK in its scope (L1) |
| "A session that adds no block reuses the delta of the first block it trains fully" | `before_task` | OK. Edge case: a session with neither a new block nor a fully trained block gets an unused fresh delta; it never occurs. |
| "shared-softmax counterpart ... we did not run one" | (N6 fix) | OK |
| "The table repeats the 2×2 ...": ER/DER++/CoLaR strong on DIL (87.9/88.0/87.8), far below joint on CIL-CORD (24.3/20.4/30.4); CPFD, IS3, CODA-Prompt, CL-LoRA do not recover CIL-CORD (≤18.7) | `tab_main_clean.tex` | OK |
| "Retention per session" | matrices | **M1** |
| Ablations: no-memory 20.4; k=11 control 14.8, "sessions after the second are barely learned" (diagonal 4.3/22.0/0.0); rank 64 learns sessions 3–4 on all seeds (88.8–96.2) | gate runs | OK |
| Cost: "Per-run training times are in App. H" | App. H points to `tab:perseed` (App. D) | OK |
| Conclusion | below | M2, M3 |

**A.3 Conclusion.**
- The drift claim is scoped to a fixed label space ("on a fixed label space only old-task supervision
  of the readout repaired it in our experiments"). OK.
- "survives a memory with true labels": one seed (61.2), the same wording as contribution 2. Acceptable.
- "resolved at the readout": consistent with Sec. 4.3 ("in every cell we ran").
- Recall sentence: M2.
- "a better memory does not repair": M3.
- "session-restricted scoring with a mask-free full-label F1 makes both measurable": acceptable,
  though full-label F1 exists for 2 runs only.

**A.4 "Old blocks cannot drift"** (`colar_sbr_recall.py`, `colar_sbr.py`, `latent_replay.py`, wrapper).

Mechanics:
- After task 0, `freeze_backbone()` freezes all of `layoutlmv3` (embeddings and all 12 layers),
  leaving the classifier trainable. `sbr_o` is re-enabled.
- `_train_only_current_delta` leaves only the current (A, B) trainable.
- The optimiser is rebuilt each session (`_make_optimizer`), over `trainable_parameters()`.

Per case:

| Case | What an old block b receives after its session | Paper |
|---|---|---|
| Single schema (CIL-CORD), no memory | Plans print `trains [(s,'full')]` only (logs of `recall_r64_seed42` and clean SBR s42). Base and Δ_b: frozen. $w_b$: not in the graph (grad None, skipped by AdamW). Tag rows $z_{\mathcal{T}_b}$: zero gradient, AdamW decay only (≤0.5%). | Correct as stated (L1) |
| Single schema, with memory (`recall_seed42`, rank 8, d50) | Replayed session-b documents train block b fully under Δ_b. Base and Δ_b are frozen, but $w_b$ and the $\mathcal{T}_b$ rows are trained, so the readout moves (session 4: diagonal 68.0 → final 65.1). | Not claimed; App. D's "within 1.5 points" is scoped to "without a memory" (exact: 64.94 → 66.44 = 1.50). OK |
| Across schemas, no memory (`mixed_..._r64b`) | Current documents of another schema train old blocks in "neg" mode under their own delta. $w_b$ and the rows are pushed to O, so the blocks see-saw. | App. D says exactly this. OK |
| Across schemas, with memory (`r64c`) | Neg terms plus full replay terms under own delta. | App. D. OK |
| Mixed revisit | `full=[0,1]` → reuses `block_delta[0]` = None (base). No delta is trainable; the readout only. | App. C. OK |

The pre-fix runs (CIL-CORD r8 / d0 / r64 ×3, all before fb2960c at 11-09 05:31) are unaffected,
as App. C says. In the old code, replayed documents ran under `_session_delta(s)` (own block's
delta), and current batches under the current delta with plan = own block only. That is
identical on a single schema.

**A.5 Mixed description vs `build_mixed`: OK.**
- Sessions: FUNSD {H, Q}, FUNSD {A}, SROIE (4 fields, 8 tags), CORD-super {menu, sub_total},
  CORD-super {total, void_menu, sub}, FUNSD revisit with all labels.
- The head reaches 25 tags; the revisit adds none.
- App. A pools from steps per epoch (75/74/313/400/399/75, batch 2) are 149/147–148/625–626/799–800/797–798/149. OK.

**A.6 Frozen-trunk control: OK in the body** ("every encoder layer but the last frozen after the
first session").
- `audit_gates_2026-11.sh:40` sets `method.docs_per_task=0 method.split_layer_k=11`.
- The log says "layers <11 + embeddings frozen; trainable params=7098638".
- `metrics.json` has `split_layer_k: 11`, `amp: false`.
- App. E wording: L4.

**A.7 Body → appendix pointers: all resolve.**

| Pointer | Holds |
|---|---|
| App. A | sessions and pools (`tab:cilsessions`, `tab:mixsessions`) |
| App. B | worked example and `tab:agree` |
| App. C | Alg. 1 and Alg. 2, `tab:plan`, memory and parameter accounting |
| App. D | per-seed `tab:perseed`, matrices, Mixed Recall matrices, delta capacity, LexSlot |
| App. E | conflict controls, with seeds; stale-label counts are Mixed (L11) |
| App. F | displacement / "late layers", ledger, ViT image section, rebalancing nulls |
| App. F and G | EWC sign flip (`fig:signflip`, `tab:grid` Mixed rows) |
| App. G | every † setting (L7 gap) |
| App. H | wall-clock |

The typed "Tab. 3" and "Sec. 4.1" in the supplement are the correct main-paper numbers, and so
is "Alg. 2" in main.

### Part B: the appendix, number by number

**B.1 Generated tables, recomputed: no mismatch > 0.05.**

Registry mapping checked. Every glob resolves to exactly one run directory, with one
`metrics.json` each; no top-level run directory holds two.
- **Exact-suffix sources:**
  - rescore and grid sources use `""`: `cil_cord_*_seed42` cannot reach `_bert`;
  - `dil_colar_e10` uses `_k4_d50_r128`;
  - the gate CoLaR / CoLaR-R controls use `_k4_d50_r128`.
- **No collisions between run names:**
  - `cil_cord_colar_sbr_recall_r64` does not reach `r64b`/`r64c` (those are Mixed);
  - `cil_cord_colar_r_sliced2` does not reach `sliced2_pseudo`;
  - `cil_cord_colar_sbr` (`""`) does not reach `_slot`, `_frozen`, `_noreplay` or `_recall*`;
  - clean `cil_cord_colar_sbr*` cannot reach `..._bros_seed*_bros`.
- **Methods, scenarios and seeds** are as labelled. Every row carries its protocol tag: clean,
  gate†, rescore† or grid†. Matrix titles name seeds and protocol.

Per-seed values (42/7/123), as recomputed:
- **`supp_perseed`.**
  - CIL-CORD:
    - naive 18.97/19.08/18.60;
    - no-memory SBR 20.36;
    - CoLaR 28.83/31.87;
    - SBR 87.62/86.27/90.15;
    - Recall 92.69/93.37/92.44;
    - joint 92.84/90.48/92.43;
    - ER 26.91/22.05/24.01;
    - LexSlot 90.15/89.72/89.23;
    - LwF 18.45/18.51/0.00ᶜ;
    - DER++ 21.27/18.76/21.07;
    - CODA 13.96/15.89;
    - CL-LoRA 18.67/18.59;
    - CPFD 2.72/0.00;
    - IS3 5.60/7.67;
    - LCA 13.72/6.97.
  - Mixed:
    - naive 44.40/43.68;
    - CoLaR 75.56/74.55;
    - SBR 89.77/89.62;
    - joint 86.94/88.46;
    - ER 75.58/73.33;
    - LexSlot 88.48/89.99/90.32;
    - Recall 85.94;
    - LwF 47.75/0.00ᶜ;
    - DER++ 75.14/73.95;
    - CODA 12.08/10.59;
    - CL-LoRA 45.73/43.90;
    - CPFD 0.27/15.09;
    - IS3 0.00/40.59;
    - LCA 37.96/36.42.
  - DIL:
    - naive 40.06/43.03/40.85;
    - CoLaR 87.58/88.30/87.38;
    - SBR 86.76/84.77/87.08;
    - joint 89.68/89.03/87.49;
    - ER 86.21/88.73/88.62;
    - LexSlot 87.24/87.25/86.62;
    - LwF 42.54/44.63/43.82;
    - DER++ 88.23/87.06/88.78;
    - CODA 33.40/31.11/33.02;
    - CL-LoRA 40.82/41.34/41.27;
    - CPFD 0.00/4.25/14.19;
    - IS3 38.29;
    - LCA 46.35/43.80/41.45.
  - h/run (`total_wall_time_s`) matches.
- **`supp_conflict`:**
  - CoLaR rescore 28.83/31.87;
  - CoLaR gate 34.06;
  - sliced rescore 44.17/39.85;
  - sliced gate 39.38;
  - both sliced 62.86/69.23/49.45;
  - + pseudo 48.01;
  - bidirectional 44.11;
  - true-label refresh 61.17;
  - no-memory SBR 20.36;
  - k=11 control 14.77.

  Configurations confirmed. Gate runs are `amp=false`, epochs 10, d50, k4, r128; the k=11 run is
  d0. Rescore runs are `amp=true`. `bilc` = `sliced_pseudo` + `current_sliced` + `current_pseudo_tau` 0.9.
- **`supp_agree`:** clean SBR s42 87.76 / 88.83 (+1.07); s7 91.32 / 91.19 (−0.13).
- **`supp_slot`:**
  - CIL-CORD +2.53 / +3.45 / −0.92;
  - Mixed (rescore) −1.29 / +0.37;
  - Mixed (gate 89.73) −1.25;
  - DIL +0.48 / +2.48 / −0.46.
- **`supp_delta`:** rank-8 + memory, rank-8 d0, and r64 ×3: every diagonal, final-row and AA cell
  matches `matrix.npy` (e.g. r8 d0: diagonal 95.3/91.6/64.9/67.7/90.3, final row 95.4/91.9/66.4/68.4/90.3, 82.50).
- **Six `supp_mat_*`.** Means over the stated seeds match every cell. Per seed:
  - CoLaR s42/s7 rescore;
  - SBR ×3 gate;
  - Recall r64 ×3 gate;
  - SBR Mixed s42/s7 rescore;
  - Recall Mixed r64b and r64c, s42 gate.
- **Unchanged since 37e7df4:** `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex` and `tab_dil_grid.tex`.
  They are absent from `git diff --stat 37e7df4..HEAD` and clean in the worktree.

**Protocol labels for tables that will switch.**
- Tables that label their protocol correctly today: `supp_perseed`, `supp_conflict`, `supp_slot`
  (one tag per row) and `supp_mat_*` (title).
- Will become wrong: `supp_slot`, at the next regeneration (M6).
- No protocol mark: `supp_delta` (L8).

**B.2 Typed numbers, traced.**

| Number (App.) | Source | Status |
|---|---|---|
| 62.9/69.2/49.4, "about 60 ... 20-point spread" (E) | `gates/..._sliced2_seed*` 62.86/69.23/49.45; mean 60.5, spread 19.8 | OK |
| 48.0 with pseudo-labels; 44.1 bidirectional (E) | 48.01; 44.11 | OK |
| 61.2; diagonal sessions 3–5 31.8/14.4/23.0 (E) | oracle s42 61.17; diagonal 31.82/14.43/23.01 | OK |
| CoLaR fp32 34.1; replay sliced fp32 39.4 (E) | 34.06; 39.38 | OK |
| 42.0 over two seeds (E) | 44.17/39.85 → 42.01 | OK |
| k=11 control 14.8; diagonal 4.3/22.0/0.0; session 2 87.0 (E) | 14.77; 4.33/22.04/0.00; 87.02 | OK |
| No-memory SBR 20.4, "old blocks fall to 0 to 5" (E) | 20.36; final row 0.48/0.0/1.25/4.64 | OK |
| True-label refresh counts 3120–3307 / 2640–2932 / 0 / 305–316 (E) | logs: s42 3120/2640/0/316, s7 3307/2932/0/316, s123 3231/2792/0/305 | OK |
| Pseudo-label refresh "none on seeds 42 and 7", 7/0/0/36 on 123 (E) | s42 **0/0/0/84**, s7 0/0/0/0, s123 7/0/0/36 | **M4** |
| Calibration 85.2 → 85.4 → 85.8; DIL ≤0.7 / ≤1.1 (E) | 85.16 recomputed (`rescore/cil_cord_colar_sbr_seed42`, `amp=true`); 85.4/85.8 and 0.7/1.1 from `STATE.md` ticks 2026-11-05 09:30 / 00:40 | traced, not recomputed (needs `eval_dump.pt` via torch) |
| 402,689 = 378,112 + 24,576 + 1 (C) | 2·128·(709+768) = 378,112; (512·4 bbox + 512 mask + 512 labels)·8 = 24,576; +1 | OK |
| ×250 = 100,672,250 (C) | `replay_memory_bytes` of gate SBR ×3 and clean s42/s7 | OK |
| 272.3 MB uncompressed (C) | 250·709·768·2 = 272,256,000 | OK |
| 4,096 B per document labels_full (C) | 512 × int64 | OK (Part D) |
| 7,077,888 per rank-64 delta; 48 sites (C) | per layer 4·98,304 + 2·245,760 = 884,736, × 8; log "48 LoRA sites (r=64)" | OK |
| 56,623,104 bytes, "every rank-64 run" (C) | `replay_memory_bytes`: r64 s42/s7/s123 and Mixed r64b | OK |
| 294,912 per rank-8 delta; 2.4 MB for four (C) | 8·768·2·3·8; `recall_d0` 2,359,296 | OK |
| Mixed with memory 177,429,804 = 300 × 402,689 + 56,623,104 (C) | r64c `replay_memory_bytes`; four deltas (the revisit reuses the base) | OK |
| 3.7× = (4 + 5·8)/12 (C) | arithmetic | OK |
| CIL-CORD pools 800/382/549/723/780 (A) | gate steps 400/191/275/362/390 (×2, ±1); clean 360/172/247/326/351 = 90% | OK |
| Mixed pools (A) | see A.5 | OK |
| 61 tags; 12 per session; 25 Mixed; 709 positions (A) | code; 512 + 197 | OK |
| 33.3 legacy joint (B) | 33.43/33.57/32.94 | OK |
| SBR-Recall on Mixed: 62.1 bug run; 44.7; 85.9 (D) | r64 62.07 (matrix shows the wrong-delta damage); r64b 44.66; r64c 85.94 | OK |
| See-saw: 87.6/90.8 → 0.0; SROIE 82.0 → <10; CORD 95.8/95.5; revisit 87.3/85.6, others ≤8.1 (D) | r64b matrix 87.55/90.80 → 0.00/0.00; 81.99 → 9.94; 95.77/95.54; 87.32/85.63; 0.26/8.12/0.00 | OK |
| With memory: no column at zero; CORD >96; FUNSD-2 46–48, 75.0 (D) | r64c 46.15/46.64/47.91 → 75.05; 96.71–97.66 | OK |
| Delta: "within 1.5 points"; sessions 3–4 at 61–68; r64 88.8–96.2 (D) | 1.50 max; 61.5/68.0/64.9/67.7; 88.77–96.19 | OK |
| LexSlot "within two points ... below SBR-Recall" (D) | +1.7/−0.1/+0.8; every seed < Recall | OK (until M6) |
| `dil_colar_sbr_seed42` AA 66.5, receipt diagonal 23.6 (C) | 66.49, diagonal 86.7/23.6/97.9 | OK (L12) |
| Wall-clock: 2 h 01 min vs 1.95 h; SBR 2–2.5 h; Recall 1.5 h; joint <1 h (H) | queue.log 03:28 → 05:29; 1.953 h; gate SBR 2.40, clean 1.95/2.27; Recall 1.49; joint 0.35 | OK |
| `tab:prompt_lora` (G) | L2P/DualPrompt/CODA/O-LoRA/CL-LoRA/ER+C-Flat++ on LMv3 and BERT | all 24 cells OK |
| Rebalancing base 87.6; metaplastic 85.9 → 79.5; budget-adaptive +0.6±7.5 (F) | `dil_colar` s42 87.58; `STATE.md` colar_meta 85.85/79.48; mean/sd of +4.0/−8.0/+5.7 = 0.57/7.47 | OK (the last two traced) |

**Typed numbers I could not trace, or did not re-trace in R4.** They were carried over from the
pre-P6 supplement and are not new:
- App. A: "roughly 200 GPU-hours on a 24 GB card". No file records it.
- App. F:
  - the Fisher magnitudes "0.22 against 0.005";
  - the consolidation probe 42.6/84.7/86.1/86.2 and ER 87.9 with a 200-doc buffer;
  - lexical routing +1.8/+0.4/−0.6 and uniform 86.4;
  - the three-seed rebalancing check 87.8/87.0;
  - PLaR 76.1/~7/58.9;
  - the image paragraph (42.3/21.6, 76.2/54.8, 38.0/25.8, 10.7);
  - the ridge gaps 9.5/16.1, 7.5/14.3, 2.9/15.0, 7.7.

  The image and ridge tables are generated by other scripts outside R4's scope.
- Ledger table (F): the rows for merge, LCA 43.9±2.5, HGT/CUBER, LexSlot 42.2, the input-anchored
  ledger, LexMem 66.0/63.2/13.0, and CoLaR-R 20.0/59.5 (plain 20.4/58.8, oracle 26.3/64.4).
  These are traced in R1–R3 to `docs/EVIDENCE_LEDGER_2026-10-22.md` only.
- App. G: the CPFD/IS3/CODA null paragraph. R2/R3 checked it except the ablation F1s (0.00, 2.1)
  and CODA n50 (79.5/40.6/0.0/18.9/36.8, AA 7.4).

### Part C: method descriptions against the code

**Supervision plan (`tab:plan`): matches cell by cell.** I re-derived it from `supervision_plan`
and `task_domain` (domain from the task name: cord, funsd, sroie), using the `label_set` of each
session (own-session tags only; logs print 12 tags per CIL-CORD session).
- CIL-CORD: a document trains only its own block, with all other cells "skipped" (same schema).
- Mixed:

  | Session | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 |
  |---|---|---|---|---|---|
  | 1 | F | – | N | N | N |
  | 2 | – | F | N | N | N |
  | 3 | N | N | F | N | N |
  | 4 | N | N | N | F | – |
  | 5 | N | N | N | – | F |
  | 6 | F | F | N | N | N |

- The logs agree on every current-session row:
  - `gates/.../mixed_colar_sbr2_seed42.log`: session 2 trains [(0,neg),(1,neg),(2,full)], ...,
    session 5 [(0,full),(1,full),(2,neg),(3,neg),(4,neg)];
  - `clean_2026-11/cil_cord_colar_sbr_seed42.log`: [(s,full)];
  - `mixed_colar_sbr_recall_r64c` prints the same.

**Alg. 1 vs `colar_sbr.py`.**
- Block creation from new tags: OK.
- 10% held-out split: OK.
- Current loss: Σ over plan of `_block_ce`, each a token-mean CE on [o_b, z_{T_b}], with full
  targets from `block_targets` (in-block tag → 1+pos; any other valid token, including other
  blocks' entities → 0) and neg targets all 0. The neg weight `sbr_neg_weight` is 1.0 in every
  config and run; Alg. 1 omits it and App. H states it. OK.
- Replay: 4 documents, U·V rebuilt, injected at layer k+1 (pre-hook on `layer[k]`, 0-indexed);
  per-document `_session_loss` under the storing session's plan over the current blocks; mean over
  documents; weight 1. OK.
- AdamW: L3.
- Restore best, then freeze at s=1, then capture 50 documents from the frozen state (order in
  `_after_task_impl`). OK.
- Inference (`aggregate_blocks`): $e_b$ = 1 − softmax(...)[0]; strict `> threshold` (0.5); ties
  go to the lowest block; restriction via `model._allowed` applied **after** the block decision
  (`restrict_in_method`). OK.
- Nothing the code does that matters is missing.

**Alg. 2 vs `colar_sbr_recall.py`.**
- s=1 trains the base (`_cur_delta=None`), then `freeze_backbone()`: OK.
- A new block gives a fresh delta (A Kaiming, B = 0); trainable: current delta, head, all $w_b$: OK.
- Else reuse `block_delta[full[0]]`: OK. Unstated edge case: no new block and no full block gives
  a fresh, unused delta; it never occurs.
- Loss: `_loss_under_deltas` groups by delta, one forward per distinct delta, reusing the current
  forward: OK.
- Replay: grouped by session, injected under each block's delta: stated in the App. C text, not in
  the algorithm. Acceptable.
- Inference: one forward per distinct delta, then `aggregate_blocks`: OK.
- Delta form: $\tfrac{\alpha}{r}xA^\top B^\top$ on every Linear named query/key/value/dense in
  layers ≥ k (48 sites): OK.

**Sec. 5 equations.**
- Eq. 1 (block softmax), Eq. 2 (masked sum, $y_b$) and Eq. 3 (block under θ₀+Δ_b) match the code.
- The mask rule matches `supervision_plan`, but the stated reason is M5.
- $o_b$ bias: L2.
- Per-token vs per-block token-mean normalisation: immaterial.

**Scoring (App. B) vs `train.py`.**
- `restrict_preds`: the classifier hook maps an out-of-session argmax to O (`bad` → O logit 1e4),
  never re-argmaxes, and is eval-only. For SBR, `model._allowed` is applied after the block decision.
- `cil_allowed`: the session's `label_set` (with O). It is a no-op when the label set covers the
  head (DIL), and for images.
- `add_full_label_f1`: final model, outside any restriction (`_allowed` None), on
  `scenario.full_eval_dataset` (CORD test, 100 receipts, 61-tag head). Exceptions are swallowed
  (R1 L9; both clean logs print the value).
- Worked example: correct (P 1/2, R 1, F1 0.67; cross-type confusion = 1 FP + 1 FN).
- "Charged at most once" and the coverage caveat: correct.

### Part D: generators and hygiene

| Check | Result |
|---|---|
| `choose()` | Unchanged (diff shows only context). `pick` is refactored through `_candidates` with identical semantics. `chosen_source` and `matrix_choice` call the same `choose()`, so the source and seeds always come from the same rule. `make_retention_fig.py` now imports `matrices` (same rule, same seed order). Risks: M6 (`write_slot` pairing), L14 (race). |
| Side effects | `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex`, `tab_dil_grid.tex` and `figures/retention_cilcord.pdf` are byte-identical to 37e7df4 (no git diff). |
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10 (page 8 opens with the Ablations paragraph, then Conclusion). |
| Undefined references / overfull boxes | PASS. 0 / 0 in `main.log` and `supp.log`; one benign `h`→`ht` float warning in supp. |
| `---` in paper inputs | PASS. 0 in `*.tex` and `fig/*.tex`, generated files included. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and `supp.pdf` (14 pages). |
| Abstract | Byte-identical to 0a896ef (sha1 `fa56d93a…`). Still contains the OPEN clause. |
| Anonymity | `main.pdf`: PASS (no names, paths, dates, package names; empty PDF metadata). `supp.pdf`: R4-L15. |
| Every float referenced | PASS. Main: all figures and tables `\cref`'d. Supp: every table/figure/algorithm except `alg:recall` (L17). |
| Every appendix section referenced | PASS. The roadmap `\cref`s A–H; main cites A, B, C, D, E, F, G, H. |
| New clean runs since R3 | **None.** `cil_cord_colar_sbr_seed123` has been running since 07:49 (pid 3395401, no `metrics.json`); v4 (pid 3470572) waits on it. On disk: smoke naive (18.05 / −76.09 / 12.55), SBR s42 (87.76 / −2.41 / 88.83), SBR s7 (91.32 / −1.61 / 91.19). Flags as R2: `amp` and `fp16` false, `val_frac` 0.1, `cil_restrict_eval` and `full_label_eval` true. |

**`labels_full` (Paper agent's question; my independent reading).**
- **It is stored.** On CIL-CORD (`build_cil_cord`: `_wrap("train", full_head=snapshots[-1])`) and
  on Mixed (`build_mixed`: `CIL_LabelRemapper(..., snapshots[-1])`), every training item carries
  `labels_full`, the document's labels in the *final* 61- or 25-tag space. That includes classes of
  sessions not yet seen.
- `LatentReplay._capture_task` copies `bbox`, `attention_mask` and `labels_full` into each stored
  record. `CoLaR._capture_task` keeps them (only `hidden` is replaced by factors). So every
  CoLaR, CoLaR-SBR and SBR-Recall store on these scenarios holds it: 512 × int64 = **4,096 B per
  document**.
- **It is never read.**
  - `CoLaR._stack_replay` builds the replay batch from hidden, labels, bbox, attention_mask,
    input_ids, plus session for SBR. `labels_full` is not in it.
  - The only reader is `colar_r.py` (oracle refresh). It casts the field to int16; every other
    `colar_r` mode pops it at capture.
  - No result of CoLaR, SBR or Recall can depend on it.
- **Uncounted bytes per record.** `memory_bytes` omits `labels_full` and the `n_labels` scalar
  (int64, 8 B). It counts the session id (int64 in memory) as 1 B.
  - The record as held is 378,112 + 24,576 + 4,096 + 8 + 8 = **406,800 B**, against 402,689
    counted (+1.0%).
  - CIL-CORD CoLaR-SBR: 101,700,000 B as held, against 100,672,250 counted.
  - Mixed CoLaR-SBR: 122,040,000 B against 120,806,700.
  - Mixed Recall with memory: 178,663,104 B against 177,429,804.
  - DIL is unaffected: no `full_head`, and 60,403,200 = 150 × 402,688 exactly.
- **What the paper should state.** The figures are the method's memory, so they are not
  understated as a claim about the method. They are understated by 1% as a description of what the
  code holds. App. C already discloses the field correctly. I would add two things:
  1. The field holds labels of later sessions, which a continual learner does not have. It is a
     diagnostic artefact that the methods never read and a deployment would not store.
  2. With it, the record is 406,800 B (101.7 MB on CIL-CORD).

  No change to the body's "101 MB" is needed if the sentence says "the method's memory".

### OPEN (not counted)
- Abstract clause "an effect that no memory can repair" (user decision). The Conclusion now has a
  second echo (R4-M3), and the abstract's "when label growth itself supplies the routing key" has
  R4-M2's scope gap.
- CR2: † headline cells, pending data.

## R5 (2026-11-11): confirmation of P7 and the first clean cell

Snapshot: `HEAD = 4c43de9` (commits 17f3720, b8617f1, 4c43de9 on top of c53cf54). `main.pdf` and
`supp.pdf` were built at 09:55, after `main.tex` (09:55) and `supp.tex` (09:42). The generated
files are from 09:55, after the clean s123 `metrics.json` (09:53). Tracked paper files are clean
against HEAD.

Scope: `git diff c53cf54..4c43de9 -- paper_cvpr/ scripts/rca/` and the rebuilt PDFs. My own
scripts recomputed every changed number from `metrics.json`, `matrix.npy` and the logs. No model,
dataset, torch or pytest was touched.

### Verdict: **PASS for draft upload**

Every R4 MEDIUM item is fixed, and so is every LOW item the Paper agent claimed. The switch to the
clean protocol moved exactly the cells it should and nothing else. Every sentence comparing
CoLaR-SBR with another cell is true of the current values. There is no CRITICAL, HIGH or MEDIUM
finding. Five LOW items remain; one is a stale sentence in App. G caused by the switch (R5-L1).

### Status of the R4 items

| Item | Status | Current text / evidence |
|---|---|---|
| **M1**, Fig. 4 caption | CONFIRMED FIXED | "With a shared softmax the curves fall steeply and irregularly, some sessions partly recovering later, and later sessions are learned poorly even on their own diagonal; with the blocked readout most sessions hold and the remaining losses are uneven across sessions; under SBR-Recall every curve is flat to within about a point." Mean matrices: CoLaR column 3 is 37.1 → 66.3; clean SBR losses 3.1/9.0/1.6/3.1; the largest Recall per-seed deviation is 1.34 (s7, session 3). |
| **M1**, body "Retention per session" | CONFIRMED FIXED | "... most keep nearly all of their F1, but the losses are uneven, with one early session losing the most". Clean mean: session 2 loses 9.0, the others ≤ 3.1. It is a statement about the figure (the mean), not per seed: on s123 session 2 loses 16.0 and session 3 loses 9.6. |
| **M1**, App. D.1 (`supp_vals.tex`) | CONFIRMED FIXED | "the first two sessions lose \colarDropOne\ [47] and \colarDropTwo\ [28] points ... only to [19] to [37] F1 on their own diagonal ... rises by up to [29] ... loses [2] to [9] points on the mean, session [2] the most ... within [0.5] points of its diagonal". The text says these are computed from the mean matrices shown. Recomputed below. |
| **M2** | CONFIRMED FIXED | Conclusion: "On a single-schema stream, where label growth supplies the routing key, per-session low-rank deltas (SBR-Recall) can replace the memory; across annotation schemas they still need it." Sec. 5.3: "On a single-schema stream it works without a memory because ...". |
| **M3** | CONFIRMED FIXED | Conclusion: "in our experiments even a memory with true labels did not repair a readout that supervises one background class with contradictory labels." |
| **M4** | CONFIRMED FIXED | App. E: "rewrites $0/0/0/84$ tokens on seed 42, none on seed 7 and $7/0/0/36$ on seed 123 ($84$ against the $6{,}076$ that the true-label refresh rewrites on seed 42)". 6,076 = 3120 + 2640 + 0 + 316. |
| **M5** | CONFIRMED FIXED; accurate now | Sec. 5.2: "the document is annotated under its own schema, so no token carries a tag of $b$, and we treat every token as a negative for $b$ (a training-time convention; related schemas such as SROIE and CORD share entity kinds like totals and dates, so on Mixed it can label a real entity of another schema as background)". App. C: "This is a convention ..., not a fact about the document". It is now correct about SROIE/CORD: it names the shared kinds and says the rule can label them background. Wording nit (optional): "a token that is an entity of $b$'s kind". |
| **M6** | CONFIRMED FIXED | `SLOT_PAIRS` reads both sides from the gate protocol: `cil_cord_colar_sbr` with exact suffix (cannot reach `_slot`), `mixed_colar_sbr2`, `dil_colar_sbr2`. New macros: `\vSbrGate` 88.0±2.0 (87.62/86.27/90.15), `\vSlotGate` 89.7±0.5, `\vSbrGateMix` 89.7 (s42 89.73), `\vSlotGateMix` 89.6±1.0, `\vSbrGateDil` 86.2±1.3, `\vSlotGateDil` 87.0±0.4. `supp_slot.tex`: one gate† row per scenario, differences unchanged from R4. Main Ablations: "In the fp32 gate runs, the only protocol in which both exist (App. D), ... ($\nSbrGate$ against $\nSlotGate$ on CIL-CORD)". |
| L-Alg. 1 | CONFIRMED FIXED | "AdamW step on layers $k{+}1..12$ (session 1: all layers), head, all $w_b$". |
| L1, "cannot drift" | CONFIRMED FIXED | Sec. 5.3: "each old block's $e_b$ is unchanged up to weight decay, and only competition from new blocks can move its tokens (App. C)". The new App. C paragraph gives "under $1\%$ over a run". Check: 10 epochs × (191+275+362+390) steps × 5e-7 = 0.61%. It also gives "about a point in either direction" (r64: −0.8 to +1.3). |
| L4, almost frozen trunk | CONFIRMED FIXED | App. E takeaway and both later sentences now read "almost frozen trunk (only the last layer and the readout plastic)"; also "on the same trunk". |
| L8, dagger on `supp_delta` and two captions | CONFIRMED FIXED | Seed column `42$^{\dagger}$` …. `tab:delta`: "fp32 gate runs$^\dagger$, ..., test-split early stopping". `tab:recallmix` and `tab:slot` captions carry †. |
| L7, legacy-settings rows | CONFIRMED FIXED | Three new rows: "Gate controls, tab:conflict \| CIL-CORD \| 10 \| fp32 \| 50 docs/session \| 1 or 3"; "SBR, layers 1..11 frozen \| 10 \| fp32 \| none \| 1"; "SBR-Recall, rank 8 \| 10 \| fp32 \| 50 docs/session, none \| 1, 1". All agree with `training_hparams` / `method_hparams` (R4 B.2). |
| GPU-hours line | FIXED, new LOW | Replaced by "Every run fits a single 6 GB GPU; a CoLaR-SBR run on CIL-CORD takes about two hours of training loop" (clean 1.95/2.27/2.00 h). The first clause is new and only partly true (R5-L3). |
| L15, internal names and paths in `supp.pdf` | CONFIRMED FIXED | pdftotext of `supp.pdf`: 0 hits for `doccl`, `results/`, `scripts/`, `.py`, `2026-1x`, `_seed`, `_CORD`, `clean_2026`, `cil_protocol`. Run names are paraphrased ("rank 64, seed 42"). Remaining hardware mentions (6 GB, RTX 2060) are acceptable. `main.pdf`: 0 hits. |
| L2, bias term | CONFIRMED FIXED | Sec. 5.2 "$o_b(x)=w_b^{\top}h(x)+c_b$"; App. C "a linear layer with bias". |
| L6, delta storage | CONFIRMED FIXED | Cost: "($57$ MB if stored in half precision, $113$ MB in fp32 as trained and evaluated)". App. C: "This is a reporting choice. ... no run evaluated half-precision-rounded deltas." 4 × 7,077,888 × 4 B = 113.2 MB. |
| L5, memory disclosure (PM decision) | CONFIRMED | Cost "$100.7$ MB on CIL-CORD, the method's memory as counted in App. C". App. C: "As held, a record is $406{,}800$ bytes against $402{,}689$ counted: $101.7$ against $100.7$ MB ... $122.0$ against $120.8$ MB for the $300$ of Mixed"; Mixed Recall "$177.4$ MB counted ... ($178.7$ MB as held)". Matches R4 Part D. |
| L9, L10, L11, L12, L13, L16, L17 | CONFIRMED FIXED | `tab:dil` caption "(... saved; App. G)"; App. B "so for CoLaR-SBR the restricted AA ..."; Sec. 4.2 "the number of stale labels the true-label refresh rewrites"; App. C "the SROIE session ... (one seed, AA 66.5)"; App. A "seeded by the run seed plus the session index (the run seed for joint training)"; App. A "three annotation schemas (FUNSD for forms; SROIE and CORD for receipts)"; App. C "\cref{alg:recall} summarises training and inference". |
| New CIL-CORD refresh counts (App. E) | CONFIRMED | `results/logs_vision/cil_cord_colar_r_oracle_seed{42,7,123}_k4_d50_r128.log` give 101/760/1461/1402, 66/944/1366/1269 and 74/943/1315/1110. The gate run behind the 61.2 (`gates/.../cil_cord_colar_r_oracle_seed42.log`) prints the same 101/760/1461/1402. So "Under restricted scoring it reaches 61.2" does not conflate two different label rewrites. |
| L14 (race) | not a text item; did not bite | This regeneration ran at 09:55, after the s123 `metrics.json` (09:53). |

**`supp_vals.tex`: how it is computed and my recomputation.** `write_supp_vals` averages the
matrices that `matrix_choice` returns, so it uses the same source and seeds as `tab:matrices`:
- CoLaR: rescore, s42/s7;
- CoLaR-SBR: clean, 3 seeds;
- Recall: gate r64, 3 seeds.

It then computes:
- `colarDropOne` = R[1,1] − R[2,1] = 94.66 − 48.12 = **46.5 → "47"**;
- `colarDropTwo` = R[2,2] − R[3,2] = 81.95 − 54.39 = **27.6 → "28"**;
- late diagonal (sessions 3 to 5): 37.07 / 34.00 / 18.57 → **"19" to "37"**;
- largest later rise in any column: column 3, rows 3 → 4, 37.07 → 66.31 = **29.2 → "29"**;
- SBR loss, diagonal − final, sessions 1 to 4 on the clean mean: 3.14 / **8.98** / **1.57** / 3.08 → "2" to "9", session "2";
- Recall: max |diagonal − final| of the mean = **0.54 → "0.5"**.

All match. The App. D.1 sentence says "on the mean", so the per-seed spread is out of its scope:
SBR session 2 loses 9.4 on s42, 1.6 on s7 and 16.0 on s123, and session 3 gains 6.6 on s42. The
body and Fig. 4 say "about a point" for Recall, which holds per seed (max 1.34).

### The switched cell: CoLaR-SBR on CIL-CORD (clean, three seeds)

- **Run.** `results/clean_2026-11/cil_cord_colar_sbr_seed123`:
  - flags: `amp` false, `fp16` false, `val_frac` 0.1, `cil_restrict_eval` true, `full_label_eval` true (log config dump);
  - method: epochs 10, docs 50, k 4, r 128;
  - `replay_memory_bytes` 100,672,250;
  - AA 86.16, BWT −8.55, `full_label_f1` 87.13 (log: "Full-label final F1 (unmasked test set): 87.13"; no "skipped" line);
  - training loop 2.00 h.
  - It has no DONE line in `queue.log`: v3 orphaned it at 08:14, and v4 started the next job at 09:53. The files are complete.
- **Recomputed** (42/7/123):
  - AA 87.76 / 91.32 / 86.16 → **88.41 ± 2.64**;
  - full-label 88.83 / 91.19 / 87.13 → **89.05 ± 2.04**.
  - These match `\vSbr` 88.4±2.6, `\vSbrFull` 89.1±2.0 and the 2×2 and main-table cells, now unmarked.
- **What moved.** `git diff c53cf54..4c43de9 --stat -- 'paper_cvpr/*.tex'` touches only:
  - `vals.tex`: `\vSbr`, `\vSbrFull`, plus the six new gate macros;
  - `tab_2x2.tex`: CoLaR-SBR cell;
  - `tab_main_clean.tex`: CoLaR-SBR CIL-CORD AA and full-label;
  - `supp_perseed.tex` and `supp_conflict.tex`: the CoLaR-SBR row → "clean & 87.8 & 91.3 & 86.2 & 2.1";
  - `supp_agree.tex`: + s123;
  - `supp_mat_sbr.tex`: clean mean, which I recomputed to 0.05;
  - `supp_slot.tex`: M6;
  - `supp_delta.tex`: daggers only;
  - `supp_vals.tex`: new;
  - `figures/retention_cilcord.pdf`: 21397 → 21403 bytes, regenerated at 09:55 from the same `C` registry. Only the SBR source changed, so only the middle panel changed.
- **What did not move.** Under `choose()`, no other cell has three clean seeds: clean Recall s42 is running with no `metrics.json`, and no other clean run exists. Every other generated value is byte-identical, including:
  - Mixed SBR (rescore 89.8/89.6†) and DIL SBR (gate 86.2±1.3†);
  - the supp Mixed matrices;
  - the gate CoLaR-SBR runs, which are still used for the LexSlot comparison only.

**Sentences that compare CoLaR-SBR on CIL-CORD with another cell.** Per seed against legacy joint
(92.84/90.48/92.43): −5.1 / +0.8 / −6.3. Against seed-matched naive, SBR closes 93.1% / 101% /
91.5% of the naive-to-joint gap.

| Location | Text | Verdict |
|---|---|---|
| Abstract (source byte-identical, renders 88.4 instead of 88.0) | "together they attain $\nSbr$ average F1 against $\nJoint$ for joint training" | True as stated: it is a plain report of two means and claims no closeness. The cross-protocol pairing is visible in `tab:twobytwo` (88.4 unmarked, 91.9†). The PM should tell the user that the rendered number changed. |
| Intro ¶3 | "the two together reach $\vSbr$, against $\vJoint$ for joint training" | True. |
| Contribution 3 | "closes most of the gap to joint training on class-incremental CORD and reaches it on a mixed stream" | True per seed (≥ 91% of the gap on every seed). "Reaches it" on Mixed: 89.77/89.62 vs joint 86.94/88.46, both legacy. The change from "within a few points" was right: 6.3 points on s123 is not "a few". |
| Teaser caption | "together they close most of the gap to it ... bars are the cell means of tab:twobytwo, which gives the seeds and protocol of each cell" | True, and protocol-visible. |
| Sec. 4.3 | "Only the cell with both approaches joint training ($\vSbr$ against $\vJoint$)" | True. "Approaches" is not a per-seed "within" claim. |
| Experiments | "the two blocked readouts close most or all of the gap: CoLaR-SBR $\vSbr$ and SBR-Recall $\vRecall$ against $\vJoint$" | True. CoLaR-SBR is clean, the other two are legacy; this is visible from `tab:main`'s marks. |
| Ablations | "what SBR still loses to joint training is retention of early sessions (Fig. 4)" | Consistent with the clean matrix: mean diagonal-to-final losses sum to 16.8, i.e. 3.4 AA of the 3.5-point gap. |
| Conclusion | no CoLaR-SBR number | — |
| App. D.1, D.3 | `supp_vals` macros (clean mean) | True (above). The LexSlot comparison is now gate-only (M6). |
| App. B | "differ by about one point or less on each seed ... for CoLaR-SBR ... not inflated" | True: +1.1 / −0.1 / +1.0. |
| App. E takeaway | "a session-blocked readout with the same memory moves it to $\nSbr$" | True. The row is tagged "clean" in `tab:conflict`, beside gate† and rescore† rows. |
| App. G | "The CIL-CORD cells of CoLaR-SBR, no-memory SBR, SBR-Recall and LexSlot ... are fp32 gate runs" | **Stale (R5-L1).** |

The Limitations promise ("Every cell is marked by its protocol and never mixes the two, so a
comparison across protocols is visible from the marks in the tables") holds for every table. It
does not hold for Fig. 4, whose panel titles give n but not the protocol (R5-L2). Each comparison
favours the reference rather than SBR, because the legacy joint runs early-stopped on the test
split.

### New findings (LOW only)

- **R5-L1. App. G is stale after the switch.**
  - Where: supp.tex:494, "The CIL-CORD cells of CoLaR-SBR, no-memory SBR, SBR-Recall and LexSlot, and the DIL SBR cells, are fp32 gate runs. In the $2{\times}2$ the shared-softmax cells are therefore fp16 or mixed and the blocked-readout cells fp32."
  - The CoLaR-SBR CIL-CORD cell is now clean. The main 2×2 caption is still correct: "Among the † cells, the blocked-readout ones ran in fp32".
  - Fix: drop "CoLaR-SBR" from the list, and write "In the $2{\times}2$ the shared-softmax $\dagger$ cells are fp16 or mixed and the blocked-readout $\dagger$ cells fp32". The `tab:legacyprotocol` row "CoLaR-SBR \| CIL-CORD \| fp32 \| 3" can stay, but should say "(LexSlot comparison only)".
- **R5-L2. Fig. 4 does not show protocol per panel.** The middle panel is now clean and the outer
  two are legacy (CoLaR rescore†, Recall gate†). Add "(CoLaR-SBR: clean protocol; CoLaR and SBR-Recall: $\dagger$)" to the caption, or put the protocol in the panel titles.
- **R5-L3. The new App. A sentence "Every run fits a single 6 GB GPU" is not supported for every run.**
  App. H says some first-grid rows ran on a rented GPU container, and Table 1's "--" cells did not
  fit. Write "The runs reported here ran on a single 6 GB GPU, except some first-grid rows (App. H)".
- **R5-L4. `supp_agree.tex` lists seeds in the order 123, 42, 7.** That is the `sorted(glob)` string
  order. Sort by `SEEDS`.
- **R5-L5. Note for the PM.** The clean s123 run has the largest forgetting of the three clean seeds
  (BWT −8.55; session 2 drops 92.0 → 76.0), and no text cites a CoLaR-SBR BWT. Clean BWT is
  −2.41 / −1.61 / −8.55 against gate −7.50 / −8.99 / −4.30.

### Hygiene

| Check | Result |
|---|---|
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10. |
| Undefined references / overfull boxes | PASS. 0 / 0 in `main.log` and `supp.log`. |
| `---` in paper inputs | PASS. 0 in `*.tex` and `fig/*.tex`, generated files included. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and in `supp.pdf` (now 15 pages). |
| Abstract | Byte-identical to 0a896ef (sha1 `fa56d93a…`). It renders "attain 88.4 average F1 against 91.9". |
| Anonymity | PASS for both PDFs (see L15 above). |

### Other clean runs

None has finished besides s123. `cil_cord_colar_sbr_recall_r64_seed42` started at 09:53 (v4,
pid 3470572); its directory holds only `tb/`.

### OPEN (not counted)

- Abstract clause "an effect that no memory can repair" (user).
- Abstract clause "when label growth itself supplies the routing key" (user). The Paper agent
  proposes adding "on a single-schema stream", which I support: it matches the corrected
  Conclusion and Sec. 5.3.
- CR2: the remaining † headline cells, pending data.

## R6 (2026-11-11): audit of P8, the clean SBR-Recall cell

Snapshot: `HEAD = d559dd7`. Tracked paper files are clean against HEAD. The generated files (15:13)
postdate the last clean Recall `metrics.json` (s123, 14:54). `main.pdf` and `supp.pdf` (15:14/15:15)
postdate `main.tex`. I rebuilt both from `git archive d559dd7` in a scratch directory, so no tracked
file was touched. The pdftotext of each rebuild is byte-identical to the committed PDF.

Scope: `git diff d50d267..d559dd7 -- paper_cvpr scripts/rca/make_clean_tables.py scripts/rca/make_retention_fig.py`.
My own script recomputed every number from `metrics.json`, `matrix.npy` and the logs. No generator,
model, dataset, torch or pytest was run. The clean `cil_cord_colar_sbr_noreplay_seed42` was running
during the audit (started 14:54, only `tb/` so far). No paper file reads it.

### Verdict: **PASS**

The three runs are clean and complete, and every Recall value they feed recomputes exactly. The
diff moves only the Recall cells, plus a reordering of `supp_agree`. Every comparative sentence
about SBR-Recall holds per seed. The single cross-protocol pairing (clean Recall and CoLaR-SBR
against fp16 legacy joint) is stated truthfully, in the 2×2 caption and again in App. G. R5 L1–L4
are fixed. There is no CRITICAL, HIGH or MEDIUM finding. Five LOW items follow.

### A. The three clean SBR-Recall runs

**Config.** The config dumps (log lines 1–40) are identical except for `seed`:
- `method.name colar_sbr_recall`, `lora_rank 64`, `lora_alpha 128`, targets query/key/value/dense;
- `epochs 10`, `docs_per_task 0`, `split_layer_k 4`;
- `amp false`, `fp16 false`, `val_frac 0.1`, `cil_restrict_eval true`, `full_label_eval true`;
- batch 2 with gradient checkpointing, `output_dir results/clean_2026-11`.

`metrics.json` `training_hparams` / `method_hparams` agree with the dumps. `rank_r: 128` is inert
because there is no memory.

**Completion.** Every run has a DONE line in `queue.log`:
- s42: 09:53 to 11:27;
- s7: 11:27 to 13:07;
- s123: 13:07 to 14:54.

Each log has "Saved matrix", "Saved metrics" and "Full-label final F1" lines (91.53 / 93.52 / 92.62),
with no "skipped" line and no traceback. The only ERROR lines are the HF `trust_remote_code` notice,
which every run shows (CoLaR-SBR clean s42 too).

**Other fields.** `replay_memory_bytes` is 56,623,104 on all three: the four deltas at 2 B per
parameter (4 × 7,077,888 × 2), i.e. the "57 MB if stored in half precision" of the Cost paragraph.
`matrix.npy` equals `metrics.json["matrix"]` on all three.

**Matrices (rows: after session i).** No collapsed session:
- diagonals 92.6/86.9/94.5/85.6/94.9 (s42), 95.0/89.0/90.5/91.7/94.3 (s7), 94.1/89.3/91.1/90.7/94.4 (s123);
- BWT +0.60 / +0.64 / −0.06;
- the only exact zeros are the first super-diagonal (the zero-shot reading of a session's not-yet-added labels). Clean CoLaR-SBR and gate Recall have the same zeros, so they are systemic, not a fault.

**Recomputed (42/7/123).**

| Value | Per seed | Mean ± s.d. | Generated | OK |
|---|---|---|---|---|
| AA | 91.35 / 92.62 / 91.86 | 91.94 ± 0.64 | `\vRecall` 91.9{\pm}0.6, `\nRecall` 91.9; 2×2 and main-table cell bold, unmarked | ✓ |
| full-label F1 | 91.53 / 93.52 / 92.62 | 92.56 ± 0.99 | `\vRecallFull` 92.6{\pm}1.0; main-table full-label cell, no n mark | ✓ |
| full − AA | +0.18 / +0.89 / +0.76 | | `supp_agree` +0.2/+0.9/+0.8, seeds in 42/7/123 order | ✓ |
| h/run (`total_wall_time_s`) | 1.45 / 1.53 / 1.67 | 1.55 | `supp_perseed` "clean & 91.4 & 92.6 & 91.9 & 1.6" | ✓ |
| mean matrix | | | `supp_mat_recall` (title "seeds 42/7/123, clean") equals my mean to 0.05 in every cell | ✓ |
| `\recallDevMax` = max \|diag − final\| of the mean | session 2: 88.37 → 89.68 | 1.31 | "1.3" | ✓ |
| `\nRecallJointGap` = max \|Recall − joint\| | −1.49 / **+2.14** / −0.57 | | "2.1" | ✓ |
| `\nRecallMaxLoss` = largest R[i,j] < R[j,j] | s123 session 4, 90.66 → 90.11 = 0.549 | | "0.5" | ✓ (rounding: see L1) |
| `\nRecallMaxGain` | s42 session 2, 86.86 → 88.89 = 2.03 | | "2.0" | ✓ |

The joint reference is −1.49 at s42 and +2.14 at s7; Recall is above joint at s7.

**Joint cell.** `\vJoint` 91.9{\pm}1.3 (92.84 / 90.48 / 92.43) is `results/rescore/cil_cord_joint_seed{42,7,123}`.
All three log "Mixed precision enabled" plus a `GradScaler`, i.e. fp16 on the Turing card. Their
config dumps match the clean joint recipe in every key except `amp: true` (and the absent
`full_label_eval`):
- `epochs 100`, `val_frac 0.1`, `cil_restrict_eval true`;
- batch 2, gradient checkpointing, lr 5e-5, wd 0.01, clip 1.0.

So this cross-protocol pairing differs **only in precision**, and the "(fp16)" in the 2×2 caption
is correct.

**What moved.** In the generated files, the diff touches:
- `vals.tex`: `\vRecall`, `\vRecallFull`, and the three new macros;
- `tab_2x2.tex` and `tab_main_clean.tex`: the Recall row only;
- `supp_perseed.tex`: the Recall row only;
- `supp_mat_recall.tex`;
- `supp_vals.tex`: `\recallDevMax` only;
- `supp_agree.tex`: three Recall rows plus the seed reorder;
- the figure.

Every other generated cell is byte-identical (`git diff --stat`). Rule 6 holds: Recall switched with
3/3 clean seeds. Clean no-memory SBR has 0 complete seeds and is ignored by `choose()`. No other
clean source exists.

**Generator review.**
- `recall_macros()`: the gap uses `chosen_source` of both cells, seed-matched, with abs: the right definition for "within X on every seed". The moves are `mat[i,j] − mat[j,j]` for i ≥ j over the matrices of the chosen source, on every seed. That is exactly "largest fall below / rise above its diagonal at any later boundary, on any seed". `max(0, −min)` is correct because the diagonal term contributes 0.
- AA (`metrics.json`) and matrices (`matrix.npy`) are chosen by two separate `choose()` calls. They agree here, and could only disagree if a run had one file and not the other (train.py writes the matrix first).
- Seed sort: `(scenario, method, SEEDS.index)` is correct.
- `ROW_LABEL` maps `colar_sbr_recall_r64` → "SBR-Recall" (see L5 for BROS).
- `make_retention_fig.py`: it now uses `matrix_choice` (the same rule as the tables). Its titles read "CoLaR (n=2, †)", "CoLaR-SBR (n=3, clean)", "SBR-Recall (n=3, clean)", which is correct (CoLaR = rescore s42/s7). Its output path is resolved from `__file__`.

### B. Paper audit

| Location | Rendered text | Per-seed check | Verdict |
|---|---|---|---|
| Teaser caption | "SBR-Recall comes within 2.1 points of it on every seed without a memory" | \|−1.49\|, 2.14, \|−0.57\| ≤ 2.14 | True. The bars read 18.9/20.4/30.4/88.4/18.6/91.9 from `vals.tex`, with the joint line at 91.9. |
| 2×2 caption | "The two bold cells are clean; joint training is still a † cell (fp16), so until its clean runs complete a comparison of either bold cell with it crosses protocols" | Bold = 88.4±2.6 and 91.9±0.6, both clean. The only blocked-readout † cell is no-memory SBR (gate, fp32), which matches "the blocked-readout one in fp32". | True |
| Sec. 4.3 | "SBR-Recall ... reaches 91.9±0.6" | — | True |
| Main comparison | "the two blocked readouts close most of the gap, SBR-Recall all or nearly all of it on every seed" | Gap closed against seed-matched naive (18.97/19.08/18.60): Recall 98.0% / 103.0% / 99.2%; CoLaR-SBR 93.1% / 101% / 91.5% | True |
| Fig. 4 caption | "no session falls more than 0.5 points below its diagonal on any seed, and the larger moves, up to 2.0, are gains" | Max loss 0.549. Every move > 0.55 is a gain (2.03, 1.88, 1.29, 0.63, ...). | True to one decimal (L1) |
| Retention ¶ | "the curves are flat, no session losing more than 0.5 points on any seed" | As above. It omits the gains up to 2.0 that the caption states. | True (L1) |
| Ablations | "rank 64 ... learns them on all three seeds, both in legacy runs ... and in the clean runs of Tab. 3" | Clean diagonals of sessions 3/4: 94.5/85.6, 90.5/91.7, 91.1/90.7, against 61–68 at rank 8 | True |
| Limitations | "visible from the marks in the tables and in the panel titles of Fig. 4" | Titles carry the protocol | True |
| App. B | "For the CoLaR-SBR and SBR-Recall runs listed the two differ by about one point or less on each seed" | Recall +0.2/+0.9/+0.8; SBR +1.1/−0.1/+1.0 | True |
| App. C "cannot drift" | "in the clean rank-64 runs no session falls more than 0.5 points below its diagonal at any later boundary and none rises more than 2.0" | 0.549 / 2.03 | True to one decimal (L1) |
| App. D.1 | "every column of the mean matrix ends within 1.3 points of its diagonal" | 1.31 (mean, session 2, a gain) | True to one decimal (L1) |
| App. D.3 LexSlot | "It stays below SBR-Recall on CIL-CORD on every seed, against both the gate and the clean SBR-Recall runs" | LexSlot gate 90.2/89.7/89.2 < clean 91.35/92.62/91.86 and < gate 92.69/93.37/92.44 | True. The cross-protocol side is named explicitly, so this is not a silent mix. |
| App. D.4 delta | "All rows of tab:delta are legacy gate runs ...; the clean rank-64 runs ... learn sessions 3 and 4 as well" | Clean range 85.6–94.5 vs the gate's 88.8–96.2 | Ambiguous (L2) |
| App. G | "the CIL-CORD cells of CoLaR-SBR and SBR-Recall are clean, and their fp32 gate runs appear only in the LexSlot comparison and the delta-capacity table" | The CoLaR-SBR gate run feeds `tab:slot` and `\vSbrGate`. The Recall gate run feeds `tab:delta` and the LexSlot sentence. `tab:conflict` "CoLaR-SBR" reads `C["sbr"]` = clean. The § footnote is scoped "On CIL-CORD", so the Mixed Recall gate cell (85.9†,b) is correctly not covered. | True |
| App. H wall-clock | "an SBR-Recall run without memory about 1.5 to 1.7 (clean runs)" | 1.45 / 1.53 / 1.67 h | True, but typed (L3) |

**Cross-protocol statement.** It appears once in the body (the 2×2 caption) and once in App. G, which
is where the protocols are stated. No remaining sentence says or implies that Recall and joint
share a protocol. "SBR-Recall 91.9±0.6 against 91.9±1.3" in the Main comparison relies on the table
marks, as every other cell does.

**The abstract's "reaches 91.9, on par with joint training"** (source unchanged, sha1 `163dd69a`
before and after; advice only).

The evidence:
- The means are 91.94 against 91.92.
- The seed-paired differences are −1.49 / +2.14 / −0.57: mean +0.03, s.d. 1.9, with both signs.
- The largest deficit (1.5, s42) is about one joint s.d. (1.26). The largest gap (2.1) favours Recall.
- The protocols differ only in precision (above).

My judgement is that **"on par" is defensible as rendered**: it claims neither "above" nor
"matches", and on every seed the two are within 2.1 points in a direction that changes sign. With
n = 3 it means "not distinguishable", not "equal". If the user wants wording that holds per seed
by construction and survives a reviewer reading rule-7 strictly, I would use "reaches $\nRecall$,
within $\nRecallJointGap$ points of joint training on every seed". Both macros exist, and the text
then tracks the clean joint cell when it lands. Either way, re-check this clause when clean joint
lands: if clean joint rises (fp32) by more than about 1.5 on s42, "on par" will need re-checking.

**R5 L1–L4.**

| Item | Status |
|---|---|
| L1, App. G stale | FIXED. The text quoted above; the 2×2 phrase is now "the blocked-readout † cell fp32"; § rows added. |
| L2, Fig. 4 protocol | FIXED. Panel titles give "(n=…, clean/†)", and the caption explains †. |
| L3, 6 GB sentence | FIXED. "The runs reported here ran on a single 6 GB GPU, except some first-grid rows (app:repro)". |
| L4, seed order | FIXED. 42/7/123 for both methods. |

### New findings (LOW only)

- **R6-L1. Upper bounds are rounded to nearest, so three "no more than / within" bounds are below the true maximum by 0.01 to 0.05.**
  - Where:
    - `vals.tex` `\nRecallMaxLoss` "0.5": true 0.549 (s123, session 4). Used in the Fig. 4 caption, the Retention ¶ and App. C.
    - `\nRecallMaxGain` "2.0": true 2.03 (s42, session 2). App. C reads "none rises more than 2.0".
    - `supp_vals.tex` `\recallDevMax` "1.3": true 1.31.
  - At the paper's one-decimal precision these are the right values, and no per-seed matrix is printed, so no reader can see a contradiction. Strictly, though, "falls no more than 0.5" is false for a 0.55 loss.
  - Fix, either:
    - (a) round bound macros up in `recall_macros()` / `write_supp_vals()` (`math.ceil(10*x)/10` gives 0.6 / 2.1 / 1.4); or
    - (b) word them "about 0.5", "about 2".
  - Optional in the same edit: give the body Retention ¶ the caption's "and the larger moves, up to \nRecallMaxGain, are gains", since "the curves are flat" omits a 2-point rise.
- **R6-L2. App. D.4, "the clean rank-64 runs ... learn sessions 3 and 4 as well" (supp.tex:298) can be read as "equally well".**
  - The clean diagonals of sessions 3 and 4 span 85.6 to 94.5. The s42 session-4 value is 85.6, below the gate range of 88.8 to 96.2 quoted two sentences earlier.
  - Fix: "also learn sessions 3 and 4 (\cref{tab:matrices})", or give the range from a generated macro.
- **R6-L3. App. H wall-clock "about 1.5 to 1.7 (clean runs)" is a typed run value (rule 6).**
  - It is true (1.45 / 1.53 / 1.67 h), and earlier rounds accepted the typed CoLaR-SBR "2 to 2.5".
  - Fix (optional): point to the h/run column of `tab:perseed` (1.6), which is generated.
- **R6-L4. Advisory: the 2×2 caption and App. G could say that the legacy joint cell differs from the clean protocol only in precision.**
  - Its config dumps have `val_frac 0.1`, 100 epochs and the clean recipe otherwise; `amp true` gives fp16 on all three seeds.
  - This is stronger and more useful than the generic "crosses protocols". App. G's "three ways" do not apply to joint.
  - Fix (optional): "(fp16; otherwise the clean recipe)".
- **R6-L5. Latent generator label (`write_agree`).**
  - The queued clean BROS runs have `run_alias colar_sbr_bros`. That maps to the key `sbr_bros`, which `ROW_LABEL` lacks, so the agreement table would print "sbr\_bros".
  - Fix: add `"sbr_bros": "CoLaR-SBR (BROS)"` before those runs land. No effect today.

### Hygiene

| Check | Result |
|---|---|
| Build | PASS. `latexmk -pdf main.tex` = 0, `supp.tex` = 0 (scratch rebuild of d559dd7). 0 undefined references or citations in `main.log` and `supp.log`. |
| Body ≤ 8 pages | PASS. `main.pdf` is now **11** pages (10 at d50d267). "References" starts on p. 8, right column, after the Conclusion; pp. 9–10 are references, and p. 11 holds only ref. [52]. No float or body text after p. 8. Note for the PM: Overleaf will now show 11 pp. The body itself did not cross p. 8. `supp.pdf` is 15 pp. |
| `---` | PASS. 0 in every tracked `paper_cvpr/*.tex` and `fig/*.tex`; 0 em-dashes in either PDF. |
| Anonymity | PASS. 0 hits for doccl / results/ / scripts/ / .py / clean_2026 / _seed / author or affiliation strings in either PDF. PDF Author fields are empty. |
| Abstract | Source byte-identical to d50d267. It renders "attain 88.4 average F1 against 91.9" and "reaches 91.9, on par with joint training". |
| Teaser / figure | The teaser bar values and labels come from `vals.tex` macros and match. Fig. 4 titles state n and protocol correctly. |

### OPEN (not counted)

- Clean joint and no-memory SBR on CIL-CORD are queued; when they land, re-check the abstract's "on par" (above), `\nRecallJointGap`, and the 2×2 caption sentence, which will then be stale.
- CR2: the remaining † headline cells, pending data.


## R7 (2026-11-11): audit of P9, the story rewrite, and a referee report

Snapshot: paper at `00dd224` (HEAD has since gained Dev commits `b27c824`, `25f4d2f`, which touch no
paper file). Tracked paper files are clean against `00dd224`. I rebuilt both PDFs from
`git archive 00dd224` in my scratchpad; the pdftotext of each rebuild is byte-identical to the
committed `main.pdf` / `supp.pdf`. Scope: `git diff 2dc86ad..00dd224 -- paper_cvpr
scripts/rca/make_clean_tables.py scripts/rca/make_retention_fig.py`, plus the whole new `main.tex`
read against the old one. Every number below was recomputed by my own script from `metrics.json`,
`matrix.npy`, config dumps and logs. No generator, model, dataset or pytest was run. During the
audit `cil_cord_colar_seed123` (clean) was still running; no paper file reads it yet.

### Verdict: **BLOCK** (one HIGH, text-only, about five minutes to fix)

Every generated number recomputes exactly, and the new prose is accurate almost everywhere. What
blocks is one headline number. The abstract's "closes only 46% of it" is a ratio whose terms come
from two protocols, and neither `main.tex` nor `supp.tex` says so. It rests on one seed, which the
abstract does not say. And a reader who computes it from the paper's own Table 2 gets 50%, not 46%.
"About half" is true on every way of computing it. Six MEDIUM and ten LOW items follow. The
bibliography additions are correct, and so is every sentence about what MiB, PLOP, SSUL and
ExtendNER do. One sentence omits that SSUL pseudo-labels, and the SSUL positioning is weaker than
it reads (M6).

### A. Numbers recomputed

| Macro / claim | Source (my recomputation) | Rendered | OK |
|---|---|---|---|
| `\nErDil` / `\nJointDil` (abstract, intro) | `results/dil_er_seed{42,7,123}` 86.21/88.73/88.62; `dil_joint` 89.68/89.03/87.49 (first grid: test-split stopping, config not saved, ER buffer 200) | 87.9 / 88.7 | ✓ |
| `\nEr` / `\nJoint` | `rescore/cil_cord_er` 26.91/22.05/24.01 (buffer 200, test-split stopping, fp16 except s42); `rescore/cil_cord_joint` 92.84/90.48/92.43 (amp true, `val_frac` 0.1) | 24.3 / 91.9 | ✓ |
| `\nOracle` 61.2 | gate `cil_cord_colar_r_oracle_seed42_k4_d50_r128` AA 61.171; amp false, no `val_frac` (test-split stopping) | 61.2 | ✓ |
| `\nColarGateSeed` 34.1 | gate `cil_cord_colar_seed42_k4_d50_r128` 34.063. The config dump equals the oracle's except `method.name/run_alias/refresh`, so "the same memory without the refresh" is true. | 34.1 | ✓ |
| `\nOracleShare` 46 | (61.171 − 34.063)/(92.837 − 34.063) = 46.1%. The denominator's joint term is **rescore seed 42, fp16, held-out stopping**; the other two terms are fp32 gate runs with test-split stopping. | 46 | value ✓, disclosure ✗ (H1) |
| `\nOracleRelabel` | oracle log: "refreshed 50/100/150/200 docs, 101/760/1461/1402 tokens relabelled" | 101/760/1461/1402 | ✓ |
| `\nOracleLateMin/Max` | oracle diagonal 95.16/79.09/**31.82/14.43/23.01** | 14 to 32 | ✓ (gate CoLaR, no refresh: 14.7/10.2/11.4) |
| `\nSlicedReplay` 42.0 | `rescore/cil_cord_colar_r_sliced` 44.17/39.85 (fp16) | 42.0 (two seeds) | ✓ (L3) |
| `\nSlicedBoth` 60.5 | gate `sliced2` 62.86/69.24/49.45 (fp32) | 60.5 | ✓ |
| `\nPseudoBoth` 44.1 | gate `bilc` s42 44.108 | 44.1 (one seed) | ✓ |
| `\nLabelRepairMax` 69.3 | max over the 6 listed sources = sliced2 s7 69.235, rounded up. I also searched every `cil_cord*colar_r*`, `*sliced*` and `*bilc*` run on disk: the earlier-grid ones are legacy-scored and all ≤ 29.6, and no restricted-scored repair exceeds 69.24. | 69.3 | ✓ (L2) |
| `\vNoBank` / `\vNoBankFull` | clean noreplay AA 21.245/19.591/23.954 → 21.6 ± 2.2; full 8.958/11.721/13.619 → 11.4 ± 2.3 | ✓ | ✓ |
| `\nobankOldMax` 15 | final-row old sessions: s42 0.3/10.8/1.3/0.9; s7 0.0/1.1/0.9/3.2; s123 11.7/**14.47**/1.2/0.0 → ceil 15 | "at or below 15 ... on every seed" | ✓ |
| `\nRecallJointGap` / `MaxLoss` / `MaxGain` / `\recallDevMax` | 2.140 / 0.549 / 2.028 / 1.302, rounded up → 2.2 / 0.6 / 2.1 / 1.4 (R6-L1 fixed). `\nRecallJointGap` is no longer used anywhere. | ✓ | ✓ |
| "On every seed the clean runs rank the methods the same way under both metrics" | s42 AA 21.2 < 33.6 < 87.8 < 91.4, full 9.0 < 32.4 < 88.8 < 91.5; s7 19.6 < 32.4 < 91.3 < 92.6, full 11.7 < 30.5 < 91.2 < 93.5; s123 24.0 < 86.2 < 91.9, full 13.6 < 87.1 < 92.6 | true | ✓ (but see L7a) |
| "restricted AA ... more lenient for weak models ... within about a point for CoLaR-SBR and SBR-Recall" | full − AA: nobank −12.3/−7.9/−10.3; CoLaR −1.2/−1.9; SBR +1.1/−0.1/+1.0; Recall +0.2/+0.9/+0.8 | true for nobank, weakly for CoLaR | L4 |
| `tab_2x2` / `tab_main_clean` / `supp_agree` / `supp_conflict` / `supp_perseed` new cells | all match the files above (CoLaR clean s42/s7 AA 33.59/32.44, full 32.38/30.54) | ✓ | ✓ |
| Intro "Distillation, prompts, low-rank adapters and the continual-NER methods ... do no better" | per seed, every such row ≤ ER's same seed (DER++ 21.3/18.8/21.1 vs ER 26.9/22.0/24.0; all others lower) | true | ✓ |
| Receipt example | `menu.price` is session 1 and `total.total_price` session 5 (`tab:cilsessions`) | ✓ | ✓ |
| Body "CPFD learns no session after the first, in fp32 as in fp16; IS3 ... and CODA-Prompt end at zero F1 on every session but the last; LCA's merged model keeps little of the early sessions" | rescore matrices: CPFD diag 93.8/0/0/0/0 (s42), 94.8/0/0/0/0 (s7); IS3 final [0,0,0,0,28.0] / [0,0,0,0,38.3]; CODA [0,0,0,0,69.8] / [0,0,0,0,79.4]; LCA [0,0,3.9,23.7,40.9] / [0,1.4,3.6,3.7,26.1] | true as stated, but not the "why" (M1) | M1 |

Rule 6: the no-memory SBR cell switched to clean with 3/3 seeds, and its gate row was removed from
`tab:legacyprotocol` correctly. No other cell switched. CoLaR s123 clean is running; when it lands
the CoLaR shared-softmax AA cell switches to clean (see M3).

### B. Findings

**HIGH**

- **R7-H1. "46%" is undisclosed cross-protocol, one-seed, and not reproducible from the paper's tables.**
  - Where: abstract (main.tex:55, "even a memory refreshed with the true labels closes only $\nOracleShare\%$ of it"), intro (main.tex:65), Sec. 4.2 (main.tex:119), App. E (supp.tex:321).
  - Evidence:
    - The numerator uses two fp32 gate runs with test-split early stopping. Their config dumps have no `val_frac`.
    - The denominator's joint term is `rescore/cil_cord_joint_seed42` (`amp: true`, GradScaler, `val_frac 0.1`).
    - Sec. 4.2 says only "(seed 42, both fp32 runs)". App. E says "under the same protocol" of the CoLaR term only. The joint term's protocol is stated nowhere for this ratio. The Setup's precision-only sentence covers comparisons of a *clean* cell with joint, not gate runs, which also differ in early stopping.
    - The abstract gives no seed count. The Limitations section does call the refresh single-seed.
    - The share depends on which runs fill the ratio:
      - 46.1% as generated;
      - 46.5% with clean CoLaR s42 (33.59);
      - 46.9% with the joint mean;
      - **50.1% from the cells a reader sees in Tab. 2** (30.4 → 61.2 against 91.9). The abstract's "it" points the reader at that factorial.
  - Fix (text only, now):
    - abstract: "closes only about half of it";
    - intro: "closes only about half of the gap (one seed)";
    - Sec. 4.2: keep the macro, but write "...to $\nOracle$ (seed 42, both fp32 gate runs): $\nOracleShare\%$ of the gap to the fp16 joint-training reference of the same seed (App. G)";
    - App. E: the same clause.
    - The Conclusion and the "What is new" paragraph already say "about half", which is right.
  - Follow-up (generator): when the clean oracle ×3 (v7 block 2) lands, `conflict_macros()` will not see it, because it hardcodes the gate `ORACLE` and `COLAR_GATE`. Meanwhile its joint term switches to clean joint by itself, so the ratio's mix of protocols changes silently. Rewrite it to use seed-matched clean oracle, clean CoLaR and clean joint, and report the per-seed range ("46 to 52%").

**MEDIUM**

- **R7-M1. The CPFD/IS3/CODA/LCA sentence (main.tex:191) is true but gives the wrong "why".**
  - Naive training also ends at zero on every session but the last: final rows [0,0,0,0,94.8], [0,0,0,0,95.4], [0,0,0,0,93.0].
  - What puts these four below naive is plasticity:
    - IS3's diagonals after session 1 are 28–39 (s42) and 27–64 (s7), against naive's 90–97;
    - CODA-Prompt never learns some sessions (diagonal 0.0 for sessions 2–3 on s42, session 3 on s7);
    - LCA learns sessions 3–5 to 26–54.
  - AA keeps essentially the last diagonal, so naive 94.8 beats IS3 28.0/38.3, CODA 69.8/79.4 and LCA 40.9/26.1.
  - Also, "well below naive" is a 3 to 5 point gap for CODA (14.0/15.9 against 19.0/19.1).
  - Fix: "CPFD learns no session after the first (in fp32 as in fp16), and IS3, CODA-Prompt and LCA learn later sessions only partly (diagonals 0 to 64 F1, against 90 to 97 for naive training); like naive training they end near zero on every earlier session (App. G)".
- **R7-M2. The 2×2 compares full-label F1 with joint training's AA.**
  - Where: `tab:twobytwo` caption "approaches joint training, under either metric", and main.tex:134 "88.4 AA and 89.1 full-label F1, against 91.9 for joint training".
  - Joint training has no full-label F1 (the table prints "--"), so the full-label half of the claim cannot be checked.
  - Fix until clean joint lands: "approaches joint training in AA, and is the only cell above 33 full-label F1".
- **R7-M3. In the CoLaR shared-softmax row, AA and full-label F1 come from different runs, with nothing to say so.**
  - Where: `tab_2x2.tex` and `tab_main_clean.tex`.
  - AA 28.8/31.9† is the fp16 rescore; full-label 32.4/30.5 (n=2) is the clean run. Read side by side, they suggest full-label F1 is above AA, but in the same runs it is 1.2/1.9 *below* (clean AA 33.6/32.4).
  - The marks are correct, but a reader assumes a row is one model.
  - Fix: add one caption clause ("a † AA cell and a clean full-label cell in the same row are different runs"), or simply regenerate after clean CoLaR s123 lands (running; it will switch the AA cell to clean ×3 and remove the problem).
- **R7-M4. The Setup calls full-label F1 the primary number, but every headline number is AA.**
  - "On CIL-CORD the primary number is the full-label F1" (main.tex:94), yet the abstract, the teaser, the 2×2 ordering, "on par with joint training" and the Recall bounds are all AA. Joint training has no full-label value at all.
  - A referee will read this as metric-switching.
  - Fix now: "We report both: session-restricted AA, which every cell has, and the mask-free full-label F1, which clean runs add". Once clean joint lands, promote full-label F1 in the abstract if wanted.
- **R7-M5. "the correction cannot come from better labels" (main.tex:123) is an unscoped universal from one seed.**
  - The decisive label-side combination, true-label memory plus a sliced current loss, was started and never finished. `results/gates/cil_protocol_2026-11/cil_cord_colar_r_sliced2_oracle_seed42.log` (2026-11-02) stops in task 2, with no run directory.
  - The current-document pull is also exactly MiB's background shift, which MiB corrects on the label side within a shared softmax.
  - Fix: "in no repair we ran did the correction come from better labels"; name the current-side pull as the shift MiB addresses; let the queued `colar_mib` control speak to it.
- **R7-M6. The SSUL positioning (main.tex:65, 80) is accurate in its parts but undersells SSUL.**
  - Verified against the paper (arXiv 2106.11562 full text):
    - SSUL freezes the backbone and the past classifiers;
    - it learns a new background classifier at each step ("only learn the 1×1 filters φ_cb, φ_cu and {φ_c}");
    - it uses per-class sigmoids with BCE because "the relative scoring of softmax could cause significant forgetting of past classes" when background targets are noisy (the paper's paraphrase is correct);
    - it also **pseudo-labels** the background ("we adopt the pseudo-labeling strategy as in [PLOP]"), which the Related-Work sentence omits. "SBR ... needs no old-model pseudo-labels" is therefore a real difference that should name SSUL.
  - But SSUL's frozen per-class sigmoids already give old classes a decision that is never trained on stale "O" and is decoupled from the shared background. That is the property SBR's "never stale" sentence claims.
  - So "corrected while keeping one shared background class ... we remove the shared background instead" is literally true (SSUL keeps one background classifier), but it overstates the gap.
  - The true differences:
    1. SBR keeps the upper trunk and the old blocks trainable and re-grounds them from a memory, with labels that are valid per block;
    2. the decision is grouped per session, with a per-block softmax;
    3. routing uses each block's background claim.
  - Fix: one sentence in Related Work stating these, plus "(SSUL also pseudo-labels the background)" in its description.

**LOW**

- **R7-L1.** main.tex:67 says CoLaR 30.4 and no-memory SBR 21.6 are "both close to naive fine-tuning (18.9)". CoLaR is 10–13 points above naive per seed (clean: 14). The old "far closer to naive fine-tuning than to joint training" was true; restore it.
- **R7-L2.** main.tex:121 "below every seed of the blocked readout": no-memory SBR is a blocked readout (21.2/19.6/24.0). Write "of CoLaR-SBR". The pairing also crosses protocols (gate/rescore against clean). Harmless for an upper bound, but say it.
- **R7-L3.** main.tex:121: "42.0 (two seeds)" is the fp16 rescore, and the next clause's 60.5 is the fp32 gate. The same-protocol pair is 39.4 → 62.9 on seed 42. Name the sources, or use the gate pair.
- **R7-L4.** main.tex:94 "restricted AA is the more lenient of the two for weak models": it is lenient by 8–12 points for no-memory SBR but only 1–2 for CoLaR, which is also weak. Write "much more lenient for no-memory SBR, by 1 to 2 points for CoLaR".
- **R7-L5. Two disclosures were shortened away.**
  - The teaser caption no longer points to the protocol of its bars: it mixes clean and † cells with no marks. Add "(seeds and protocol per cell in Tab. 2)".
  - The Fig. 4 caption no longer names the metric (session-restricted).
  - Everything else that was disclosed at `2dc86ad` still exists in `main.tex` + `supp.tex` (list below).
- **R7-L6.** The `tab:dil` caption reads "On average replay ...; EWC and LwF stay within ten points of naive training". The "on average" scope of the second clause is now ambiguous; per run the bound is 12 (R3-N4). Write "both on average".
- **R7-L7. Latent generator issues.**
  - (a) `write_agree` silently drops clean runs whose alias has no `ROW_LABEL`.
    - Queued aliases that will be dropped: `cl_lora` (the label key is `cllora`), `er250`, `er_sbr`, `colar_r_oracle`, `colar_mib`.
    - Yet the caption says "every clean-protocol CIL-CORD run", and the Scoring paragraph's "on every seed the clean runs rank the methods the same way" would then hold only over a filtered set.
    - Fix: add the labels, or print unlabelled runs under their raw alias.
    - Its docstring says "LayoutLMv3", but there is no backbone filter; BROS rows are included by design, and the caption says so.
  - (b) See H1: the oracle sources are hardcoded.
  - (c) `if o` / `if cg` treat 0.0 as missing. Trivial.
- **R7-L8.** supp.tex:323 "leaves the newest session weak on its own diagonal (31.8, 14.4 and 23.0 for sessions 3 to 5)": write "the three newest sessions".
- **R7-L9.** The `cermelli2020mib` entry lacks pages (CVPR 2020, pp. 9233–9242). Optional; everything else in the four entries matches the publisher records.
- **R7-L10.** The Setup's "tokens of classes not yet introduced are labelled O" (main.tex:88) understates the masking. Tokens of *every* out-of-session class, earlier ones included, are O (`scenarios.py:73`, "out-of-session entities relabelled to O"). The current-document half of the conflict, now central to the story, depends on the earlier-class half. Pre-existing text, now load-bearing.

### C. Disclosure audit ("move, never delete")

Every protocol or scope disclosure present at `2dc86ad` was checked against `00dd224`. All of the
following still exist:

| Disclosure | Now in |
|---|---|
| joint differs only in precision | Setup + App. G (R6-L4 done) |
| shared-softmax † cells fp16 | App. G |
| ER 200 vs as-many-as-memory | Setup |
| LwF, CPFD and IS3 zero seeds and precision | body + App. G |
| test-split early stopping except joint | Setup + App. G + `tab:legacyprotocol` |
| first grid saved no configuration | App. G |
| the coverage caveat of restricted scoring | App. B |
| the legacy "floor" band 16–19 | App. B |
| the CL-LoRA port design | Setup + App. G |
| CL-LoRA's 81–95 diagonals | App. G (recomputed: 80.9–94.6) |
| LexSlot "fp32 gate runs only" and "stores token ids" | App. D.3 |
| modality masks, Fisher-by-construction, EWC sign flip, five-family numbers, marginal-snap, ViT numbers | App. F.1 |
| single-seed diagnostics | Limitations (now including the refresh) |

The old Limitations sentence "comparison across protocols is visible from the marks" was dropped,
but the Setup paragraph covers it. Lost: only the two caption pointers in L5. Nothing was deleted
that changes a claim.

### D. Citations (new entries)

All four were verified against the publisher records and the papers themselves.

| Key | Record | Body sentence |
|---|---|---|
| `cermelli2020mib` | CVPR 2020; authors and title correct; no pages (L9) | Unbiased CE ("background label matched with background or any old class") and unbiased KD ("old background with background or any new class"): correct. |
| `douillard2021plop` | CVPR 2021, pp. 4040–4050; correct | Pseudo-labelling with the old model's confident (low-entropy) predictions plus Local POD multi-scale pooled distillation: correct. |
| `cha2021ssul` | NeurIPS 2021, Cha, Kim, Yoo, Moon; correct | Unknown class inside the background, frozen backbone and past classifiers, tiny exemplar memory (SSUL-M), sigmoid instead of softmax with the stated rationale: all verified in the full text. Omits pseudo-labelling (M6). |
| `monaikul2021extendner` | AAAI 35(15):13570–13577, 2021, DOI 10.1609/aaai.v35i15.17600; correct | KL to the teacher on O-labelled tokens, CE on new-type tokens (Sec. 3.4): correct. |

`refs_precompiled.bbl` = the old 52 entries unchanged plus exactly these 4 (56). 56 distinct keys
are cited.

### E. Invariants and generator review

| Check | Result |
|---|---|
| Build | PASS. `latexmk -pdf` main = 0 and supp = 0 from `git archive 00dd224`. 0 undefined references or citations; 0 "??" in either PDF. |
| Pages | `main.pdf` has 10 pp (11 at R6). The Conclusion ends and "References" begins in the **left column of p. 8**; pp. 9–10 are references; no float or body text after p. 8. `supp.pdf` has 16 pp. |
| `---` / em-dash | 0 in every tracked `paper_cvpr/*.tex` and `fig/*.tex`; 0 em-dashes in either PDF. |
| Anonymity | 0 hits for doccl / results/ / scripts/ / .py / clean_2026 / _seed / names; the PDF Author field is empty. |
| Teaser | The bars read `\nNaive` 18.9, `\nNoBank` 21.6, `\nColar` 30.4, `\nSbr` 88.4, `\nRecall` 91.9, with the joint line at `\nJoint` 91.9. Labels match the 2×2 rows; SBR-Recall is separated by a dotted rule and labelled "(no memory)". |
| Appendix cross-references after the move | App. A–H letters match the supp sections. App. F = drift with the new F.1, E = conflict, G = baselines/protocols, D.4/D.5 = Mixed Recall/delta. Every body "App. X" pointer targets the right section; supp "Sec. 4.1 / Tab. 1 / Tab. 3 of the main paper" still resolve (Tab. 1 = DIL, 2 = 2×2, 3 = comparison). |
| R6 L1–L5 | All fixed: `up1()` bounds; "also learn"; h/run pointer in place of the typed hours; precision-only joint in Setup/App. G; `sbr_bros` label. |
| Generator diff | `up1()` is correct (`ceil(10x − 1e-9)/10`). `conflict_macros()` reads the right runs and parses the right log line. `write_agree` adds a scenario filter (correct: it keeps `cil_funsd` and `er_sbr` out). The strict 2×2 uses `pick_clean_only` for full-label (no legacy source can leak in) and prints n. Issues are listed in H1 (follow-up) and L7. |

### F. Question 3 of the brief: the abstract's two load-bearing clauses

- **The opening sentence is true as rendered.**
  - Both pairs are legacy:
    - DIL ER and joint come from the first grid (test-split stopping, configuration not saved, ER buffer 200);
    - on CIL-CORD, ER is the 200-doc rescore (test-split stopping) and joint is fp16 with held-out stopping.
  - The protocol differences within a pair favour ER (test-split stopping), so they cannot have created the contrast.
  - The minimal wording that is true today is the current one, or "Plain replay of 200 stored documents ..." (exact). The intro sentence should carry "(legacy cells, †)" once.
  - When the clean DIL joint/ER and ER-250 land, the macros switch by themselves. The clean DIL ER keeps the default buffer of 200 (`configs/method/er.yaml`) and the CIL ER-250 keeps 250, so "a few hundred" stays true. The intro's "the same method" with "200 stored documents" would then need "250".
- **"Closes only 46% of it" is not acceptable as rendered (H1).** Until the clean oracle ×3 lands, write "closes only about half of it". Afterwards, report the seed-matched clean range.

### Part B. Referee report (advice; does not decide PASS/BLOCK)

I read the rebuilt `main.pdf` once through, as a CVPR reviewer who has not seen the project, and
compared it with the `2dc86ad` abstract, introduction and Sec. 4.

**Summary of claims (in my words).** Replay works for continual key-information extraction when
only the input domain shifts (ER 87.9 vs joint 88.7), but fails when the label set grows (24.3 vs
91.9). The authors attribute this to a second cause, *background conflict*. Under label growth,
stored and current documents label each other's entities "O", and a single shared background logit
is trained in contradictory directions. A 2×2 on class-incremental CORD (memory × readout) is the
evidence:
- memory alone gives 30;
- a session-blocked readout alone gives 22;
- both together give 88.4 against joint 91.9;
- a memory refreshed with true labels recovers only about half the gap.

The proposed readout (SBR) gives each session its own background logit and softmax, and routes
tokens by the strongest claim. A per-session-LoRA variant (SBR-Recall) needs no memory on a
single-schema stream and reaches 91.9. A compressed latent memory (per-document SVD, CoLaR) matches
raw replay on DIL at 60 MB.

**Strengths.**
1. A genuine hook, stated in the first sentence, with a clean factorial that a reader can check in one table.
2. The true-label-memory control is the right experiment, and the receipt example makes the mechanism concrete.
3. A large effect on CIL-CORD (~25 → 88–92), with flat retention for SBR-Recall.
4. Unusually careful protocol accounting and honest limitations, which reviewers reward.
5. Drift is shown on four backbones.

**Weaknesses, ranked by how much they lower the score.**
1. **Novelty against the background-shift literature, with no working baseline from it.**
   - The current-document pull is MiB's background shift.
   - SSUL already diagnoses softmax relative scoring under noisy background labels and removes the coupling with per-class sigmoids and frozen old heads.
   - SBR is, to a segmentation reviewer, "SSUL/ICE-style per-session heads plus replay".
   - The only background-shift methods in the table (CPFD, IS3) learn nothing after session 1 or never relearn old classes. Below naive, they look like broken ports, not like evidence.
   - Without a MiB- or SSUL-style control on the same memory, the claim "the shared background must be removed, not corrected" is untested against the strongest alternative.
2. **One class-incremental dataset.** The 2×2, the oracle control and every SBR number rest on CORD: 100 test receipts, one session partition, one order. Mixed has 1–2 seeds of legacy cells. CVPR reviewers will ask for a second scenario.
3. **Protocol heterogeneity.** Many cells are †, the joint reference is fp16, the DIL table is legacy, and the headline ratio mixes protocols (H1). The paper reads as unfinished even though it is honest about it.
4. **Metric.**
   - Session-restricted AA is the authors' own metric. The mask-free full-label F1, which a reviewer trusts more, exists only for clean runs, and joint training has none.
   - "Primary = full-label" (M4) while the headlines are AA invites "you changed the metric".
5. **The key diagnostics are single-seed:** the oracle, the freezing experiment and the memory ladder. SBR's s.d. is 2.6 against a gap of 3.5.
6. **CoLaR dilutes the story.** It is evaluated as a contribution only on DIL, which is the "known cause", and its DIL evidence is the single-seed ladder in Fig. 3. On CIL-CORD it is just "the memory".
7. **SBR-Recall is narrow:**
   - single-schema only;
   - about 3.7× encoder compute at five sessions;
   - its base is frozen after session 1, and order sensitivity is untested;
   - its Mixed result is one seed.

**Three rebuttal questions.**
1. With the same CoLaR memory and a shared softmax, how far do MiB's unbiased cross-entropy and an SSUL-style per-class sigmoid head get on CIL-CORD? If they reach ~85, what remains of "remove, don't correct"?
2. Does the strict 2×2, with full-label F1 in every cell including joint, replicate on a second class-incremental dataset or partition and a second backbone, under one protocol?
3. Are the CPFD/IS3/CODA-Prompt ports faithful: do they reproduce their published numbers on a native NER benchmark? If not, why should their collapse count as evidence?

(Reserve: the true-label memory plus a sliced current loss is the label-side upper bound for a shared softmax. Why was it not run to completion?)

**Score (CVPR 2025-style six-point scale: 1 Reject … 3 Borderline Reject, 4 Borderline Accept … 6 Accept).**
- **As it stands: 3, Borderline Reject.** Expected spread across reviewers 2–4. A CISS-aware reviewer gives 2 on weakness 1; a document-AI reviewer gives 4 on the hook and the effect size.
- **`2dc86ad`: about 2–3, Weak/Borderline Reject.**
  - Its abstract opened on a strawman ("commonly treated as a single problem").
  - Contribution 1 was a "corrected protocol" that sounded like metric-gaming.
  - The hook was buried in Sec. 4.2.
  - Above all, the background-shift prior art (MiB/PLOP/SSUL/ExtendNER) was absent; any CISS reviewer would have scored novelty at 1–2 and called it a missing-related-work reject.
- The rewrite fixes the framing and the prior-art hole. It moves the expected score up about half a point, not more, because the comparison the new related work invites is still missing.

**What would raise the score before the freeze, in order of value per hour.**

*Text only:*
1. H1: "about half" (5 min).
2. M6 + M5: one honest SSUL sentence (what SSUL already does; what SBR adds: a trainable trunk and old blocks re-grounded by memory, per-session grouping, claim routing), and scope "cannot come from better labels" (20 min). This turns the biggest reviewer attack into an acknowledged comparison.
3. M1: the correct "why" for the sub-naive baselines, plus one sentence on port fidelity (what was checked; if nothing, say "our ports"; 15 min).
4. M4 + M2 + M3: make the metric story consistent. Report both, and stop calling full-label primary until joint has it (15 min).
5. Consider demoting CoLaR from contribution 3 to "the memory we use (Sec. 5.1)". This sharpens the story to diagnosis + readout. It is the user's call; it changes the contributions list.
6. The LOW items (30 min in total).

*GPU runs (all queued in v7 except where noted). They matter most in this order:*
1. **`colar_mib` (MiB-style UCE + CoLaR, shared softmax) ×3.** This is the run that decides the novelty claim against the literature. It is at block 7 in v7, behind Mixed, cil_funsd, BROS and er_sbr. **Move it right after clean joint.** If time allows one seed of something not queued, it would be an **SSUL-style per-class sigmoid head with the CoLaR memory**, the other half of the attack. Not queued.
2. **Clean joint ×3** (block 1): makes every headline comparison same-protocol and gives the 2×2 its joint full-label F1 (M2, M4). Already first; keep it there.
3. **Clean oracle ×3** (block 2): turns the abstract's clause into three seeds. Update `conflict_macros()` before it lands (H1 follow-up).
4. **er_sbr ×3** (block 6): shows the readout effect does not depend on CoLaR (raw memory × blocked readout). It strengthens the factorial's generality cheaply. Move it ahead of BROS.
5. **The second class-incremental scenario (cil_funsd, two sessions)**: necessary but weak. FUNSD has three entity types, so two sessions on 50 test forms is a thin replication; a referee will still call the CIL evidence "one dataset". If anything is cut, keep it, but do not oversell it ("secondary replication" is the right word).
6. **Clean naive ×3** (block 1): makes the 2×2 fully clean, with full-label F1. Cheap; keep.
7. **CoLaR-SBR on BROS ×3:** a second backbone for the method. Moderate value.
8. **Mixed CoLaR-SBR/joint ×3:** moderate. Mixed is "where both causes act", but its cells are 1–2 legacy seeds. Useful for the per-seed claims there.
9. **Clean DIL joint/ER:** low for the story (the hook cannot flip: test-split stopping favoured ER), but it removes † from the first sentence.
10. **Recall session-order o2/o3 (seed 42):** low-moderate; answers a stated limitation.
11. **CL-LoRA clean, ER-250, LwF fp32:** low. Drop them first if the queue is late.

*Still missing (not queued):*
- the SSUL-style sigmoid-head control;
- the true-label memory + sliced current loss (the gate run died in task 2), which is the label-side upper bound for a shared softmax and about 2 h;
- any fidelity check of the CPFD/IS3 ports. Probably too costly; handle it in text.

### OPEN (not counted)

- Clean CoLaR s123 (running) will switch the CoLaR AA cell to clean. Regenerate, and re-check M3 and `\nColar` (the intro and teaser use it).
- When clean joint lands: re-check "on par" (R6), M2, M4, the precision-only sentences in Setup/App. G (then stale), and `\nOracleShare` (its denominator switches by itself).

## R8 (2026-11-12): audit of P10 (`5cfef5b`): clean joint/naive/CoLaR cells, oracle wording, headline metric

Snapshot: the paper files at HEAD equal `5cfef5b` (the later commit `51218a6` touches no paper
file). Scope: `git diff 269fd11..5cfef5b -- paper_cvpr scripts/rca/make_clean_tables.py`, plus a
full read of the new `main.tex` and the changed parts of `supp.tex`. My own script recomputed every
number below from `results/clean_2026-11/*/metrics.json`, `matrix.npy`, `per_class_f1.json`, the
config headers of the logs, and `results/rescore/*` matrices. I also read
`doccl/methods/colar_r.py` and `latent_replay.py`. I ran no generator, model, dataset or pytest.

**Oracle seed 7 status at audit time (07:13–07:22).** `cil_cord_colar_r_oracle_seed7` is still in
task 0 (log at "LR T0 ep9/10"). Its run directory holds only `tb/`, so it has no metrics yet. Every
"one seed" statement is therefore still current, and nothing newer exists to report. I judge the
commit against the files as they were, which are also the files as they are now.

### Verdict: **BLOCK** (one HIGH, text-only fix available; optional re-run is a PM decision)

Every generated number recomputes exactly, and so does every changed cell and macro. Every new
clean run is clean and complete. Every per-seed comparative sentence I was asked to check holds
under both metrics. The headline-metric rule is applied consistently. The generator changes are
correct. What blocks is the description of the control the paper's central claim rests on.
Abstract, Sec. 4.2 and App. E say the stored documents are refreshed with the true labels of
every class seen so far. In fact the code never refreshes the documents of the session that has
just finished, and those documents carry "O" on every earlier-session entity during the whole of
the next session. The paper's account of what stale labels remain after the refresh omits exactly
this part. One MEDIUM (an ambiguous "below the blocked readout") and nine LOW items follow.

### A. Numbers recomputed

| Macro / cell | My recomputation | Rendered | OK |
|---|---|---|---|
| Clean protocol of the new runs | joint ×3, naive ×3, CoLaR ×3 and oracle s42: log config `amp: false`, `fp16: false`, `val_frac: 0.1`, `cil_restrict_eval: true`, `full_label_eval: true`, `epochs` 10 (joint 100); `metrics.json` `training_hparams.amp=false`; full 5×5 matrices; queue DONE lines; 0 tracebacks. Joint s42 log shows held-out early stopping (`T0 ep7 ... bad=2 -> STOP`). CoLaR s123 ran as the v4 orphan and finished at 23:36 (DONE logged). | | ✓ |
| `\vNaive` / `\vNaiveFull` | AA 18.79/18.27/17.78 → 18.3±0.5; full 13.22/12.81/12.97 → 13.0±0.2 | same | ✓ |
| `\vNoBank` / `Full` | 21.25/19.59/23.95 → 21.6±2.2; 8.96/11.72/13.62 → 11.4±2.3 | same | ✓ |
| `\vColar` / `Full` | 33.59/32.44/34.42 → 33.5±1.0; 32.38/30.54/29.32 → 30.7±1.5 | same | ✓ (R7-M3 gone: one row = one set of runs) |
| `\vSbr` / `Full` | 87.76/91.32/86.16 → 88.4±2.6; 88.83/91.19/87.13 → 89.1±2.0 | same | ✓ |
| `\vRecall` / `Full` | 91.35/92.62/91.86 → 91.9±0.6; 91.53/93.52/92.62 → 92.6±1.0 | same | ✓ |
| `\vJoint` / `Full` | 91.13/90.54/92.12 → 91.3±0.8; 91.70/91.80/92.38 → 92.0±0.4 | same | ✓ |
| `tab_2x2`, CIL-CORD columns of `tab_main_clean`, teaser bars and joint line | all the values above. The teaser plots the five `*Full` macros and `\nJointFull`, and the caption says "full-label F1 (mean of three clean seeds)". | | ✓ |
| `\nOracle` / `Full` | oracle s42 AA 65.812, full 79.091 | 65.8 / 79.1 | ✓ |
| `\nColarSeed(Full)`, `\nJointSeed(Full)`, `\nSbrSeed(Full)` | 33.59/32.38, 91.13/91.70, 87.76/88.83 | same | ✓ |
| "the same memory without the refresh" | the clean CoLaR s42 log config equals the oracle's except `name: colar_r`, `run_alias`, `refresh: oracle`. Matrices are identical through row 2, as expected, since the first refresh acts from session 3 on. | | ✓ |
| `\nOracleShare` / `Full` | (65.81−33.59)/(91.13−33.59) = 56.0%; (79.09−32.38)/(91.70−32.38) = 78.7% → 79. All terms clean, seed-matched (R7-H1 follow-up done). | 56 / 79 | ✓ |
| `\nOracleRelabel` | clean log: "refreshed 50/100/150/200 docs, 105/1019/1394/1239 tokens relabelled" | same | value ✓, wording ✗ (H1) |
| `\nOracleLateMin/Max` | diagonal 95.43/48.51/**6.76/2.20/14.68** | 2 / 15 | ✓ |
| `\nOracleNextMin/Max` | R[4,3] = 83.007, R[5,4] = 74.644 | 75 / 83 | L3 (74.6 is printed as "75") |
| `\nOracleLast` / `\nOracleLastShare` | R[5,5] = 14.68. Entity support per session from `per_class_f1.json`, excluding the avg rows: 744/140/143/180/99, so the last session holds 7.6%. | 14.7 / 8 | ✓ |
| Why AA and full-label F1 disagree (main.tex:119, supp.tex:323) | Supported by the run's own files. The entity-weighted mean of the final row is 77.3, against AA 65.8 and full-label 79.1, so weighting explains about 11.5 of the 13.3 points. Session 1 holds 57% of the entities and ends at 86.7. | | ✓ |
| Oracle below joint and below every CoLaR-SBR seed | AA 65.8 < 91.1 and < min SBR 86.2; full 79.1 < 91.7 and < min SBR 87.1 | | ✓ |
| `\nAgreeLowMin/Max` | AA − full: naive 5.57/5.46/4.81, no-memory 12.29/7.87/10.33 → down 4.8, up 12.3 | 4.8 / 12.3 | ✓ |
| `\nAgreeColarMin/Max` | 1.202/1.902/5.105 → 1.2 / 5.2 | same | ✓ |
| `\nAgreeHighMax` | max |AA − full| over SBR 1.07/0.13/0.97, Recall 0.18/0.89/0.76, joint 0.58/1.26/0.26 → 1.3 | 1.3 | ✓ |
| Group order (main.tex:94, supp.tex:125) | s42 AA {18.8, 21.2} < 33.6 < 65.8 < {87.8, 91.4, 91.1}; full {13.2, 9.0} < 32.4 < 79.1 < {88.8, 91.5, 91.7}. s7 AA {18.3, 19.6} < 32.4 < {91.3, 92.6, 90.5}; full {12.8, 11.7} < 30.5 < {91.2, 93.5, 91.8}. s123 AA {17.8, 24.0} < 34.4 < {86.2, 91.9, 92.1}; full {13.0, 13.6} < 29.3 < {87.1, 92.6, 92.4}. "naive and no-memory SBR swap on two seeds" (s42, s7) ✓; top-group swaps are 0.2–0.8 apart ✓. | true | ✓ |
| `\nRecallJointGap` | max over seeds and both metrics: AA +0.22/+2.08/−0.27, full −0.17/+1.72/+0.24 → up1(2.085) = 2.1 | 2.1 | ✓ "within 2.1 on every seed under either metric" |
| `\nRecallMaxLoss/Gain`, `\recallDevMax` | unchanged runs; 0.549/2.028/1.302 (R7) | 0.6/2.1/1.4 | ✓ |
| `\nCpfdLateMax` | rescore CPFD diagonals [93.8,0,0,0,0], [94.8,0,0,0,0] | 0.0 | ✓ |
| `\nBaseWorstLateMax` | worst later diagonal: IS3 27.98/27.04, CODA 0.0/0.0, LCA 32.27/26.14 → max 32.27 → 32.3 | 32.3 | ✓ |
| `\nNaiveDiagMin` | clean naive min diagonal 78.67 (s42, session 4) → 78.6 | 78.6 | ✓ |
| "Four baselines end below naive training on every seed" | CPFD 2.7/0.0, IS3 5.6/7.7, CODA 14.0/15.9, LCA 13.7/7.0, against clean naive 18.8/18.3 (and legacy 19.0/19.1) | true | ✓ |
| "every continual method with a shared softmax stays at or below CoLaR in AA" | max baseline seed ER 26.9 < min CoLaR 32.4 | true | ✓ (cross-protocol, favours the baselines) |
| `\nLabelRepairMax` | now also includes the clean oracle (65.8). Max is still sliced2 s7 69.235 → 69.3, below min SBR 86.2. | 69.3 | ✓ |
| `\nSlicedReplayGate` 39.4, `\nSlicedReplay` 42.0, `\nSlicedBoth` 60.5, `\nOracleGate` 61.2 | gate/rescore files as in R7 | same | ✓ (R7-L3 fixed) |
| `supp_mat_colar` (now clean, 3 seeds) and `supp_vals` | mean matrix diagonal 93.9/70.8/25.8/8.4/9.2; `\colarDropOne` 93.9−72.3 = 22; `\colarLateDiagMin/Max` 8/26; `\colarMaxRise` col 3: 25.8→57.2 = 31.4 → "31" | same | ✓ (31 is a rounded-down "up to"; trivial) |
| Fig. 4 | panel titles "CoLaR (n=3, clean)", "CoLaR-SBR (n=3, clean)", "SBR-Recall (n=3, clean)"; caption names session-restricted F1 | | ✓ (R7-L5b fixed) |
| `supp_agree` | 19 rows = every clean CIL-CORD run with both metrics on disk (oracle under its label) | | ✓ |

Rule 6: the 2×2 and its joint reference are entirely clean, three seeds each, under both metrics,
and so are the CIL-CORD naive, CoLaR, CoLaR-SBR, SBR-Recall and joint rows of Tab. 3. No cell mixes
protocols. The only clean single-seed value (the oracle) appears as a per-seed row in
`tab:conflict` and as `\nOracle*` macros with `\nOracleScope`, never as a table cell. ✓

### B. Findings

**HIGH**

- **R8-H1. The true-label refresh is not "the true labels of every class seen so far": the code never refreshes the documents stored at the boundary itself, and the paper's account of the stale labels that remain omits them.**
  - What the code does:
    - `LatentReplay.after_task` banks the session's 50 documents first (`latent_replay.py:335-336`). Then `CoLaRRefresh.after_task` refreshes only the documents with `n_labels < n_cur` (`colar_r.py:150-157`).
    - A document banked in this session has `n_labels == n_cur`, so it is skipped.
    - Its labels are the CIL-masked session labels: every out-of-session entity, *earlier ones included*, is "O" (`cord.py:283-291`; Setup, main.tex:88).
    - So during session t+1 the 50 documents of session t are replayed with "O" on every entity of sessions 1..t−1. They are corrected only at the end of session t+1.
    - The log confirms it: "refreshed 50/100/150/200 docs" while the memory held 100/150/200/250 (`replay_memory_bytes` 100,928,000 = 250 × 403,712).
    - In replayed documents, stale "O" on earlier classes therefore remains on 1/2 of the memory in session 3, 1/3 in session 4 and 1/4 in session 5.
    - The fourth refresh (1239 tokens) runs after the last session and affects no training. "Rewrites ... at the four boundaries" counts it, and the boundary after session 1 has no refresh at all.
  - Where the paper states otherwise:
    - abstract (main.tex:55): "a memory refreshed with the true labels of every class seen so far";
    - intro (main.tex:65): "refreshing the stored labels with the true labels of every class seen so far";
    - Sec. 4.2 (main.tex:119): "a document stored at session t carries 'O' on every entity introduced after t. We therefore refresh the stored labels at every boundary with the documents' true labels for every class seen so far; a refresh at a boundary cannot cover the session trained next". This names only the future-class part of the staleness that is left;
    - main.tex:121: "The refresh removes the first pull only for the classes known at the last boundary, and current documents keep the second". The newest stored documents also keep a pull on earlier classes;
    - "What is new" and the Conclusion: "with the true labels in the memory"; the Conclusion: "against a memory that already has the true labels"; contribution 1: "a memory with true labels shows that better memory labels alone do not close the gap";
    - supp.tex:321: "The refresh runs at the end of each session and gives the stored documents the true labels of every class seen so far".
  - Why HIGH:
    - This is the control behind the paper's claim that better labels do not close the gap.
    - As described it would be the label-side upper bound. As run it leaves 25–50% of the replayed documents stale on earlier classes in every session after the second. That is the same direction of pull as the current documents', which the paper blames for the remaining gap.
    - The headline inference may survive (CoLaR-SBR is 8–12 points above it in full-label F1 and 20–26 in AA on every seed), but the description is factually wrong in the abstract.
  - Fix (text, required for PASS):
    - Abstract: "a memory whose stored documents are given the true labels of every class seen so far, from the boundary after they are stored, ...".
    - Sec. 4.2: replace the two sentences with "A document stored at session t carries 'O' on every entity of another session, earlier or later. At the end of each session we give every document stored *before* it the true labels of all classes seen so far. The documents of the session just finished keep their own labels until the next boundary, and no refresh can cover the session trained next." Then: "The refresh rewrites 105/1019/1394 stale tokens before sessions 3 to 5 (and 1239 after the last session; seed 42)".
    - main.tex:121: "The refresh removes the first pull for the documents stored before the last boundary ...".
    - Soften "with the true labels in the memory" (What is new, Conclusion ×2, contribution 1) to "with true labels in the older stored documents", or to "with the memory's stale labels refreshed".
    - supp.tex:321: the same correction as Sec. 4.2.
  - Optional (PM decision, GPU): a complete refresh would also unmask the just-banked documents, i.e. call `oracle_refresh(..., n_cur)` on them at capture. It must be a **new mode/alias** (e.g. `refresh: oracle_full`), not an edit of `oracle`. Oracle s7 is running and s123 is queued under the current code. Editing `oracle` in place would change the experiment between seeds, because s123 would load the edited file. About 1.8 h per seed.

**MEDIUM**

- **R8-M1. "below the blocked readout" reads as the no-memory cell, which the oracle is far above.**
  - Abstract (main.tex:55): "a session-blocked readout alone ... stays far below ..., and a memory refreshed ... stays below joint training and below the blocked readout, which never sees those labels". The nearest antecedent is "a session-blocked readout alone" (11.4 full-label / 21.6 AA), and the oracle is at 79.1 / 65.8.
  - The same phrase appears in the intro (main.tex:65, "below the blocked readout introduced here"), in "What is new" (main.tex:123) and in the Conclusion (main.tex:209, "below a blocked readout that never sees them").
  - Only main.tex:119 and 121 say "CoLaR-SBR", which is the true comparison. It holds on every seed under both metrics.
  - This is the R7-L2 issue, fixed in one place and repeated in four.
  - Fix: "below the same memory with the blocked readout (CoLaR-SBR), which never sees those labels".

**LOW**

- **R8-L1. Typed "(one seed)" will go stale by itself.** The oracle's scope is a macro (`\nOracleScope`) in Sec. 3 and Sec. 4.2. It is typed in the abstract, the intro, "What is new", the Conclusion and Limitations ("single-seed diagnostics"). When oracle s7/s123 land, the macros switch and these five do not. Also "below joint training on the same seed" then needs "seeds". Note this in the next Paper round.
- **R8-L2. The generator will print v9's mechanism-dump runs in `tab:agree`.**
  - v9 queues `colar_dump`, `naive_dump` and `joint_dump` (seed 42, `full_label_eval` on).
  - `write_agree` now keeps every clean CIL-CORD run. These would appear as raw rows "colar\_dump" etc. under a caption that says "every clean-protocol CIL-CORD run": duplicate seed-42 models outside the stated groups.
  - The cell loaders are safe: `cil_cord_naive_seed42*` does not match `cil_cord_naive_dump_seed42`.
  - Fix: skip aliases ending in `_dump` in `write_agree`.
- **R8-L3. `\nOracleNextMin` rounds 74.64 to "75" in "rise to 75 to 83"**, and `\nOracleLateMin/Max` and `NextMax` use plain rounding too. Use floor for the low end and ceil for the high end, as `down1`/`up1` do elsewhere ("74 to 84"), or one decimal.
- **R8-L4. `agree_macros` does not check the sign of the "more lenient" claim.** If a future clean run makes AA − full ≤ 0 for a no-memory or CoLaR run, `\nAgreeLowMin` prints a negative or zero value under "more lenient by". Add `_warn` when `min(low) <= 0` or `min(diff['colar']) <= 0`.
- **R8-L5. Two AA-only comparisons do not name the metric in the sentence.** main.tex:191: "On Mixed, CoLaR-SBR reaches 89.7 against 87.7 ..." and "On DIL ... (87.8, 86.2, 87.9 against 88.7)". The Scoring rule says such comparisons are "labelled so". Add "AA" once. (The abstract's first sentence carries "average F1"; the intro says "session-restricted AA".)
- **R8-L6. Stale legacy descriptions after the switch.**
  - The `tab:perseed` caption and the App. G "Settings" paragraph still describe "rescore = re-runs ... of CoLaR on CIL-CORD and Mixed". No CIL-CORD cell, figure or matrix reads the CoLaR rescore any more.
  - The same paragraph lists "100 for naive, joint training" as a CIL-CORD and Mixed legacy cap, but CIL-CORD naive and joint training are now clean.
  - Fix: write "of CoLaR on Mixed" and "on Mixed" for naive/joint. (`tab:legacyprotocol` itself is correct: Mixed only.)
- **R8-L7. App. B's "joint training ... reads 33.3 under legacy scoring (tab:grid) and \vJoint under restricted scoring"** now sets an earlier-grid run against the clean cell, so the 58-point contrast is no longer one model scored two ways. It was not one model before either (the rescore run); the illustration still holds. Optional: "(different runs)".
- **R8-L8 (R7-L6 still open).** The `tab:dil` caption changed only ";" to ", and". "On average" still need not scope the EWC/LwF clause, and per run that bound is 12 (R3-N4). Write "On average, replay ... and EWC and LwF stay within ten points of naive training".
- **R8-L9 (contribution 1, main.tex:71).** "a memory with true labels shows that better memory labels alone do not close the gap" is the only place outside Limitations where the single-seed scope is missing. With H1 this needs the softer wording anyway.

### C. Rule 7 on the sentences of the brief (per seed, both metrics where both are used)

| Sentence | Check | Result |
|---|---|---|
| "on par with joint training" (abstract, SBR-Recall) | full −0.17/+1.72/+0.24, AA +0.22/+2.08/−0.27 | ✓ |
| "within 2.1 points of joint training on every seed under either metric" | max 2.085 | ✓ |
| "closes most of the gap, on every seed" (CoLaR-SBR, 2×2) | gap closed from CoLaR, full: 95/99/92%; AA: 94/101/90%. Final full 88.8/91.2/87.1 vs joint 91.7/91.8/92.4 | ✓ |
| "SBR-Recall all or nearly all of it on every seed" | above | ✓ |
| "both far closer to naive fine-tuning (13.0) than to joint training" (intro, full-label) | CoLaR 32.4/30.5/29.3 and no-memory SBR 9.0/11.7/13.6, against naive 13.2/12.8/13.0 and joint 91.7/91.8/92.4 | ✓ (R7-L1 fixed) |
| "below joint training and below the blocked readout (one seed)" | true against CoLaR-SBR on every seed, both metrics; false against no-memory SBR | M1 |
| "below every seed of CoLaR-SBR" (label repairs, AA) | 69.24 < 86.16 | ✓ |
| Group-order statement | §A | ✓ |
| "Four baselines end below naive training on every seed" + diagonal facts | §A | ✓ |
| "Only the cell with both remedies approaches joint training, under either metric" (2×2 caption) | joint now has full-label F1 (R7-M2 resolved) | ✓ |

### D. Question 4: the abstract's opening contrast

- DIL pair: `\nErDil` 87.9 (86.21/88.73/88.62) against `\nJointDil` 88.7 (89.68/89.03/87.49). Both are first-grid legacy: test-split stopping, no saved config, ER 200.
- CIL-CORD pair: `\nEr` 24.3 (26.91/22.05/24.01; rescore, 200 documents, test-split stopping, fp16 except s42) against `\nJoint` 91.3 (clean ×3). All four numbers are right.
- "A few hundred" = 200 ✓.
- Disclosure: the intro says "Both ER cells are legacy-protocol runs (†) that early-stop on the test split, which favours ER". `tab:dil` says all of its cells are legacy, and Setup gives the legacy precision and epoch-cap variation. The abstract carries no protocol words, which is normal for an abstract.
- The ER cell's fp16 seeds (22.0, 24.0, against fp32 s42 26.9) may disfavour ER by a few points. That cannot close a 67-point contrast. No finding.

### E. Question 5: the headline-metric rule

Full-label F1 leads wherever both metrics exist:
- abstract: SBR 89.1, Recall 92.6 against 92.0;
- teaser (axis label and caption);
- intro paragraph 3;
- Sec. 4.2 (full-label first, then AA);
- Sec. 4.3 and `tab:twobytwo` (full-label column first);
- Experiments;
- `tab:main` (full-label column first).

AA is used on both sides wherever one side has only AA: the abstract/intro ER contrast, "at or below CoLaR in AA", Mixed, DIL, the label-repair bounds, the ablations ("21.6 AA", "82.5 AA").

The Conclusion quotes no number. The only gaps are the two unlabelled sentences in L5. The rule is applied consistently. R7-M4 is resolved.

### F. R7 items

| R7 | Status |
|---|---|
| H1 ("46%") | **Fixed.** No share in the abstract. Sec. 4.2/App. E give seed-matched clean shares in both metrics (56% / 79%) with scope. `oracle_macros()` reads the clean oracle, clean CoLaR and clean joint by seed. |
| M1 (why below naive) | **Fixed**, generated diagonals, plus the port-fidelity sentence ("not checked against their published NER numbers"). |
| M2 (2×2 joint full-label) | **Fixed** (clean joint ×3). |
| M3 (mixed-run CoLaR row) | **Fixed** (CoLaR clean ×3 in both metrics). |
| M4 (primary metric) | **Fixed** (rule stated in Scoring and applied; §E). |
| M5 ("cannot come from better labels") | **Fixed** ("In no label-side repair we ran ..."; MiB named; Limitations: MiB/SSUL not ported). H1 changes what "better labels" covered. |
| M6 (SSUL) | **Fixed.** "also pseudo-labels the background", "SSUL is the closest", the three differences, "Unlike PLOP, CPFD and SSUL it needs no old-model pseudo-labels". |
| L1 | **Fixed.** |
| L2 | **Fixed** at main.tex:121 ("of CoLaR-SBR", with the protocol caveat). The same ambiguity reappears in four other places (M1). |
| L3 | **Fixed** (rescore and gate sources named, gate pair 39.4 given). |
| L4 | **Fixed** (generated ranges per group). |
| L5 | **Fixed** (teaser: "mean of three clean seeds; AA and per-seed values in Tab. 2 and App. D"; Fig. 4 names session-restricted F1). |
| L6 | **Open** (L8 above). |
| L7a/b/c | **Fixed.** Labels added, raw-alias fallback with `_` escaped, oracle sources clean, `is not None` tests. New edge in L2. |
| L8 | **Fixed** ("sessions 3 to 5"). |
| L9 | Not done (optional, as declared). |
| L10 | **Fixed in Setup** ("earlier or later"). Not carried into Sec. 4.2's description of stored-document staleness ("introduced after t"), which is where H1 bites. |

**Disclosures deleted rather than moved.** Two sentences were removed, and both had become false:
- the precision-only note on the CIL-CORD joint reference (joint is now clean);
- "the two shared-softmax † cells of the 2×2 ran in fp16" (both are now clean).

The single-seed scope of the oracle is kept (Limitations, abstract), and so is the legacy gate oracle (61.2, `tab:conflict`, `tab:legacyprotocol`). The disclosure list in R7 §C still holds otherwise. Nothing true and needed was deleted.

### G. Invariants and generator review

| Check | Result |
|---|---|
| Build | PASS. From `git archive 5cfef5b paper_cvpr` in my scratchpad: `latexmk -pdf` main exit 0 (4 runs), supp exit 0. 0 undefined references or citations in either log; 0 "??" in either PDF. The pdftotext of each rebuild is byte-identical to the committed `main.pdf` / `supp.pdf`. |
| Pages | `main.pdf` 10 pp. Body: the left column of p. 8 ends with Cost and the start of Limitations. The right column continues Limitations, then the Conclusion, then "References" begins in the **right column of p. 8**. pp. 9–10 are references only, with no float or body text after p. 8. `supp.pdf` 16 pp. |
| `---` / em-dash | 0 in every tracked `paper_cvpr/*.tex` and `fig/*.tex`; 0 em-dashes in either PDF. |
| Anonymity | 0 hits for doccl / results/ / scripts/ / .py / clean_2026 / _seed / author or institution names in either PDF; the PDF Author field is empty. |
| Appendix cross-references | Supp sections A Setup, B Scoring, C Method, D Extended results, E Conflict, F Drift, G Baselines/protocols, H Reproducibility. Every body pointer (A×1, B×1, C×5, D×5, E×4, F×5, G×5, H×1) targets the right section. "Tab. 3 of the main paper" = `tab:main` ✓. |
| `oracle_macros()` | Correct. It uses clean runs only. Seeds are the intersection with clean CoLaR, joint and SBR, in both metrics. It prints the mean only at n = 3, else per-seed, and `\nOracleScope` names the seeds. Shares are seed-matched and per-seed (a range at n = 3). It warns if any seed is not below joint and below *every* CoLaR-SBR seed under both metrics, and if session 3..T−1 does not rise at the next boundary. The entity share excludes the avg rows. The relabel counts come from the seed-42 log, and the paper says "(seed 42)". The `--` defaults keep the build alive with no clean oracle. (The rounding of the ranges is L3.) |
| Warning logic | `conflict_macros`: warns if any label repair reaches the minimum CoLaR-SBR seed. `baseline_macros`: warns if any seed of a sub-naive baseline reaches the *minimum* naive seed, which is stricter than per seed. `agree_macros`: the group-order check inserts the oracle between CoLaR and the top group, drops empty groups and compares max(lower) < min(upper) per seed and metric. This is correct and matches the prose. It does not check the "lenient" sign (L4). Every check passes on the current files by my recomputation. |
| Unlabelled-alias fallback | `ROW_LABEL.get(alias, alias).replace("_", "\\_")`: builds safely. The silent drop (R7-L7a) is gone. The v9 dump aliases are L2. |
| `nErCilBuf` | 250 only when the ER cell's chosen source is clean (ER-250), else 200; it renders 200 ✓. |
| `_per_seed`, `_span`, `down1` | correct; `down1` = floor(10x + 1e-9)/10 mirrors `up1`. |

### OPEN (not counted)

- When oracle s7/s123 land: regenerate, re-check `\nOracleScope`, the shares (a range at n = 3), "below joint training on the same seed", the group-order warning, and the five typed "(one seed)" (L1).
- If the PM queues a complete refresh (H1, optional), compare it with `oracle` on the same seeds before any wording claims a label-side upper bound.

## R9 (2026-11-12): audit of P11 (`8ce5bb4`, `3db7321`): the investigation restructure, and a referee report

Snapshot: paper files at HEAD = `3db7321`. Scope: `git diff 5cfef5b..3db7321 -- paper_cvpr
scripts/rca/make_clean_tables.py`, a full read of the new `main.tex` and the changed parts of `supp.tex`,
`scripts/rca/mechanism_analysis.py`, `doccl/methods/colar_r.py` / `latent_replay.py`, `doccl/data/cord.py`
(split construction), and the v11 queue. Numbers were recomputed by my own CPU script
(scratchpad `recompute.py`) straight from `eval_dump.pt` (map_location cpu), `per_class_f1.json`,
`matrix.npy`, `metrics.json` and logs, not from `mechanism_analysis.json` (which I then checked against).
No generator, model, dataset or pytest was run. Citations were checked at source (arXiv / PMLR / ACL
Anthology / Crossref / OpenAlex).

**Oracle seed 7 at audit time (07:52).** Still running, in session 4 (log: `T3 ep1`). No metrics yet.
Partial rows from the log: R[1,1] 94.1; R[2,·] 75.5 / 75.3; R[3,·] 55.7 / 81.4 / **13.8**. Refreshes so
far "50 docs, 67 tokens" and "100 docs, 859 tokens". Nothing in the paper is false against it: session 3's
diagonal (13.8) lies inside "2 to 15", session 2 rises one boundary later (75.3 → 81.4), and the
"(seed 42)" on the relabel counts is right (s7's counts differ). Every "one seed" is still current.

### Verdict: **BLOCK** (two HIGH, both text-only, about 40 minutes together)

Every generated number recomputes, every ladder and mechanism cell matches the run files, the R8 findings
are all fixed, the build invariants hold and every new citation exists and is described correctly. What
blocks: (H1) the paper's first contribution says its direct measurements "tell the causes apart" by error
type, but the same measurement on naive training, which is in the generated JSON and not in the paper,
shows that "old entities missed at O" is what any shared-softmax run does, memory or not; the error-type
signatures are tied to the readout, not to the cause. (H2) The new selection-procedure paragraph says the
comparison set "was fixed before the class-incremental runs"; it was frozen on 2026-10-21, after a full
CIL-CORD grid that included the methods it excluded. The facts favour the paper in both cases (the dropped
methods were all weak; the suppressed-current-session evidence survives), so the fixes are wording plus
one table row. Seven MEDIUM and twelve LOW follow.

### A. Numbers recomputed

**Mechanism table (`tab_mech.tex`), my recomputation from `eval_dump.pt` / `per_class_f1.json` per seed (42 / 7 / 123):**

| Row | R | P | owner | →O | same | other | false | Rendered | OK |
|---|---|---|---|---|---|---|---|---|---|
| CoLaR, shared | 22.5 / 20.5 / 18.7 | ≤72.7 / 74.0 / 80.7 (FP unknown for 7/7/3 classes) | | | | | | 18.7–22.6, ≤72.8–80.8 | ✓ (outward rounding) |
| Oracle (s42) | 77.2 | ≤88.2 | | | | | | 77.2, ≤88.2 | ✓ |
| SBR, no memory | 5.6 / 0.4 / 13.2 | 6.1 / 3.5 / 10.7 | 10.6 / 16.8 / 17.5 | 0.0 ×3 | 6.6 / 16.1 / 10.0 | 93.4 / 83.9 / 90.0 | 17.04 / 24.30 / 22.23 | as rendered | ✓ |
| CoLaR-SBR | 89.6 / 92.5 / 88.5 | 88.0 / 90.4 / 84.9 | 96.8 / 98.3 / 95.5 | 5.6 / 6.2 / 4.9 | 58.0 / 71.5 / 56.7 | 36.4 / 22.3 / 38.4 | 1.03 / 0.68 / 1.68 | as rendered | ✓ (but M2) |
| SBR-Recall | 92.7 / 94.0 / 93.2 | 90.5 / 93.2 / 92.1 | 97.8 / 98.3 / 97.1 | 4.4 / 17.0 / 28.3 | 65.2 / 63.0 / 51.2 | 30.4 / 20.0 / 20.5 | 0.79 / 0.44 / 0.50 | as rendered | ✓ |

Denominators: old entity tokens = 2,137 first-subword gold tokens of sessions 1–4 on their own splits; "→O / same / other" are
shares of the wrong ones (they sum to 100); "false" = claims by block t, as the arg-max block above 0.5, on the non-session-t
tokens of split t. Restricted span F1 from the dumps reproduces `matrix.npy` on 58 of 60 cells; the two exceptions are no-memory
SBR s123 session 1 (13.1 vs 11.7) and SBR-Recall s7 session 4 (91.2 vs 91.7) (L9).

**Macros in the new prose:**

| Macro / claim | Recomputed | Rendered | OK |
|---|---|---|---|
| `\mColarImpPMin/Max` (implied precision, all sessions) | 71.8 / 72.1 / 79.1 | 71.8–79.2 | ✓ |
| `\mColarMissedOMin` | (FN − FP_implied)/FN = 0.896 / 0.907 / 0.942 | 89 | ✓ value (M1 on the word "bound") |
| `\mColarLastMin/Max` | R[5,5] 2.0 / 9.5 / 15.9 | 2.0–16.0 | ✓ |
| `\mSessOneShare`, `\mTestSpans` | 744 of 744+140+143+180+99 = 1,306 → 57% | 57, 1,306 | ✓ |
| `\mPoolGapMax` / `\mPoolGapNoBankMax` | max |pooled − full| over the rest: SBR s7 0.35 → 0.4; no-memory 3.67 / 0.58 / 3.67 → 3.8 | 0.4 / 3.8 | ✓ |
| `\mNoBankBoundMax` | s7 0.818 → 81 (per-token: 0 O errors) | 81 | ✓ |
| `\nSbrGapFullMin/Max` | joint − SBR full: 2.87 / 0.61 / 5.25 | 0.6–5.3 | ✓ |
| `\nOracleNextMin/Max` (R8-L3) | R[4,3] 83.007, R[5,4] 74.644, floor/ceil | 74 / 84 | ✓ |
| `\nOracleRelabelTrain/After` | s42 log 105/1019/1394, then 1239 after the last session | same | ✓ |
| `\nRecallMixNoMem/Mem`, `\nSbrFrozen` | gate s42 44.66 / 85.94 / 14.77 (fp32) | 44.7 / 85.9 / 14.8 | ✓ |
| "raw documents do no better (ER ... below every CoLaR seed)" | ER 26.9 / 22.0 / 24.0 vs clean CoLaR 33.6 / 32.4 / 34.4 | | ✓ (cross-protocol; M4) |
| "on DIL CoLaR-SBR is below CoLaR" | SBR 86.76 / 84.77 / 87.08 vs CoLaR 87.58 / 88.30 / 87.38 | | ✓ per seed (L6) |
| "Mixed without a memory ... blocks of the other schemas fall silent until their domain returns" | r64b matrix: FUNSD 87.6/90.8 → 0 at SROIE → 87.3/85.6 at the FUNSD revisit; SROIE and CORD end at 0.3 / 8.1 / 0.0 | | ✓ |
| H3 "above the relabelled memory on every seed both have" | SBR s42 88.8/87.8 vs oracle 79.1/65.8 | | ✓ |

**Ladder (`tab_ladder.tex`)**: every value is the existing clean or legacy cell already audited in R8 (naive, CoLaR, ER†,
oracle n=1, sliced2† 62.86 / 69.23 / 49.45 → 60.5 ± 10.1, CoLaR-SBR, no-memory SBR, SBR-Recall, joint) plus the two
Mixed gate values above. ✓. Rule 6: legacy cells carry †, the oracle carries n=1, no ratio mixes protocols. The two Mixed
rows are single legacy seeds with no n mark (L1).

### B. Findings

**HIGH**

- **R9-H1. The error-type "signatures" are the readout's, not the cause's; the naive run that shows it is omitted.**
  - Where: contribution 1 (main.tex:70: "direct measurements that tell the causes apart (drift sends old entities to
    another session's class through over-claiming blocks; the conflict leaves them missed ...)"); H0 (main.tex:121: "The
    errors do not look like naive forgetting ... With a memory the model mostly misses old entities rather than confusing
    them"); intro (main.tex:64: "misses most old entities and suppresses the session being trained, which drift does not
    do"); abstract ("Forgetting is not simply harder: ... the old entities it still predicts are mostly right, but most are
    missed").
  - Evidence (`mechanism_analysis.json`, same indicator the paper uses for CoLaR):
    - naive fine-tuning, shared softmax, no memory: old-span recall 0.0 on all three seeds; the whole final model makes only
      8 / 9 / 21 false-positive spans on the full test set; the "missed entirely at O" indicator is **0.99 / 0.99 / 0.98**,
      against CoLaR's 0.90 / 0.91 / 0.94. With so few FP spans the overlap proviso is far safer for naive than for CoLaR.
    - So "old entities missed at O" is what a shared softmax does whether or not a memory is present, and "old entities sent
      to another session's class by over-claiming blocks" was measured only on the blocked readout without memory. Each
      signature is observed in one readout only; the measurements do not separate drift from conflict within one readout.
    - Under the paper's own account naive CIL training also carries the current-document pull, so naive is not a
      drift-only control either. The only clean drift-only setting is DIL, where no error decomposition was run.
    - What does separate CoLaR from naive is real and should carry the argument: the session being trained is suppressed
      (newest session 2.0–16.0 vs naive's every diagonal ≥ 78.6), and CoLaR keeps 19–23% of old spans with high precision
      where naive keeps none.
  - Fix (text + one generated row, about 30 min):
    - add naive to `MECH_ROWS` (its R 0.0, and the indicator in a new column or in the prose);
    - H0: drop "rather than confusing them" as the contrast with naive; say naive also misses old entities at "O" (≥ 98% by
      the same indicator), and that what differs is the suppressed current session and the retained high-precision 20%;
    - contribution 1: "direct measurements of where the errors go under each readout (with a blocked readout and no memory
      drift sends old entities to another session's class through over-claiming blocks; with a shared softmax old entities
      are missed at 'O' with or without a memory, and a memory adds a suppressed current session)";
    - intro: "suppresses the session being trained, which naive training does not";
    - abstract: keep "mostly right, most missed", add nothing, or replace by the suppressed-session fact.
    - Contribution 1 also says the relabelled memory "removed" the stored-label side; per R8-H1 it removes part of it.
      Write "reduced".

- **R9-H2. "The comparison set was fixed before the class-incremental runs" (main.tex:95) is false.**
  - Evidence: `results/cil_cord_ewc_seed42` dates from 2026-06-17; the June grid ran EWC, L2P, DualPrompt, O-LoRA,
    ER + C-Flat++, CPFD, IS3, LCA and others on CIL-CORD (legacy scoring). The set was frozen on 2026-10-21 (project memory
    "fixed-12 baselines"), and that decision explicitly settled "CPFD/IS3/CODA nulls on CIL-CORD", i.e. with CIL-CORD
    results in hand. EWC, L2P, DualPrompt, O-LoRA, SD-LoRA and ER + C-Flat++ were excluded then.
  - Why it is HIGH although it does not help the paper: the paragraph exists to disclose researcher degrees of freedom, and
    a reviewer or reader who learns the date will discount everything else in it.
  - The facts are benign: every excluded method stayed at or below ER on that grid (EWC 0.0 / 1.1 / 0.3, L2P 2.1 / 1.3 /
    0.0, DualPrompt 0.0 / 2.0 / 3.3, O-LoRA 0.0 / 0.0 / 14.2, ER + C-Flat++ 16.6 / 14.2 / 17.0, against ER 16.9 / 16.6 / 14.4,
    legacy scoring), and methods known to fail (CPFD, IS3) were kept.
  - Fix (5 min): "The comparison set (one published method per family) was fixed after a first class-incremental grid
    and before the runs reported here; the methods dropped then (EWC, L2P, DualPrompt, O-LoRA, SD-LoRA, ER with C-Flat++)
    stayed at or below ER on that grid (App. G)." Put the grid numbers in App. G.

**MEDIUM**

- **R9-M1. The span-level indicator is called a bound in the body, and it fails in the anti-conservative direction.**
  main.tex:121 "a span-level bound puts at least 89% of its missed spans entirely at 'O' (an indicator ...)". On no-memory
  SBR s7 it reads 81.8% where the per-token truth is 0%: it can overstate by 80 points. App. B handles it honestly; the body
  should not say "bound" or "at least". Write "an indicator (not a bound: on one blocked run it reads 81% where the per-token
  count is 0; App. B) puts about 90% ...". Also: the paper can give a real lower bound on old-span precision instead of the
  all-session one: TP_old / (TP_old + FP_implied,all) = **71.7 / 71.8 / 78.5%**, which is what "the old entities it still
  predicts are mostly right" needs (an under-claim).
- **R9-M2. The "same" column counts B/I swaps of the right type as confusions with "another class of their session".**
  `decompose()` puts B-x→I-x and I-x→B-x into `to_same_session` (its docstring says so; the caption does not). For
  CoLaR-SBR they are 17.3 / 12.3 / 12.5 points of the 58.0 / 71.5 / 56.7; without them within-session class confusions are
  40.7 / 59.2 / 44.2%, so "most remaining errors ... are confusions inside a session" (main.tex:155) holds on 1 of 3 seeds.
  Fix: caption "another tag of their session (including a B/I swap of the right type)"; prose "most remaining errors give the
  token another tag of its own session". Better: add a "B/I" column or subtract it.
- **R9-M3. H2's verdict word and its "remaining suspect".**
  - The H2 cell "partly" rests on legacy, test-split-stopped runs (s.d. 10.1) compared across protocols with clean CoLaR,
    and the sliced2 run changes both sides (sliced replay = the stored side, H1; sliced current = H2). The only
    same-protocol, H2-specific number is seed 42's increment from slicing the current loss (gate 39.4 → 62.9).
  - Fix: verdict cell "partly$^\dagger$" (or "open"), and one clause in the prose: "on seed 42 slicing the current loss adds
    23.5 points to the sliced replay (both gate runs)".
  - main.tex:149 "the current documents' pull is the remaining suspect": per H1's own description, the relabelled memory
    still leaves the newest bank stale on earlier classes and every stored document stale on the session being trained.
    Write "the current documents' pull and the stale labels the refresh cannot reach remain; the pre-session refresh and the
    MiB-style loss separate them."
  - main.tex:145 "prior work gives no reason to expect a separated readout to do better than it" overstates: offline,
    Greenberg et al. found marginalisation at least as good as per-dataset outputs, but in segmentation SSUL's separated
    sigmoids beat MiB (with other changes). Write "prior work is mixed: ...".
- **R9-M4. "changes one thing against the run it is compared with" (main.tex:107) does not hold for H0's raw-memory row.**
  ER vs CoLaR changes the memory format, the frozen layers 1..4, the replay path (whole network vs upper layers), the
  buffer (200 vs 250), the replay batch, and the protocol (legacy, two fp16 seeds). So "Memory fidelity is not the
  bottleneck, since raw documents do no better" (main.tex:123) is weaker than it reads. No uncompressed latent memory was
  run on CIL-CORD (only on DIL). Fix: "(ER differs in more than the memory format; the raw-memory control with the same
  readout, `er_sbr`, is queued)" and scope the preamble sentence to "unless stated".
- **R9-M5. MORE's parameter isolation is omitted, and SBR-Recall's real difference from MORE is not stated.**
  Related Work describes MORE as "a head per task with an out-of-distribution output on saved samples", but MORE also isolates
  per-task parameters (HAT-masked adapters on a pre-trained transformer, per Lit A, B12), which is the SBR-Recall pattern. The
  difference the paper can truthfully claim is that MORE needs saved samples to train each head's "none of mine" output,
  whereas SBR-Recall's background logits are trained on each session's own negatives with no memory (Lit A threat 2). Add both
  halves in one sentence.
- **R9-M6. The sliced-current-loss control is ER-ACE-style masking and is not named so.**
  `latent_replay.py:_current_ce` (`current_sliced: true`) masks old classes out of the softmax of current "O" tokens; its own
  docstring cites ER-ACE. Lit A (threat 9) asks to say so and cite `caccia2022erace`. Naming it turns a reviewer question into
  a reported control (an under-claim). Note also that it is not the "separated softmax with one shared background" listed
  as not run in Limitations, so that line stays correct.
- **R9-M7. Mixed SBR-Recall carries a boundary-condition claim on one legacy seed in the abstract.**
  Abstract: "across schemas they cannot". Evidence: one fp32 gate seed (44.7†). Contribution 3 and Sec. 5.3 say "(one seed)";
  the abstract does not. Add "(one seed)" or "did not in our run".

**LOW**

- **R9-L1.** Ladder rows "the same on Mixed" show 44.7† / 85.9† with no seed count; the caption's n=k mark covers clean seeds
  only. Print n=1 on legacy single seeds too, or say "(seed 42)" in the row.
- **R9-L2.** Abstract "What removes it is to stop sharing the background class": CoLaR-SBR is 0.6 to 5.3 points below joint
  per seed. "What closes most of it" is the wording the body uses.
- **R9-L3.** `refs_precompiled.bbl` has 61 entries, but `main.tex` cites 60: `zhai2023investigating` lost its citation in P11
  and still prints as [54] with no back-reference. Re-cite it (GDI-Bench's sentence fits) or regenerate the `.bbl`.
- **R9-L4.** Bibliography fields (all entries exist and match their records): `mo2024bpf` lacks LNCS 15074, pp. 463–480;
  `sassioui2026documents` lacks pp. 1124–1129 and abbreviates four given names (Crossref: Mohamed El Kamili, El Mehdi
  Benyoussef, Meriyem Chergui, Mohamed Ouzzif); `kalb2023principles`, `liu2023abr` and `chen2023star` have no pages.
  Optional.
- **R9-L5.** For blocked runs, the "P" column is precision on the splits, which omit receipts without that session's
  entities, so it is an upper bound on the full test set as well (it matters only for no-memory SBR, whose pooled F1
  overstates full-label F1 by 3.7). Add the ≤ there, or one caption clause.
- **R9-L6.** main.tex:159 marks only CoLaR's DIL cell † ("$\nColarDil$$^\dagger$"); the CoLaR-SBR DIL cell is a legacy gate run
  too.
- **R9-L7.** Generator: the verdicts of "cell" rows (refutes / supports / partly) are typed, with checks only for the oracle
  (between CoLaR and joint) and ER < CoLaR. A slot row that lands renders with an empty verdict cell and only a warning.
  Make an empty slot verdict print "pending", and add one check per typed verdict (e.g. SBR above every oracle and CoLaR seed;
  SBR-Recall within the gap).
- **R9-L8.** `write_mechanism` macros use `down1` for `m*PMin` even when precision is an upper bound (72.7 vs the table's
  72.8). Unused in the text; harmless.
- **R9-L9.** `mechanism_analysis.py` labels `spans_restricted` "= matrix", but it reproduces the matrix on 58 of 60 cells
  (exceptions in A). Not in the paper. Dev may want to log `repro_minus_matrix` in the table.
- **R9-L10.** Layout: the ladder (`table*`, Tab. 1) floats to p. 5, after the H0–H2 text on p. 4, so a reader meets the rungs
  before the ladder. Move the `table*` source before Sec. 3, so it lands at the top of p. 4.
- **R9-L11.** Sassioui et al.'s abstract says no method "consistently" preserves prior knowledge under label-space shift, and
  that domain transfer forgets the most for the (non-replay) methods they test. The paper drops "consistently". One clause
  ("for the non-replay methods they test, domain transfer forgets most") would pre-empt a reviewer who reads it against the
  paper's hook.
- **R9-L12.** Sec. 4.2 "a session suppressed while current and recovered one boundary later is the stale-label mechanism seen
  directly" sits next to App. E's "the run does not separate these two contributions". Both explanations are stored-label
  effects, so the sentence is defensible; "seen directly" is still stronger than one seed allows.

### C. Question-by-question (brief)

1. **`mechanism_analysis.py`.** The definitions match the headers. Units: first-subword gold-labelled tokens; seqeval
   spans per flat split; "old sessions" = splits 1–4 of the final model, pooled; "owner" = arg-max block above the
   threshold is the gold class's session (the block index is the session index on CIL-CORD, checked: 5 blocks, `allowed` ids
   0/13/25/37/49 + 12); "false claims" = claims by block t on the non-session-t tokens of split t. Precision for
   shared-softmax rows is correctly an upper bound, since seqeval gives P = 0 when TP = 0. The exceptions are M2 (B/I in
   "same") and L5. **The own-split false-claim column is meaningful**, because every such claim is an error whatever the
   token truly is. It is the only precision-side token measure (17–24% without memory vs 0.4–1.7% with), so keep it. Its
   denominator mixes true background with other sessions' entities, on receipts selected by containing a session-t entity
   (`cord.py:_filter_by_labels`). Say that once in the caption. **The indicator is handled honestly in App. B, not in the
   body** (M1).
2. **Rungs.** The predictions are stated before the results in every subsection, as paragraphs titled "Prediction". The
   one-word verdicts follow per seed under both metrics for H0, H1 (s42), H3 and H4 (CIL-CORD). For Mixed, see M7. H2 is
   M3; "open" is defensible, "partly†" is the minimum. Evidence against the story: DIL SBR < CoLaR is reported, and so is the
   oracle being partly true. The omitted contrary evidence is naive's error pattern (H1). Legacy cells are marked and never
   share a ratio with clean ones. For the fixed label space, nothing is claimed beyond "follows from the design and was not
   run" (✓). The plan's "predicted failures presented as confirmed predictions" was correctly not done for it.
3. **Boundary refresh (R8-H1).** The code (`colar_r.py:after_task` → `_refresh_memory`, `n_labels < n_cur`;
   `latent_replay.py:335` banks first) and both logs match every description. The descriptions are in the abstract, the
   intro, the contributions ("relabelled memory", but see H1's "removed"), Sec. 4.2 ("documents stored before it ... the
   session just finished keep their own labels ... no refresh can cover the session trained next"), the receipt paragraph,
   the Conclusion, the ladder footnote ʳ and App. E. **Fixed.** `refresh_before_task` (`colar_r_oracle_pre`) is the
   never-stale control. It is queued, and its ladder slot is in place. Oracle s7: see the header; nothing false yet.
4. **Attribution, verified at source.**
   - **ICE-O** (full text): frozen feature extractor; per-session `Softmax([δ; o_{t+1}])` on D_{t+1} only; inference by
     arg-max over `[δ; o_1..o_{t+1}]`, Other iff max < δ; no stored samples. The paper's description is exact. The stated
     SBR differences (learned per-session background; trainable trunk and old blocks from memory) are true.
   - **MORE**: PMLR 199:548–563; the author order Kim, Liu, Ke matches PMLR (arXiv lists Kim, Ke, Liu). Head per task, OOD
     trained with saved samples: ✓. Isolation omitted: M5.
   - **ABR** (abstract): foreground shift = replayed images' background contains current objects, "the main reason" image
     replay failed in IOD ✓.
   - **BPF** (ECCV 2024): co-occurring past / present / future objects make the stage objectives conflict ✓.
   - **SS-IL**: ✓.
   - **MiB / PLOP / SSUL**: verified in R7 ✓.
   - **Greenberg**: EMNLP 2018, 2824–2829; marginal likelihood over disjoint tag sets ✓.
   - **Beryozkin**: ACL 2019, 140–150, tag hierarchy ✓.
   - **Kalb & Beyerer** (full text, App. D): "freezing layers ... shifts the initial representation shift to later layers" ✓.
     The mirror-image phrasing is right.
   - **ShadowKV**: PMLR 267:57355–57373 ✓; per-sequence SVD of keys ✓.
   - **STAR**: NeurIPS 2023, class prototypes replayed in CISS ✓ ("rivals" is modest).
   - **Ma et al.**: ACL 2023, 5959–5979, DOI ✓.
   - **Cha & Cho**: TMLR 2025, tuning and evaluating on the same scenario overstates CL methods ✓.
   - **The two 2026 LayoutLM papers exist.**
     - IWCMC 2026, pp. 1124–1129, DOI 10.1109/IWCMC69287.2026.11579834. Its abstract (OpenAlex) has the three regimes, four
       layout models, SROIE / FUNSD / CORD / BuDDIE, and "none ... consistently preserve prior knowledge under semantic or
       label-space shifts" (L11).
     - Technologies 14(5):250, 2026. The abstract matches "gradient projection against a memory of task subspaces" on the
       four datasets ✓.
   - **Nothing marked KNOWN ALREADY is claimed as new.** The intro's "None of the ingredients is new" paragraph and
     contribution 2's "not a new mechanism" follow Lit A §7 exactly. The forbidden phrases are all gone: "SSUL is the
     closest", "real negatives at no extra cost", "one background class per model", "new format".
   - **Under-claims:** the old-span precision lower bound (M1), the ER-ACE-style masking control (M6), and SBR-Recall
     needing no replay to train its "none of mine" outputs, unlike MORE (M5).
   - **Bibliography:** 61 `\bibitem`s = R8's 56 − 8 no longer cited (aljundi2018mas, bian2024cflat, farajtabar2020ogd,
     kornblith2019similarity, mcdonnell2023ranpac, wang2022dualprompt, yoon2018den, zenke2017si) + 13 new. All 13 match
     their publisher records; field gaps are in L4, and the orphan in L3.
5. **R8 items.**

   | R8 | Status |
   |---|---|
   | H1 | **Fixed** (see 3) |
   | M1 | **Fixed** (the abstract no longer compares with the blocked readout; intro and Conclusion name CoLaR-SBR) |
   | L1 | **Fixed** (`\nOracleN` everywhere) |
   | L2 | **Fixed** (`_dump` skipped) |
   | L3 | **Fixed** (floor / ceil) |
   | L4 | **Fixed** (warning) |
   | L5 | **Fixed** ("AA" named) |
   | L6 | **Fixed** (App. G and the `tab:perseed` caption) |
   | L7 | **Fixed** ("different runs") |
   | L8 | **Fixed** (`tab:dil`) |
   | L9 | **Fixed** (contribution 1 scope) |

   - **Disclosures:** the 2×2, the teaser, the drift figure, the memory-content ladder and the Pareto figure moved to App. E
     and F rather than being deleted. Precision, test-split stopping, the CL-LoRA port, port fidelity, "no privacy claim",
     single-seed scopes, the LwF fp16 exclusion and cost are all still in the body. Nothing true and needed was deleted.
   - **Cross-references:** supp "Tab. 1 / 2 / 3 / 4 of the main paper" = ladder / mech / DIL / comparison ✓; "Sec. 4.1" = H0 ✓;
     App. letters unchanged ✓.
6. **Invariants**, from `git archive 3db7321 paper_cvpr` in my scratchpad:

   | Check | Result |
   |---|---|
   | Builds | `latexmk -pdf` main 0, supp 0 |
   | References and citations | 0 undefined; 0 "??" |
   | Pages | main 11 pp. The Conclusion ends and References begin in the **right column of p. 8** (line 632). pp. 9–11 hold only references, with no body text or float after p. 8. The body has essentially zero slack. supp 18 pp. |
   | `---` and em-dashes | 0 in every tracked .tex; 0 em-dashes in either PDF |
   | Anonymity | 0 hits; Author field empty |
   | Empty slots | `tab_beyond.tex` is the header comment only and is never referenced (`tab:beyond` has no `\cref`). Slot rows absent, dump rows absent; no empty float or dangling reference. |

   **Generator diff:**
   - Slot logic: renders at `min_n` = 3 clean seeds, else nothing.
   - Three-seed rule: `pick` for cells, `pick_clean_only` for full-label F1.
   - Exclusions: `_dump` in `write_agree` and in the gap loop; smoke runs by directory name, and glob patterns cannot match
     `cil_cord_smoke_*`. The o2/o3 runs are filtered by scenario name.
   - Verdicts: see L7.
   - Oracle "between CoLaR and joint" warning added; correct.

### Part B. Referee report (advice; does not decide PASS/BLOCK)

I read the rebuilt `main.pdf` once through as a CVPR reviewer new to the project.

**Claims in my words.**
- Replay that works under domain shift for token-level KIE fails under class-incremental label growth.
- The authors test four explanations on CORD split into five sessions:
  - harder drift: rejected, because a memory that works on DIL gives 31 full-label F1 here, and raw replay is no better;
  - stale labels in stored documents: partly, because relabelling older stored documents lifts 32 → 79 on one seed;
  - current documents mislabelling old entities: partly, from legacy loss-slicing runs, with the decisive MiB control
    pending;
  - the shared background normalisation: supported, because one background per session with the same memory reaches 89 vs
    joint 92.
- Per-token measurements on the blocked readouts show where the errors go.
- Given the separated readout, per-session LoRA deltas replace the memory on one schema (93) but not across schemas (one
  seed).
- The readout itself is attributed to ICE-O and MORE.

**Strengths.**
1. It now reads as an argument. Each rung has a prediction paragraph, a result and "what it rules out", and the relabelled
   memory being *partly* right is reported as such. This is rarer than it should be, and reviewers who check will reward it.
2. Prior art is conceded up front ("None of the ingredients is new"), accurately (I verified ICE-O's training and inference
   rule at source). That removes the R7 novelty landmine.
3. The relabelled-memory control and the suppressed-current-session pattern (sessions 3–5 learned to 2–15 on their own
   diagonal, recovering one boundary later) are genuinely informative, and they are the paper's best evidence.
4. Selection procedure and dropped families are disclosed (once H2 is fixed); protocol accounting is meticulous.
5. Effect size and stability are large: SBR-Recall's flat retention, and CoLaR-SBR's 89 under the mask-free metric.

**Weaknesses, ranked by how much they lower the score.**
1. **The decisive rungs are empty.** For an investigation paper, the experiments that discriminate between the hypotheses
   have not been run:
   - the MiB-style loss (H2);
   - the never-stale memory (does a shared softmax with fully correct stored labels close the gap?);
   - faithful ICE-O and SBR with a constant background (is SBR's learned background or its trainable trunk what matters?);
   - raw memory with the blocked readout.

   Today the ladder's discriminating evidence is one oracle seed and a legacy sliced-loss run with s.d. 10. A reviewer will
   say the hypotheses are listed rather than tested.
2. **The mechanism table measures only the blocked readouts.** The shared-softmax rows have recall and a precision bound
   only. The one cross-readout signature claim is confounded (H1 above): the "conflict signature" is also what naive
   training shows. The diagnosis of *why* the shared softmax fails is therefore still inferred from which remedy works, which
   is R7's weakness 1.
3. **One class-incremental dataset, one partition, one order, one backbone for every rung.** CORD's test split is 100
   receipts, and a second CIL stream is not in the paper. "Evidence beyond one stream" holds Mixed and DIL legacy cells.
4. **Contribution size.** With the mechanism conceded, the contribution is an analysis of one benchmark plus "a known
   readout works well here". A CVPR reviewer weighs that against the CV-centric venue. Document AI is in scope, but the
   insight has to be general to carry an analysis-only paper.
5. **No figure until p. 8.** The mechanism, the conflict on one receipt, and the readout itself (the SBR schematic moved to
   the appendix) are all text only. The first five pages are dense prose with tables.
6. **Residual protocol heterogeneity:** † cells in H0 (ER), H2 and H4 (Mixed), and the whole DIL table.

**Three rebuttal questions.**
1. With every stored document carrying true labels *before* each session (`colar_r_oracle_pre`) and a MiB-style unbiased CE on
   current documents, all with a shared softmax, how far below CoLaR-SBR does the model stay on each seed? If it reaches
   ~85, what is left of H3?
2. A faithful ICE-O (frozen encoder, constant Other, no memory) on CIL-CORD: how close is it to SBR-Recall? If close, what does
   SBR-Recall add beyond ICE-O plus LoRA?
3. With a shared softmax, what fraction of old-entity *tokens* are predicted "O", for naive, CoLaR and the relabelled memory
   (per-token dumps, not the span indicator)? Does it differ between naive and CoLaR at all?

**Score (CVPR six-point scale).**
- **As it stands: 3, Borderline Reject**, spread 2–4. The expected score is up about a quarter point on R7's 3, not a full point.
- **Why it rose:**
  - CISS / continual-NER reviewers no longer have a missed-prior-art reject: the attributions are correct and upfront;
  - the hypothesis structure and the honest "partly" verdicts read as research;
  - the selection-procedure paragraph pre-empts the hyperparameter critique (after H2 is fixed).
- **Why not more:**
  - an investigation is judged by its discriminating experiments, and the discriminating rungs are slots;
  - conceding the mechanism removes the effect-size "method" carrot that earned R7's document-AI reviewer's 4;
  - evidence is one dataset.
- **If `colar_mib`, `colar_r_oracle_pre`, ICE-O and the const-background SBR land with three seeds each before the freeze**,
  and the ladder's verdicts stay as they are: **4 (Borderline Accept)** is realistic. It needs a figure and H1-type
  overclaims removed. If MiB comes close to SBR, the paper stays a 3, but an honest one ("background shift transfers to
  documents; here is which part each fix removes").

**(a) Research or a table with captions?** Mostly research now. H1 and H3 are convincing on their own: H1 because it
reports the partial success and its mechanism, H3 because it shows both signatures across memory on / off within one readout.
H0 is weaker than it reads (ER confound, naive shares the signature). H2 is a placeholder with legacy evidence. H4 is
convincing on CIL-CORD, anecdotal on Mixed. The ladder table itself is the weak spot: five of its future rows are slots, and
the verdict column currently reads like captions on known numbers.

**(b) Pitch.** Not "a method" and not "we found background shift". Pitch it as *"where the label conflict lives in replay
under label growth, measured, on token-labelled documents"*. The deliverable is the decomposition:
- the stored side, removed by relabelling and the never-stale memory;
- the current side, removed by the MiB loss;
- the shared normalisation, removed by the readout with no true labels, plus the error accounting.

Plus a practical finding for document AI: a known per-session background readout plus whole-document replay closes 90–100% of
the gap where every published continual method in the comparison stays at the naive level. The title should follow, e.g.
*"Why Replay Fails When the Schema Grows: Taking Apart the Background Conflict in Continual Document Understanding"*; "Memory
Is Not Enough" is fine if the never-stale memory confirms it.

**(c) The single figure (about 0.35 column, top of the right column of p. 1 or p. 2).** The one-receipt conflict
instance as a two-panel schematic:
- left: a stored receipt (item prices = `menu.price`, total = O) and a current one (total = `total.total_price`, item
  prices = O) both pushing one shared O logit on look-alike price strings, with arrows for the stored pull and the current
  pull;
- right: the same two receipts each training only their own block's O.

It replaces the receipt paragraph (about 8 lines), states H1 / H2 / H3 visually, and brings back the readout picture the body
lost. The error-decomposition bar chart is the better figure *once* per-token shared-softmax dumps exist (naive / CoLaR /
oracle / no-memory SBR / SBR as stacked →O / same / other bars). Today two of its five bars would be bounds, so do not build
it yet. The ladder as a diagram duplicates Tab. 1; skip it.

**(d) What to cut (the body has zero slack; five ladder rows plus a figure need about 0.5 column).**
- `tab:dil` and its paragraph in H0 go to App. F (about 0.3 column). Keep one sentence: "replay retains under domain shift
  on four encoders (App. F)".
- The memory paragraph (Sec. 5.2) goes to 4 lines: CoLaR is a component.
- Related Work "Latent and compressed replay" and "Where forgetting lives" merge into one 6-line paragraph.
- The receipt paragraph becomes the figure.
- Ablations: drop the LexSlot sentence and the rank-8 detail (App. D has both).
- "Cost" goes to two lines.
- `tab:main`: drop the DIL column (it duplicates `tab:dil` / App. F) or the four sub-naive baseline rows' Mixed cells.

**(e) Queued runs by value now (each CIL-CORD seed ≈ 1.5–2 h; about 90 GPU-hours remain before the freeze).**
1. **`colar_r_oracle_pre` ×3 and `colar_mib` ×3, interleaved by seed** (s42 of both first). Together they *are* the
   investigation (H1 complete, H2 decided). With only one of them the ladder still has a hole.
2. **Per-token full-label dumps of the shared-softmax runs** (`colar_dump`, `naive_dump`, and an oracle dump if cheap; s42).
   Cheap, and they fix weakness 2 and rebuttal question 3. Already queued: keep them before ICE-O.
3. **Faithful ICE-O ×3** (cheap: frozen encoder, head-only) and **SBR with constant background ×3**. These are the two
   comparators that define what SBR / SBR-Recall add over the prior readout. ICE-O first, since it is minutes per seed if
   head-only.
4. **`er_sbr` ×3**: fixes the H0 confound and answers "is CoLaR needed".
5. **Oracle s7 / s123** (running / queued): needed for the H1 row to stop being n = 1. s123 can wait behind 1–3 if time is
   short; s7 is nearly done.
6. **BROS CoLaR-SBR ×3**: a second backbone for H3. Moderate.
7. **CIL-FUNSD**: thin (3 types, 50 forms); a referee will still say "one dataset". Run it only if the above are done.
8. **Mixed** SBR / joint clean: low for the story now; Mixed is a boundary-condition anecdote either way.

Still missing, not queued:
- a token-level shared-softmax error split on DIL (the only drift-only setting), which would give the drift signature
  under a shared softmax and repair H1 properly; two seed-42 dump runs;
- a weight-tied-background SBR (SS-IL with shared O; Lit A §9.3a), which isolates "separate normalisation" from "separate
  background";
- an uncompressed latent memory on CIL-CORD (one seed) for H0's "fidelity is not the bottleneck".

**Changes in order of value per hour.**

*Text only:*
1. R9-H2 (5 min).
2. R9-H1 plus the naive row in `tab_mech` (30 min).
3. R9-M1 (both halves) and R9-M2 (15 min).
4. R9-M3, M4 and M7 (20 min).
5. R9-M5 and M6 (MORE isolation, the ER-ACE naming plus citation; 15 min, including the bib entry).
6. Move the ladder `table*` earlier (R9-L10) and the cuts in (d) (45 min).
7. The receipt figure (2–3 h of TikZ; the highest-value single addition to the reading experience).
8. The LOW items (30 min).
9. After `colar_mib` lands, apply the plan's H2 decision rule *before* touching the title.

*GPU:* as in (e). Interleave `oracle_pre` and `colar_mib` by seed; keep the dumps; run ICE-O before constant-background SBR;
then `er_sbr`; everything else is optional.

### OPEN (not counted)

- When oracle s7 lands: regenerate; check that `\nOracleLateMax` ("2 to 15") still holds (s7 session 3 = 13.8 so far), that
  sessions 3 and 4 rise at the next boundary on s7 (the generator warns otherwise), `\nOracleN` = "two seeds", and that
  `mechanism_analysis.json` is regenerated before `tab_mech` (the oracle row is n = 1 until then).
- When any ladder slot lands: its verdict must be written by hand (L7); re-audit the rung's prose against the per-seed numbers.

## R10 (2026-11-12): audit of P12 (`e7835b7`): R9 fixes, transposed mechanism table, dropped-methods table, first-page figure

Snapshot: paper files at `e7835b7`. Scope: `git diff 6217ed5..e7835b7 -- paper_cvpr scripts/rca/make_clean_tables.py`, the
new figure, the changed `supp.tex` sections, and the run files. Method: I re-ran the generator with `OUT` redirected to my
scratchpad (it reads only `metrics.json` / `matrix.npy` / `mechanism_analysis.json`): **every generated file is
byte-identical to the committed one, no warning, no verdict abort**. I then recomputed the new cells independently from
`per_class_f1.json` + `metrics.json` (implied full-label FP) and from `mechanism_analysis.json`, and read the first-grid run
files, logs and git history behind the new App. G table. Builds from `git archive e7835b7 paper_cvpr` in my scratchpad.
No model, dataset or pytest was run. GPU at audit time (08:49): oracle s123 running (queue: START 08:30); s7 landed.

### Verdict: **BLOCK** (one HIGH, text-only, about 15 minutes)

R9's two HIGH findings are fixed in substance: contribution 1, the abstract, the intro and H0 no longer claim that error
type tells the causes apart, the naive column is in the table, and the selection paragraph now gives the true order. Every
number recomputes. What blocks is new and sits in the R9-H2 fix itself: the disclosure names two dropped methods that the
runs behind it are not. "ER with C-Flat++" is ER + SAM (curvature weight 0 in every run), and the O-LoRA row comes from
runs made before the port's orthogonality penalty was repaired. Same standard as R9-H2: a disclosure paragraph must be
literally true. Four MEDIUM and eleven LOW follow.

### A. Numbers recomputed (per seed 42 / 7 / 123)

| Cell / macro | Recomputed | Rendered | OK |
|---|---|---|---|
| Naive column: R; P; "missed at O" indicator | 0.0 ×3; no predicted old span (TP = FP = 0, 8 / 9 / 21 FP spans on the full test set); 0.9934 / 0.9926 / 0.9827 | 0.0; blank; 98.2–99.4 | ✓ (outward rounding) |
| CoLaR indicator | 0.8964 / 0.9069 / 0.9421 | 89.6–94.3; macros 89–95 | ✓ |
| Relab. (oracle, s42 / s7): R; P (upper bound); indicator | 77.22 / 71.58; 88.17 / 87.01; 0.642 / 0.682 | 71.5–77.3; ≤87.1–88.2; 64.2–68.2 | ✓ |
| No mem. indicator | 0.0 / 0.818 / 0.0 | 0.0–81.9 (body: "81%") | ✓ (L2) |
| `\mColarOldPLb` (TP_old / (TP_old + FP_implied,all)) | 71.77 / 71.68 / 78.47 → floor 71.6 | 71.6 | ✓ (conservative) |
| `\nOracle*` two-seed cells | AA 65.8 / 62.8, full 79.1 / 74.9; CoLaR 33.6 / 32.4, 32.4 / 30.5; joint 91.1 / 90.5, 91.7 / 91.8; SBR 87.8 / 91.3, 88.8 / 91.2 | same | ✓ |
| `\nOracleShare` / `Full` | AA 56.0 / 52.3; full 78.8 / 72.4 | 56/52, 79/72 | ✓ |
| Oracle diagonals sessions 3–5 | s42 6.8 / 2.2 / 14.7; s7 13.8 / 13.9 / 3.9 → "2 to 15" | 2–15 | ✓ |
| Oracle next-boundary rise R[4,3], R[5,4] | s42 83.0, 74.6; s7 87.5, 81.6 → "74 to 88" | 74–88 | ✓ every seed rises |
| `\nOracleLast` | 14.7 / 3.9 | 14.7/3.9 | ✓ |
| Relabel counts in prose | s42 105/1019/1394 + 1239 (s7 log: 67/859/1349 + 1273); prose says "(seed 42)" | | ✓ |
| `\nSlicedBothGate` | gate sliced2 s42 62.9 vs sliced replay 39.4 (both fp32 gate); log `current_sliced: true` = O tokens only | 62.9 | ✓ |
| H0: CoLaR newest session R[5,5] vs naive diagonals | CoLaR 2.0 / 9.5 / 15.9 (R[4,4] 2.2 / 6.3 / 16.8; R[3,3] 14.0 / 8.9 / **54.6**); naive diagonal min 78.7 / 90.4 / 88.9 | 2.0–16.0 vs ≥78.6 | ✓ |
| Abstract "barely learns the last sessions it is trained on, which naive training learns well" | true per seed for sessions 4 and 5 (≤ 16.8 vs ≥ 78.7); session 3 only on s42 and s7 | | ✓ (read as the last two) |
| H3 "above the relabelled memory on every seed both have" | SBR 88.8 / 91.2 vs 79.1 / 74.9 full; 87.8 / 91.3 vs 65.8 / 62.8 AA | | ✓ |
| `\nRecallJointGap` 2.1, `\nSbrGapFullMax` 5.3 | 0.3 / 2.1 / 0.2 (AA), 0.2 / 1.7 / 0.2 (full); 2.87 / 0.61 / 5.25 | | ✓ |
| `supp_dropped.tex` | naive 18.79 / 19.08 / 19.08; ER 16.94 / 14.43 / 16.58; EWC 1.11 / 0.00 / 0.28; L2P 0.00 / 2.09 / 1.30; DualPrompt 0.00 / 2.03 / 3.31; O-LoRA 0.00 / 0.00 / 14.22; er_cflat 14.17 / 16.65 / 16.96 | same | ✓ values (H1, M3 on what they are) |
| `supp_agree` s7 oracle row | 62.8 / 74.9 / +12.1; group order holds on s7 | | ✓ |

### B. Findings

**HIGH**

- **R10-H1. Two of the five dropped methods in the new disclosure are not what the runs are.**
  - Where: main.tex:102 (Sec. 3, "the methods it dropped then (EWC, L2P, DualPrompt, O-LoRA, ER with C-Flat++) stayed
    below naive training"); `supp_dropped.tex` row "ER + C-Flat++" and row "O-LoRA" (generator `DROPPED`); App. G
    "How the comparison set was fixed". The same mislabel pre-exists in App. G `tab:prompt_lora` (caption
    "Prompt/LoRA/C-Flat++ baselines") and its prose ("ER with C-Flat++ sharpness-aware optimisation ... its sharpness term
    adds nothing over plain ER").
  - Evidence, C-Flat++: `configs/method/er_cflat.yaml:17` `cflat_lambda: 0.0`; `doccl/methods/sam_step.py:98-112`: with
    `cflat_lambda == 0` the step is exactly SAM (the curvature term is added only when λ > 0). No run overrode it (the
    `_bert` runs record `cflat_lambda: 0.0` in `method_hparams`; the June runs predate hparam logging). Every `er_cflat`
    row in the paper is ER + SAM.
  - Evidence, O-LoRA: `results/cil_cord_o_lora_seed*` (and `dil_o_lora_seed*` LayoutLMv3) are dated 2026-06-18; the
    orthogonality penalty of the port computed `A_curr.T @ A_past` instead of `A_curr @ A_past.T` until commit `c974159`
    (2026-06-24, "repair O-LoRA/CL-LoRA ortho penalty"). These rows were trained without the constraint that defines O-LoRA.
    (Their diagonals, 0.0 on sessions 2 and 3 on every seed, may be the bug.)
  - Fix (15 min): rename everywhere to "ER with SAM" (App. G may add "our C-Flat++ port with its curvature weight at 0");
    generator label `("ER + SAM", "er_cflat")`. For O-LoRA, either drop it from the list and the table (the conclusion
    holds on EWC, L2P, DualPrompt), or keep it with a note "(run before the fix of the port's orthogonality penalty,
    2026-06-24)". Mark the DIL O-LoRA / CL-LoRA LayoutLMv3 rows of `tab:prompt_lora` the same way (06-18/19 runs).

**MEDIUM**

- **R10-M1. Residue of R9-H1 in three sentences.**
  - main.tex:153 ("What discriminates what"): "The error type separates drift from conflict only within the blocked
    readout, memory against no memory". Within the blocked readout neither run has the conflict (H3 removes it by design);
    the table shows drift (no memory: 84–94% to another session, 17–24% false claims) against reduced drift (memory). Write:
    "The error type shows drift only within the blocked readout (memory against no memory); under neither readout does it
    separate drift from conflict: ...".
  - main.tex Conclusion: "what separates the conflict from drift is a suppressed current session that the memory does not
    cure". Naive training has no suppressed session; the memory brings it (with a shared softmax). Write "a suppressed
    current session, which appears only with a memory and a shared softmax".
  - Contribution 1 and the Conclusion state "old entities are missed at 'O' with or without a memory" as a measurement.
    For naive it is safe (recall 0, 8–21 FP spans in total). For CoLaR it is the span indicator the paper itself calls
    "not a bound", which read 82% on a run whose per-token truth is 0%. Add "by a span-level indicator" in both places
    (H0 already hedges it; Limitations already says so).
- **R10-M2. H0's naive-vs-memory contrast has an unnamed confound, and the paper already holds the control.**
  CoLaR differs from naive in more than the memory. Embeddings and layers 1–4 are frozen after session 1 (logs: "freeze
  map — layers <4 + embeddings frozen"), and four stored documents are replayed per step. A reader can blame the
  suppressed current session on lost plasticity. The Sec. 4 preamble says only "H0's raw-memory row changes several"
  things, but the CoLaR row does too. The controls exist and refute that reading. CoLaR-SBR has the same freeze map,
  memory and replay batch and learns every session to ≥ 81.6 on the diagonal (92.6/92.1/81.6/88.8/93.4,
  95.0/92.9/93.8/88.7/92.7, 94.1/92.0/93.1/90.5/95.4). No-memory SBR has the same freeze map and learns every session to
  ≥ 81.2. Fix (one clause in H0 and in "What discriminates what", and widen the preamble's "unless stated"): "the memory
  run also freezes layers 1–4; with the same freezing, memory and replay, the per-session background learns every session
  to at least 81 (App. D), so lost plasticity does not explain it."
- **R10-M3. The dropped-methods comparison is weaker than the sentence reads, and "one protocol" cannot be verified.**
  - The first grid uses raw scoring (joint reads 33). Under it, a model that keeps old classes is charged on every other
    session's split, so "below naive" does not mean "weak". ER, which was kept, is also below naive on every seed
    (16.9 / 14.4 / 16.6 vs 18.8 / 19.1 / 19.1). ER + SAM tracks ER (diagonals 55–97, final last-session 65–80) and is
    below naive for the same reason.
  - What actually separates the other dropped methods is that they learn no later session. EWC is 0.0 on sessions 4–5,
    and L2P and DualPrompt 0.0 on sessions 3–5, every seed. O-LoRA is 0.0 on sessions 2–3 every seed (see H1).
  - Protocol: the `metrics.json` files predate hparam logging (`c974159`), and the rows come in two batches. Naive, ER,
    EWC, L2P and DualPrompt ran 06-16/17 with a recorded peak of 10.5–27 GB, so not on the 6 GB box. O-LoRA and ER + SAM
    ran 06-18/19 with a peak of 3.7–5.6 GB. The 06-12 autonomous log also shows 3-epoch runs under the same run names
    (overwritten later). "Same grid" holds by directory; same recipe is not shown.
  - Fix: body "...the methods it dropped then (EWC, L2P, DualPrompt, O-LoRA, ER with SAM) either learned no later session
    on some seed or tracked ER, which was kept (App. G)". Caption of `tab:dropped`: add "this scoring puts every method,
    ER included, below naive; two batches of runs, hyper-parameters not recorded".
- **R10-M4. "What discriminates what": the MiB-style loss alone does not separate H2 from H3.** The control described
  in H2 applies MiB "in both directions", so it also marginalises the stale labels of replayed documents (the H1 side).
  It separates "label content" (H1 + H2) from "normalisation" (H3). Separating the current side needs the never-stale
  memory too, as H2's own last sentence says. Write "which the never-stale memory and the MiB-style loss decide".

**LOW**

- **R10-L1 (rule 7, pre-existing, touched).** main.tex:121 "within a point of joint training on DIL on average": CoLaR
  87.6 / 88.3 / 87.4 vs joint 89.7 / 89.0 / 87.5, so seed 42 is 2.1 below. The abstract's "nearly matches" for ER on DIL:
  86.2 / 88.7 / 88.6 vs 89.7 / 89.0 / 87.5 (3.5 on seed 42). Rule 7 asks per seed. Write "within about two points on
  every seed" / "comes within 3.5 points of joint training on every seed".
- **R10-L2.** The indicator row prints 0.0 for CoLaR-SBR and SBR-Recall, a negative value (implied FP > misses) clipped
  at 0. It sits next to their per-token "wrong → O" of 4.9–6.2 / 4.4–28.4. Blank it where per-token outputs exist, or add
  "clipped at 0" to the caption. The body says the indicator "reads 81%" (`\mNoBankBoundMax`, floor); the table says
  81.9. Use one rounding.
- **R10-L3.** H0 cites Tab. 2 for the old-span precision lower bound (71.6%); the table shows only the upper bound
  (≤72.8–80.8). Cite App. B.
- **R10-L4. Figure 1.**
  - What works: legible at 110 dpi (scriptsize, about 7 pt), colour-blind-safe, and correct on who labels what
    (session 1 stores `menu.price` on 4.50 and "O" on 9.00; session 5 labels 4.50 "O" and 9.00 `total_price`, which is in
    session 5 per `session_types`). It is consistent with the SBR supervision rule (stored → block 1, current → block 5),
    and the caption is true.
  - Nits:
    - In (a) the dashed current-document arrow starts under "total_price" (row 9.00), not under that document's "O"
      (row 4.50), so it reads as "total_price trains O". Start it at the 4.50 row's "O".
    - In (b) the arrows land on the class cells, not on O₁ / O₅.
    - "total.∗" for block 5 suggests all total fields are there; block 4 holds six other total.* classes. Write
      "total_price, ...".
    - "One receipt line" shows two receipts. Call it "one pair of price tokens".
- **R10-L5. Table 2.** It is set at about 6.5–7 pt after `\resizebox`, and its caption (11 lines) is longer than the
  table. It is readable but dense. Move the definitions of "missed at O" and "false claims" to App. B and drop
  `\resizebox` with `\footnotesize`.
- **R10-L6. Generator verdict checks can pass vacuously.**
  - `_gap_share` returns [] when a seed is missing in any of the three dicts, and `all([])` is True. `_clean` returns {}
    when a cell's first source is not clean. The `recall` check iterates over an empty dict, and the `sbr` check uses
    `max(..., default=-1)` for the oracle.
  - Add `assert len(shares) == len(SEEDS)` (or require non-empty) in every check.
  - `("sliced2","open")` is `True` by construction; that is acceptable because "open" claims nothing, but say so in the
    docstring.
  - Each check tests its verdict's main prediction. Minor gaps: `nobank supports` does not test "fails the way drift
    does", and `oracle partly` does not test "below SBR". The `sbr` check covers the latter.
- **R10-L7.** App. E "The legacy gate run of the same seed" is ambiguous now that two clean seeds exist: "of seed 42".
- **R10-L8.** "Fixed after a first grid": the kept CPFD, IS3, LCA and CODA-Prompt also ran on CIL-CORD (10-16 to 10-21)
  before the 10-21 freeze. App. G's "methods that fail on the scenario (CPFD, IS3) were kept" covers it. Optional:
  "after a first grid and first runs of the kept methods".
- **R10-L9.** Slack: References start at line 593 in the right column of p. 8. The column runs from line 565 to about
  line 617, so the slack is about 24 lines, **≈ 0.45 column**, not 0.6. That figure is in the commit message only.
- **R10-L10.** R9-L4 remainder (optional): `kalb2023principles`, `liu2023abr` and `chen2023star` still have no pages.
- **R10-L11.** `\mNaiveIndMax` = 100 (ceil of 99.34) is unused; the table's 99.4 is the right print. Harmless.

### C. R9 items

| R9 | Status |
|---|---|
| H1 | **Fixed** in abstract, intro, contribution 1 ("reduced"), H0 and the naive column; three residual sentences (R10-M1) |
| H2 | **Fixed** (true order, App. G table, SD-LoRA correctly "never run on CIL-CORD", no `results/*sd_lora*` exists); new mislabels (R10-H1) |
| M1 | **Fixed** ("not a bound", App. B rewritten, old-span lower bound 71.6 added) |
| M2 | **Fixed** (caption + prose "another tag of its own session, a B/I swap of the right type included") |
| M3 | **Fixed** ("open", seed-42 increment 39.4 → 62.9, "prior work is mixed", "the current pull and the stale labels ... both remain") |
| M4 | **Fixed** (footnote e, preamble scope, "does not isolate memory fidelity"); see R10-M2 for the CoLaR row |
| M5 | **Fixed** (MORE: HAT masks on adapters of a frozen pre-trained transformer, OOD output from saved samples; SBR-Recall difference stated) |
| M6 | **Fixed**. `caccia2022erace` exists (ICLR 2022; authors and title match). The description fits the code: `latent_replay.py:_current_ce`, `current_sliced: true` masks old classes only for current "O" tokens; `"all"` would be full ER-ACE. Gate logs show `true`. |
| M7 | **Fixed** ("in our one run") |
| L1 | **Fixed** (n=1 on the two Mixed legacy cells) |
| L2 | **Fixed** ("closes most of the gap") |
| L3 | **Fixed** (`zhai2023investigating` re-cited; 62 `\bibcite` = 62 cited keys, 0 orphans) |
| L4 | **Partly** (`mo2024bpf`, `sassioui2026documents` fixed; three page fields remain, R10-L10) |
| L5 | **Fixed** (caption: P overstates on the full test set) |
| L6 | **Fixed** (both DIL cells †) |
| L7 | **Fixed** with the vacuity caveat (R10-L6) |
| L8 | **Fixed** (upper bounds round up at both ends) |
| L9 | Dev item, not in the paper; unchanged |
| L10 | **Fixed** (ladder on p. 3, before H0 on p. 4) |
| L11 | **Fixed** |
| L12 | **Fixed** ("is what stale stored labels predict (two seeds so far)") |

### D. Question by question (brief)

1. **Recomputation:** section A. The dropped table compares against `results/cil_cord_naive_seed*`, the June grid
   (raw scoring), not the clean naive (18.8 / 18.3 / 17.8). Within that directory family the comparison is internally
   consistent, but one recipe is not demonstrable (R10-M3).
2. **H0:**
   - The argument is now sound in direction: suppression of the current session is not a drift effect, and the oracle's
     own suppressed sessions 3–5 fit the stale-label account (the session being trained is "O" in every stored document).
   - The other plausible explanations are the frozen embeddings and layers 1–4 and the replay weight. The paper should
     name them and cite its own controls (R10-M2).
   - The replay-loss scale is 1.0 (`replay_task_balance` off). Four replayed documents against two current ones enter
     as two token-mean losses with equal weight, so the replay weight is not an extra factor beyond the stored labels.
3. **"What discriminates what", item by item:**
   - (i) "Session being trained": correct with "under a shared softmax" added (R10-M2).
   - (ii) "Within 5.3 points on every seed once the background is not shared": ✓.
   - (iii) "Relabelled memory measures part of the stored side": ✓.
   - (iv) H2 vs H3 "the MiB-style loss decides": partly (R10-M4).
   - (v) Error type: R10-M1.
4. **Rule 6/7:** every new number comes from a macro or a generated table. The two-seed oracle statements hold per seed,
   and every joint comparison is per seed. Exception: the pre-existing DIL "within a point on average" / "nearly matches"
   (R10-L1). No "upper bound"/"oracle" for joint training.
5. **Figure:** R10-L4. Table 2: R10-L5.
6. **Invariants** (archive build):

   | Check | Result |
   |---|---|
   | `latexmk -pdf` | main 0, supp 0 |
   | Undefined references/citations and "??" | 0 in both |
   | Pages | main 11 pp.; body ends on p. 8, right column, line 592; References at 593; slack ≈ 24 lines (R10-L9). supp 18 pp. |
   | `---` in tracked .tex and em-dashes in the PDF | 0 |
   | Anonymity | "Anonymous CVPR submission"; Author field empty |
   | Cross-references | "Tab. 1/2/3 of the main paper" = ladder / mech / main; Sec. 4.1 = H0; App. A–H letters unchanged; `tab:dil` and `tab:dropped` resolve |

   **Moved, not deleted:** protocol details (App. A), pooling gap (App. B), rank-8 adoption (App. D, new sentence), DIL
   table (App. F), the "reuse a delta" rule (App. C "Which delta a session trains"), the LexSlot detail (App. D), the
   ER/DER++/LwF/CL-LoRA settings (App. G). The CL-LoRA single-adapter disclosure stays in the body (Sec. 5.3). Nothing
   true and needed was lost.

### E. Score

As it stands the referee score stays at **3 (Borderline Reject)**, perhaps +0.25 over R9. The figure fixes the dense
opening, the scoping of error types removes the R9-H1 overclaim, and the H2 "open" verdict is honest. But the
discriminating rungs are still slots, and the only H2 evidence is legacy. The single change that would move it most is
landing `colar_mib` and the never-stale `colar_r_oracle_pre` with three seeds each. They turn the ladder from "listed" to
"tested", decide H2 against H3, and with the rest unchanged make a 4 realistic. Text fixes cannot substitute for them.

### OPEN (not counted)

- When oracle s123 lands: regenerate. "two seeds" becomes three. Check that `\nOracleLateMax` (2 to 15), `\nOracleNextMin`
  (74), "below CoLaR-SBR on every seed" and the group order of App. B still hold. `\nOracleLast` becomes three values.
- When `colar_dump` / `naive_dump` land: the indicator sentences (R10-M1, third bullet) can become per-token statements
  or must be withdrawn.

## R10b (2026-11-12): confirmation of P13 (`fdd4713`) against R10

Scope: `git diff 0df6972..fdd4713 -- paper_cvpr scripts/rca/make_clean_tables.py`.
- I re-ran the `fdd4713` generator with `OUT` pointed at my scratchpad. Every generated file is byte-identical to the
  committed one, with no warning.
- I recomputed the new cells from `matrix.npy` / `metrics.json`.
- I tested the verdict checks by deleting seeds in memory.
- I built from `git archive fdd4713 paper_cvpr` and rendered page 1.

### Verdict: **PASS**

Every R10 finding is fixed except the optional L10 (three bibliography entries without pages, as before). No remaining
finding is above LOW.

| R10 | Status and evidence |
|---|---|
| H1 | **Fixed.** "ER with SAM" is used in main.tex, App. G prose, `tab:prompt_lora` and the generator label `DROPPED`. No "C-Flat" is left outside the one explanatory clause in App. G. O-LoRA is out of the dropped list and table, and App. G says why. The ᵖ marks are correct against the git dates (fix `c974159`, 2026-06-24 07:35): DIL `dil_cl_lora`/`dil_o_lora` runs are from 06-18/19, `mixed_cl_lora` from 06-23, `cil_cord_cl_lora` s42/s7 from 06-19 and s123 from 10-22. So `tab_main_core` ("DIL and Mixed cells and two of its CIL-CORD seeds") is right. In `tab:main` only the DIL cell is marked (`results/dil_cl_lora`, June). Its CIL-CORD and Mixed CL-LoRA cells come from `results/rescore` (11-04 to 11-08, after the fix) and are correctly left unmarked. `tab:prompt_lora` marks the LayoutLMv3 O-LoRA/CL-LoRA rows; its BERT runs are 07-29, after the fix. |
| M1 | **Fixed.** "What discriminates what" now says "shows drift only within the blocked readout ... under neither readout does it separate drift from conflict". The Conclusion now says "appears only with a memory and a shared softmax". "By a span-level indicator" is added in contribution 1, "What discriminates what" and the Conclusion. |
| M2 | **Fixed.** H0, the preamble ("H0's memory rows change several") and App. D now name the freezing. Logs: "freeze map: layers <4 + embeddings frozen" in `colar`, `colar_sbr` and `colar_sbr_noreplay`, i.e. four layers, so "layers 1..4" is right. `replay_batch_size: 4`. `\nSbrDiagMin` 81.1: diagonal minima are CoLaR-SBR 81.58 / 88.71 / 90.46 and no-memory SBR 81.17 / 86.67 / 92.18; 81.167 floors to 81.1. ✓ |
| M3 | **Fixed.** The body sentence and the App. G paragraph state the raw-scoring caveat and the two batches. Unlearned sessions (own diagonal < 1): EWC 3–5 / 4–5 / 4–5; L2P 3–5 / 2–5 / 2–5; DualPrompt 3–5 / 2–5 / 2–5. All exact zeros, so "at least sessions 4 and 5 on every seed" holds. `\nSamErGapMax`: ER with SAM vs ER is 2.77 / 2.22 / 0.38, ceil 2.8 ✓. Rule 7 holds per seed. |
| M4 | **Fixed** ("the never-stale memory and the MiB-style loss decide together"). |
| L1 | **Fixed.** `\nColarDilGapMax` 2.2: joint − CoLaR on DIL is 2.10 / 0.73 / 0.10, rounded up (conservative). `\nErDilGapMax` 3.5: 3.47 / 0.30 / −1.13. Intro and H0 now give per-seed bounds. The abstract's "comes close to joint training on average" is explicitly a mean claim and uses none of rule 7's listed words. ✓ |
| L2 | **Fixed** (the indicator is blank where per-token outputs exist; the clipping is explained in App. B). |
| L3 | **Fixed** (H0 cites App. B for the lower bound). |
| L4 | **Fixed.** Rendered page 1: the dashed arrow now leaves the current document's "O" (row 4.50) and curves to the shared O in (a) and to O₅ in (b). The stored arrow lands on O₁. The blocks read "s1 tags" / "s5 tags", and the caption says "one pair of price tokens". It is legible, and the caption is true. |
| L5 | **Fixed** (`\scriptsize` at its natural width, no `\resizebox`, no overfull box; the caption is 4 lines; definitions are in a new App. B paragraph, which is accurate). |
| L6 | **Fixed and tested.** With joint s7 removed in memory, the `colar`, `sbr`, `nobank`, `recall` and `oracle` checks all return False. With the oracle runs removed, `sbr` and `oracle` return False. A missing seed now fails the check and aborts the build through `SystemExit`. |
| L7, L8 | **Fixed.** |
| L9 | n/a (commit message). Slack now: References at line 603, column ends near 623, so about 20 lines (≈ 0.4 column). |

**Invariants** (archive build):
- `latexmk -pdf`: main 0, supp 0.
- 0 undefined references or citations, 0 "??".
- Main 11 pp.: the body ends in the right column of p. 8 (Conclusion from line 583) and References start at line 603.
  Supp 19 pp.
- 0 `---` in any .tex, 0 em-dashes in the PDF, Author field empty.
- Cross-references resolve.

**LOW (new, optional)**
- R10b-L1: H0 says "with the same freezing and replay, the blocked readouts with and without the memory learn every
  session to at least 81.1". The no-memory run has the freezing but no replay. Write "with the same freezing (and, with
  the memory, the same replay)".
- R10b-L2: `gap_macros` takes the max of the signed gap joint − method for a "within" claim. Use `abs(...)`. The value is
  the same today.

## R11 (2026-11-12): audit of P14 (`dffff14`) and P14b (`964049f`): new exhibits, oracle at three seeds, LCA optimiser

Scope: `git diff b644525..964049f -- paper_cvpr scripts/rca` (worktree = `964049f` for these paths).
- **Independent recomputation.** A throwaway script in my scratchpad reads `results/clean_2026-11/*/metrics.json`,
  `matrix.npy`, `per_class_f1.json`, `eval_dump.pt` and `full_label_dump.pt` on CPU. It imports neither generator.
- **Generator re-run.** I re-ran `make_clean_tables.py` with `OUT` redirected to the scratchpad. Every generated `.tex` is
  byte-identical to the committed one, with no warning.
- **Builds.** I built both PDFs from `git archive 964049f paper_cvpr` and rendered every page that carries a new exhibit
  (90 dpi, with 200–250 dpi crops).
- No GPU, dataset or pytest was used. The queue was running `cil_cord_joint_dump_seed42` (START 13:04).

### Verdict: **BLOCK** (two HIGH, text and one table cell, about 15 minutes)

Every number in the new exhibits recomputes, and the legacy cells are unchanged from `b644525`. The dump re-runs enter no
mean, range or table. The move to the appendix lost nothing, and the build invariants hold. Two things block:
- P14b turned a per-seed macro into a mean inside an "ends at" sentence (rule 7).
- The new baseline card says our LCA port changes nothing, which our own port contradicts.

### A. Numbers recomputed (seeds 42 / 7 / 123)

| Exhibit / cell | Recomputed | Rendered | OK |
|---|---|---|---|
| Clean cells, `tab:ladder` / `tab:main` / F2(a) bars (mean ± s.d., ddof 1) | naive AA 18.79/18.27/17.78 → 18.3±0.5, full 13.0±0.2; joint 91.3±0.8 / 92.0±0.4; CoLaR 33.5±1.0 / 30.7±1.5; **relabelled 65.81/62.76/62.37 → 63.6±1.9, full 79.09/74.91/75.45 → 76.5±2.3**; no-mem SBR 21.6±2.2 / 11.4±2.3; CoLaR-SBR 88.4±2.6 / 89.1±2.0; SBR-Recall 91.9±0.6 / 92.6±1.0 | same; F2(a) bars and dots match by eye at 200 dpi; reference lines 92.0 / 13.0 | ✓ |
| H2 legacy bar | 62.9 / 69.2 / 49.4 → 60.5±10.1, hatched | same | ✓ |
| Legacy (†) cells of `tab:main` | cell-by-cell against `b644525`: unchanged | | ✓ |
| Memory column | `replay_memory_bytes`: CoLaR 100,672,000; CoLaR-SBR 100,672,250; SBR-Recall 56,623,104 (identical over seeds) | 100.7 / 100.7 / 56.6 MB | ✓ |
| `\nOracleShare(Full)` | AA 56.0 / 52.2 / 48.4; full 78.7 / 72.4 / 73.1 | "48 to 56", "72 to 79" | ✓ |
| Oracle below joint **and** CoLaR-SBR, both metrics, per seed | 65.8<87.8, 62.8<91.3, 62.4<86.2; 79.1<88.8, 74.9<91.2, 75.4<87.1 | "partly"; H3 "above the relabelled memory on every seed" | ✓ |
| Oracle diagonals, sessions 3–5 | s42 6.8/2.2/14.7; s7 13.8/13.9/3.9; **s123 36.3/6.2/3.9** | "2 to 37" (outward) | ✓ |
| Oracle next-boundary rise R[4,3], R[5,4] | 83.0, 74.6; 87.5, 81.6; 75.8, 77.2 (every one rises) | "74 to 88" | ✓ |
| `\nOracleLast` | 14.7 / 3.9 / 3.9 | **7.5 (mean)** | ✗ H1 |
| `\nOracleLastShare` | 99 / 1306 = 7.6% | 8 | ✓ |
| `tab:mech` relabelled R / P (`per_class_f1.json`) | R 77.22/71.58/72.91; P 88.17/87.01/86.27 | 71.5–77.3; 86.3–88.2 (P bound rounded up) | ✓ |
| F2(b) (mean over seeds) | span R: naive 0, CoLaR 20.6, relab. 73.9. Per-token "correct" from `eval_dump.pt`: no-mem 4.5, SBR 92.0, Recall 94.4. Wrong-token shares (→O / same / other): SBR 5.6/58.0/36.4, 6.2/71.5/22.3, 4.9/56.7/38.4; Recall 4.4/65.2/30.4, 17.0/63.0/20.0, 28.3/51.2/20.5; no-mem 0/6.6/93.4, 0/16.1/83.9, 0/10.0/90.0 | bars; `tab:mech` ranges (outward) | ✓ |
| F2(c) (newest R[5,5], old mean) | naive (93.9,0.0)(91.4,0.0)(88.9,0.0); CoLaR (2.0,41.5)(9.5,38.2)(15.9,39.0); relab. (14.7,78.6)(3.9,77.5)(3.9,77.0); no-mem (92.9,3.3)(92.8,1.3)(92.4,6.8); SBR (93.4,86.3)(92.7,91.0)(95.4,83.9); Recall (94.8,90.5)(94.3,92.2)(94.4,91.2); joint (92.5,90.8)(91.5,90.3)(94.5,91.5) | plotted | ✓; caption holds per seed |
| Heatmaps (mean matrices) | naive diagonal 93.9/91.9/93.2/86.1/91.4, final row 0 except 91.4; CoLaR final 20.9/59.2/43.3/34.8/9.2; relab. final 82.3/76.8/73.8/77.8/7.5; no-mem 4.0/8.8/1.1/1.4/92.7; SBR 90.8/83.4/87.9/86.2/93.8; Recall 94.1/89.7/92.3/89.1/94.5; joint 93.6/90.3/93.2/86.4/92.8 | same | ✓ |
| Trajectory (mean of R[i,:i+1]) | CoLaR 93.9/71.6/47.2/41.2/33.5; relab. 93.9/71.6/50.6/56.0/63.6; no-mem …/21.6; SBR 93.9/93.0/89.8/89.3/88.4; Recall …/91.9; end points = AA | plotted | ✓ |
| Routing (pooled, % of row) | no-mem rows 6/61/33/0/0/0, 7/60/33/0/0/0, 32/33/33/0/1/0, 19/45/31/5/0/0, 2/3/1/0/94/0; SBR row 2 21/78/0/0/0/1, others ≥ 96 on the diagonal | same | ✓ |
| Spans per session | 744/140/143/180/99 = 1,306; 57% | same | ✓ |
| Per-class F1 (Fig. 13, `full_label_dump.pt`, seqeval by document) | CoLaR dump 32.38 (= clean s42 32.38, AA 33.59 = 33.59); naive dump 13.22 (= clean s42). CoLaR: s1 31/0/0/19/0/17, s2 75/56/87/84, s3 0/0/42/50/45, s4 35/31/9/0/9/0, s5 0/2; naive: only total_price 95 | same | ✓ App. I prose holds |

The dumps also give the per-token picture the span indicator stands in for, on seed 42 only. Of the old-session gold tokens
that are wrong, CoLaR puts 97.7% at "O" and naive training 99.6%. That supports the indicator sentences on that seed. It
is not a cell, and I do not ask for it to become one.

### B. Findings

**HIGH**

- **R11-H1 (rule 7). "the last session ... ends at 7.5" is a mean that no seed reaches.**
  - Where: main.tex:141 (H1 Result: "ends at $\nOracleLast$") and supp.tex:357.
  - Evidence: `\nOracleLast` comes from `_per_seed` (make_clean_tables.py:642), which prints the mean once n = 3. The
    per-seed values are 14.7 / 3.9 / 3.9. Rule 7 asks that every "at" sentence hold per seed. P14b introduced this
    (it read "14.7/3.9" before).
  - Fix: print the range, e.g. `v["nOracleLast"] = _span([m[-1,-1] for m in mats.values()], "{:.1f}")`, which renders
    "3.9 to 14.7". Or keep the per-seed list.
- **R11-H2. The baseline card says our LCA port changes nothing; our own port says otherwise.**
  - Where: `supp_baselines.tex` LCA row, "Our port changes" = "--" (generator `CARD`, make_clean_tables.py:1289).
  - Evidence: `doccl/methods/lca.py:21-25` lists the differences. A single growing head replaces LCA's per-task heads.
    The class Gaussians are per BIO token class, and "O" is excluded from sampling (`ca_skip_O: true`,
    `configs/method/lca.yaml`). The card discloses the same head change for CPFD and IS3 ("one growing token head
    (published: per-task classifiers)"). Hiding it for LCA makes the card wrong and inconsistent.
  - Fix: "one growing token head (published: per-task heads); per-token-class Gaussians, O excluded".
  - The optimiser disclosure itself is correct. `lca.py:141-150` is SGD (momentum 0.9, weight decay 5e-4, cosine, lr
    0.01). `lca.py:268-271` is the realignment SGD at 5e-3. Every LCA run records lr 0.01 / ca_lr 0.005 / wd 5e-4 (CIL-CORD
    and Mixed rescore; DIL s42/7/123).
  - No other baseline departs from the stated protocol undisclosed. Recorded hparams: AdamW 5e-5 and wd 0.01 for ER,
    DER++, LwF, CODA-Prompt, CL-LoRA, CPFD and IS3 (code: `er.py`, `der.py`, `lwf.py`, `prompt_base.py`, `cl_lora.py`,
    `cpfd.py`, `is3.py`). The epoch caps (100, or 30 for CODA-Prompt and CL-LoRA) and the rescore precision are legacy
    settings already disclosed in `tab:legacyprotocol`.

**MEDIUM**

- **R11-M1. "72to79%" and "48to56%" are typeset in math italics.** P14b changed `\nOracleShare(Full)` to "48 to 56" /
  "72 to 79", but they are used inside `$...\%$`. Rendered p. 5, lines 284–285 read "72*to*79%". The same happens at
  supp.tex:355. Fix: drop the `$` around them (`\nOracleShareFull\%`), or have the generator emit `\text{ to }`.
- **R11-M2. "one published method per family" contradicts the new grouped `tab:main`.** main.tex:106. The table now shows
  Replay (ER, DER++), Prompt/LoRA (CODA-Prompt, CL-LoRA) and Continual NER (CPFD, IS3), so two methods in three of the five
  families. Fix: "eight published methods from five families (\cref{tab:main})".
- **R11-M3. Fig. 2(a) shows five "queued" placeholders** (raw mem. + SBR, relabel before, MiB loss, const. bg, ICE-O).
  These are experiments not run, and Tab. 1, which the caption says panel (a) draws, does not have these rows. In a
  submission this reads as unfinished work and points straight at the missing H2 control.
  - Fix: render a slot only once its runs exist, the rule `tab_ladder.tex` already follows (make_paper_figs.py:149-152,
    skip `c["slot"]`). Keep the "queued" text in H2's prose only.
  - If the PM wants the slots shown, the caption must say "(not in Tab. 1)".
- **R11-M4. Fig. 2(b) is cited for things it does not show.**
  - main.tex:151 cites `fig:findings`b for "old-session recall stays at 71.5 to 77.3%, against 88.4 to 92.6% for
    CoLaR-SBR". Panel (b) draws span recall only for the shared-softmax runs. For CoLaR-SBR it draws the per-token
    "correct" share, 92.0 on the mean, a different quantity from the 88.4–92.6 span recall.
  - main.tex:184 says "we measure the routing directly (\cref{fig:findings}b; App. B)". Panel (b) shows tag
    correctness, not which block claims a token.
  - Fix: cite "App. B, Tab. 3" for the recall comparison. For routing, cite "App. B, Tab. 3; App. I, Fig. 12".

**LOW**

- R11-L1. main.tex:73 (contribution 1): "rests on \nOracleN{} so far" → "rests on three seeds".
- R11-L2. supp.tex:240 says "... 3.7 encoder-layer equivalents ..., which is the figure in the main paper". The body
  dropped the Cost paragraph, so delete the clause.
- R11-L3. supp.tex:676: "every method but LCA, below". Nothing below describes LCA. Write "(LCA: App. G)".
- R11-L4. supp.tex:359 and 387 say "(cited in the main paper)". The supplement now has its own bibliography (App. J),
  so cite `caccia2022erace` and `ramasesh2021anatomy` directly.
- R11-L5. supp.tex:708: "With a shared softmax the curves fall at every boundary ...; the relabelled memory rises again
  after session 3". The relabelled memory is itself a shared softmax. Write "Without the refresh, the shared-softmax
  curves fall at every boundary".
- R11-L6. `fig/method.tex`.
  - The diagram is faithful to Sec. 5 and `colar_sbr.py`: freeze map below layer 4, rank-128 layer-4 SVD with boxes,
    labels and session, Δ₂..Δ₅ with Δ₁ = 0, claim e_b = 1 − p_b(O_b) > 0.5.
  - One label is off: "layers 5..12, base θ₀" is true only of SBR-Recall. Under CoLaR and CoLaR-SBR these layers keep
    training. Write "layers 5..12 (SBR-Recall: frozen base θ₀)".
- R11-L7. Fig. 13: a class absent from the test set (NaN, e.g. `menu.vatyn`, `void_menu.*`) looks the same as F1 = 0.
  Grey its tick label or print "n/a".
- R11-L8. Fig. 2(c): joint training, CoLaR-SBR and SBR-Recall overlap top right, and naive and no-memory SBR overlap
  bottom right. The caption claims nothing about them, so this is acceptable. An inset or jitter is optional.
- R11-L9. The baseline card renders at about 5 pt after `\resizebox`. It is legible zoomed, small in print. Consider
  splitting the hyper-parameter column.
  - Also, CL-LoRA "keeps: orthogonality penalty, distillation" should be checked against He et al. 2025.
  - The port also keeps gradient reassignment (`shared_grad_scale`), which the card omits.
- R11-L10. In `figures/spans.pdf`, the "180 (14%)" label touches the dashed equal-share line.
- R11-L11 (pre-existing). main.tex:149: "Its seeds are not complete". No `cil_cord_colar_mib` run exists on disk; write
  "It has not run yet".
- R11-L12. R10b-L1 ("same freezing and replay", H0) and R10b-L2 (signed max in `gap_macros`, make_clean_tables.py:1071)
  are still open. Both are optional.

### C. Checks by question

1. **Numbers:** section A. All match, except the rule-7 print of `\nOracleLast` (H1). The `tab:mech`, F2(b) and routing
   per-token values match my recomputation from `eval_dump.pt` to the outward rounding.
2. **Rule 6/7:** the bars, dots and cells are clean only with three clean seeds. Legacy is hatched or dashed with †.
   - `MECH_ROWS` holds no `_dump` run. The dumps appear only in Fig. 13, labelled "seed 42, one run", and they reproduce
     the clean seed-42 runs exactly.
   - Relabelled-memory sentences at three seeds:
     - "partly" is justified: 48–56% (AA) and 72–79% (full) of the gap, below joint and CoLaR-SBR on every seed under
       both metrics.
     - "recovers much of the gap" ✓; "below joint training" ✓.
     - "learned **only** to 2 to 37": still fair. All nine values are ≤ 36.3 against naive diagonals ≥ 78.7 on every
       seed. Note that the 37 is session 3 on s123, which CoLaR without the refresh learned to 54.6 on that seed; the
       paper claims nothing about that comparison.
     - "rise to 74 to 88 one session later" ✓ (all six rise).
     - "ends at 7.5" ✗ (H1).
3. **`tab:bench` and the card.**
   - `tab:bench` against `scenarios.py`, App. A and `dil_remapping.py`:
     - 30 classes / 61 tags, 6 per session ✓.
     - DIL 4 classes / 9 tags ✓.
     - Mixed: 6 sessions, 3 + 4 + 5 = 12 classes, 25 tags, 3 schemas ✓.
     - 149/50, 626/347, 800/100 ✓.
   - Card venues against `refs_precompiled.bbl`: NeurIPS 2019 / 2020, ECCV 2016, CVPR 2023 / 2025, EMNLP 2023, Findings
     ACL 2024, ICLR 2026 ✓.
   - Card hyper-parameters equal the recorded `method_hparams` of the first seed ✓. Seed counts ✓.
   - LCA: H2.
4. **Moved content:** nothing was lost, and nothing was strengthened.
   - Related work → App. J.
   - Comparator list → App. J.
   - Sliced-both 3-seed mean ("about 60 ... 20-point spread", true: 62.9/69.2/49.4) and the pseudo-label rows → App. E.
   - Conflict-on-one-receipt paragraph → App. E.
   - Ablations (`\nSbrFrozen`, calibration ≤ 0.6) → App. E.
   - Cost (100.7 MB, 57 MB, 3.7×) → App. C.
   - Dropped-methods detail → App. G.
   - `tab:mech` → App. B.
   - Body pointers resolve (App. B, C–E, G, I, J). The supplement's "Tab. 3 / Fig. 2b / Fig. 2 and 4 / Fig. 1 of the main
     paper" match the new numbering. One stale pointer remains (L2).
5. **Figures:**
   - Legible at print size (6.5–7 pt), axes labelled, one colour per method (Okabe-Ito) in F2(a), F2(c), Figs. 10, 11
     and 13.
   - Captions are self-contained and true, except the M4 citations and L5.
   - `fig:method` is faithful (L6).
6. **Build/form:**
   - `latexmk -pdf`: main 0, supp 0. 0 undefined references or citations, 0 "??".
   - Main is 11 pp.; the body ends on p. 8 (Conclusion, right column), and References start at the top of p. 9.
     Supp is 25 pp.
   - 0 `---` in any .tex; Author empty; no names in the figure PDF metadata.
   - Every new float is referenced.

### D. Referee view

The paper now shows its evidence.
- Fig. 2 lets a reader see the ladder, the error destinations and the suppressed newest session in one place.
- `tab:main` with Memory and Readout columns makes the comparison legible.
- App. I gives the matrices, spreads, routing and trajectories a reviewer would ask for.

What is still missing:
1. **H2 has no picture because it has no data.** The MiB-loss and never-stale controls are not run, and Fig. 2(a) now
   advertises that (M3).
2. **The comparison is almost entirely legacy.** No baseline has a clean cell or a full-label F1, which is the paper's
   headline metric.
3. **H3 on CIL-CORD has one backbone.** The B+ plan's multi-backbone requirement is unmet for the central claim.
4. **The shared-softmax error split rests on an indicator.** The seed-42 dumps support it but are not reported.

**Score: 3 (Borderline Reject), +0.25 over R10.** Presentation improved, and the evidence did not. What would move it to
4:
- `colar_mib` and the relabel-before control at three seeds;
- clean ER (and DER++) on CIL-CORD with full-label F1;
- CoLaR-SBR on a second backbone on CIL-CORD.

## R11b (2026-11-12): confirmation of P15 (`2ec05b1`) against R11

Scope: `git diff 964049f..2ec05b1 -- paper_cvpr scripts/rca` (worktree = `2ec05b1` for these paths).
- **Generator re-run.** I re-ran the `2ec05b1` `make_clean_tables.py` with `OUT` in my scratchpad. Every generated
  `.tex` (including the new `supp_baselines_hp.tex`) is byte-identical to the committed one, with no warning.
- **Builds.** I built both PDFs from `git archive 2ec05b1 paper_cvpr` and rendered main p. 5, main p. 6 (crop), supp
  p. 3 (crop), supp p. 17 and supp p. 23.
- **Card against code.** I checked every card and Memory cell against `doccl/methods/*.py`, the recorded `method_hparams`
  and `CL4IE/raw/CL_LoRA_*.md`.
- **New dump.** I recomputed `cil_cord_joint_dump_seed42` (landed 13:28).

### Verdict: **BLOCK** (one HIGH, one cell in a generator dict, about 5 minutes)

Every R11 finding is fixed and none of the fixes adds anything false. Checking the Memory column against the code, as the PM
asked, turns up one false cell. It was already there in R11 and I missed it then. It is the same class of error as R11-H2.

| R11 | Status and evidence |
|---|---|
| H1 | **Fixed.** `\nOracleLast` = "3.9 to 14.7" (`_span` over the three seeds; per seed 14.7 / 3.9 / 3.9). Rendered p. 5 l. 295: "ends at 3.9 to 14.7". Holds per seed. Same in supp.tex:357. |
| H2 | **Fixed.** LCA card: "one growing token head (published: per-task heads); Gaussians per token class, ``O'' excluded" (`lca.py:21-25`, `ca_skip_O: true`). Keeps: "per-session fine-tuning (SGD), TIES backbone merge, Gaussian classifier realignment" ✓. App. G prose says the same. Memory "per-session backbones, class Gaussians": `lca.py:103-130,229-232` stashes each session's backbone and merges them ✓. |
| M1 | **Fixed.** The `$` around the range macros is gone. Rendered "72 to 79% and 48 to 56%", and "(means, three seeds)" now labels the mean values ✓. Same in App. E. |
| M2 | **Fixed.** "eight methods from five families (\cref{tab:main})" in main.tex:106 and App. G ✓ (ER, DER++, LwF, CODA-Prompt, CL-LoRA, CPFD, IS3, LCA; five families). |
| M3 | **Fixed.** Fig. 2(a) renders no slot (7 bars: CoLaR, ER, relabelled, sliced losses, CoLaR-SBR, SBR no mem., SBR-Recall). The caption's "queued" clause is removed. |
| M4 | **Fixed.** H2 cites "App. B, Tab. 3" (= `tab:mech`, supp label 3 ✓). Sec. 5.1 cites "App. B, Tab. 3; App. I, Fig. 12" (= `fig:routing`, supp label 12 ✓). |
| L1–L5 | **Fixed** ("rests on three seeds"; the 3.7× clause dropped; "LCA: \cref{app:baselines}"; ER-ACE, Ramasesh and Kalb now cited directly; "Without the refresh, the shared-softmax curves ..."). |
| L6 | **Fixed.** The base box reads "layers 5..12". The θ₀ note now sits with SBR-Recall ("layers 5..12 frozen as θ₀"). Rendered without collision. |
| L7 | **Fixed.** Absent classes are marked "n/a" in grey (`menu.sub.unitprice`, `menu.vatyn`, `sub_total.othersvc_price`, `sub.*`, `void_menu.*`), and the caption says so. |
| L8 | **Acceptable** (question a). With semi-transparent markers the two clusters still overlap. The caption makes claims only about CoLaR / relabelled (left) and naive (bottom right), and those stay distinguishable. No change needed. |
| L9 | **Fixed.** The card is split into two tables at natural width (`\scriptsize`, no `\resizebox`), legible at 90 dpi. CL-LoRA, question (c): **confirmed.** `cl_lora.py:132` multiplies the KD term by `shared_grad_scale`, and both CIL-CORD rescore runs record `shared_grad_scale: 1.0`. So "at scale 1, i.e. off" and App. G's "leaves the distillation gradient unchanged" are literally true. The published design per `CL4IE/raw/CL_LoRA_*.md` ("random orthogonal down-projection ... early-exit knowledge distillation with gradient reassignment"; "block-wise weights under orthogonal constraints") matches the card's parenthesis. "Keeps orthogonality penalty" is fair (the task-specific adapters carry orthogonal constraints). |
| L10 | **Fixed.** The span labels sit on a white box above the dashed line. |
| L11 | **Fixed** ("It has not run yet"; no `colar_mib` on disk). |
| L12 | **Fixed.** H0 reads "(and, with the memory, the same replay)". `abs(...)` sits in the DIL "within" gap (make_clean_tables.py:1087). `\nColarDilGapMax` and `\nErDilGapMax` are unchanged (2.2, 3.5; ER's −1.13 seed is smaller in magnitude). |

The other card cells check against the code:
- CPFD: confidence pseudo-labels, logit KL and three-view attention MSE, adaptive CE factor, and `sample_weights` simplified
  to 1 (`cpfd.py:1-40`) ✓.
- IS3: O-token KD, head-row gradient surgery at 0.6, prototype replay (`is3.py:15-37`). Memory "previous model, class
  prototypes" ✓.
- CODA-Prompt, question (b): the attention-weighted components and orthogonality penalty, the frozen encoder, and the
  "AAAI-Lite input-prepend variant" (`coda_prompt.py:9-16`) ✓. **Acceptable** with no claim about the published design.
  Optional: "(published: prefix prompts in several layers)".
- ER and DER++ ✓.

Fig. 13 now carries the joint-training dump:
- Recomputed with seqeval by document, full-label F1 91.70 (= clean joint s42 91.70; AA 91.13 = 91.13).
- Per class: s1 96/70/0/89/84/98; s2 92/81/92/94; s3 100/69/100/93/97; s4 92/92/83/33/76/33; s5 0/94. The bars match.
- The App. I paragraph still describes only the CoLaR re-run, which stays true.
- "Re-runs still in the queue" is true: `cil_cord_colar_sbr_dump_seed42` started 13:28.

**Invariants** (archive build):
- `latexmk -pdf`: main 0, supp 0.
- 0 undefined references or citations, 0 "??".
- Main 11 pp.: the Conclusion is on p. 8 and References start at the top of p. 9. Supp 25 pp.
- 0 `---` in any .tex.

### Remaining findings

**HIGH**

- **R11b-H1. CL-LoRA's Memory cell says "none (one shared adapter)"; the port keeps the previous model and the past adapter
  matrices.**
  - Where: `tab:main` (main p. 8) and the card's Memory column, from generator `BASE_INFO["cl_lora"]`.
  - Evidence:
    - `cl_lora.py:54` and `:17-20` keep a frozen copy of the model after the previous task as the KD teacher.
    - `o_lora.py:69,203` appends every session's LoRA A matrices to `past_A_matrices` for the orthogonality penalty.
    - LwF, CPFD and IS3, which keep the same teacher, are listed as "previous model". "none" is therefore false, and
      inconsistent within the column.
    - This was already in `964049f`; R11 missed it.
  - Fix: `"cl_lora": ("Prompt, LoRA", "previous model, past adapter matrices")`, then regenerate.

**LOW (optional)**

- R11b-L1. App. I "Per class" describes only the CoLaR bars. One clause on the other two would help a reader:
  - naive training finds only `total.total_price` (95);
  - joint training misses only rare classes (0 on `menu.itemsubtotal` and `total.total_etc`, test support 6 and 3).
- R11b-L2 (code only, not the paper). The `is3.py` docstring names the authors "Chen, Cao, Li"; the paper (arXiv
  2402.10447) is Qiu, Zheng, Liu, Luo and Ma, as the bibliography has it. A Dev nit.

The referee score is unchanged at **3 (+0.25)**. P15 is presentation and accuracy, not new evidence (see R11 §D).

## R11c (2026-11-12): confirmation of P16 (`642048c`) against R11b

Scope: `git diff 2ec05b1..642048c -- paper_cvpr scripts/rca` (worktree = `642048c`). I built both PDFs from
`git archive 642048c paper_cvpr` and rendered supp p. 18 (`tab:legacyprotocol`).

### Verdict: **PASS**

| Item | Status and evidence |
|---|---|
| R11b-H1 (CL-LoRA Memory) | **Fixed.** `BASE_INFO["cl_lora"]` = "previous model, past adapter matrices", in `tab:main` and the card. Evidence: `cl_lora.py:54` (frozen previous-task teacher) and `o_lora.py:69,203` (`past_A_matrices`). |
| LCA Memory "base and per-session backbones, class Gaussians" | ✓ `lca.py:81` declares `_base_backbone`, `:128-130` stashes it once, `:182-184` appends each session's backbone to `_task_backbones` before the TIES merge; the class Gaussians come from `_compute_class_stats`. |
| `tab:legacyprotocol` Memory, hand-edited | ✓ LwF "previous model" (`lwf.py:129-137`); CPFD "previous model" (`cpfd.py:194`); IS3 "previous model, prototypes" (`is3.py:87`, `prototypes` at `:195`); CODA-Prompt "none" (no teacher or buffer in `coda_prompt.py` / `prompt_base.py`); CL-LoRA now its own row, "previous model, past adapters", with the same cap 30, fp32/fp16 and seeds 2, 2 as the old combined row; LCA "backbones, class Gaussians". Consistent with `tab:main`. The table fits the text width at `\tabcolsep` 3.5pt; no overfull box in either log. |
| New App. I sentence | ✓ Recomputed from `full_label_dump.pt` (seqeval by document). The naive re-run is non-zero only on `total.total_price` (95), a session-5 class. Joint training is non-zero on 21 of the 23 classes that occur in the test set; the two zeros are `menu.itemsubtotal` (support 6) and `total.total_etc` (support 3). "Almost every class that occurs in the test set" holds. |
| Invariants | `latexmk -pdf`: main 0, supp 0. 0 undefined references or citations, 0 "??", 0 overfull. Main 11 pp.: Conclusion on p. 8, References at the top of p. 9. Supp 25 pp. 0 `---` in any .tex. |

**LOW (optional)**
- R11c-L1. In `tab:legacyprotocol` the Memory column now lists kept models (previous model, backbones). Yet "SBR-Recall,
  rank 8 / rank 64" still reads "none" for the runs without the document memory. Those runs keep a frozen base and one
  delta per session (56.6 MB at rank 64, as `tab:main` reports). Write "deltas" or "deltas, 50 docs/session".
- R11b-L2 (the `is3.py` docstring authors) is still open. It is code only.

The referee score is unchanged at **3 (+0.25)**.
