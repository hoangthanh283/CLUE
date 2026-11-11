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
