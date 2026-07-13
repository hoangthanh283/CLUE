# EXPLORE.md — Session exploration log (2026-07-10 → 07-13)

Every experiment run this session, disk-verified from `results/*/metrics.json`. Numbers are
dil (FUNSD→SROIE→CORD), LayoutLMv3, seed 42 unless noted. `row3` = retention after the last
task `[FUNSD, SROIE, CORD]`; `diag` = at-learning F1 per task. **Budget matters** — "5ep" is a
capped diagnostic; "grid" (10ep, val-F1 early stop) is the real operating point. Do not compare
across budgets.

**Reference anchors (grid budget):** joint oracle 88.7±1.1 · naive ~41 · raw latent-replay
k4/d50 87.3 · ER/DER++ 87–88.

---

## Arc of the session

Started from a brainstorm for an ICML idea → proposed **SLR** (spectral latent replay) → it and
two siblings were **falsified** at Gate 0, which *proved a design law* (whole-document
consistency) → the law prescribed **PLaR** (zero-private-storage proxy replay, bounded success)
and **CoLaR** (per-doc SVD compression, the win) → then a deep dive on the two "87.3" rows in the
results ledger. Net: one real new method (CoLaR), a proven mechanism (the consistency law), a
clean falsification chain, and a corrected understanding of the LexSlot row.

---

## 1. The falsification chain — buffer-free FEATURE replay fails for doc-IE

All k=8, 5ep, seed 42. Every one collapses to the mid-task (SROIE ~3) — the LexMem-v5 signature.

| method                                   | what it replays                           | AA   | row3                       | mem     |
| ---------------------------------------- | ----------------------------------------- | ---- | -------------------------- | ------- |
| `spectral_memory` (SLR)                | synthetic, rank-16 subspace               | 41.9 | [29.3,**3.2**, 93.1] | 0.15 MB |
| `aglr_replay` (AGLR-CL port)           | synthetic, full-`d` per-class Gaussians | 39.4 | [22.4,**3.0**, 92.9] | 0.39 MB |
| `coreset_memory` (50 carriers/16-cent) | **real** k-means centroids          | 39.7 | [22.8,**3.1**, 93.2] | 4.05 MB |
| `coreset_memory` (4 carriers)†        | real centroids                            | 36.7 | [14.5, 2.9, 92.7]          | 0.48 MB |

† artifact dir was deleted mid-session; value from run log / STATE.

**Four controls rule out every alternative explanation:**

- NOT synthesis quality — real centroids fail like synthetic Gaussians.
- NOT rank — full-`d` fails like rank-16. (And features aren't low-rank: pooled layer-8 variance
  r16=43%, r256=92%. The "forgetting is low-rank" theorem is about the OUTPUT space, not the
  head-INPUT feature space — SLR conflated them.)
- NOT carrier diversity — 50 carriers (8× mem) fail like 4.
- **The necessary ingredient is whole-document (feature, position, label) co-occurrence.** The
  single-variable isolation:

| config (matched count = 4)                        | AA   | SROIE final |
| ------------------------------------------------- | ---- | ----------- |
| 4**whole real docs** (`latent_replay d4`) | 63.8 | 37.1        |
| 4 carriers + real centroid features               | 36.7 | 2.9         |

**+27 AA from consistency alone.** Same count, same frozen boundary, same real features — only the
binding differs. Written up as Finding 3b in `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`.
Committed as evidence: `7c1f68e`.

---

## 2. PLaR — proxy latent replay (zero private storage)

Bank pseudo-labeled PUBLIC docs (WildReceipt) as whole-doc frozen-trunk latents. Preserves
consistency by construction; **stores zero customer bytes**. Bounded success, not parity.

| run                           | budget | AA   | row3                        | notes                                   |
| ----------------------------- | ------ | ---- | --------------------------- | --------------------------------------- |
| `d5` hard                   | 5ep    | 45.8 | [40.2, 4.1, 93.1]           | beats every falsified marginal          |
| `d50` hard                  | 5ep    | 57.2 | [**73.3**, 5.1, 93.3] | free public count grounds FUNSD         |
| `d50` soft (dark-knowledge) | 5ep    | 58.9 | [**76.1**, 7.0, 93.5] | best; FUNSD > matched private d4 (60.8) |
| `d50` soft                  | grid   | 50.3 | [51.8, 5.5, 93.5]           | convergence HURTS proxy replay          |
| `k4 d50` soft               | 5ep    | 57.3 | [68.7,**5.9**, 97.4]  | k4 interface doesn't fix coverage       |

**Verdict:** FUNSD grounds *above* matched private replay with zero private storage, but **SROIE
never grounds (~5) at any count/format/depth**. Diagnosis = **feature-region coverage**:
WildReceipt covers FUNSD's latent region (under the FUNSD-tuned frozen trunk) but not SROIE's.
d0-at-parity is out of reach with this pool. Committed: `1eb941e`, `23d77de`.

---

## 3. CoLaR — compressed latent replay (THE WIN)

Per-DOCUMENT rank-`r` SVD of banked latents. Whole-doc binding preserved (never pools across
docs); per-doc matrices ARE low-rank even though the pooled space isn't (r64=87.7%, r128=95.2%
variance). All k=4, d50, grid budget.

| run                                     | AA             | BWT             | row3               | mem               | vs raw d50                |
| --------------------------------------- | -------------- | --------------- | ------------------ | ----------------- | ------------------------- |
| `latent_replay k4 d50` (raw baseline) | 87.3           | −2.2           | [88.8, 76.7, 96.6] | ~163 MB           | —                        |
| **`colar r128`**                | **87.6** | **−1.7** | [89.2, 76.3, 97.2] | **60.4 MB** | **lossless, 2.7×** |
| `colar r64`                           | 80.6           | −12.3          | [82.3, 62.4, 97.1] | 32.0 MB           | −6.7 AA, 5×             |

**Verdict: d50 accuracy at 2.7× less memory, losslessly** (r128 matches raw on every retention
cell, better BWT). r64 is a lossy dial to 5×. Compression loss concentrates on the mid-task
(SROIE 76.7→62.4 at r64) — same locus as every other degradation. Untested cheap multiplier: int8
on the factors (~2× more). Committed: `06ae53e` (+ crash fix `0acd58a`).

---

## 4. The "reduce d50 → d5/d0" study (user directive)

Can the 87.3 latent-replay bank (50 docs, ~163 MB) shrink to d5 (~16 MB) or d0?

| route                              | experiment                         | result                 | verdict                     |
| ---------------------------------- | ---------------------------------- | ---------------------- | --------------------------- |
| Cut docs by**selection**     | `latent_replay_kc` (k-center d5) | 77.6 vs random 76.0    | ✗ curation ≈ count        |
| Cut docs to**zero** (public) | PLaR @ k4 d50                      | 57.3, pool-bounded     | ✗ d0-at-parity unreachable |
| Cut**bytes** (per-doc SVD)   | CoLaR r128                         | **87.6 @ 60 MB** | ✓**lossless 2.7×**  |

**Answer:** you cannot cut the documents (selection ≈ random; d0-public caps ~57), but you can
cut the **bytes** — losslessly to 2.7×, lossy dial to 5×. "Compress whole docs, don't curate or
substitute them." E1 also showed selection only *redistributes* retention (SROIE +6.9, FUNSD −1.4),
confirming the d5→d50 gap is coverage MASS, not curation.

---

## 5. Deep dive on the two "87.3" rows in the ledger

### 5a. Latent replay k4/d50 (87.3) — REAL, and improved

Genuine near-joint buffer-free retention: FUNSD ends at **88.8 > at-learning 86.3** (positive
backward transfer — the head re-carves the joint optimum). CoLaR-r128 reproduces it losslessly at
2.7× less memory (§3). Dose-response (k4): d1=61, d5=78, d50=87 (saturating). Depth: k2≈k4≫k8
(generic frozen interface + more plastic layers).

### 5b. LexSlot `_off` (87.3±1.1) — REAL, but buffer-BASED (not buffer-free)

**Correction made this session.** The row is **LexSlot slots + full DocCL machinery**
(`use_replay=True, buffer=200`, KD, Fisher) — at the run date the class was `LexSlot(DocCL)`. The
three seeds on disk:

| seed | AA   | row3               | use_replay        |
| ---- | ---- | ------------------ | ----------------- |
| 42   | 88.4 | [87.2, 80.7, 97.3] | True (buffer 200) |
| 7    | 87.2 | [85.3, 81.7, 94.8] | True              |
| 123  | 86.3 | [85.6, 76.7, 96.8] | True              |

Mean = **87.3 ± 1.1**, exactly reproducing `thesis/generated/table_main.tex`.

**Process error + fix (documented for honesty):** I first "verified" this by running
`method=lexslot` on the *current* code — but the class was refactored to `LexSlot(NaiveFineTune)`
(standalone, no buffer) after the 2026-07-02 RCA. That standalone run gave **42.4** and I wrongly
called the row an artifact; worse, it **overwrote** the seed-42 hybrid result. The user caught it.
Restored seed-42 from archive; the wrong run is parked at
`results/dil_lexslot_seed42_off_STANDALONE_wrongconfig` (42.4). The 3-seed evidence is intact.

**What IS wrong:** chapter-7 prose calls LexSlot *"replay-level retention buffer-free / without
storing any past data"* — factually incorrect, the row uses a 200-exemplar buffer. **The number
(87.3) stands; the "buffer-free" framing must become "small-buffer (200 exemplars)."** Separately,
standalone LexSlot ≈ naive (42.2, the RCA finding) — that's a different config, unchanged.
Still-open (low priority): re-run the DocCL-hybrid on post-gate-fix code to confirm ~87 holds
(expected yes — the buffer carries retention). Corrections committed: `3289616`, `68dd431`.

---

## Code shipped this session (all in `doccl/methods/`, tested + lint-clean, pushed)

| file                       | what                                                                       | status             |
| -------------------------- | -------------------------------------------------------------------------- | ------------------ |
| `spectral_memory.py`     | SLR (rank-`r` spectral latent replay)                                    | falsified evidence |
| `aglr_replay.py`         | AGLR-CL port (per-class Gaussians)                                         | falsified evidence |
| `coreset_memory.py`      | real k-means centroid replay                                               | falsified evidence |
| `proxy_latent_replay.py` | PLaR (public proxy + soft/task-mask knobs)                                 | bounded success    |
| `colar.py`               | CoLaR (per-doc SVD compression)                                            | **the win**  |
| `latent_replay.py`       | +`doc_selection: kcenter` knob, `memory_bytes()`                       | E1                 |
| `scripts/train.py`       | registry +`_STD_FORWARD` + run-name + `replay_memory_bytes` in metrics | wiring             |

~40 unit tests added across the new methods; all green. New wiki idea pages:
`CL4IE/wiki/ideas/2026-07-10-{spectral-latent-replay-icml,consistency-law-replay}.md`,
`CL4IE/wiki/analyses/2026-07-11-two-87point3-rows-deep-dive.md`.

---

## Open items / next runs (Vast.ai — local box is ~90 min/grid-run)

- **CoLaR:** seeds 7/123 + cil_cord (single-seed/scenario so far); int8-on-factors probe (~30 MB @ r128).
- **PLaR:** coverage-targeted proxy retrieval (the untested lever for SROIE); mixed pool (XFUND+WildReceipt).
- **LexSlot:** re-run DocCL-hybrid on post-gate-fix code (confirm ~87); fix ch7 "buffer-free" prose.
- **Data hygiene:** regenerate the deleted 4-carrier coreset artifact; the stale 1-epoch
  `dil_latent_replay_seed42` (64.9) needs a clean grid re-run for the paper table.
- `bd` still schema-fork-blocked — tracking in STATE/ROADMAP.

---

## 6. Venue & scope decision (2026-07-13)

After the results above, a game-theoretic acceptance analysis (real 2025 venue data) settled how to
submit. Modeled each paper as 3 reviewers in a Bayesian game; acceptance requires clearing the
borderline zone, which empirically needs **no strongly-negative reviewer** (negative sentiment is the
robust killer) + ≥1 champion.

**P(accept) estimates** (ordering robust; point %s ±5–8pp):

| Path | ICML/CVPR | AAAI | Binding risk |
|---|---|---|---|
| A. Method paper (CoLaR as headline) | **8–12%** | 15–20% | incremental (known latent-replay + known compression) + *matches-not-beats* ER/DER++ → triggers the method-reviewer's strong-negative |
| B. Diagnostic, as currently scoped | 12–18% | 20–25% | correctness/generality — 1 seed / 1 dataset / 1 backbone |
| **B+. Diagnostic, fully scoped** | **22–30%** | **30–38%** | just needs multi-seed × multi-scenario × multi-backbone RE-RUNS |

**Decision: diagnostic-led, fully scoped (B+); CoLaR = constructive control, PLaR = bounded-negative.**
B+ strictly dominates A and B, and the gap *widens* at more prestigious venues — the method paper's
weaknesses are exactly what method-reviewers punish hardest, while the diagnostic's only real risk
(generality) is closable by re-running existing code, not inventing a new idea. This also matches the
framing already locked in `CLAUDE.md`.

**The +12pp B→B+ requirement (compute, not research):** every headline row at **3 seeds (42/7/123)
× {CIL-CORD, DIL, mixed} × {LayoutLMv3, LiLT, BROS, BERT}**, via `run_grid_multigpu.sh` on Vast.ai.

**Why these numbers (sources):**
- Base rates: ICML 2025 26.9% (3260/12107), CVPR 2025 22.1% (2878/13008), AAAI 2025 23.4%
  (3032/12957); AAAI 2026 tightened to 17.6%.
- Borderline band [4.5, 6) → **~22% acceptance** (713/2588, ICLR data); decided by review sentiment
  + rebuttal + AC discretion, not the numeric score.
- Structural tilt: ICLR review-text analysis shows reviewer attention skews to "performance /
  architecture / method" — a documented bias *toward* method papers.
- Countervailing rubric ("accept = correct AND (result OR idea)"): a strong analysis substitutes for
  a novel method **iff correctness is airtight** — which for a diagnostic paper means *generality*,
  hence B+.
- Bar calibration: **"Revisiting Neural Networks for Continual Learning: An Architectural
  Perspective" (ICML 2024)** — a diagnostic forgetting paper that landed, and was
  architecture-general/multi-dataset. Ours must match that generality; CL×document-understanding is
  otherwise an empty intersection at CVPR/ICML.
