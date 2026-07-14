# KT — CoLaR + LexSlot integration (handoff 2026-07-14)

**For:** the next session resuming the "combine CoLaR r128 + LexSlot for a big improvement" idea.
**Read first:** this file, then `STATE.md` (top blocks), `EXPLORE.md`, `docs/RESULTS_LEDGER_DIL.md`.
**Repo:** `CLUE/`, branch `doccl`. All code below is committed & pushed (commits cited inline).

## 1. The two ingredients (what they are, mechanically)

**CoLaR** (`doccl/methods/colar.py`, `CoLaR(LatentReplay)`, commit 06ae53e) — compressed latent
replay. Freezes encoder layers `<k` after task 0; banks each past doc's layer-`k` activation as a
per-DOCUMENT rank-`r` SVD (`us`,`v` fp16 factors); at each train step reconstructs `us@v` and
replays it (real past-task gradient) into the plastic head+late layers via a forward-pre-hook on
`encoder.layer[k]`. **r128 = lossless (95% per-doc variance).** Overrides only `_capture_task`
(bank+SVD), `_sample_replay` (reconstruct), `memory_bytes`. Inherits `train_task`
(`loss = ce + replay_ce; backward()`), `_pre_hook`, `_apply_freeze_map` from `LatentReplay`.

**LexSlot** (`doccl/methods/lexslot.py`, `LexSlot(NaiveFineTune)`, + `lexslot_memory.py`,
`lexslot_mask.py`) — task-private additive slot memories on head + late layers, routed at inference
by OCR lexical similarity. Two mechanisms: (1) gradient-isolation mask at train (a slot owned by a
prior task trainable ∝ inter-task OCR-cosine S); (2) inference gate `cos(sig(doc), sig_task) ∈ [0,1]`
— a RAW cosine, NOT softmax; a non-matching doc gets gate≈0 so the slot doesn't fire. `LogitSlots`
(head, zero-init proj) + `ReprSlots` (late layers, LoRA-style zero-init up). **Standalone LexSlot ≈
naive (42.2)** — it needs a buffer to work.

## 2. THE HEADLINE RESULTS (disk-verified, dil FUNSD→SROIE→CORD, LayoutLMv3, seed42, grid budget)

| method | AA | BWT | final-row [FUNSD,SROIE,CORD] | note |
|---|---|---|---|---|
| joint (oracle) | 89.7 | 0.0 | [87.9,83.4,97.7] | upper bound |
| DER++ | 88.2 | −2.9 | [85.6,81.5,97.6] | strong replay baseline |
| **LexSlot+DocCL** | **88.4** | −1.1 | [87.2,80.7,97.3] | **REAL but BUFFER-BASED (200 exemplars + KD + Fisher)** |
| **CoLaR r128** | **87.6** | −1.7 | [89.2,76.3,97.2] | **60 MB, lossless 2.7× — THE CONSTRUCTIVE WIN** |
| latent_replay k4/d50 | 87.3 | −2.2 | [88.8,76.7,96.6] | CoLaR's uncompressed base |
| ER | 86.2 | −5.1 | [84.4,76.6,97.7] | classic replay |
| CoLaR r64 | 80.6 | −12.3 | [82.3,62.4,97.1] | 32 MB lossy dial |
| latent_replay k4/**d5** | 76.0 | −19.3 | [72.6,57.6,97.9] | the LARM baseline (any fusion must beat THIS) |
| **LARM (all variants)** | 32.6–66.3 | — | see §4 | **FAILED — do not repeat as-is** |
| naive | 40.1 | −75.0 | [18.3,4.3,97.6] | floor |

**Two crucial caveats a future session MUST know:**
- The LexSlot 88.4 row is **LexSlot slots layered on the FULL DocCL machinery** (`use_replay=True,
  buffer_size=200`, KD teacher, Fisher) — at the run's date the class was `LexSlot(DocCL)`. It is
  NOT buffer-free. The current on-disk `LexSlot(NaiveFineTune)` is the post-RCA STANDALONE stripped
  version (=42.2). The thesis ch7 "buffer-free" prose is WRONG (flagged in ROADMAP).
- CoLaR/LARM headline runs use **k=4, d=50** (via CLI override); the configs default to k=8/d=5.

## 3. Why the user's intuition (CoLaR + LexSlot = big win) is REASONABLE

- CoLaR gives strong RETENTION (BWT −1.7) but NO forward transfer (FWT). LexSlot's lexical gate is
  the only FWT lever in the codebase. On paper they're complementary: replay holds old tasks,
  lexical slots transfer to new ones.
- The 88.4 LexSlot+DocCL row already PROVES slots+replay coexist and hit near-joint — so a
  slots+CoLaR combo is not obviously doomed. CoLaR is a cleaner/cheaper replay than DocCL's raw
  buffer, so slots+CoLaR could match 88 at far less memory.

## 4. WHY LARM FAILED (the trap to avoid — this is the most important section)

LARM (`doccl/methods/larm.py`, commits e3d7efa/f21456e/ca4e19e) was a *fused* CoLaR+LexSlot: one
lexically-keyed low-rank memory M, READ additively at layer-`k` (routed by OCR sig), WRITTEN by
replay CE. It FAILED, three variants (dil k4/d5 grid):
- add-mode (h+Δ): AA 66.3 [60.5,40.7,97.7] — WORSE than its CoLaR baseline (76.0)
- replace-mode (Δ only): AA 32.6 — collapsed below naive (confirms consistency law: replacing
  whole-doc features is catastrophic; adding is merely harmful)
- **fixed gate (softmax→clamped-cosine, commit ca4e19e): AA 62.7 [63.5,27.1,97.6] — STILL failed.**

**Root causes (some proven, one still open):**
1. **Softmax routing had no null-option** (fixed): a doc matching no cell still got mass=1 averaged
   foreign correction. Fixed with clamped-cosine + mass-cap. This helped cross-task leak (→0.03) but
   did NOT recover AA.
2. **The deeper, still-unproven cause:** even correctly routed to a doc's OWN task cells (gate ~0.96),
   the additive correction fires at ~full strength on every eval doc — and diag is FINE [87.3,82.0,
   97.6] (learns each task) while row3 is DESTROYED [63.5,27.1,97.6] (retention only). So the
   memory poisons OLD-TASK EVAL. Leading hypothesis: the correction's up-factors were trained
   against the plastic-layer state AT that task's time, but applied at final-eval through
   plastic layers that DRIFTED since — the correction desyncs from the moved head. **NOT YET
   PROVEN** (I wrongly asserted mechanisms twice; the discriminator below was never run).
3. **The discriminator that would settle it (NEVER RUN — run it if reviving LARM):** LARM with
   memory ON at train, OFF at eval (`mem_eval_off` flag, ~5 lines). eval-off→~76 ⇒ stale-correction
   (drift, fixable); eval-off still ~63 ⇒ training-time poisoning (deeper).

**The lesson for the NEXT attempt:** LARM tried to make ONE object do both jobs (replay + rewrite)
at the layer-`k` boundary, BELOW the plastic drift. That coupling is what broke. **LexSlot's slots
work at head+late (ABOVE/AT the plastic layers) and are ADDITIVE-ISOLATED per task, with a raw-cosine
gate that stays silent for non-matching docs.** The user's "integrate CoLaR + LexSlot" is
NOT "rebuild LARM" — it's "run LexSlot's slot machinery ON TOP of CoLaR's replay, each doing its own
job."

## 5. THE NEXT DESIGN (user-chosen — build design A first)

**USER-CHOSEN DIRECTION (2026-07-14): build design A first.** LARM failed; the user's "CoLaR +
LexSlot" means **run LexSlot's OWN slot machinery on top of CoLaR replay** — NOT rebuild LARM.

- **A (PRIMARY — build this first). Slots-on-CoLaR.** New method `CoLaSlot(CoLaR)` (or similar):
  layer LexSlot's `LogitSlots` (head) + `ReprSlots` (late layers) + the gradient-isolation mask +
  the RAW-COSINE inference gate on top of CoLaR's compressed replay. Replay trains the shared
  plastic weights (retention); the task-private, gradient-isolated slots add capacity at HEAD+LATE
  (retention + the FWT lever). **This is chosen precisely because it fixes BOTH LARM failure modes:**
  (1) slots sit at head+late = ABOVE the plastic drift (LARM's layer-k rewrite sat below it and
  desynced); (2) the gate is LexSlot's RAW COSINE ∈[0,1] = null-preserving (LARM's softmax always
  fired). Reuse `lexslot_memory.py`/`lexslot_mask.py`/`LexSlot._install_infer_gate` VERBATIM — do NOT
  reinvent them. The earlier `CoLaSlot(CoLaR)` "stack" objection is MOOT: LARM proved the "fuse into
  one object" alternative is worse, and the 88.4 LexSlot+DocCL row proves slots+replay coexist near
  joint. Gate 0 (dil k4/d50/r128): match/beat CoLaR r128 (87.6) with BWT ≥ −2. Then xlingual for FWT.
- **B (only if A works, higher novelty).** Use CoLaR replay to RE-FIT LexSlot's slot values each task
  so they don't go stale — LARM's intent at the RIGHT layer (head+late).
- **C (do BEFORE trusting the 88.4 target).** Re-run `LexSlot(DocCL)` on post-gate-fix code (the 88.4
  predates fix 90f5a15) to confirm ~88 is the real bar to match-cheaper.

**Order for next session: C (confirm target) → A (build, the user's pick) → B (if A works).** Frame
as the constructive result INSIDE the diagnostic paper (EXPLORE.md §6), not a standalone method paper.

## 6. Code seams & how to run (for the next session)

- Add a method: subclass in `doccl/methods/`, export nowhere (train.py imports directly), add to
  `METHOD_REGISTRY` + `_STD_FORWARD` + a run-name branch in `scripts/train.py`, add
  `configs/method/<name>.yaml`, add `tests/methods/test_<name>.py`.
- Reusable pieces: `LogitSlots`/`ReprSlots` (`lexslot_memory.py`); `slot_trainable_mask`,
  `task_similarity_matrix`, `signature_cosine` (`lexslot_mask.py`); `sparse_doc_vectors`
  (`hybrid_routed_prompt.py`); CoLaR's `_capture_task`/`_sample_replay`/`_pre_hook` (`colar.py`,
  `latent_replay.py`). The lexical inference gate to COPY (not softmax): `LexSlot._install_infer_gate`
  (`lexslot.py:229`).
- Run (headline recipe): `uv run python scripts/train.py method=<name> scenario=dil
  model=layoutlmv3_base method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128
  training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0
  wandb.mode=offline seed=42`. Grid budget ≈ 1.5h on the RTX 2060 (VRAM-safe). Read the matrix:
  `results/dil_<name>_seed42_k4_d50/metrics.json` — AA/BWT + final-row + diag.
- **FWT only moves on xlingual** (dil FWT is structurally ~−85, impossible; xlingual ~−22, movable).
  Positive FWT is NOT achievable in dil — target BWT×FWT Pareto-dominance, not literal positive FWT.

## 7. Operational gotchas (bit me this session)

- The RTX 2060 is slow (~1.5h/dil grid-run) and OOMs above ~5GB VRAM. Use k4/d5 for cheap gates,
  k4/d50 for headline. Rent Vast.ai for the full B+ grid (seeds×scenarios×backbones).
- Background runs die on session restart unless harness-tracked (`run_in_background: true`), and
  NEVER pipe a chained run's stderr through `tail` (it swallows the crash — cost me a misdiagnosis).
- `bd` (beads) is schema-fork write-blocked — track work in STATE.md/ROADMAP.md, not beads.
- `configs/` are immutable — add a new option file, never edit an existing one.
- CoLaR banking log assumed raw `hidden` in every stored doc — subclasses that re-format the store
  (LARM/CoLaR) must handle mixed store; regression already fixed (0acd58a) but watch for it.

## 8. Provenance (commits, all pushed to origin/doccl)

06ae53e CoLaR + kcenter · 0acd58a CoLaR crash fix · 2fea62a CoLaR r128 lossless · e3d7efa LARM ·
f21456e LARM vectorize · 41f69ff LARM Gate-0 result · ca4e19e LARM gate fix · 9338549 venue/scope
decision · b9c6285 EXPLORE.md · 3289616/68dd431 lexslot 87.3 artifact→retracted(real,buffer-based).
STATE.md has the live findings (LARM fixed=62.7 result + discriminator plan).
