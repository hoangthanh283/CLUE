# ICLR 2027 draft plan — DocCL (diagnostic + falsification paper)

Status 2026-09-18. Companion to `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` (claims) and
`STATE.md` (evidence). This file is the *writing* plan: story, abstract, outline, gates.

> **Deadline check first.** ICLR deadlines historically fall in late September / early
> October (ICLR 2026: abstract 19 Sep, paper 24 Sep 2025). Confirm the ICLR 2027 dates on
> the official CfP *before* committing effort to this cycle. If the window has closed, the
> same draft targets the next ICLR cycle or (per `FINDINGS §"Venue implication"`) CoLLAs /
> TMLR, where a mechanism-backed negative result is a natural fit.

---

## 0. Where the evidence stands (inputs to the draft)

| Asset | State | Action before numbers are frozen |
|---|---|---|
| Generality grid (6 methods × 4 backbones × 3 scenarios × 3 seeds) | **62/72 cells** filled; 10 cells (22 runs) missing: BROS DER++ ×3 scen, LiLT LWF (dil, mixed, cil_cord), LiLT cil_cord EWC, LiLT DER++ partials | 3 LiLT DER++ retries running now (`scripts/retry_timeouts.sh`). The other 21 runs are hard-OOM on 6 GB → rented GPU **or** disclose as missing in the table (grey cells). |
| Aggregates (`all_runs.csv`, `pivot_*.csv`, `table_*.tex`) | **Stale — dated 2025-08-29**, newest metrics 2026-09-18 | `uv run python scripts/analyze_results.py --source local` after the retries land. Nothing goes in the paper from the old tables. |
| Joint BWT/AF/FWT on LiLT/BROS/BERT | NaN in metrics.json | Report joint as AA-only (oracle), or recompute BWT from `matrix.npy` (`matrix[-1]` vs diagonal). |
| CA-CoLaR T5 (seeds 7/123) | Done: 79.04 / 71.18 / 75.29 → **75.17 ± 3.94**; control (fixed [d10]) only seed 42 = 71.93 | Either run control on seeds 7/123 (~1.5 h each, fits locally) or present CA-CoLaR as seed-42 frontier only, in appendix. |
| Latent-replay "87.3 @ k8/d5" | Does not reproduce: 66.47 ± 2.95. Real value is d=50 | Fix the label (d=50, cite `dil_colar_seed42_k4_d50_r128` 86.38 / 3-seed 86.75). |
| LexSlot 87.3 row | Buffer-based (200 exemplars), not buffer-free | Relabel everywhere; standalone LexSlot ≈ naive 42.2. |
| Underfit-floor confound (NO-GOs at 3-epoch / frozen backbone) | Open | Explicit paragraph in Limitations + one table showing the strongest falsified family at full budget. |
| FWT | Invalid for CIL (≈ −85 for everyone) | Drop FWT from the paper; AA/BWT/AF only. |

---

## 1. The story (one paragraph the whole paper serves)

Continual-learning remedies were built on unimodal classification and treat a network
as one homogeneous block. We open a multimodal document encoder and *look* at where
forgetting lives. It lives at the output: the shared classifier head and late layers, on
every backbone we try. Protecting that locus doesn't remove forgetting — it *moves* it
(the drift migrates into the layers left plastic, total loss conserved). That anatomy
makes a prediction: only something that re-grounds the head can recover old competence.
The prediction holds — replay reaches the joint bound, regularisation and distillation
stay at the floor — and it survives an honest attempt to break it: five buffer-free
families we built or ported, including the field's strongest exemplar-free tool, all fail
for the same identified reason. A constructive control then shows the buffer doesn't need
raw pixels — compressed whole documents are lossless at 2.7× — but it *does* need whole
documents: every marginal summary collapses. So the right question for document CL is
not "how to avoid the buffer" but "how small can it be, and what must it hold".

Genre: **detective story → falsification → constructive control**. Every method in the
repo is a *witness*, not a contribution. Never pitch a method.

---

## 2. Motivation → problem statement → gap (Intro skeleton)

**Motivation (¶1).** Deployed KIE systems must absorb new document types / schemas /
domains; retraining on all history is blocked by cost and PII/GDPR. Distill from
`thesis/chapters/chapter1.tex:6-30`.

**The blind spot (¶2).** EWC/LwF/prompt/LoRA families treat the network uniformly or cut
depth by heuristic; PEFT/prompt methods *presuppose* knowing where to intervene. A
tri-modal encoder is manifestly heterogeneous (text / visual / layout / fusion / head).
Distill `chapter1.tex:31-60`.

**Problem statement (¶3).** Under continual token-level KIE over FUNSD→CORD→SROIE
(and class-incremental CORD, and mixed): *Where* does forgetting occur in a multimodal
document encoder, is that location a property of the task or of the architecture, and
which remedies act on it?

**Questions the paper answers (¶4).**
- Q1 *Anatomy*: which parameter groups drift (Fisher displacement, CKA), and is it
  backbone-invariant?
- Q2 *Conservation*: does protecting the locus remove forgetting or relocate it?
- Q3 *Prediction*: does the anatomy predict the method ordering, and does that ordering
  hold across backbones and scenarios?
- Q4 *Falsification*: can any buffer-free mechanism substitute for replay's head gradient?
- Q5 *Buffer content*: what is the minimal thing a buffer must contain?

**Contributions (¶5) — three, plus an implication.**
1. A multi-backbone, multi-scenario, multi-seed *anatomy* of forgetting in document IE:
   output-localised, architecture-general, and migrating under constraint (Findings 1–2).
2. A prediction-tested benchmark: replay re-grounds the head and reaches the joint bound;
   buffer-free families do not, and remedy benefit is backbone-dependent (Finding 3).
3. A convergent negative result over five buffer-free families with a single mechanistic
   explanation, plus a constructive control (CoLaR) isolating whole-document consistency
   as the necessary buffer ingredient (Findings 3b/3c).
- Implication: reframes document-CL research from "avoid the buffer" to "minimal buffer".

---

## 3. Draft abstract (v0 — numbers marked ⟨v⟩ must be re-read from regenerated tables)

> Continual-learning methods are almost always developed on unimodal classification and
> treat the network as a homogeneous block. We ask what forgetting actually looks like
> inside a multimodal document encoder performing continual key-information extraction,
> and whether standard remedies act on it. Using per-group Fisher displacement and
> layer-wise CKA across four backbones (LayoutLMv3, LiLT, BROS, BERT), three scenarios
> (domain-, class-, and mixed-incremental) and three seeds, we find that forgetting is
> **output-localised** — it concentrates in the shared classifier head and late encoder
> layers while input encoders stay nearly stable — and **architecture-general**: the
> naive-to-joint gap agrees to within 2.5 points⟨v⟩ across all four backbones. Protecting
> the locus does not remove forgetting: freezing the head and late layers migrates the
> drift into early layers (CKA 1.00→0.19) with no change in accuracy, i.e. forgetting is
> **conserved** under architectural constraint. This anatomy predicts, and our benchmark
> confirms, that only mechanisms which re-ground the head recover old competence: replay
> (ER, DER++) reaches the joint bound on domain-incremental learning (≈88 vs 88.7 AA⟨v⟩),
> regularisation and distillation stay at the naive floor (41–44⟨v⟩), and a remedy's
> benefit is backbone-dependent (EWC ranges from −12 to +5 points⟨v⟩). We then attempt to
> break the prediction with five buffer-free consolidation families — weight merging,
> subspace transfer, parametric slot memories, input-anchored memory, and
> feature-Gaussian generative replay — and each fails for the same identified reason:
> nothing they store reproduces replay's gradient on the head. A constructive control
> isolates what a buffer must contain: per-document SVD replay (CoLaR) is lossless at
> 2.7× compression, whereas every marginal or per-class summary collapses, with +27 AA⟨v⟩
> attributable to whole-document consistency alone. For document understanding, the
> question is therefore not how to avoid the buffer but how small it can be and what it
> must hold.

~300 words → trim to ≤250 for submission (cut the family list to "five buffer-free
families" and one number per claim).

**Title candidates** (pick one; the lead must be the migration/anatomy finding):
- *Forgetting Has an Address: Anatomy, Conservation, and the Necessity of Replay in Continual Document Understanding*
- *Where Document Encoders Forget — and Why Only Replay Fixes It*
- *Head First: Output-Localised, Architecture-General, Conserved Forgetting in Multimodal Document IE*

---

## 4. Section outline with page budget (ICLR main text ~9–10 pp; verify current limit)

| § | Pages | Content | Source to distill |
|---|---|---|---|
| 1 Introduction | 1.25 | §2 above; Fig. 1 = one-panel "Fisher displacement by group × backbone" teaser | ch1 |
| 2 Related work | 0.75 | CL families (reg / distill / replay / prompt / PEFT / merge / exemplar-free feature replay); depth-localised forgetting in ViTs; doc-IE CL is thin — cite honestly | ch2 (711 lines → 0.75 p) |
| 3 Setup | 0.75 | Scenarios (DIL FUNSD→CORD→SROIE, CIL-CORD, mixed), 4 backbones, metrics (AA/BWT/AF; **no FWT**), early-stop protocol EPOCHS_CAP=100, diagnostics (Fisher displacement, CKA) | ch3/ch4 |
| 4 Anatomy of forgetting | 1.75 | Finding 1 (localised, general; permutation p=0.38/0.51 → shared *location*, not magnitude — state caveat); Finding 2 (migration, CKA 1.00→0.19, control identical); Finding 3c conservation (3-seed verdict `STATE.md:712`) | ch4, ch5 |
| 5 Anatomy predicts the remedy ordering | 1.5 | Grid table (Table 1, 62–72 cells, grey = missing); DIL: replay ≈ joint; mixed: replay closes most of gap; CIL-CORD: near-floor regime for *everyone* incl. replay — report, don't hide; EWC sign flip across backbones (Fig. 3); LiLT joint-inversion footnote | ch6 |
| 6 Falsification: five buffer-free families | 1.5 | One table: family → mechanism → best AA → why it fails. DocMERGE 21.6 ≈ 21.9; LCA 41.9; LexMem v3b 66.0 (positive control); v5 63.2 with mid-task 13.0; Ledger 21/23. Underfit-floor rebuttal paragraph | ch6, FINDINGS §3 |
| 7 What the buffer must contain | 1.0 | CoLaR r128 87.6 @ 60 MB vs raw 87.3 @ 163 MB (lossless 2.7×); r64 80.6; SLR 41.9 / AGLR 39.4 / coreset 36.7–39.7; 4 real docs 63.8 vs 4 carriers 36.7 → +27 from consistency. Label fix: d=50. PLaR ~59 (coverage-limited) one paragraph. CA-CoLaR → appendix frontier | STATE, EXPLORE §86 |
| 8 Limitations & implications | 0.5 | Seeds/scope per finding (Finding 2/3b are seed-42 DIL only — say so); missing grid cells; underfit floor; FWT dropped; conservation provisional beyond DIL | FINDINGS §"reviewer risks" |
| Appendix | — | Full per-backbone tables, all seeds; CA-CoLaR Pareto branch; PLaR sweep; LexSlot/LexMem configs; compute | — |

---

## 5. Figures & tables (minimum set)

1. **Fig 1** Fisher displacement per parameter group, 4 backbones (bar/heat) — the teaser.
2. **Fig 2** CKA-by-depth under naive vs freeze-head+late vs control — migration.
3. **Table 1** Generality grid: AA (BWT) mean ± std, methods × backbones × scenarios; grey missing cells.
4. **Fig 3** Remedy benefit vs naive per backbone (EWC/LwF/ER/DER++) — the sign flip.
5. **Table 2** Falsification ledger: family, mechanism, best AA, failure reason, evidence pointer.
6. **Fig 4** Buffer-content Pareto: AA vs MB for raw / CoLaR r128 / r64 / marginals / carriers.

Pipeline: `analyze_results.py` → `pivot_*.csv`/`table_*.tex`; `ingest_to_thesis.py`
already copies into `thesis/` — reuse the same outputs for the paper tree.

---

## 6. Reviewer attack surfaces → pre-emptive answers (write these *into* the draft)

| Attack | Answer in text |
|---|---|
| "Negative result on one setup" | Table 1 is 4 backbones × 3 scenarios × 3 seeds; bar = ICML'24 architectural-perspective paper. |
| "NO-GOs reached under-trained" | Limitations ¶ + one falsified family re-run at full budget (pick LexMem v3b/v5, already at full). |
| "It's a method paper in disguise" | Contributions list names no method; every method row is labelled *evidence*. |
| "87.3 buffer-free LexSlot" | Removed; labelled small-buffer (200 exemplars). |
| "Latent replay 87.3 at d=5" | Corrected to d=50; 3-seed 86.75 cited to on-disk run. |
| "CIL replay below naive" | Reported as near-floor regime; explained (label-space growth + tiny sessions); not claimed as replay success. |
| "Joint < ER on LiLT DIL" | Footnote, all 3 seeds. |
| "Head-localisation p-values are null results" | Framed as *shared location* consistent across backbones; magnitude not claimed equal. |

---

## 7. Work order (CPU work runs while the GPU retries finish)

1. **Now (CPU):** confirm ICLR 2027 dates; create `paper/` tree (ICLR style file — none in repo; fetch from the CfP), `paper/main.tex`, `paper/refs.bib` seeded from `thesis/refs.bib` (58 entries).
2. **Now (CPU):** write Intro §1 + Abstract v0 + Related-work skeleton from ch1/ch2. Numbers as ⟨v⟩ placeholders.
3. **Now (CPU):** fix prose corrections in one pass — LexSlot label, 87.3→d=50, drop FWT.
4. **When retries land (GPU→CPU):** `analyze_results.py`; regenerate Table 1/Fig 3; replace every ⟨v⟩.
5. **Decision:** rent GPU for the 21 OOM runs (BROS DER++, LiLT LWF/EWC) *or* ship with grey cells + explicit disclosure. Recommend: grey cells unless a rental slot opens ≥1 week before deadline.
6. **Optional, fits locally (~3 h):** CA-CoLaR control seeds 7/123 → turns appendix frontier into a 3-seed claim.
7. Sections 4–7 from ch4–ch6; Limitations; appendix tables.
8. Full read as the [EXAMINER] from `docs/THESIS_METHOD_CHAPTER_REFRAME_2026-07.md`; then `latexmk`, page-limit check, anonymisation (strip GitHub/W&B URLs from footnote).

---

## 8. Verification before submission

- Every number in the PDF traces to a `results/*/metrics.json` that exists on disk (script: grep numbers in `.tex` against `all_runs.csv`).
- `grep -n "buffer-free" paper/*.tex` — none refers to LexSlot 87.3.
- `grep -n "d=5\b\|d5" paper/*.tex` — none attached to 87.x.
- Table 1 cell count matches `ls results/*_seed*/metrics.json` count for the six methods.
- No FWT column anywhere.
- PDF builds with the ICLR style, ≤ page limit, anonymised.
