# Brainstorm round 1 — lens: READOUT & HEAD (2026-10-14)

Rubric: Novelty 0-3 · Significance 0-3 (DIL-only caps at 1) · Feasibility 0-3 (≤5 d) · Vision 0-2 ·
Evidence-consistency 0-2 · Informative-if-null 0-1 · total /14.
Constraints honoured: 5 head-side kill-tests ≤ −0.1 AA; Gaussian head alignment null (B1/B2);
protection relocates drift; no frozen trunk; readout-coverage gap (images); CIL-CORD replay < naive.

| # | Idea | Mechanism (one sentence) | Falsifier (control / kill bar) | Expected | N | S | F | V | C | I | Σ |
|---|------|--------------------------|--------------------------------|----------|---|---|---|---|---|---|---|
| A | **Snap index** (label-free forgetting monitor) | Cosine between the head's output marginal on *unlabelled* old-task inputs and the current task's gold marginal, per boundary; report as a forgetting predictor on 4 backbones × 3 scenarios | Control: correlate with measured BWT per boundary; kill if Spearman ρ < 0.6 across the 68-cell grid | Strong (RCA already shows 0.997 vs 0.60); turns a diagnostic into a deployable, label-free drift alarm | 2 | 1 | 3 | 0 | 2 | 1 | **9** |
| B | **Keep-correctable** (saturation cap) | Cap old-class logit margin / add entropy floor on O tokens during training so old-class mass never falls below the branch threshold (~1e-6), then re-run the five already-implemented corrections | Control: same corrections on uncapped model (known ≤ −0.1); kill if capped+corrected ≤ +0.5 AA on CIL-CORD | Uncertain: if information dies in *features*, cap is inert → that null sharpens the conservation story | 2 | 2 | 3 | 0 | 1 | 1 | **9** |
| D | **Per-session O rows** (O_s, logsumexp at inference) | Each session adds its own O_s row; training in session s targets O_s; P(O)=Σ_s P(O_s); old O_r semantics never overwritten | Control: single-O head (current); kill if ΔAA on CIL-CORD < +1 over 3 seeds | Partial: softmax competition still suppresses old rows on O_s tokens; cheap to test | 2 | 1 | 3 | 0 | 1 | 1 | **8** |
| F | **Closed-form head refit** on exact features (current full data + memory through current trunk) | Ridge/logistic refit of the plain linear head at each session end from full-coverage features — targets the readout-coverage gap | Control: continued-SGD head; kill if < +0.5 AA | ~0 on DIL (already at oracle); on CIL-CORD bounded by stale labels (not this lens) | 1 | 1 | 3 | 0 | 2 | 1 | 8 |
| K | **Frozen-boundary analytic probe** (ridge head on layer-k features) | Drift-immune linear readout over the frozen layer-4 features CoLaR already banks | Control: current late head; report linear decodability of BIO at k | Low accuracy expected (layer 4 too shallow); informative bound | 1 | 1 | 3 | 0 | 2 | 1 | 8 |
| M | **Head reset-and-refit control** | Re-init head at each boundary, refit on current data + memory; tests whether head *state* carries anything irreplaceable | Kill if reset ≈ continued (then drift lives in features) | Equal → strengthens Finding 2 | 1 | 1 | 3 | 0 | 2 | 1 | 8 |
| E | One-vs-rest sigmoid heads, session-masked negatives | Per-tag binary heads; O-tokens give no gradient to old heads | Control: softmax head; kill if < +1 AA CIL-CORD | Possible gain; close to OvR-CL and multi-label CIL work | 1 | 2 | 3 | 0 | 1 | 1 | 8 |
| C | Factorised type × boundary head | Type head trained only on entity tokens (O tokens never push type logits); shared B/I/O head | Control: flat head; kill if < +1 AA | Partial fix of E2O; span-based CNER already decouples | 1 | 2 | 2 | 0 | 1 | 1 | 7 |
| G | Born-from-O init | Init new class rows as O row + ε so new classes carve out of O mass gradually | Control: N(0,0.02) init; kill if < +0.5 | Small; one-line change | 1 | 1 | 3 | 0 | 1 | 1 | 7 |
| J | Freeze O row after session 0 | Fixed O reference for all later geometry | Control: plastic O row; kill if < +0.5 | Tiny constraint on a linear head | 1 | 1 | 3 | 0 | 1 | 1 | 7 |
| H | Cosine / weight-aligned head | Normalised classifier, new-class norm scaled to old (WA/LUCIR) | Control: plain linear | Baseline, not idea | 0 | 1 | 3 | 0 | 1 | 0 | 5 |
| L | O-token gradient masking on old rows | Zero old-row gradient from O-labelled tokens (IS3/CFNER adjacent) | Control: plain CE | Baseline-ish | 0 | 1 | 3 | 0 | 1 | 0 | 5 |
| I | ICE-style per-session frozen heads | Independent heads frozen per session, max over heads | — | Falsified in spirit by LexSlot (42.2): isolated heads cannot re-ground a drifting trunk | 0 | 1 | 2 | 0 | 0 | 1 | 4 |

Excluded as already-run nulls: post-hoc marginal matching, prior-ratio, per-doc EM, KL anchor,
balanced-softmax logit adjustment; drift-compensated Gaussian head alignment.

## Novelty lookups (WebSearch extended, 2026-10-14)
- A snap index: no hit for a label-free output-marginal forgetting monitor (queries: "label-free
  forgetting detection output marginal distribution shift predicts backward transfer"); nearest:
  surveys shifting evaluation from outputs to representations (Awesome-Forgetting TPAMI 2024).
- B keep-correctable: no hit (query: "logit saturation old class probability collapse confidence
  cap keep classifier correctable"); nearest: implicit-adversariality of forgetting
  (arXiv 2510.09181) — gradient-alignment mechanism, not logit saturation.
- D per-session O rows: no direct hit; CFNER "Distilling causal effect from miscellaneous
  Other-class" (arXiv 2210.03980) treats O as one miscellaneous class; concept-driven KD + pseudo
  labels (2025) targets background shift by distillation, not head structure.
- C factorised head: span-based CNER (arXiv 2302.12200), BGNER boundary guidance (2025),
  Type-Balanced Contextual Learning (arXiv 2608.31038) already decouple span/type → novelty 1.
- E OvR heads: continual OvR class-skill architectures (kobros incremental_feature_cl) and
  Class-Independent Increment for multi-label CIL (arXiv 2503.00515) → novelty 1.

Sources: https://aclanthology.org/2023.emnlp-main.509.pdf · https://arxiv.org/pdf/2210.03980 ·
https://arxiv.org/pdf/2302.12200 · https://arxiv.org/pdf/2608.31038 · https://arxiv.org/html/2503.00515v2 ·
https://arxiv.org/pdf/2510.09181 · https://github.com/EnnengYang/Awesome-Forgetting-in-Deep-Learning ·
https://arxiv.org/pdf/2607.22075 (FSE fast-slow experts CNER)

Note for parent: this lens cannot produce the ≥1-AA CIL-CORD gain on its own — every head-only
lever either is a registered null or is bounded by the stale-label problem owned by the replay lens.
A and B are the lens's genuine contributions (a diagnostic instrument and a conservation test).
