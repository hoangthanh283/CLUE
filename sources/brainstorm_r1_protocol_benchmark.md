# Brainstorm round 1 — lens: protocol / benchmark / diagnostic contributions

Date 2026-10-14. Scope: contributions whose value is the instrument, the benchmark, or the
certified finding. Excluded (other forks): replay/label mechanisms, vision/modality mechanisms,
head geometry, optimizers. Rubric (integers): Novelty 0-3 · Significance 0-3 · Feasibility 0-3
(≤5 days, 6 GB GPU) · Vision 0-2 · Evidence-consistency 0-2 · Informative-if-null 0-1 · total /14.

| # | Candidate | N | S | F | V | E | I | Σ | Run / write | Falsifier | Preempts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Method-rank inversion across domains** — SLCA/RanPAC-style head statistics beat latent replay on image CIL (SLCA 87.5 vs CoLaR-bank 80.0, CIFAR) but are null on documents (B2: 86.2 vs 86.4); binding is the discriminating variable | 3 | 3 | 2 | 2 | 2 | 1 | **13** | port SLCA head-alignment to token level (~1 d) + 3 DIL runs; or cite B2 as the document cell and write a 2×2 table + paragraph | if SLCA-token ≥ CoLaR−1 on DIL, the inversion is false and binding is not the discriminator | "compare to PTM-based CL (SLCA/RanPAC/EASE)" |
| 2 | **Headroom certification step** — report Joint−Independent before ranking CL methods; DIL headroom ≤0.5 (negative on LiLT/BROS) ⇒ the benchmark cannot rank; propose headroom-normalised recovery (AA−Indep)/(Joint−Indep) and cost-normalised (bytes incl. 3× weights) | 2 | 3 | 3 | 1 | 2 | 1 | **12** | write only (numbers exist); one supp table of recovery ratios for all DIL cells | if Indep ≪ Joint on Mixed-like streams, the step is scenario-specific (fine, still a step) | "why is every replay method tied?" |
| 3 | **Released localisation protocol (doccl.diag) as a tool** — three-step Localise/Test/Perturb with dominance + shared-location tests; demonstrated on 4 doc backbones + ViT | 2 | 2 | 3 | 1 | 2 | 0 | 10 | supp: API + pipeline figure; apply `shared_location_test` ViT-vs-LayoutLMv3 (data exists) | ViT vs LayoutLMv3 profile rejected ⇒ "architecture-general" needs qualifying | "is this just Ramasesh et al. redone?" |
| 4 | **Conservation as a metric** — total old-task-Fisher displacement is conserved under constraint; report "displacement budget" per boundary; already on 3 orders + long CIL (fig:conservation) | 2 | 2 | 3 | 1 | 2 | 1 | 11 | write + one analysis script over existing displacement records | budget not conserved under some protection ⇒ a genuine buffer-free lever exists (would be big) | "protection is just under-tuned" |
| 5 | **DocCL benchmark release** — DIL / CIL-CORD / Mixed / DIL-XLing / CIL-WildReceipt / long streams × 4 backbones + protocol + run names; first document-image CL suite | 3 | 2 | 3 | 2 | 2 | 0 | **12** | supp + README + leaderboard table; nothing to run | — | "no standard benchmark for doc CL" |
| 6 | **Stale-label audit** — fraction of buffered tokens whose label is stale per session, for every replay method on CIL/Mixed (measurement only; fix is another fork's) | 2 | 2 | 3 | 0 | 2 | 1 | 10 | offline script over scenarios (no GPU) | audit ≈0 but replay still < naive ⇒ stale labels are not the cause | "why is ER below naive?" |
| 7 | **Order-invariance test** — location invariant to task order, magnitude not; formalise via shared_location_test across 3 orders; extend to LiLT/BROS reversed | 1 | 2 | 3 | 1 | 2 | 1 | 10 | 2 naive reversed-order runs/backbone (~1 h each) | profile rejected across orders ⇒ locus is order-specific | "single-order conclusions" |
| 8 | **Long-horizon stress** — 10-session cil_cord_long / wildreceipt_long: does the head locus hold at depth? | 1 | 2 | 2 | 1 | 2 | 1 | 9 | naive + ER × 3 seeds with protocol (~6 GPU-h) | locus migrates with horizon | "only 3–6 sessions" |
| 9 | **Modality-stream attribution as protocol step 0** (input masking C1–C4 → which stream carries forgetting; model-agnostic via masking) | 2 | 2 | 3 | 2 | 2 | 0 | 11 | write (data exists); overlaps vision fork — packaging only | — | "where is the vision?" |
| 10 | **Readout-marginal snap scalar** — cosine of old-task output marginal to new-task gold marginal (0.997) as a mechanism indicator any token-CL paper can report | 2 | 2 | 2 | 0 | 2 | 1 | 9 | needs logits per boundary; cheap for new runs, not for saved ones | snap absent while forgetting present ⇒ not the mechanism | "mechanism is asserted, not measured" |
| 11 | **Seed-power analysis** — from observed sds, how many seeds detect Δ=1 AA per scenario; recommend seeds per claim | 1 | 2 | 3 | 0 | 2 | 1 | 9 | write only | — | "three seeds is too few" |
| 12 | **Method×backbone interaction test** — formal test of the EWC sign flip (−11.8…+5.3) on the 68-cell grid; argue multi-backbone protocols are mandatory | 2 | 2 | 3 | 1 | 2 | 1 | 11 | write + small stats script | interaction n.s. ⇒ flips are noise | "single-backbone results generalise" |
| 13 | **Byte-accounted leaderboard** — storage in bytes incl. model weights (Independent = 3×) | 1 | 2 | 3 | 0 | 2 | 0 | 8 | write | — | "exemplar counts hide cost" |
| 14 | **ViT transfer of the protocol as scope evidence** (H1: head share 97–100 %) | 1 | 2 | 3 | 2 | 2 | 0 | 10 | write (data exists) | — | "document-specific" |

## Novelty lookups (top 5; WebSearch extended, 2026-10-14)

- #1 inversion — no hit for SLCA / Gaussian classifier alignment on token classification or sequence labeling; SLCA results are image-only (ICCV'23, SLCA++ 2024). Queries: "SLCA classifier alignment Gaussian feature replay token classification sequence labeling continual NER". **Novelty 3 confirmed.**
- #2 headroom — "Achieving Upper Bound Accuracy of Joint Training in CL" (arXiv 2502.12388) treats Joint as the bound; no work certifies Joint−Independent headroom before ranking. **Novelty 2–3.**
- #5 benchmark — surveys (2408.06345, 2507.09861) and GDI-Bench show no CL benchmark over LayoutLM-family KIE; "work combining CL with document understanding benchmarks appears limited". **Novelty 3.**
- #3 protocol — Ramasesh et al. 2021 (layer anatomy), Davari et al. 2022 (CKA representation forgetting) cover the instruments; no released protocol with dominance/shared-location tests + a perturb step. **Novelty 2.**
- #12 interaction — "Same Methods, Different Rankings: Trainable Depth as an Evaluation Variable in CL" (arXiv 2604.21927) shows rankings flip with trainable depth; ours flips with *backbone* at fixed regime — adjacent, so **Novelty 2** (cite it).

## Notes

- #1 and #2 compose: the inversion explains *why* head statistics fail on documents (binding), the
  headroom step explains *why* nothing can win on DIL. Together they are the "evaluation
  methodology" section a CVPR reviewer would adopt.
- #5 is free (artefact packaging) and raises the paper's adoption ceiling; it needs the repo
  README + a leaderboard table in supp, not research.
- Out of scope but noticed: the stale-label audit (#6) is the natural *diagnostic* companion to
  the CoLaR-R mechanism owned by the replay fork.
