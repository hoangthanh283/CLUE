# Brainstorm round 1 — lens: VISION & LAYOUT MODALITY (2026-10-14)

Rubric: Nov 0-3 · Sig 0-3 (DIL-only caps at 1) · Feas 0-3 (≤5 d) · Vis 0-2 · Evid 0-2 · Null 0-1 → /14.
Facts used: patch CKA 1.00, C3≈C4, c2_no_text collapses, D2 (no visual positions) SROIE→12,
ViT head share 97–100 %, CoLaR trails SLCA by 7.5 on CIFAR. Falsified list respected.

| # | Idea | Mechanism (one line) | Falsifier (control / kill) | Expected | Nov | Sig | Feas | Vis | Evid | Null | Tot |
|---|------|----------------------|----------------------------|----------|----|----|----|----|----|----|----|
| 4 | **Position-axis anatomy** | Run the localisation protocol along the *sequence* axis: CKA/displacement for visual-patch positions vs text positions vs CLS at each depth, on existing checkpoint pairs | Control = text positions; kill if visual-position CKA at layer 11 drops within 0.05 of text-position CKA (then no within-sequence locus) | Visual positions stay ≈1.0 while text positions drift → forgetting is text-position-localised *inside* a shared encoder; complements the depth axis | 2* | 2 | 3 | 2 | 2 | 1 | **12** |
| 1 | **Silent-modality probe** | Keep the 197 visual positions in CoLaR replay but replace their content with the model's dummy-image latents (structure kept, content removed); compare to D2 (positions removed) and full CoLaR | Kill: if structure-only ≥ full −1 pp → visual tokens are only *structure*; if ≈ D2 (12) → they carry *information* | Likely structure (text load-bearing), but either outcome is a finding and fixes the D2 confound (D2 changed length AND content) | 2 | 2 | 3 | 2 | 2 | 1 | **12** |
| 8 | **Conservation on pure vision (ViT CIL)** | Repeat Finding 2 on ViT/CIFAR-100: freeze head+late after task 1, measure CKA migration to early/mid | Control = naive ViT; kill if early/mid CKA stays > 0.9 under freeze (no migration → conservation is document-specific) | Migration reproduces (anatomy already transfers, H1) → conservation law becomes architecture- and modality-general | 2* | 2 | 2 | 2 | 2 | 1 | **11** |
| 2 | Modality-asymmetric rank | Per-doc SVD with r_text=128, r_vis=16 (visual latents drift-free, near-deterministic) at matched bytes vs uniform r | Kill if AA drops > 1 vs uniform at equal bytes | Bytes −30–40 % at equal AA on DIL | 2 | 1 | 3 | 2 | 2 | 1 | 11 |
| 13 | Position-ablation ladder | Which replayed positions matter: {all} ⊃ {labelled+visual} ⊃ {labelled} ⊃ {visual}; 4 seed-42 cells | Kill: ladder monotone with no step > 3 pp → "whole sequence" is the carrier (already partly D1/D2) | Extends Finding 3b to the position axis | 2 | 2 | 2 | 2 | 2 | 1 | 11 |
| 12 | Latent-inversion privacy audit | Train a small decoder on public pages from layer-4 latents → pixels; attack CoLaR's r=128 store; report SSIM/OCR-recoverability | Kill (for the privacy claim): if recovered text is OCR-readable, "no raw bytes" ≠ private | Partial reconstruction likely → honest privacy caveat + motivates rank/noise defence | 1 | 2 | 2 | 2 | 1 | 1 | 9 |
| 14 | Dual-stream locus (LiLT) | Localise forgetting in LiLT's text vs layout stream separately | Kill if both streams drift equally | Text stream carries it; layout stream ≈ drift-free (mirrors patch CKA) | 2 | 1 | 2 | 1 | 2 | 1 | 9 |
| 5 | Cross-modal attention drift | Attention mass text→image before/after task (pooled attention CKA) | Kill if attention maps CKA > 0.95 | Model stops attending to image under drift? finding only | 2 | 1 | 2 | 2 | 1 | 1 | 9 |
| 7 | Patch-budget vs forgetting | Downscale page (fewer patches); does locus/magnitude change? | Kill if BWT changes < 2 | Finding; cheap | 1 | 1 | 3 | 2 | 1 | 1 | 9 |
| 11 | OCR-free backbone anatomy (Donut) | Head-locus with no text stream | — | Powerful but infeasible (6 GB, generative decoder) | 3 | 2 | 0 | 2 | 1 | 1 | 9 |
| 3 | Layout-rendered replay | Store boxes+labels only; render synthetic pages for replay | Kill if < naive +5 | Predicted dead (text load-bearing; carriers control 36.7) | 2 | 1 | 2 | 2 | 0 | 1 | 8 |
| 10 | Modality-dropout as CL regulariser | Random modality masking during CL to force redundancy | Kill if AA Δ < 1 | Does not touch the head → predicted null | 1 | 1 | 2 | 2 | 0 | 1 | 7 |
| 9 | Layout-signature routing | Drift-free router on box geometry | — | Lexical routing null → predicted null | 1 | 0 | 3 | 1 | 0 | 1 | 6 |

\* Novelty adjusted after search (below).

## Novelty checks (WebSearch extended, 2026-10-14)
- #4: adjacent — "Hidden Forgetting in Continual Multimodal Learning: When Accuracy Survives but Grounding Fails" (arXiv 2607.02020) studies reliance drift over visual regions/OCR tokens in *autoregressive MLLMs*; "Visual Dependence-Aware ... Continual Post-Training" (2608.26095) uses token-level visual dependence. No per-position representation-drift locus in an encoder-only document model → Nov 3→2.
- #1: no direct hit (queries: latent replay visual tokens structure vs content; replay carriers multimodal). Analogue outside CL: "Leveraging Latent Visual Reasoning in Silence" (2605.18641) finds latent visual tokens replaceable by noise → supports the "structure" prior. Nov 2.
- #8: adjacent — "Putting a Face to Forgetting" (2601.22012) reports earlier layers *gain* capacity as tasks accrue; "Reset It and Forget It" (2310.07996). No protect-and-migrate test on ViT → Nov 2.
- #2: "Latent Replay Detection" (2603.00138) compresses per *task*, not per modality; no hit for modality-asymmetric rank → Nov 2.
- #12: feature inversion with diffusion priors exists (2412.10448); applying to SVD latent stores is an audit, not a method → Nov 1.

## Notes
- #1 and #13 share runs: the ladder's {labelled+visual} and the structure-only cell cost 4 seed-42 cells total (~2.5 h each on DIL/CoLaR k4).
- #4 needs only checkpoint pairs + the new `doccl.diag` protocol with position-sliced activations (`collect_activations` already returns per-token vectors; add a position mask). Zero training if pilot checkpoints exist; otherwise one naive DIL run per seed.
- #8 reuses `cil_cifar100` + ViT wrapper; freeze arm = `freeze_groups(model, ["head","late"])`.
