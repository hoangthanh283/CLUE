# RCA — CoLaR / LexSlot drawbacks, pros & cons (2026-10-19)

Scope: everything already in `results/` tagged CoLaR or LexSlot (75 cells).
Procedure: `scripts/rca/build_ledger.py` → `scripts/rca/analyze_ledger.py` →
`rca/colar_lexslot_ledger.csv`, `rca/pass2_gap_decomposition.md`,
`rca/pass3_flaw_map.md`, `rca/pass5_sign_rule_v2.md`.
Baselines from the Reports sheet and `all_runs.csv` (3-seed means).
Ridge-based coverage/trunk/head terms require the dump_final probe and
are left as Pass 2b.

## 1. Headline numbers (recipe-matched; from Pass 5 v2)

### Documents, LayoutLMv3

| Scenario | Joint | Naive | ER | Best buffer-based | Plain CoLaR (3s) | Best CoLaR-R (3s) |
|---|---|---|---|---|---|---|
| DIL | 88.7 | 41.3 | 87.9 | 88.4 | 86.4 (seed 42) / 87.75 (e10, 3s) | — |
| CIL-CORD | 33.3 | 19.0 | 16.0 | 19.0 | 20.4 ± 1.1 | **26.3 ± 2.9** (oracle, 3/3 sign) |
| Mixed | 67.5 | 33.9 | 60.3 | 62.2 | 58.75 ± 0.9 | **64.4 ± 1.6** (oracle, 3/3 sign) |

LexSlot (3-seed means): DIL 70.7 ± 23.2 (mixed d5/d50 recipes), CIL-CORD **18.9 ± 0.2**, Mixed **26.7 ± 1.5**.

### Images, ViT-B/16 (CIFAR-100 / ImageNet-R, 3 seeds)

| Arm | CIFAR-100 | ImageNet-R |
|---|---|---|
| Joint | 89.5 | 77.6 |
| SLCA | **87.5** | **65.0** |
| RanPAC (frozen) | 85.3 | 58.6 |
| SimpleCIL | 81.7 | 48.2 |
| CoLaR r128 | 80.5 | **55.3 ± 1.2** |
| CoLaR + S2 anchor (s42) | 84.1 | — |
| CoLaR + I1 drift-comp (s42) | 84.4 | — |
| Probe ceiling (RP-ridge on adapted trunk, full data) | 88.6 | not yet run |

## 2. Pros (what the ledger supports)

**P1. CoLaR is a lossless compressor on documents, not elsewhere.** DIL CoLaR r=128 at 60 MB
matches the raw-latent bank at 163 MB within 1 pp (86.4 vs 87.3). The same statement is false
on images: CIFAR CoLaR r=128 is a tie with raw bank at 80 (the compression is clean, the
ceiling is the one 80 under the plastic head). *The compressor isn't the problem; the readout
is.*

**P2. CoLaR has a recovery property no frozen-trunk method has.** Pass 3 flaw map:
plain CoLaR DIL cells have diag_mean 95+ and 2–4 recovery events per matrix (`M[t][k]
> M[k][k]`, i.e. an old-task cell *improves* after later replay). SLCA/RanPAC can never
show this because their diagonals never go high in the first place. On Mixed seed 42 the
FUNSD-s0 cell traversed 88.8 → 0.0 → 1.2 → 29.4 → 25.3 → 63.1 — a 63-pp recovery that is
pure replay signal. This is the one capability a frozen-trunk method cannot replicate.

**P3. On CIL-CORD, label-refreshed CoLaR is the only arm beating the naive floor by ≥ 4 AA
with 3/3 sign.** `colar_r_oracle` 26.3 ± 2.9 (Δ naive +7.3), `colar_r_sliced` 25.4 ± 2.0
(Δ +6.4), both 3/3 sign. Vanilla replay (ER 16.0) is **below** naive on this scenario — it
teaches "O" on every post-capture class. CoLaR + a stale-label fix is the only working family.

**P4. LexSlot's lexical gate routes at 0.92 F1 and is drift-immune.** From the earlier
STATE record. Useful as a *diagnostic* that OCR-vocab similarity signals task identity, but
as a method axis it's dead on CIL-CORD and Mixed (next section).

## 3. Cons (what the ledger refuses to hide)

**C1. CoLaR's plastic head is coverage-bound.** CIFAR probe (seed 42): RP-ridge on **full
train data** of the adapted trunk = **88.6**; same ridge on **50 stored/class** = 80.7. The
8 pp gap is readout coverage, not representation or statistics. Fixes targeting statistics
(I1 drift-comp Gaussians, S2 anchoring) cap at 84.4; fixes targeting coverage have never
been tested. See `docs/IMAGE_SCOPE_PREREG.md` Amendment 8 addendum.

**C2. CoLaR trails SLCA by 7 pp on CIFAR, 10 pp on ImageNet-R.** `plastic_gap` (= joint − AA)
on image CIL: plain CoLaR 9.0 / 22.3 vs SLCA 2.0 / 12.6. The trade is diag_mean 98 vs 94 —
CoLaR fits the current task harder, pays on the old. SLCA avoids this with a near-frozen
trunk; CoLaR's whole point is plastic feature adaptation, so matching SLCA's recipe would
collapse the method into SLCA-with-latents.

**C3. On CIL-CORD, plain CoLaR = plain ER = plain naive = 20 ± 1 AA.** Replay-family
methods (ER 16.0, DER++ 18.2, CoLaR 20.4) are statistically indistinguishable from naive
(19.0) because the capture-session labels have introduced class-labelled-as-O on every
post-capture token. The CoLaR *memory* contains the right features and the wrong labels.
The only arms clearing this: oracle refresh (needs raw labels; not deployable) and sliced-
softmax CE (26.3 ± 2.9 and 25.4 ± 2.0 — don't need labels, don't train on O for new classes).

**C4. On Mixed, the sliced-softmax arm fails the 3/3 sign rule by one tie.** sliced
+2.5 mean (bar +1.0) but seed 123 at Δ −0.08. The deployable arm (sliced + pseudo refresh)
is a null on Mixed and a kill on CIL-CORD (R2d, 2026-10-18). Nothing stacked on sliced-
alone has survived. Mixed stays open as "ceiling 61.3, oracle 64.4, pre-reg bar 63.2 missed
by two of three seeds".

**C5. LexSlot is dead on CIL-CORD and Mixed.** 18.9 ± 0.2 and 26.7 ± 1.5 (3-seed, 3/3 sign
*against* CoLaR — plain CoLaR wins by 1–30 pp on every seed). The lexical gate works on
DIL where domains are OCR-distinct; it does nothing on CIL-CORD (one domain, split classes)
and makes Mixed worse because it protects wrong slots. Falsified-chain entry, not a
candidate for revival.

**C6. Routing (Amendment 10) was a dead end.** Three-seed confirmation: far-routing +0.5 pp
mean vs uniform (seed-42 +1.8 was noise), 2/3 sign. See `IMAGE_SCOPE_PREREG.md` Amendment 10
verdict (2026-10-13). The gate adds nothing beyond replay as *router*.

**C7. Per-document SVD compression is not low-rank on all tasks.** D1/D2 (token-position
compression) collapses SROIE 81 → 59 → 12 pp. The whole-doc consistency law has a floor —
cannot drop context tokens, cannot drop visual positions. Byte reductions beyond r=128 are
not free.

**C8. CoLaR-ablation siblings didn't move anything.** colar_bal, colar_cb, colar_wsvd,
colar_knn, colar_meta all land within noise of plain. Store-balancing and read-side kNN
are not where the ceiling sits.

**C9. Head-coverage interventions on images don't reach SLCA.** S1 (RP head only) 78.6, S2
(anchor) 84.1, S3 (both) 81.6, S4 (first-task adapt + freeze + RP) 72.9, I1 (drift Gaussian)
84.4. Go bar was 87.5. **Amendment 8 has no closing verdict in the prereg** — this RCA is
also closing that record: NO-GO on all five arms, best 84.4, bar 87.5.

**C10. CoLaR's win over SLCA on documents is apples-to-oranges.** On DIL LayoutLMv3 CoLaR
r=128 is 86.4–87.6 and SLCA has never been ported to a per-token BIO head; the comparison
in the paper is implicit (there is no SLCA document row). This is a limitations-paragraph
issue, not a result issue, but it should be stated.

## 4. Flaw map (Pass 3 summary, representative cells)

| Cell | diag_mean | final_old | plastic | recovers | reading |
|---|---|---|---|---|---|
| CoLaR DIL s42 d50 r128 | 95 | 86 | HIGH | yes | the P2 behaviour; works |
| CoLaR CIFAR s42 d500 | 98 | 79 | HIGH | some | plastic trunk + undercovered head |
| SLCA CIFAR s42 (sheet) | 94 | 87 | HIGH-but-lower | no | conservative trade |
| RanPAC CIFAR s42 | 90 | 85 | LOW | no | frozen — can't recover |
| CoLaR CIL-CORD s42 | 32 | 24 | LOW | yes (4) | diagonal collapses on each session; stale labels |
| CoLaR-R-oracle CIL-CORD s42 | 36 | 33 | LOW | yes (4) | +6 pp over plain, same shape, different label stream |
| LexSlot Mixed s42 | varies | 27 | — | — | gate hurts |

## 5. Open axes (never probed)

1. **Readout refit from reconstructed features** — the probe showed the ceiling (88.6 on
   CIFAR). CoLaR already stores decodable codes; the readout can be solved from reconstructed
   features of every stored latent through the current trunk, every task close. Nobody has
   promoted this probe to a method.
2. **Per-token ridge on documents** — the ridge decomposition has never been run on
   LayoutLMv3. Pass 2b work.
3. **Modality-sliced ridge** — Finding 1 says text-and-head carries the drift; a text-only
   vs. layout-only ridge on CoLaR's stored latents would quantify this per cell.
4. **Vocabulary-weighted ridge solve** — LexSlot's gate is dead as a *router* but has never
   been tried as a *per-sample weight* in a ridge normal-equation solve.
5. **CoLaR's d/r at matched bytes on Mixed/CIL-CORD** — only d=50/r=128 was swept on the
   non-DIL scenarios; the compressor's own parameters have not been tuned there.

## 6. Verdict on each surveyed drawback

| Issue | Recipe-level? | Method-level? | Framing-level? |
|---|---|---|---|
| C1 head coverage | — | **yes** | — |
| C2 vs. SLCA on image CIL | — | **yes** (plastic trunk by design) | — |
| C3 stale-O labels CIL-CORD | — | **yes** (store labels at capture) | — |
| C4 Mixed sign-rule miss | partly (seed noise) | partly (sliced + mixed-doc behaviour) | — |
| C5 LexSlot on non-DIL | — | **yes** (gate meaningless) | — |
| C6 routing | — | **yes** (dead axis) | — |
| C7 compression floor | yes | — | — |
| C8 CoLaR-* ablations | yes | — | — |
| C9 image head arms | partly | — | — |
| C10 SLCA document comparison | — | — | **yes** (prose) |

## 7. What the next method must do (necessary conditions)

Any new arm claiming to beat the current ceiling needs:

- **On image CIL**: close the readout-coverage gap (reach the 88.6 probe ceiling) without
  freezing the trunk. Head-only arms have been shown to cap at 84.4.
- **On CIL-CORD**: avoid teaching "O" for post-capture classes; the only two arms doing
  this are the oracle (not deployable) and sliced-softmax (needs the Mixed sign to pass).
- **On Mixed**: pass the +1.0 bar on all three seeds. Current ceiling 61.3 (sliced-alone),
  oracle 64.4; the 3-pp gap is approximately the stale-label cost on the within-CORD
  class-incremental step.
- **On DIL**: nothing to beat — Joint 88.7, best any 88.4. Only a byte-reduction result
  can land here.

## 8. Deferred (Pass 2b, needs dump_final)

A ridge probe on **9 lead cells** (3 doc + 3 doc + CIFAR + ImageNet-R + 1 DIL) would fill
the three TODO columns of `pass2_gap_decomposition.md`. Cost: 1 GPU-day on this box.
Not needed to publish this RCA; needed to publish the next method claim.

## Appendix

Generated from:
- `rca/colar_lexslot_ledger.csv` — Pass 1
- `rca/pass2_gap_decomposition.md` — Pass 2 ledger-only terms
- `rca/pass3_flaw_map.md` — plasticity × recovery
- `rca/pass5_sign_rule_v2.md` — recipe-matched 3-seed verdicts
- `scripts/rca/build_ledger.py`, `scripts/rca/analyze_ledger.py`
