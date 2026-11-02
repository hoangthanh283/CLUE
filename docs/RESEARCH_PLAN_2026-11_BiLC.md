# BiLC — Bidirectional Label Completion over a whole-document latent bank (2026-11-02)

Status: proposed after the CIL-CORD protocol fix; E1/E2 running (`scripts/rca/g0_bilc_2026-11.sh`).

## 1. What we measured this week (seed 42, CIL-CORD, corrected scorer)

| Row | Legacy | Corrected | What it tells us |
|---|---|---|---|
| Naive | 19.0 | 18.9 | unchanged — floor is real |
| ER (200 docs) | 16.0 | 26.9 | replay forgets from session 2 on |
| CoLaR | 20.4 | 34.1 | retains middle sessions, new session suppressed to 10–15 |
| CoLaR-R sliced | 25.4 | 39.4 | replay-side slice helps; new-session val oscillates 82→27, 45→5→51→5 |
| CoLaR-R oracle (true labels) | 26.3 | 61.2 | retention fixed by refresh; new session still 14–32 |
| **CoLaR-R double-sliced** | — | **60.5 ± 10.1** (3 seeds) | current-side slice fixes new-session learning; no labels |
| Joint | 33.3 | 93.1 | the real ceiling |

Mixed seed 42: double-sliced 73.6 (legacy joint 67.5).

Mechanism, now separated into two leaks:
1. **Replay-side stale O** — banked docs label post-capture entities O → suppresses new classes. Fixed by the sliced replay CE (capture-time classes).
2. **Current-side stale O** — CIL-masked current data labels old-class entities O → fights replay on the same surface forms (menu.price vs total.price). Fixed by the mirror slice.

What is left: (a) session-0 retention collapses at the last session (9.9 / 27.1 / 38.6); (b) one seed still oscillates on its last session (seed 123: 16→4→31→19→4); (c) the two slices *remove* gradients, they never *add* the missing positives.

## 2. Literature (what exists, what does not)

- **CL-NER background shift** is the standard framing: ExtendNER / Monaikul 2021 (KD from the old model), CPFD (EMNLP'23, confidence pseudo-labels of old types on current O), IS3 / "Tale of Two Shifts" (ACL'24), "Learning O helps" (entity-aware contrastive inside O), POF (AAAI'25, prototype replay + *old-class-focusing* feature KD), FINER (COLING'25, prototype-guided adaptive pseudo labels), FSE (2026, fast/slow span experts), "Diagnosing representation dynamics in NER extension" (2025: O-tag classifier plasticity is the bottleneck).
- **Image CIL**: ER-ACE (asymmetric CE: incoming batch uses current classes only, replay uses all) is the image-side ancestor of our two slices. SS-IL separated softmax.
- **Continual detection**: RCLPOD (2024) does *forward* label propagation into the replay buffer — the only prior work that re-annotates stored samples with later classes. Multi-label CL does online relabeling of the buffer.
- **Nobody** combines, on a whole-document latent bank: forward propagation into the bank + backward pseudo-labels on current data + asymmetric slices for the residual. And nobody has the document setting's defining property: every CORD receipt carries entities of *all* sessions, so a banked document is a *complete* supervision source once its labels are completed. In sentence NER a stored sentence rarely contains later types; in documents it almost always does.

## 3. The idea

**BiLC**: treat the stale-O problem as a *label-completion* problem on both sides of the replay boundary, with the two slices only for what cannot be completed.

- **Forward completion** (bank side): after task *t*, the model that just learned session *t* at ~93 F1 writes its confident session-*t* predictions into the stale-O tokens of every banked document (`refresh: sliced_pseudo`). This refresh returned 0 candidates before, because under the legacy conflict session *t* was never learned (diag 3–15). With the double slice it has something to write.
- **Backward completion** (current side): a frozen snapshot taken at task start labels the old-class entities hidden under the current session's O (confidence ≥ τ = 0.9). CPFD-style, but applied to the *current* stream of a replay method, and only to O tokens.
- **Residual slices**: whatever stays O after completion is trained with the sliced CE on its side (no gradient toward or away from the classes that could still be hiding there).

Expected: completed labels turn every banked receipt into a near-joint training document; the oracle run (61.2) is the ceiling for forward completion alone, the joint (93.1) for both. The three-seed double-sliced result (60.5) is the floor.

Why it should also fix (a) and (b): session-0 collapse happens when session-4 (mixed) entities on session-0 receipts are still O in the bank at the final session; forward completion writes them in. Oscillation on the last session is the current/replay fight on uncompleted tokens; backward completion resolves it on the current side instead of merely silencing it.

## 4. Experiments (all CIL-CORD seed 42, corrected scorer; `results/gates/cil_protocol_2026-11/`)

| ID | Config | Tests |
|---|---|---|
| E1 | `colar_r_sliced2_pseudo` | forward completion alone (vs double-sliced 62.9, oracle 61.2) |
| E2 | `colar_r_bilc` | forward + backward (τ = 0.9) |
| G0d | `colar_r_sliced2_oracle` | ceiling with true labels + both slices (first run was mis-specified) |
| G0c | double-sliced seed 123, replay batch 2 | stabiliser control |

Decision rule: E2 ≥ 70 on seed 42 → run 3 seeds + Mixed, make BiLC the method row. E1 ≈ E2 → drop the teacher (cheaper). Neither above 65 → the residual is representation drift of session 0, next lever is POF-style old-class-focusing feature KD on replay tokens at layer k.

## 5. Then

- 3-seed rescore of the 12 reference rows on CIL-CORD + Mixed (`scripts/rescore_cil_2026-11.sh`, rented GPU). CPFD / IS3 were "nulls" under the broken scorer; under the corrected one they are the natural baselines and may be strong.
- Paper reframe: CIL-CORD is no longer a floor regime; the label-growth section becomes the method section.
