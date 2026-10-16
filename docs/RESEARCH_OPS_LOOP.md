# Research-ops loop — CVPR 2027 (operating procedure for the autonomous ticks)

Deadlines (sourced, cvpr.thecvf.com, read 2026-10-14): **registration Nov 10, 2026 AoE ·
paper Nov 16, 2026 AoE · supplementary Nov 23, 2026 AoE** · reviews Jan 25 · rebuttal Jan 25–Feb 1 ·
decisions Feb 25, 2027. Venue Seattle, Jun 20–25, 2027. OpenReview; all authors need profiles.

Instrument: `uv run python scripts/r2_status.py` (read-only; prints cells, means, gate verdicts).
State: `STATE.md` (top entry), plan `docs/RESEARCH_PLAN_2026-10_next_mechanism.md` §0',
evidence `docs/RESULTS_SYNTHESIS_CVPR2027_2026-10-14.md`.

## What "CVPR-qualified" means here (rubric the loop scores against)

| Criterion | Pass condition | Current status |
|---|---|---|
| Novel claim | a mechanism or finding not in prior work, stated as a falsifiable claim | anatomy + conservation + binding: yes; CoLaR-R: pending R1 |
| Significance | moves the hardest scenario by ≥ 1 AA over the strongest comparator, or certifies a null that changes practice | CIL-CORD gap 14 is the target |
| Rigor | ≥ 3 seeds; ≥ 2 backbones for any claim called general; pre-registered bars; controls isolate the mechanism | grid 68/72 cells; R1 has controls (plain, oracle, ER+refresh, CPFD, IS3) |
| Vision relevance | the document image/layout modality is part of the story, not just the backbone | partial (modality masks, ViT transfer); silent-modality probe optional |
| Reproducibility | code + configs + run names; released protocol (`doccl.diag`) | yes |
| Honesty | nulls reported as nulls; limitations not duplicated in claims | yes |

A "qualified idea" = passes all six. The paper already passes five on the diagnostic
contribution; the loop's job is to either make CoLaR-R pass "Significance" or to certify it as
a null and keep the five.

## Tick procedure (every ~60 min while runs are live; stop conditions below)

1. `uv run python scripts/r2_status.py`. Read: new DONE cells, FAILED cells, gate verdicts.
2. **FAILED cell** → read `results/logs_vision/<cell>.log` tail; if OOM → relaunch that cell
   alone with `training.batch_size=1` (CoLaR) after the queue finishes, never alongside it; if
   code error → fix, add test, commit `AGENT FIX:`, let the resume-safe queue pick it up.
3. **New DONE cells** → append one line per cell to the STATE.md R2 table (AA/BWT, time).
4. **Gate reached** (all three seeds of the cells involved exist):
   - R1 **PASS** on cil_cord → write §7 paragraph "Refreshing the memory's labels" with the numbers,
     add rows to Table 1 (CIL-CORD + Mixed) for CoLaR, CoLaR-R; queue the ER+oracle-refresh control
     (buffer-general or latent-specific?); queue secondary backbones (LiLT/BROS) for the winner
     only — `SCENARIOS=cil_cord MODELS=...` variant of the queue script. Rebuild; keep 8 pp.
   - R1 **FAIL/KILL** → add CoLaR-R as a row in the falsification ledger (Table 2) with the
     one-line mechanism reading; keep the diagnostic framing; no further GPU on label refresh.
   - oracle **VIOLATED** (oracle ≤ pseudo) → stop pseudo cells, inspect refresh counts in the
     run logs (`colar_r[...]: refreshed N docs, M tokens relabelled`), fix before any writing.
5. **Resource check**: `nvidia-smi` VRAM < 5 GB, `free -g` RAM < 14 GB; never start a second
   dataset-building job while the queue runs.
6. **Writing while GPU is busy** (pick the first undone): [CoLaR headline reconciled 2026-10-15, provenance comment in
   main.tex §7 — done]; [HGT/CUBER cell filled 2026-10-16 02:15, verified vs disk 21:18; no `\todo` left — done]; F9 inversion paragraph
   (SLCA vs CoLaR × images/documents); supplementary PDF skeleton; Overleaf re-upload.
7. Commit anything changed; `git pull --rebase && git push`. One-line STATE.md update.

## Schedule and stop conditions

| Date | Milestone | Loop action |
|---|---|---|
| Oct 15–17 | seed-42 block (cil_cord, mixed) | watch; writing items from step 6 |
| Oct 18–21 | seeds 7/123 | **Gate 1 (Oct 21)**: verdict from r2_status |
| Oct 22–26 | winner on LiLT/BROS, ER+refresh control | or ledger row if null |
| Oct 27–30 | ablation controls; no new mechanisms after Oct 30 | freeze mechanism list |
| Oct 31–Nov 6 | buffer; full read-through; supplementary | writing only |
| **Nov 10** | **register paper on OpenReview (title/abstract/authors)** | hard stop for abstract |
| Nov 13 | internal freeze | no content edits after |
| **Nov 16** | **submit** | — |
| Nov 23 | supplementary | — |

Stop the loop when: (a) Gate 1 verdict is written into the paper and the queue is empty, or
(b) the user says stop, or (c) a FAIL repeats twice for the same cell (escalate to the user).
Compression knob if the queue runs long: seeds 7/123 at `method.epochs=5` (Amendment-9 recipe),
stated in the paper as a budget cap.
