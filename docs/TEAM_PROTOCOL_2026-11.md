# Team protocol for finishing the CVPR 2027 paper (2026-11-11)

Plan of record: `~/.claude/plans/rippling-pondering-comet.md`. Current state: top ticks of `STATE.md`.
Paper: `paper_cvpr/main.tex` + `supp.tex`, title *Memory Is Not Enough: Separating Drift from
Background Conflict in Continual Document Understanding*. Story: forgetting in continual document
IE has two separable causes, **drift** (remedy: CoLaR, Compressed Latent Replay) and **background
conflict** under label growth (remedy: SBR, Session-Blocked Readout; SBR-Recall replaces the memory
by per-session low-rank trunk deltas). Centrepiece: a 2×2 (memory × readout) on CIL-CORD.

Deadlines: full paper **Nov 16 AoE = Nov 17 19:00 ICT**; numbers freeze Nov 17 07:00 ICT;
supplement Nov 23. Earlier verified review: `docs/REVIEW_SENIOR_CVPR_2026-11-10.md` (items C1–C6,
I1–I8).

## Roles and ownership

| Role | Owns (may edit) | Never touches |
|---|---|---|
| **PM** (main session) | `STATE.md`, `ROADMAP.md`, this file, `git push`, Overleaf uploads, user reports, priorities | GPU jobs, paper prose |
| **Dev** | `doccl/`, `scripts/` except the two generators, `configs/` (new files only), `tests/`, the GPU queue, `results/` | `paper_cvpr/` |
| **Paper** | `paper_cvpr/**`, `scripts/rca/make_clean_tables.py`, `scripts/rca/make_retention_fig.py` | `doccl/`, GPU, datasets |
| **Reviewer** | append-only `docs/REVIEW_LOG_2026-11.md` | everything else (read-only) |

Agents cannot message each other. Report to the PM; the PM relays.

## Hard rules (all roles)

1. **Machine limits.** RTX 2060 6 GB, about 14 GB RAM. One GPU job at a time, VRAM < 5 GB,
   RAM < 14 GB; an out-of-memory crashes the box. Only **Dev** may start anything that loads a
   model or a dataset, and only through the single queue driver. Paper and Reviewer run only
   `latexmk`, `pdftotext`/`pdfinfo`, the two generators, `git` read commands, and small Python that
   reads `metrics.json` / `matrix.npy` / logs. No `pytest` for Paper or Reviewer.
2. **Processes.** Never edit a driver script while it runs. Kill by PID, never by pattern. Do not
   restart `scripts/rca/audit_gates_2026-11.sh` or `scripts/rescore_cil_2026-11.sh` (old protocol,
   stopped on purpose).
3. **Configs.** `configs/` is add-only; use CLI overrides for one-off changes.
4. **Git.** Several agents share one index, so a bare `git commit` sweeps up files another agent
   has staged (this happened on 2026-11-11: commit 8edfcfc carries Dev's figure under the
   Reviewer's message). **Always commit with a pathspec**: `git add <paths>` then
   `git commit -F <msgfile> -- <paths>` (or `-m ... -- <paths>`), which commits only those paths
   whatever else is staged. Never `git add -A`, never `git add .`, never a bare `git commit`.
   If `.git/index.lock` exists, wait and retry. No amend, rebase, reset or force. Put `git commit`
   in its own Bash call, with no `-n` flag anywhere in that call (a hook rejects it). Dev prefixes
   code commits `AGENT IMPL:` / `AGENT FIX:`. Do not push; the PM pushes. End every commit message
   with a co-author line naming the model you actually run on (sub-agents: Opus 5.5; the PM:
   Fable 5.2) and the session line:
   ```
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_018guzpoJbFfkTJTp1h4Vaaz
   ```
5. **Shell.** Non-interactive: `rm -f`, `cp -f`, `mv -f`. Run Python through `uv run`.
6. **Numbers.** No result number is typed into the paper. Headline values come from
   `paper_cvpr/vals.tex` macros; tables from `tab_2x2.tex` and `tab_main_clean.tex`; all three are
   written by `scripts/rca/make_clean_tables.py`. The retention figure comes from
   `scripts/rca/make_retention_fig.py`. A cell switches from the old protocol (dagger) to the clean
   one only when all three clean seeds exist.
7. **Wording.** No `---` in any paper file. Every "at / within / matches / above / level with"
   sentence must hold per seed, not only on the mean. Joint training is a *reference*, never an
   "upper bound" or "oracle". No privacy claim beyond "stores no raw pixels or token ids".
8. **Paper invariants.** Body ≤ 8 pages (CVPR), zero undefined references, `supp.tex` builds.
   Bibliography is read from `refs_precompiled.bbl`; to add a citation, verify it exists, add it to
   `refs.bib`, move the `.bbl` aside, run `pdflatex` + `bibtex main` + `pdflatex` ×2, then copy
   `main.bbl` to `refs_precompiled.bbl`. Never run `latexmk -C`.
9. **No sub-agents.** Do your task yourself; do not spawn agents.

## Clean protocol (what "clean" means)

fp32 (`training.amp=false`), `+training.val_frac=0.1` (early stopping on 10% held out from the
session's training data, never the test split), `method.epochs=10` (joint: 100),
`+training.cil_restrict_eval=true`, `+training.full_label_eval=true` (writes `full_label_f1` to
`metrics.json`: final model on the unmasked CORD test set), seeds 42/7/123,
`output_dir=results/clean_2026-11`. Queue log: `results/clean_2026-11/queue.log`.

Known hazard (corrected 2026-11-11 from `docs/LEGACY_PROTOCOL_TABLE_2026-11.md`): the LwF zeros
(CIL-CORD seed 123, Mixed seed 7) and the CPFD CIL-CORD seed-7 zero are fp16 runs; LwF never
collapses in fp32. **IS3 never enables mixed precision** (`is3.py` has no `_amp_setup`), so its
Mixed seed-42 zero is a genuine fp32 result and must not be dropped or blamed on precision. Several
legacy cells mix precisions across seeds (seed 42 fp32, later seeds fp16); the legacy table lists
each.

## One cycle (at every milestone)

1. PM reads the queue and results, updates `STATE.md`, sets the scope.
2. Dev fixes or extends the queue if needed (parallel with 3).
3. Paper runs both generators, rebuilds, edits the text the new numbers touch.
4. Reviewer audits the results and the paper diff; appends a verdict to
   `docs/REVIEW_LOG_2026-11.md`: **PASS** or **BLOCK**, findings ranked CRITICAL / HIGH / MEDIUM /
   LOW, each with file, line or cell, evidence, and the fix.
5. Paper or Dev resolves CRITICAL and HIGH in one round; the rest goes to the PM.
6. PM verifies the build, pushes, uploads to Overleaf, reports to the user. Nothing is uploaded on
   a BLOCK.

## Report format (every agent, every task)

```
DONE: what was completed, in two or three lines
CHANGED FILES: paths, and the commit hash if committed
NUMBERS: any result values read or produced, with their source file
OPEN ISSUES: what is unresolved or risky
NEEDS DECISION: anything only the PM or the user can decide (else "none")
```
