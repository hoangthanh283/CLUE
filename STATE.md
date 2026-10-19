## RESUME HERE (2026-10-15 00:30): tournament converged → P1 "live labels"; sliced-softmax control added
- **Tick 16:10 (2026-10-19) — R2d sliced_pseudo COMPLETE (22:10 2026-10-18) → KILL on CIL-CORD, null on Mixed; all R2 chains finished, GPU idle.** colar_r_sliced_pseudo k4_d50_r128, seeds 42/7/123: **CIL-CORD** 19.53/21.26/21.23 (mean 20.68) vs plain 20.38 → −2.2/+2.3/+0.8, mean +0.3 ≤ plain+1.0 → **KILL**; vs sliced-alone 25.39 → −8.2/−1.7/−4.3. **Mixed** 60.47/58.91/60.90 (mean 60.09) vs plain 58.76 → +3.0/−0.7/+1.7, mean +1.3 ≥ 1.0 but s7 negative → **null (fail-to-reach-bar)**; vs sliced 61.30 → −1.6/−3.8/+1.8. Pseudo refresh on top of sliced replay hurts in 5/6 cells: the deployable P1 arm is dead; sliced-alone (R2b, previous tick) stays the only arm with a scenario-level pass (CIL-CORD). R2 retry chains all COMPLETE: cil_cord_is3_seed42 DONE 07:27 10-18, mixed_is3_seed42 DONE 02:25 10-19, cil_cord_is3_seed7 DONE 04:33 10-19 (post token_features no_grad fix). r2_status pending 1/30 = cil_cord_cpfd_seed42 (FAILED 06:29, never re-queued). Nothing running; GPU 45 MiB. NEXT: (1) decide whether to re-run cil_cord_cpfd_seed42 or report CPFD n=2; (2) write R2b/R2d verdicts into docs/RESEARCH_PLAN_2026-10_next_mechanism.md P1 row + Reports sheet; (3) ROADMAP: P1 → sliced-alone only, pseudo arm closed.**
- **Tick 17:55 (2026-10-18) — R2b SLICED 3-SEED VERDICT + R2d sliced_pseudo 4/6 done.** R2b queue completed 22:53 2026-10-17. Sliced (colar_r_sliced, k4_d50_r128) vs plain CoLaR, seeds 42/7/123: **CIL-CORD** 27.70/22.93/25.53 (mean 25.39) vs plain 21.68/19.00/20.47 (mean 20.38) → +6.0/+3.9/+5.1, mean +5.0 ≥ 4.0, 3/3 sign → **PASS** (clears absolute bar 23.0; oracle 26.3). **Mixed** 62.04/62.72/59.13 (mean 61.30) vs plain 57.50/59.56/59.21 (mean 58.76) → +4.5/+3.2/−0.1, mean +2.5 ≥ 1.0 but s123 tie (−0.08) breaks 3/3 sign → **FAIL-to-reach-bar (null, not KILL)**; misses absolute bar 63.2 and oracle 64.4. Net under the 2026-10-15 rule: NOT a full PASS (split: CIL-CORD pass, Mixed null). R2d (colar_r_sliced_pseudo, auto-chained, started 07:29 2026-10-18): CIL-CORD s42 19.53/−18.7 (plain −2.2, sliced −8.2), CIL-CORD s7 21.26/−45.2 (+2.3 / −1.7), Mixed s42 60.47/−28.8 (+3.0 / −1.6), Mixed s7 58.91/−34.3 (−0.7 / −3.8) — pseudo refresh loses to sliced-alone in all 4 cells and sign-flips vs plain on both scenarios, so it cannot reach 3/3 regardless of s123 → on current evidence NULL. cil_cord s123 STARTED 17:50 (T0), mixed s123 after; queue ends ~23:30. Cells: results/{cil_cord,mixed}_colar_r_sliced{,_pseudo}_seed{42,7,123}_k4_d50_r128/metrics.json. r2_status pending 4/30 (CPFD/IS3 retry cells). GPU 2.7 GB. Not yet in the paper or the Reports sheet.**
- **Tick 13:24 (2026-10-17) — R2b mixed_colar_r_sliced_seed42 at T5 ep3 (last task; DONE ~13:50). After task 3 [26.46 69.90 74.36 96.35] vs plain s42 [25.73 57.37 71.29 95.33] (+0.7 / +12.5 / +3.1 / +1.0); after task 4 [9.15 71.19 73.16 32.17 60.47] vs plain [24.71 51.88 78.65 14.16 71.08] (−15.6 / +19.3 / −5.5 / +18.0 / −10.6) — gains are no longer uniform at task 4: FUNSD session 0 drops to 9.15 (plain 24.71) while tasks 1 and 3 gain ~18–19. Running AA after t4: sliced 49.2 vs plain 48.1 (+1.1). Final row decides; plain final AA 57.5, oracle 63.5, pseudo 59.5, Mixed PASS needs ≥ +1.0 mean over three seeds. No new DONE/FAIL; r2_status pending 6/30; 6 chain scripts armed. GPU 2.7 GB / RAM 5 GB.**
- **Tick 12:22 (2026-10-17) — R2b mixed_colar_r_sliced_seed42 at T3 ep4 (started 11:14, DONE ~14:30). T0 87.80 ✓ (= plain s42). After task 1 [21.65 79.09] vs plain s42 [1.13 86.70] (old +20.5, new −7.6); after task 2 [34.51 61.71 81.98] vs plain [25.96 52.01 82.04] (+8.6 / +9.7 / −0.1) — same shape as CIL-CORD s42: sliced replay CE protects old tasks at a small cost on the current one. One seed, no verdict. Plain s42 remaining rows for comparison: t3 [25.73 57.37 71.29 95.33], t4 [24.71 51.88 78.65 14.16 71.08], final [64.26 53.62 64.92 11.22 62.40 88.56] AA 57.5; oracle 63.5, pseudo 59.5. Next R2b START after this cell: cil_cord_colar_r_sliced_seed7 — its T0 must be 95.42 (= plain cil_cord s7 T0). **R2b SLICED ADJUDICATION RULE (from docs/RESEARCH_PLAN_2026-10_next_mechanism.md P1 row, 2026-10-15; no separate §0' text exists for sliced):** over seeds 42/7/123, PASS iff mean AA(colar_r_sliced) − mean AA(colar plain) ≥ 4.0 on CIL-CORD (plain 20.4±1.3: 21.7/19.0/20.5) AND ≥ 1.0 on Mixed (plain 58.75±1.10: 57.5/59.6/59.2), with the per-seed gain of the same sign on all three seeds (same consistency test r2_status applies to pseudo); KILL iff mean(sliced) ≤ mean(plain) + 1.0 on a scenario (mirror of r2_status `kill`); anything between = FAIL-to-reach-bar (null, not kill). Reference comparators per scenario: oracle 26.3±3.6 / 64.36±1.89, pseudo 20.0±4.1 / 59.5±0.0, pre-reg absolute bars 23.0 / 63.2, joint 67.5 (Mixed). Cells: results/{cil_cord,mixed}_colar_r_sliced_seed{42,7,123}_k4_d50_r128/metrics.json. So far CIL-CORD s42 +6.0 (27.70 vs 21.68). No new DONE/FAIL; r2_status pending 6/30 (retry cells); 6 chain scripts armed. GPU 2.7 GB / RAM 5 GB.**
- **Tick 11:20 (2026-10-17) — R2b cil_cord_colar_r_sliced_seed42 DONE 11:14: AA 27.70 / BWT −6.88, final row [8.21 25.99 47.49 54.84 1.94]. Seed-42 CIL-CORD comparison: plain 21.68/−15.9, pseudo 23.74/−12.6, oracle 26.78/−11.3, SLICED 27.70/−6.9 — the label-free sliced-softmax replay CE beats the true-label oracle on this seed and has the smallest BWT of the four. ONE SEED, NO VERDICT: P1 pre-reg asks ≥ 4 AA over plain on three seeds (+6.0 here); seeds 7/123 follow in R2b (cil_cord s7/s123 after the mixed cells). Do not write into the paper before n=3. mixed_colar_r_sliced_seed42 STARTED 11:14: T0 best val_f1 87.7987 = plain s42 T0 87.80 ✓; T0 ep8 at 11:20, DONE ~14:30. GPU 2.7 GB / RAM 6 GB; 6 chain scripts armed; r2_status pending 6/30 (retry cells).**
- **Tick 10:18 (2026-10-17) — R2b cil_cord_colar_r_sliced_seed42 at T3 ep5 (DONE ~13:00). T0 95.16 ✓ (= plain s42). After task 1 [70.60 48.37] vs plain/oracle/pseudo s42 [58.79 63.22] — divergence at task 1 is BY DESIGN: `sliced` is not a refresh (colar_r.py:17,139 — replay CE over each document's capture-time classes during the task, no relabelling, nothing logged in after_task), so the determinism reference does not apply beyond T0. Early read, one seed: old task +11.8, new task −14.9 at task 1; after task 2 [43.62 72.30 14.77]. No gate depends on it (P1 control, 6 cells). No new DONE/FAIL; 6 chain scripts armed. GPU 2.7 GB / RAM 5 GB. Note: scripts/r2_status.py does not cover colar_r_sliced / sliced_pseudo / dil_colar_e10 — a 20-line read-only sibling (r2b_status) would make the stop condition checkable from one command; not written (user's tool, ask first).**
- **Tick 09:16 (2026-10-17) — MAIN R2 QUEUE COMPLETE 08:50; R2b started 08:52. mixed_is3_seed123 DONE 08:50: AA 30.26 / BWT −33.8, final row [55.04 38.94 0.00 0.00 14.65 72.95]; every old task exactly 0.00 at every class-adding boundary, partial recovery of tasks 0/1/4 only in the last session (same shape as s7 [57.0 45.9 0 0 4.7 79.5]). IS3 Mixed n=2 = 30.72±0.65; IS3 CIL-CORD s123 2.0. CPFD n=2/n=2 at ≈0, IS3 n=2 Mixed / n=1 CIL-CORD collapsed — the two ports are fully characterised for decision (a)/(b); retry cells (6 remaining in r2_status) add only seed coverage. r2_queue.log "R2 QUEUE COMPLETE 08:50:49"; run_r2_queue.sh exited, 6 chain scripts still armed (r2b, r2c, retry, r2d, retry2, retry3). R2b: cil_cord_colar_r_sliced_seed42 STARTED 08:52 (log results/logs_vision/r2b_queue.log), T0 ep1-3 val_f1 90.34/91.00/92.55 bit-identical to plain s42 (determinism OK, T0 should end 95.16); T0 ep6 at 09:16. GPU 2.7 GB / RAM 5 GB.**
- **Tick 08:14 (2026-10-17) — mixed_is3_seed123 at T4 ep1 (DONE ~09:15). After task 2 [0.00 0.00 9.26], after task 3 [0.00 0.00 0.00 87.11]; prototypes 4→6→14→18 classes. Every old task is exactly 0.00 after each class-adding boundary, as on CIL-CORD s123 — the port-level single-head flaw reproduces on Mixed. All 7 chain scripts still armed (r2, r2b, r2c, retry, r2d, retry2, retry3). No new DONE/FAIL. GPU 3.2 GB / RAM 5 GB. Cells pending 7/30.**
- **Tick 07:12 (2026-10-17) — mixed_cpfd_seed123 DONE 06:49: AA 0.07 / BWT −17.4; T0 87.47 then every task 0.00 from the first boundary (one 0.41 blip on task 3 at the end). CPFD collapse now n=2 on Mixed (s7 0.0, s123 0.1) and n=2 on CIL-CORD (s7, s123 both 0.0) — seed- and scenario-independent, evidence for decision (a). Mixed comparators in r2_status now [0.0, 31.2]; Mixed verdict unchanged (KILL). mixed_is3_seed123 STARTED 06:49 (~2 h): T0 87.31, after task 1 [0.00 53.88] — old task wiped at the first class-adding boundary, same as CIL-CORD s123. p9 cut MOOT since 06:09 rebuild. docs/RESEARCH_OPS_LOOP.md has no stale "Mixed pending" text. GPU 3.2 GB / RAM 6 GB. Cells pending 7/30; queue after is3 s123: R2b ×6, R2c ×3, retry ×2, R2d ×6, retry2 ×2, retry3 ×1.**
- **Tick 06:09 (2026-10-17) — MIXED GATE 1 = KILL, WRITTEN INTO PAPER. mixed_colar_r_pseudo_seed123 DONE 05:40: AA 59.49 / BWT −28.2, final row [64.59 53.54 64.76 13.72 71.59 88.75]. r2_status Mixed: pseudo 59.51±0.04 (59.5/59.6/59.5) vs bar 63.2, plain 58.75±1.10, sign-agree=False, kill (≤ plain+1.0) → KILL; oracle 64.36±1.89 > pseudo OK. Pseudo relabel counts s123 7/0/0/36 tokens vs oracle 3231/2792/0/305; s42/s7 pseudo 0 everywhere. Paper: tab:ledger ‡ row widened to CIL-CORD / Mixed (20.0 / 59.5; plain 20.4 / 58.8; oracle 26.3 / 64.4; +5.9 / +5.6), caption ‡ updated, verdict comment rewritten (main.tex ~:319). Rebuilt: 10 pp, 0 undefined, no big Overfull; body now ENDS ON P8 (References start top of p9) — the pre-existing 2-line p9 overflow is gone, so the p9 cut decision is moot unless later edits push it back. ROADMAP item 0 closed. Queue: mixed_cpfd_seed123 STARTED 05:40 (GPU 3.9 GB / RAM 5 GB); loop continues until queue empty. Cells pending 8/30.**
- **Tick 05:37 (2026-10-17) — mixed_colar_r_pseudo_seed123 at T5 ep4 (DONE ~06:00). After task 3 [38.57 44.33 74.76 97.19]; after task 4 [46.98 27.12 76.06 23.66 58.06]. Refresh counts s123: 7/0/0/36 tokens relabelled (15/0/0/73 candidates) — nonzero only at 50 and 200 docs, vs oracle s123 3231/2792/0/305; s42/s7 pseudo all 0. Verdict next tick (KILL iff pseudo_s123 ≤ 60.20). No new DONE/FAIL. GPU 2.7 GB / RAM 6 GB. Cells pending 9/30.**
- **Tick 04:35 (2026-10-17) — mixed_colar_r_pseudo_seed123 at T3 ep5. After task 2 [13.80 54.42 81.57] (plain s123 [36.32 28.96 82.72], oracle s123 [20.34 72.07 80.30]); second refresh (100 docs) relabelled 0 tokens / 0 candidates — only the first refresh touched anything (7 tokens). Verdict due ~06:30; patch awaits pseudo values only. No new DONE/FAIL. GPU 2.7 GB / RAM 6 GB. Cells pending 9/30.**
- **Tick 03:33 (2026-10-17) — mixed_colar_r_oracle_seed123 DONE 03:01: AA 63.0 / BWT −15.2, final row [65.03 52.65 73.03 50.00 48.72 88.59]; refresh at 200 docs relabelled 305 tokens (s42 316, s7 316 — in range). Oracle Mixed n=3 = 64.36±1.89 (63.5/66.5/63.0) vs plain 58.75 (+5.6) vs bar 63.2 vs joint 67.5. mixed_colar_r_pseudo_seed123 STARTED 03:01: T0 84.91 ✓, after-task-1 [1.09 87.36] ✓ (bit-identical to plain/oracle s123); first refresh 7 tokens relabelled (15 candidates) — s42/s7 relabelled 0, still ≈0; at T2 ep4 at 03:33, DONE ~06:30. Verdict closes then: KILL iff pseudo_s123 ≤ 60.20, else FAIL. Scratchpad patch pre-filled with oracle/plain values; only {PSEUDO_MIXED} {PSEUDO_S123} {PSEUDO_SD} {VERDICT_MIXED} {TIME} remain. GPU 2.7 GB / RAM 6 GB. Cells pending 9/30.**
- **Tick 02:31 (2026-10-17) — mixed_colar_r_oracle_seed123 at T4 ep3 (started 00:40, due ~03:45): after task 3 = [59.56 54.99 68.66 96.04]; refresh counts 3231 / 2792 / 0 tokens (50/100/150 docs) — in range of s42 (3120/2640/0/316) and s7 (3307/2932/0/316); determinism row OK. Mixed Gate 1 PENDING 2/3 pseudo. GPU 2.7 GB, RAM 5 GB used. Scratchpad verdict patch verified present (1917 B). **Remaining queue map (for the stop condition):** all seven queue scripts are armed as waiting processes (run_r2_queue → r2b → r2c → retry → r2d → retry2 → retry3). Main R2 after oracle s123: mixed_colar_r_pseudo_seed123 (closes Mixed Gate 1, ~04:00–07:30), mixed_cpfd_seed123, mixed_is3_seed123 (3 cells). Then R2b colar_r_sliced × {cil_cord, mixed} × seeds 42/7/123 (6 cells, ~18 h); R2c dil_colar_e10 × 3 seeds (3 cells, headline disk-backing, ~8 h); retry cil_cord cpfd/is3 seed42 (2); R2d colar_r_sliced_pseudo × 2 scenarios × 3 seeds (6, ~18 h); retry2 mixed cpfd/is3 seed42 (2); retry3 cil_cord is3 seed7 (1). Total ≈ 23 cells ≈ 60–65 GPU-h → queue empty ~2026-10-19/20, so the loop (stop = verdict written AND queue empty) runs ~3 more days unless the user stops it at the Mixed verdict. **Seven of the 23 cells (~15 GPU-h) are CPFD/IS3 re-runs** (mixed s123 ×2, retry ×2, retry2 ×2, retry3 ×1) on ports whose collapse is already n=2/n=2 per scenario; under decision (a) the user can reclaim them by killing the retry/retry2/retry3 waiters (`pkill -f run_r2_retry`) — not done by me (user decision). Pending user: CPFD/IS3 (a)/(b), p9 cut, F9 mini-table, stop-at-verdict vs run-to-empty.
- **Tick 01:28 (2026-10-17) — mixed_colar_seed123 (plain) DONE 00:40: AA 59.21 / BWT −26.88 / FWT −77.8, final row [64.8 52.6 72.5 50.4 26.6 88.3]. Mixed plain now n=3 = 58.75±1.10 (57.5/59.6/59.2). mixed_colar_r_oracle_seed123 STARTED 00:40: after task 1 = [1.09 87.36] — BIT-IDENTICAL to plain seed 123 (determinism OK); first refresh at task 2 relabelled 2792 tokens in 100 docs; after task 2 = [20.34 72.07 80.30] (plain [36.32 28.96 82.72]); at T3 ep1, due ~03:45; pseudo seed 123 follows (~04:00–07:30) and closes Mixed Gate 1. Updated KILL threshold: KILL iff pseudo_s123 ≤ 60.20 (= plain_s123 59.21 + 0.99; pseudo s42+s7 sum 119.07, plain n=3 sum 176.27); otherwise FAIL (PASS impossible, seed-7 pseudo == plain). GPU 2.7 GB, RAM 5 GB used. Dry-run verdict patch written to the session scratchpad: /tmp/claude-1000/-mnt-DataDrive-Workspace-Master-HUST-Thesis-CLUE/f794f80b-3f83-42e7-a4dc-3402b9448067/scratchpad/mixed_verdict_patch.tex — three replacements (ledger row main.tex:281, caption ‡ sentence :263, verdict comment :319) with placeholders {PSEUDO_MIXED} {PLAIN_MIXED}=58.8 {ORACLE_MIXED} {VERDICT_MIXED} {PSEUDO_S123} {ORACLE_S123} {PSEUDO_SD} {ORACLE_SD} {TIME}; the verdict tick is fill-and-apply + rebuild + p9 check. (Scratchpad is session-scoped; if lost, the Tick 00:26 pre-read has the full spec.) Pending user: CPFD/IS3 (a)/(b), p9 cut, F9 mini-table.
- **Tick 00:26 (2026-10-17) — mixed_colar_seed123 (plain) at T4 ep3 (started 22:15; ~45 min/task from task 2 on, so DONE ~01:45, not 01:15): after task 3 = [55.06 31.40 78.76 96.79]; after-task-1 reference [1.09 87.36] unchanged. Oracle seed 123 not started. Mixed Gate 1 PENDING 2/3. GPU 2.7 GB, RAM 5 GB used. **Mixed Gate 1 pre-read (from scripts/r2_status.py gate()):** verdict = KILL if pseudo_mean ≤ plain_mean + 1.0, else FAIL unless (bar ≥ 63.2 AND same-sign on all three seeds AND ≥ comparators), else PASS. With pseudo s42/s7 = 59.51/59.56 (sum 119.07) and plain s42/s7 = 57.50/59.56 (sum 117.06): (i) **KILL** iff pseudo_s123 ≤ plain_s123 + 0.99 — the likely path given seed 7 was bit-identical to plain and seed 42 gained only +2.0; (ii) **FAIL** otherwise — PASS is impossible already: seed 7 pseudo − plain = 0.00 exactly, so neither all(>0) nor all(<0) holds and sign-agree is False whatever seed 123 does (the bar would additionally need pseudo_s123 ≥ 70.5 for a 63.2 mean); (iii) no PENDING once both seed-123 cells exist. Paper action on KILL or FAIL is the same: extend the existing ‡ ledger row (tab:ledger, "CoLaR-R pseudo (CIL-CORD, 3 seeds)") to "CIL-CORD and Mixed, 3 seeds each" with Best-AA cell "20.0 / 59.5 (plain 20.4 / 58.5; oracle 26.3 / 65.0)" — Mixed numbers to be replaced by the n=3 means — and the caption ‡ sentence to "CIL-CORD and Mixed, LayoutLMv3, three seeds; oracle = true-label refresh ceiling"; replace the 3-line GATE-1 VERDICT comment with a CIL-CORD + Mixed version; rebuild (watch the p9 overflow: the longer cell may add a line → the user's 2-line cut becomes 3); STATE.md ledger entry + ROADMAP item 0 → closed. Oracle n=3 Mixed mean (currently 65.0 n=2 vs joint 67.5) goes into the ‡ note as the true-label ceiling. Then the loop stops only when the queue (R2b/R2c/R2d/retries) is empty — or the user says to stop at the verdict. Pending user: CPFD/IS3 (a)/(b), p9 cut, F9 mini-table.
- **Tick 23:23 (2026-10-16) — mixed_colar_seed123 (plain) at T3 ep1 (started 22:15): after task 1 = [1.09 87.36] (DETERMINISM REFERENCE for mixed oracle/pseudo seed 123 — their after-task-1 rows must be bit-identical), after task 2 = [36.32 28.96 82.72]; due ~01:15, then mixed oracle seed 123. Mixed Gate 1 PENDING 2/3. GPU 2.7 GB, RAM 5 GB used. Loop-doc step 6 item 4 (supp skeleton): paper_cvpr/supp.tex builds (2 pp, 0 undefined; sections app:setup, app:extra). Added a COMMENTED-OUT draft paragraph "Continual-NER comparators (CPFD, IS3): port status" before \end{document}, marked `%% DRAFT — depends on user decision (a)/(b)`: CPFD CIL-CORD s7/s123 0.0/0.0 + Mixed s7 0.0; IS3 CIL-CORD s123 2.0 + Mixed s7 31.2; old tasks exactly 0.00 after every class-adding boundary; attributed to the single-growing-head adaptation (reference impls keep per-task classifiers). PDF unchanged until uncommented. IS3 has no refs.bib entry (CPFD = zhang2023cpfd) — add before uncommenting. Remaining step-6 items: Overleaf re-upload (user), p9 overflow cut (user). Pending user: CPFD/IS3 (a)/(b), p9 2-line cut, optional F9 mini-table.
- **Tick 22:21 (2026-10-16) — cil_cord_is3_seed123 DONE (22:15, 2.9 h): AA 2.0 / BWT −55.1 / FWT −94.2; matrix diagonal 93.0 / 28.9 / 32.4 / 66.2 / 10.0, every off-diagonal exactly 0.0 — all five CIL-CORD sessions add classes (prototypes 10→21→33→45→52) so no recovery session exists, unlike Mixed task 5. IS3 now n=1 CIL-CORD (2.0) + n=1 Mixed (31.2); diagnosis (is3.py:135 CE mask / :216 KD slice) holds on both scenarios. r2_status comparators for cil_cord now [None, 2.0]; KILL verdict unchanged. mixed_colar_seed123_k4_d50_r128 (plain) STARTED 22:15, T0 84.91, at T1 ep2 at tick — its after-task-1 row is the determinism reference for the oracle/pseudo seed-123 Mixed cells that follow. Mixed Gate 1 PENDING 2/3. GPU 2.7 GB, RAM 6 GB used. Loop-doc step 6 item 3 closed: the F9 inversion paragraph already exists as §7 "The ingredient is domain-discriminating" (main.tex:305) with every synthesis number; no inversion figure file exists; the optional 2×2 mini-table (Gaussian-head vs whole-doc replay × images vs documents) is left as a %% comment under the paragraph for the user to accept or drop against the page budget. Next GPU-free item: supplementary PDF skeleton. Pending user: CPFD/IS3 (a)/(b), kill-ahead (moot now — Mixed cells have started), p9 2-line cut.
- **Tick 21:18 (2026-10-16) — cil_cord_is3_seed123 at T3 ep9 (slower than the 2.5 h estimate; ~1 h more): after task 2 = {0: 0.00, 1: 0.00, 2: 32.39} — every old task exactly 0.00 at each class-adding boundary so far (prototypes 10→21→33), diagnosis holds. mixed_colar_seed123 not started. Mixed Gate 1 PENDING 2/3. GPU 3.2 GB, RAM 5 GB used. Loop-doc step 6 item 2 closed: the HGT `\vnum{--}` ledger cell was filled at Tick 02:15 and verified against disk now (results/hgt_ablation, DIL seed 42, 5-ep frozen-backbone: naive 38.65, ER 86.21, CUBER 39.80, HGT α=0/0.5/1 = 24.80/25.29/24.90 → cell "39.8 (own naive 38.7); HGT ≤25.3" is exact); no `\todo` call remains in main.tex or supp.tex (only the macro definition). Comment-only %% PROVENANCE note added under the ledger row; item struck from docs/RESEARCH_OPS_LOOP.md. Next GPU-free item: F9 inversion paragraph (SLCA vs CoLaR × images/documents). Pending user: CPFD/IS3 (a)/(b), kill-ahead, p9 2-line cut.
- **Tick 20:15 (2026-10-16) — cil_cord_is3_seed123 at T2 ep5 (started 19:10): T0 92.98; after task 1 = {0: 0.00, 1: 28.86} — task 0 reads exactly 0.00 at the first class-adding boundary, as the is3.py:135/216 CE-mask/KD-slice diagnosis predicts (prototypes 10 → 21). Mixed Gate 1 PENDING 2/3; mixed_colar_seed123 not yet started. GPU 3.2 GB, RAM 6 GB used. Loop-doc step 6 item 1 closed: the CoLaR headline was already reconciled 2026-10-15 (M3, RESULTS_SYNTHESIS): 87.6 = seed-42 10-ep grid run; 87.8±0.5 = 10-ep seeds 42/7/123 = 87.58/88.30/87.38 (log-sourced, R2c colar_e10 re-run restores metrics.json); 86.62±0.25 = 5-ep control (local results/dil_colar_seed42_k4_d50_r128 = 86.38 is that control, not the headline). Wrote a comment-only %% PROVENANCE note under the §7 "CoLaR at r=128" paragraph (no prose change) and struck the item from docs/RESEARCH_OPS_LOOP.md. Pending user: CPFD/IS3 (a)/(b), kill-ahead, p9 2-line cut.
- **Tick 19:12 (2026-10-16) — cil_cord_cpfd_seed123 DONE (19:10): AA 0.0 / BWT −23.7; T0 94.74, then F1=0.00 on every task from task 1 onward (thresholds computed for 13/25/37/49 old classes, so the pseudo-label path ran). CPFD now n=2 on CIL-CORD (s7, s123 both 0.0) and n=1 Mixed (0.0): collapse is seed- and scenario-independent. cil_cord_is3_seed123 STARTED 19:10 (T0 ep1 at tick). Mixed Gate 1 still PENDING 2/3 (needs mixed seed-123 plain/oracle/pseudo, after is3). GPU 2.6 GB, RAM 5 GB used. GPU-free scouting (not applied): body overflows 2 lines onto p9. Three longest §6–§7 paragraphs: "The ingredient is domain-discriminating" (main.tex:302, 1096 chars), "CoLaR, the proposed memory" (:304, 958 chars), "CoLaR at r=128 matches raw-exemplar replay" (:306, 815 chars); ledger `\resizebox{\linewidth}` (:266) already at column width, nothing to gain there. Recommended 2-line cut (user-owned prose, awaiting OK): in :304 drop the trailing clause "; implementation in \texttt{doccl/methods/colar.py} on top of a shared latent-replay base" (≈85 chars ≈ 1.5 col-lines; also a repo path in a review copy) — if still 1 line over, in :302 drop the parenthetical "(ER $76.2/54.8$, DER++ $76.5/52.8$)" or in :310 shorten "The seed-42 reading was promising (far $+1.8$ over uniform $86.4$ at $d{=}50$), so we pre-registered that single arm for confirmation." to "A promising seed-42 reading (far $+1.8$ at $d{=}50$) was pre-registered for confirmation.". Nothing killed or re-queued; cpfd.py/is3.py untouched.
- **Tick 18:09 (2026-10-16) — GATE 1, CIL-CORD: KILL (r2_status). `cil_cord_colar_r_pseudo_seed123` DONE 17:35: AA 20.6 / BWT −12.1 → pseudo n=3 = 23.7 / 15.6 / 20.6 = 20.0±4.1 vs bar 23.0, plain 20.4±1.3, sign-agree=False; oracle 26.3±3.6 > pseudo (OK).** Pseudo relabelled 12 / 0 / 0 / 1 tokens across the four boundaries (26 / 2 / 0 / 3 candidates) — the teacher is confident on ≈0 stale-O tokens, so self-refresh is a no-op; only true-label (oracle) refresh reaches the stale slot (+5.9 on the mean, > plain on every seed). Executed the KILL branch: (1) one row added to the falsification ledger `tab:ledger` (paper_cvpr/main.tex:278, "Label refresh (latent memory) / CoLaR-R pseudo (CIL-CORD, 3 seeds)‡ / 20.0 (plain 20.4; oracle 26.3)") plus the ‡ sentence in the caption; (2) the commented §7 CoLaR-R paragraph replaced by a 3-line GATE-1 VERDICT comment (prose recoverable from git history); (3) rebuilt — 10 pp, References start on p9 after a 2-line body overflow that was ALREADY present in the committed PDF before this tick (pre-existing page-budget item, not caused by the row), 0 Overfull, 0 undefined refs; (4) ROADMAP.md updated. No further GPU on label refresh for CIL-CORD. Mixed Gate 1 still needs seed 123 ×3 (plain/oracle/pseudo). **Queue note:** the main queue moved on to `cil_cord_cpfd_seed123` at 17:35 (then `cil_cord_is3_seed123`) BEFORE the Mixed seed-123 cells — ~5 GPU-h on two ports already shown broken; left running per the no-hand-kill rule, but killing them to reach the Mixed verdict ~5 h sooner is the user's call (same decision as CPFD/IS3 (a)/(b)). VRAM 3.0 GB, RAM 6 GB. Queue 13/30 (+3 reruns), no new FAILs.
- **Tick 17:07 (2026-10-16) — `cil_cord_colar_r_pseudo_seed123` on task 3/5 (ep5): after task 1 [71.72 49.88] bit-identical to plain seed 123 (required); 12 tokens relabelled at the first boundary (26 candidates; seeds 42/7 relabelled 1 / 2), 0 at the second (2 candidates); after task 2 [57.5 64.1 1.1] vs plain [63.1 63.1 1.8].** Gate 1 CIL-CORD still PENDING until this cell is DONE (~18:00); r2_status prints the verdict then. Pre-read of the KILL procedure (docs/RESEARCH_OPS_LOOP.md step 4): on FAIL/KILL add CoLaR-R as one row in the falsification ledger `tab:ledger` (paper_cvpr/main.tex ~line 271-279, columns Family / Instance / Best AA / Why it fails) with the one-line mechanism reading ("teacher confident on ~0 stale-O tokens → pseudo refresh is a no-op; oracle refresh +5.9 shows the slot is reachable only with true labels"), keep the diagnostic framing, no further GPU on label refresh; the commented §7 GATE-1 DRAFT paragraph stays commented (delete or shrink it to the verdict line), and the `\vnum{--}` placeholders are the separate HGT item (loop-doc step 6), untouched by the verdict. Caveat for the row: the ledger caption says "DIL/LayoutLMv3" and CoLaR-R ran on CIL-CORD/Mixed — add a `$^\ddagger$` footnote to the caption ("CIL-CORD, 3 seeds") rather than silently mixing scenarios. VRAM 2.7 GB, RAM 6 GB. Queue 14/30 (+3 reruns), no new FAILs. **CPFD / IS3 port decisions pending with the user** (recommendation (a)).
- **Tick 16:05 (2026-10-16) — `cil_cord_colar_r_oracle_seed123` DONE 15:37: AA 29.6 / BWT −0.8 → CIL-CORD oracle is n=3: 26.8 / 22.5 / 29.6 = 26.3±3.6, above the 23.0 bar on the mean and > plain on every seed (+5.1 / +3.5 / +9.1).** Final row [63.0 33.6 22.1 29.0 0.3]; refresh counts 74 / 943 / 1315 / 1110 tokens (same regime as seeds 42/7). Last-session diagonal 0.35 — the oracle plasticity loss now reproduces on all three seeds (s42 3.8, s7 0.3, s123 0.35): refreshing the bank with true labels protects the stale slots but the final session barely trains. This is the strongest oracle seed (BWT −0.8 ≈ zero forgetting). GATE oracle is a ceiling statement, not the pre-reg pass criterion; Gate 1 (pseudo vs bar) still PENDING: `cil_cord_colar_r_pseudo_seed123` started 15:37 (no boundary yet; first row ~16:20, DONE ~18:00). For reference the pseudo arm relabelled 1 (s42) and 2 (s7) tokens at the first boundary — expect ~0-5 again. VRAM 2.7 GB, RAM 5 GB. Queue 14/30 (+3 reruns), no new FAILs. Paper: "oracle n=3" comment lines added after "plain n=3" (comment-only). **CPFD / IS3 port decisions pending with the user** (recommendation (a)).
- **Tick 15:03 (2026-10-16) — `cil_cord_colar_r_oracle_seed123` on task 3/5 (ep5): refresh counts and determinism check both in range.** After task 1 [71.72 49.88] — bit-identical to plain seed 123 (refresh is in after_task), as required. Refresh relabel counts 74 tokens after task 1 (seeds 42/7: 101 / 66) and 943 after task 2 (760 / 944) — same regime. After task 2 [58.4 72.0 1.5] vs plain [63.1 63.1 1.8]: the refreshed slot (task 1) gains +8.8, task 0 gives back −4.7, same redistributive pattern as the other seeds (s42 after task 2 [41.1 70.5 13.6], s7 [14.7 34.7 32.1]). Due ~16:00; then `cil_cord_colar_r_pseudo_seed123` (closes Gate 1 on CIL-CORD, ~18:30). VRAM 2.7 GB, RAM 5 GB. Queue 15/30 (+3 reruns), no new FAILs. No paper edit this tick. **CPFD / IS3 port decisions pending with the user** (recommendation (a)).
- **Tick 14:00 (2026-10-16) — `cil_cord_colar_seed123` DONE 13:32: AA 20.5 / BWT −13.8 / FWT −94.1 → CIL-CORD plain is n=3: 21.7 / 19.0 / 20.5 = 20.4±1.3.** Final row [9.8 40.5 36.3 12.4 3.2]; after task 3 [45.3 29.0 23.4 6.6]. The three-seed plain mean sits below the 23.0 bar and at the naive/best-prior level (19.0), as expected for the no-refresh arm. `cil_cord_colar_r_oracle_seed123` started 13:32 — T0 trajectory bit-identical to plain seed 123 (89.1→96.0 by ep6, same seed), no refresh yet (first refresh lands after task 1; seeds 42/7 relabelled 101 and 66 tokens there, then 760 / 944). Gate 1 CIL-CORD: PENDING (3/3 plain, 2/3 pseudo) — closes with pseudo seed 123, two cells from now (~19:00). VRAM 2.7 GB, RAM 5 GB. Queue 15/30 (+3 reruns), no new FAILs. Paper: one comment line "plain n=3" added after the CIL-CORD n=2 summary (comment-only). **CPFD / IS3 port decisions pending with the user** (recommendation (a)).
- **Tick 12:58 (2026-10-16) — `cil_cord_colar_seed123` (plain, third CIL-CORD seed) on task 3/6 (ep3): T0 96.1; after task 1 [71.7 49.9]; after task 2 [63.1 63.1 1.8].** Within the seed spread so far (s42 after task 2 [47.1 60.6 5.9], s7 [36.6 58.4 25.5]) — task-0 retention is the highest of the three seeds, the task-2 diagonal (1.8) the lowest; the session-2 diagonal is seed-fragile on all three (5.9 / 25.5 / 1.8). Due ~14:00; cil_cord plain becomes n=3 then; Gate 1 on CIL-CORD still waits for oracle/pseudo seed 123 (next two cells). VRAM 2.7 GB, RAM 5 GB. Queue 16/30 (+3 reruns), no new FAILs. GPU-free: "Comparator decision needed" summary added under the paper Comparators note (comment-only). **Both NER-CL port decisions (CPFD, IS3) pending with the user** — recommendation (a) drop + supplement disclosure.
- **Tick 11:55 (2026-10-16) — `mixed_is3_seed7` DONE 11:35: AA 31.2 / BWT −31.3 / FWT −88.8; old tasks read exactly 0.00 after every class-adding boundary; `cil_cord_colar_seed123` started 11:35 (T0 val F1 94.7 at ep4, normal).** IS3 rows: after task 3 [0 0 0 88.3]; after task 4 [0 0 0 0 25.6]; final [57.0 45.9 0.0 0.0 4.7 79.5] — tasks 0/1 come back only at task 5, which adds no classes (prototypes stayed at 22 after tasks 4 and 5). Read-only diagnosis of `doccl/methods/is3.py` (not edited): for non-first tasks `ce_mask = valid & (labels != 0)` (line 135) → CE sees new-entity tokens only, and `_kd_distill` softmaxes `flat_logits[distill_mask, :n_old]` (line 216) → KD constrains the old rows only. The new head rows therefore get no negative signal on O or relabelled-old tokens, their logits win the argmax on every token, and old classes are never predicted (F1 exactly 0). When n_old == n_total (task 5) the old-class tokens fall into ce_mask and the head relearns them — matching the recovery. The prototype CE over C_total (lines 223-227, weight 0.1) is too weak to counter this, and it feeds L2-normed prototypes to a head trained on raw features (scale mismatch, secondary). This is a port-level flaw in the single-growing-head adaptation (the reference uses one classifier per task), not the encoder bug fixed earlier. **IS3 decision now joins the CPFD decision for the user:** (a) drop both NER-CL ports and disclose in the supplement, or (b) fix the IS3 CE mask (include O tokens in CE for the new rows, as the reference's per-task classifier effectively does) and rerun 3 seeds × 2 scenarios (~18 GPU-h). Result dir left in place. VRAM 2.7 GB, RAM 5 GB. Queue 16/30 (+3 reruns), no new FAILs, gates PENDING (2/3 seeds; seed-123 cells now running). GPU-free: "IS3 post-fix" entry under the paper Comparators note updated with the final rows and the diagnosis (comment-only).
- **Tick 10:53 (2026-10-16) — `mixed_is3_seed7` on task 3/6 (cord_s0, ep8/10, val F1 88.3): the encoder trains, but IS3 wipes every old task to exactly 0.00 at each boundary and the sparse sroie session only reaches 9.3.** Rows: T0 85.9; after task 1 [0.00 55.10]; after task 2 [0.00 0.00 9.25] (T2 val F1 peaked 9.2, early-stopped ep4); T3 val F1 77.6→88.3 by ep6. Not a crash — old-task F1 exactly 0 while the current task learns is a classifier-side wipe (prototype/head realignment after expansion), to inspect once the run finishes; not killed, is3.py untouched. VRAM 3.2 GB, RAM 6 GB. Queue 17/30 (+3 reruns), no new FAILs, gates PENDING (2/3 seeds). GPU-free: 3-line "IS3 post-fix" entry added under the paper Comparators note (comment-only). **CPFD decision pending with the user** (recommendation (a) unchanged).
- **Tick 09:51 (2026-10-16) — `mixed_cpfd_seed7` DONE 09:22: AA 0.0 / BWT −30.2 / FWT −88.8 — second CPFD collapse confirmed end-to-end; `mixed_is3_seed7` (first post-fix IS3 in the main queue) started 09:22 and its encoder trains.** CPFD matrix rows: T0 86.3; after task 1 [2.9 9.0]; after task 2 [0.0 0.4 55.7]; after tasks 3/4/5 all 0.00 (the dense sroie/cord sessions also collapse once the head reaches 15+ old classes). Result dir left in place (`results/mixed_cpfd_seed7/`), not quarantined — the CPFD call is the user's. IS3 post-fix check: T0 val F1 81.8→85.9 (best, early-stop ep5), After task 0 = 85.90 — vs 69.6 for the frozen-encoder run → the `token_features` fix works and the encoder trains. But After task 1 = [0.00 55.10] (task 0 wiped at the first boundary) and T2 val F1 is only 6.4/9.2/8.5 at ep1-3 (plain colar T2 diag ≈ 82) — IS3 is converging badly on the sroie session; watch, do not intervene this tick. VRAM 3.2 GB, RAM 5 GB. GPU-free: 5-line "CPFD status (09:51)" entry added under the paper Comparators note (comment-only). Queue 17/30 (+3 reruns), no new FAILs. Gates still PENDING (2/3 seeds everywhere). **CPFD decision pending with the user** — recommendation unchanged: (a) drop + supplement disclosure.
- **Tick 08:49 (2026-10-16) — `mixed_cpfd_seed7` (started 07:47, first post-fix CPFD on Mixed) is degrading the same way: SECOND CPFD collapse, scenario-independent.** T0 86.3 (plain 88.8); after task 1 [2.9 9.0] — the funsd_s1 diagonal is 9.0 vs plain 89.8 and funsd_s0 drops to 2.9; after task 2 [0.0 0.4 55.7] — dense CORD still learns (55.7 vs plain 81.5) while both sparse FUNSD sessions are at ~0; pbar CE prints 0.000 from task 1 on. Not the exact 0.00-everywhere of CIL-CORD seed 7 but the identical signature as the invalid Mixed seed-42 run (sparse sessions collapse, dense sessions learn, CE≈0). 'old classes' = 5/7/15 = pre-expansion head sizes → the teacher fix is in; this is the second problem reproducing on a second scenario and seed. **Not killed** (per tick plan — ~3 h to finish, settles that the bug is scenario-independent). Together with the CPU probe (classif_adaptive_factor ruled out when the teacher is confident), the residual suspects are the O-token KD and the attention MSE. VRAM 3.8 GB (teacher loaded; under the 4.8 GB note threshold), RAM 6 GB. GPU-free: §7 draft caveat now says all three Mixed arms are n=2 (pseudo 59.5/59.6, 59.5±0.1). Queue 18/30 (+3 reruns), no new FAILs. **CPFD decision still pending with the user** — recommendation unchanged: (a) drop CPFD with a supplement disclosure; the two post-fix runs now document a reproducible, scenario-independent non-convergence of the port.
- **Tick 07:47 (2026-10-16) — `mixed_colar_r_pseudo_seed7` finished 07:47: AA 59.6 / BWT −23.8 — BIT-IDENTICAL to plain seed 7.** 0 candidates / 0 relabels at all four boundaries (seed 42: 0/0/0/84 tok), every "After task" row equals plain to the digit, final row [63.1 54.9 75.1 25.9 50.4 88.0]. Pseudo gain on this seed is exactly 0.0 (seed 42: +2.0, from the last boundary's 84 relabels). Two-seed Mixed: plain 58.5±1.5, **pseudo 59.5±0.1**, oracle 65.0±2.1; oracle > pseudo on both seeds. Pre-reg reading: pseudo mean ≤ plain + 1.0 → **heading to KILL on Mixed as on CIL-CORD**, and the mechanism null is the cause (the teacher is confident on ~0 stale-O tokens, so the pseudo refresh has nothing to relabel) — gate stays PENDING until seed 123. Recorded as "Pseudo seed 7" lines in the paper GATE-1 DRAFT comment block (comment-only). Next in queue: mixed_cpfd_seed7 (not started at 07:47; watch for a second CE 0.000 / F1 0 collapse at task 1 — record, do not kill) then mixed_is3_seed7 (first post-fix IS3). No new FAILs; CPFD decision pending with the user. GPU 2.7 GB, RAM 5 GB. Queue 18/30 (+3 reruns).
- **Tick 06:45 (2026-10-16) — no new DONE cells, no new FAILs; pseudo seed 7 bit-identical to plain so far.** `mixed_colar_r_pseudo_seed7` at T3 ep4 (started 05:23, due ~08:00): 0 candidates / 0 relabels at both boundaries so far, and the rows match plain seed 7 to the digit (after task 1 [0.00 89.75], after task 2 [1.16 55.34 81.50]) — determinism reconfirmed on a second seed; the first non-zero candidate boundary (seed 42: 84 relabels at cord_s1 → funsd_revisit only) is still ahead. GPU-free: §7 draft header caveat in `paper_cvpr/main.tex` now says Mixed oracle is n=2 (63.5/66.5, 65.0±2.1 vs plain 58.5±1.5; both seeds above the 63.2 bar), Mixed pseudo still n=1; comment-only. Next in queue: mixed_cpfd_seed7 (watch for a second CE 0.000 / F1 0 collapse at task 1 — record as determinism evidence, do not kill) then mixed_is3_seed7 (first post-fix IS3 in the main queue — T0 val F1 must head to ~94). CPFD decision pending with the user. GPU 2.7 GB, RAM 5 GB. Queue 19/30 (+3 reruns).
- **Tick 05:43 (2026-10-16) — `mixed_colar_r_oracle_seed7` DONE 05:23: AA 66.5 / BWT −17.7, final row [66.0 53.9 76.2 78.4 34.8 89.9]** (seed 42: 63.5 / −15.1, [65.6 54.2 78.1 38.6 54.9 89.8]); two-seed Mixed oracle **65.0±2.1 vs plain 58.5±1.5** (+6.1 / +6.9 per seed), both seeds above the 63.2 bar, ER 60.3 and best-tuned 62.2. Refresh 3307/2932/0/316 tok (seed 42: 3120/2640/0/316 — the cord_s1 refresh relabels the identical 316 tokens on both seeds). cord_s0 (stale-O slot) 25.9 → 78.4 after the cord_s1 refresh (seed 42: 11.2 → 38.6) — the mechanism target is where the gain lands on both seeds. **cord_s1 anomaly does NOT reproduce as oracle-specific**: oracle diag 39.4 vs plain 32.2 on seed 7 (seed 42: 12.8 vs 71.1) → the seed-42 gap is a seed-specific early-stop event, not a refresh side-effect; the one slot oracle loses on seed 7 is final cord_s1 34.8 vs plain 50.4 (same "retention up, last-session plasticity down" pattern as CIL-CORD). Recorded as "Oracle seed 7" lines in the paper GATE-1 DRAFT comment block (comment-only). Mixed gates still PENDING (2/3 plain, 2/3 oracle, 1/3 pseudo). **`mixed_colar_r_pseudo_seed7` started 05:23** (expect 0 candidates at the first three boundaries, as seed 42). No new FAILs; CPFD decision pending with the user. GPU 2.7 GB, RAM 5 GB. Queue 19/30 (+3 reruns).
- **Tick 04:41 (2026-10-16) — no new DONE cells, no new FAILs; oracle seed 7 tracking seed 42.** `mixed_colar_r_oracle_seed7` at T3 ep5 (started 02:55, due ~06:10): refreshes 3307 / 2932 tok at the first two boundaries (seed 42: 3120 / 2640); after task 2 [38.7 66.6 83.3] vs oracle seed 42 [30.2 69.9 78.1] and plain seed 7 [1.2 55.3 81.5] — the funsd_s1 refresh lifts funsd_s0 by +37 and funsd_s1 by +11 over plain on this seed too (seed 42: +4 / +18). cord_s1 diagonal (seed-42 anomaly 12.8) not reached yet — check next tick. GPU-free: §7 draft header caveat in `paper_cvpr/main.tex` now says Mixed plain is n=2 (57.5/59.6, 58.5±1.5), Mixed oracle/pseudo still n=1; bodies and `\vnum{--}` untouched, comment-only. CPFD decision pending with the user. GPU 2.7 GB, RAM 6 GB. Queue 20/30 (+3 reruns).
- **Tick 03:39 (2026-10-16) — `mixed_colar_seed7` DONE 02:55: AA 59.6 / BWT −23.8, final row [63.1 54.9 75.1 25.9 50.4 88.0]** (seed 42: 57.5 / −33.3, [64.3 53.6 64.9 11.2 62.4 88.6]); two-seed Mixed plain 58.5±1.5. Same shape as seed 42 — cord_s0 (the within-CORD CIL step) is the weakest slot on both seeds (25.9 / 11.2), FUNSD/SROIE domain steps hold at 50–75; the funsd_s0 transient (88.8→0.0→1.2→29.4→25.3→63.1) recovered by replay as in seed 42. Recorded as a "Mixed seed 7" line in the paper GATE-1 DRAFT comment block (comment-only). **`mixed_colar_r_oracle_seed7` started 02:55**, at T2 ep6 at 03:39; first boundary refreshed 50 docs / 3307 tokens (seed 42: 3120) — oracle refresh counts consistent across seeds. Mixed gates still PENDING (2/3 plain, 1/3 oracle, 1/3 pseudo). No new FAILs; CPFD decision still pending with the user. GPU 2.7 GB, RAM 6 GB. Queue 20/30 (+3 reruns).
- **Tick 02:37 (2026-10-16) — no new DONE cells, no new FAILs; paper comment block updated.** `mixed_colar_seed7` at T4 ep4 (started 00:30, ~on schedule for ~03:45): after task 3 [29.4 52.1 70.2 96.3] vs seed-42 plain [25.7 57.4 71.3 95.3] — funsd_s0 recovered from 1.2 to 29.4 at the CORD→SROIE boundary exactly as seed 42 did (26.0→25.7 there), rows now within ±5 pp of seed 42; the seed-7 "collapse" of funsd_s0 after task 1 was transient. GPU-free: added to `paper_cvpr/main.tex` GATE-1 DRAFT comment block the two-line CIL-CORD n=2 summary (plain 20.3±1.9, oracle 24.6±3.0, pseudo 19.7±5.8 sign flip; pseudo heading to KILL, oracle clears 23.0 on the two-seed mean only) and the CPFD-probe line under Comparators (classif_adaptive_factor ruled out; KD / attention-MSE remain suspects). Comment-only, no rebuild. CPFD decision still pending with the user. GPU 2.7 GB, RAM 5 GB. Queue 21/30 (+3 reruns).
- **Tick 01:33 (2026-10-16) — no new DONE cells; CPFD CE-scaling hypothesis probed on CPU and NOT confirmed.** `mixed_colar_seed7` running (T3 ep2 at 01:33; started 00:30): After task 0 [88.8]; after task 1 [0.0 89.8]; after task 2 [1.2 55.3 81.5] vs seed-42 plain after task 2 [26.0 52.0 82.0] — funsd_s0 is lower this seed (1.2 vs 26.0), funsd_s1/cord_s0 match; seed 42 recovered funsd_s0 to 64.3 at the end via replay, so judge on the final row only (due ~03:45). GPU-free probe `tests/methods/test_cpfd_sparse_signal.py` (2 CPU tests, both pass): with `_n_old=13`, thresholds at the 0.001 floor, 190 confident-O tokens (teacher entropy ~4e-4) + 12 new-class tokens with a uniform student, the per-token CE on the new tokens after `classif_adaptive_factor` scaling is exactly log(25)=3.22 (factor 1.0) — **the factor does not crush new-entity CE when the teacher is confident, so it is not the smoking gun.** Second test documents the ceiling: a batch whose O tokens the teacher is uncertain about gets factor 0 (`adaptive_ce_min: 0.0`) → zero CE on its new entities; with per-class median thresholds ~50 % of O tokens are confident by construction, so the average factor is ~0.5, not 0 — consistent with "not the cause of a total collapse". Residual suspects for the AA 0.0 run (T0 94.0, then 0.00 everywhere incl. task 0 — i.e. the old head is destroyed too, which CE scaling cannot do): the O-token KD (`kl_div` over `n_old` logits, batchmean) and the three-view attention MSE (`distill_weight 2.0 × sqrt((n_old-1)/(n_new))`), both of which touch every token. cpfd.py untouched this tick — user decision (a)/(b) from tick 00:20 still pending. GPU 2.7 GB, RAM 5 GB. Queue 21/30 pending (+3 reruns), no new FAILs.
- **Tick 00:20 (2026-10-16) — BOTH continual-NER comparators invalid again on CIL-CORD; one bug found and fixed, one open. ESCALATION for the user below.** (1) `cil_cord_cpfd_seed7` (first post-fix CPFD run) finished 00:12 with **AA 0.0**: task 0 F1 94.0, then every F1 exactly 0.00 from task 1 on, including task 0 itself; val_f1 0.0000 at epoch 1 of every later task; mean train loss ~0.007; pbar CE prints 0.000 from the first steps of task 1. The teacher fix *did* take (old-class counts 13/25/37/49 = pre-expansion head sizes), so this is a second, different problem. The invalid Mixed seed-42 run showed the same CE≈0 and collapsed only on the sparse FUNSD sessions while learning dense SROIE/CORD, so the CE path (`classif_adaptive_factor` × per-token CE averaged over a valid set dominated by pseudo-labelled O tokens) is the leading suspect on sparse sessions — every CIL-CORD session is sparse (6 classes). **Not root-caused; time-boxed.** Quarantined to `results/_invalid/cil_cord_cpfd_seed7_all_zero_collapse/` (WHY.md inside). The queued `cil_cord_cpfd_seed42` retry (run_r2_retry.sh) will run the same code — left in place as a determinism check (~1.5 h), not as a result. (2) `cil_cord_is3_seed7` was at T0 val F1 69.7 after 7 epochs (naive/CoLaR/CPFD: 94–95). Root cause found: `LayoutLMv3Wrapper.token_features` was `@torch.no_grad()` (written for LCA's read-only statistics) and IS3 builds its *student* forward from it → the encoder never got gradients, IS3 trained only its head. Only the LayoutLMv3 wrapper had the decorator; all other callers (LCA, HGT, CoLaR++, IS3 prototypes/teacher) already run under their own no_grad. **AGENT FIX:** decorator removed; integration test `test_layoutlmv3_token_features_is_differentiable` (passes, real weights) + unit test `test_first_task_step_updates_the_encoder_not_only_the_head`; 23/23 unit tests green. Run killed 00:30 and quarantined (`_invalid/cil_cord_is3_seed7_encoder_frozen/`); re-queued in `scripts/run_r2_retry3.sh` (pid 1478668, waits 'R2 RETRY2 COMPLETE'). The queued cil_cord/mixed IS3 seed-42 retries are fresh processes and pick up the fix. This also means the mixed IS3 seed-42 T0 F1 48.7 had two causes (O-excluded CE + frozen encoder). **Queue moved to `mixed_colar_seed7` at 00:30.** GPU 45 MB (dataset build), RAM 6 GB. Queue 20/30 (+3 reruns). **User decision needed:** CPFD is now 0-for-2 valid cells; either (a) drop CPFD as a comparator and keep IS3 (post-fix) + the pre-registered ER/DER++/best-tuned bars, or (b) spend a GPU-free session root-causing the CPFD CE path (suspect: `classif_adaptive_factor` scaling + O-dominated mean) and re-queue. Recommendation: (a) for the CVPR deadline, with CPFD disclosed as "port did not converge on sparse sessions" in the supplement; the Gate-1 pre-reg says "≥ CPFD/IS3" and the status script already treats a missing comparator as non-blocking.
- **Tick 23:18 — `cil_cord_colar_r_pseudo_seed7` DONE 22:41: AA 15.6 / BWT −43.3, final row [4.8 8.0 8.9 51.7 4.4], diag [95.4 81.4 39.2 30.5 4.4], refresh 4→2 / 0→0 / 2→0 / 4→2 cand→tok:** −3.4 vs plain seed 7 (19.0); seed 42 was +2.0 → **pseudo-vs-plain sign flips across seeds** (two-seed pseudo 19.7±5.8 vs plain 20.3±1.9). Oracle 22.5 > pseudo 15.6: gate order holds at n=2 (no violation). Pre-reg reading at n=2: pseudo mean ≤ plain + 1.0 and sign disagrees → the pseudo arm is heading to KILL unless seed 123 is anomalous; no gate action until n=3 (seed 123 pseudo due ~2026-10-17). Not bit-identical to plain: 2 relabels at boundary 1 diverge the run from task 2 on, so CIL-CORD single-seed spread is relabel sensitivity on a deterministic run. Mechanism null holds (≤ 4 candidates per boundary). **`cil_cord_cpfd_seed7` started 22:41 — first post-fix CPFD run; task 2 log says "thresholds computed for 13 old classes" = pre-expansion head (post-expansion would be 25): fix confirmed in a real run.** Task 1 F1 94.0; VRAM 3.8 GB with the teacher (under the 5 GB cap). GPU 3.8 GB, RAM 5 GB. Queue 21/30 (+2 reruns), no new failures.
- **Tick 22:16 — `cil_cord_colar_r_pseudo_seed7` still running, task 5/5 since 22:11 (due ~22:40); refresh so far 4→2 / 0→0 / 2→0 (candidates→relabelled tok; seed 42: 1/0/0/2):** NOT bit-identical to plain seed 7 this time — the 2 tokens relabelled at boundary 1 change the replay targets and the run diverges from task 2 on (after task 2: 34.9/51.5/39.2 vs plain 36.6/58.4/25.5; after task 3: 19.2/17.9/25.7/30.5 vs plain 11.0/18.8/12.8/27.4). Consistent with the mixed finding (0 relabels → bit-identical): the runs are deterministic and a handful of relabels is enough to move single-seed rows by 5–15 pp, which is the CIL-CORD "spread". Mechanism null holds (≤2 candidates at 3 boundaries). Nothing DONE this tick; GPU-free item skipped (cell still running). `cil_cord_cpfd_seed7` (first post-fix CPFD) has not started. GPU 2.7 GB, RAM 5 GB. Queue 22/30 (+2 reruns), no new failures.
- **Tick 21:14 — `cil_cord_colar_r_pseudo_seed7` on task 3/5 (ep7), due ~22:00; first boundary 4 candidates → 2 tokens relabelled (seed 42: 1 tok):** After task 1 row 38.79/81.43 = plain seed 7, as expected (refresh acts from the next task's replay on); the bit-identity test is the after-task-2 row (plain 36.59/58.36/25.54) next tick. GPU-free item done: the commented §7 paragraph now states the CIL-CORD oracle at n=2 (lifts every seed so far, 21.7→26.8 and 19.0→22.5, two-seed 20.3→24.6) with `\vnum{--}` kept for the three-seed means; header caveat updated (Mixed n=1, seed 7 alone below the 23.0 bar). Comment-only, no rebuild. GPU 2.7 GB, RAM 6 GB. Queue 22/30 (+2 reruns), no new failures.
- **Tick 20:12 — `cil_cord_colar_r_oracle_seed7` DONE 19:37: AA 22.5 / BWT −25.9, final row [33.8 21.9 12.4 44.1 0.3], diag [95.4 81.4 32.1 6.7 0.3], refresh 66/944/1366/1269 tok:** +3.5 over plain seed 7 (19.0); seed 42 was +5.1 → oracle > plain holds at n=2 (oracle 24.6±3.0 vs plain 20.3±1.9). Seed 7 alone is below the 23.0 bar, the two-seed mean clears it. Same shape as seed 42: retention of sessions 1 and 4 (33.8 / 44.1 vs plain 10.8 / 9.4) bought with plasticity on the last two sessions (diag 6.7 / 0.3 vs plain 27.3 / 14.6); mid-stream it trailed plain, so only final AA is interpretable. Open item: last-session diagonal 0.3 is the lowest of any arm — refreshed-buffer replay crowding out a 50-doc final session; a limitations sentence if seed 123 repeats it. Recorded in the GATE-1 comment block (comment only, no rebuild). `cil_cord_colar_r_pseudo_seed7` started 19:37, on task 1/5 ep9 at 20:12, due ~22:00 (seed-42 pseudo relabelled 1/0/0/2 tok). GPU 2.7 GB, RAM 6 GB. Queue 22/30 (+2 reruns), no new failures.
- **Tick 19:10 — `cil_cord_colar_r_oracle_seed7` on task 4/5 (ep4), due ~20:00; refresh 66/944 tok at the first two boundaries (seed 42: 101/760):** After task 2 row [14.7 34.7 32.1] vs plain seed 7 [36.6 58.4 25.5] — the oracle refresh is *behind* plain on both old sessions mid-stream (seed-42 oracle after task 2 [41.1 70.5 13.6] vs plain [47.1 60.6 5.9] was mixed too; its +5.1 AA came from the final row, task 0: 45.6 vs 6.2). Judge on final AA only. "After task 1" rows are identical to plain in both seeds — the refresh relabels the buffer in `after_task`, so its effect appears only from the next task's replay on. Paper GATE-1 comment block gained the seed-7 oracle progress line and a list of the CIL-CORD seed-7 cells still needed (oracle, pseudo, then cpfd/is3 seed 7 = first post-fix CPFD/IS3 runs — verify 'old classes' == pre-expansion head size and non-zero task-2 diagonal). Comment-only, no rebuild. GPU 2.7 GB, RAM 5 GB. Queue 23/30 (+2 reruns), no new failures.
- **Tick 18:08 — `cil_cord_colar_seed7` DONE 17:28: AA 19.0 / BWT −37.3, final row [10.8 30.5 29.7 9.4 14.6], diag [95.4 81.4 25.5 27.3 14.6]:** two-seed plain mean 20.3±1.9 = naive 19.0, below the 23.0 bar (seed 42: 21.7 / −15.9 / [6.2 54.5 30.4 14.1 3.2]). Seed 7 learns sessions 3–5 much better (diag 25.5/27.3/14.6 vs 5.9/4.5/3.2) but forgets far more (BWT −37 vs −16) — same AA band, opposite stability/plasticity mix, so CIL-CORD single-seed rows are not row-interpretable; the "last session 3.2 in all arms" observation is seed-42-specific (seed 7: 14.6), dropped as a mechanism. Recorded as a "seed 7" line in the GATE-1 DRAFT comment block (comment only, no rebuild). `cil_cord_colar_r_oracle_seed7` started 17:28, on task 1/5 ep10 at 18:08 (no refresh before the first boundary); ~2.4 h → due ~20:00. GPU 2.7 GB, RAM 6 GB. Queue 23/30 (+2 reruns).
- **Tick 17:06 — colar seed7 still on task 5/5 (ep3), nothing new to record; `r2_status.py` now prints the mean±sd final retention row per arm:** `cil_cord_colar_seed7` After task 3: 10.9 / 18.8 / 12.8 / 27.4 (seed-42 row at the same point was 20.4 / 56.7 / 30.4 / 4.5 — larger early forgetting, stronger task-3 learning; AA verdict waits for the final row, ~17:40). GPU-free item done: `scripts/r2_status.py` `row_summary()` (+17 lines, black/ruff clean) prints `final row [a±s …]` under each arm from `matrix.npy[-1]`; reproduces the seed-42 plain row [6.2 54.5 30.4 14.1 3.2]. Note all three CIL-CORD arms end task 4 (cord_cil_t4) at F1 3.2 — identical across plain/oracle/pseudo, so the last task is barely learned regardless of refresh; check whether that holds on seed 7 before reading it as a mechanism. GPU 2.7 GB, RAM 6 GB. Queue 24/30 (+2 reruns).
- **Tick 16:04 — teacher-snapshot audit clean; no stale CPFD/IS3 numbers anywhere:** grep of paper_cvpr/main.tex, supp.tex and docs/RESULTS_SYNTHESIS_CVPR2027_2026-10-14.md finds CPFD/IS3 only as a related-work citation and in the gate-1 comment block; no `results/*cpfd*`/`*is3*` dir with a metrics.json predates the fix (the synthesis doc itself records "zero cells"). Every other teacher deepcopy in doccl/methods/ (lwf, doccl, cl_lora) already lives in `after_task`; the `out_features` reads inside `before_task` (lexmem, lexslot, lexslot_fm, nullspace_analytic, o_lora) intentionally read the grown head to size their own memories — not teachers. Bug confined to the two continual-NER ports. `cil_cord_colar_seed7` at task 3/5 (After task 1: 38.8 / 81.4), ~17:15 finish. GPU 2.7 GB, RAM 5 GB. Queue 24/30 (+2 reruns).
- **Tick 14:53 — CPFD/IS3 BASELINES WERE BROKEN; fixed, invalid cells quarantined, reruns chained:** `mixed_cpfd_seed42` finished at 6.0 AA (final row [0 0 0 36.2 0 0], diag 0.0 on both FUNSD tasks while training on them, CE≈0, 3 epochs per task). Root cause: `scripts/train.py` expands the head (l.875) before `before_task` (l.920), and both `cpfd.py` and `is3.py` snapshotted the teacher + `_n_old` in `before_task` → n_old == n_total (log: "7/15/19/25/25 old classes" = post-expansion sizes), the teacher carried the random-init new rows and KD distilled them on every O token, suppressing new-class learning in proportion to label sparsity (CORD diag 76.3, SROIE 41.8, FUNSD 0.0). IS3 also used CE without O tokens on the first task (T0 F1 48.7 vs 88 for everyone else). AGENT FIX: teacher snapshot moved to `after_task` in both; IS3 first task uses full CE; 3 regression tests added. `mixed_is3_seed42` (invalid, at task 3/6) killed at 15:00 → queue moved to `cil_cord_colar_seed7`; bad dirs moved to `results/_invalid/*_teacher_after_expand/`. `scripts/run_r2_retry2.sh` launched (sleeps until 'R2d QUEUE COMPLETE', reruns mixed cpfd/is3 seed42). Remaining R2 cpfd/is3 cells pick up the fix as fresh processes. Oracle cord_s1 anomaly: both plain and oracle early-stopped after 3 epochs on cord_s1 (best = ep1) → low best-epoch val-F1, unexplained at n=1. Mixed IS3 earlier T0 F1 48.7 recorded. GPU chain: R2 → R2b → R2c → retry → R2d → retry2. Queue 24/30 (+2 reruns).
- **Tick 13:50 — MIXED PSEUDO seed42 DONE = 59.5 AA (BWT −30.9); CPFD passes Task 2:** pseudo +2.0 over plain 57.5, below ER 60.3 / bar 63.2; oracle 63.5 > pseudo → gate holds. Final row [63.7 54.5 72.6 19.4 58.1 88.5]; refresh 0[0]/0[0]/0[0]/84[169] tokens[candidates] — zero candidates at the first three boundaries, 84 relabels only at cord_s1. The pseudo run is bit-identical to plain through task 4 (same "After task 4" row), so runs are deterministic and the whole +2.0 is the last-boundary relabel; the CIL-CORD 2-pp plain-vs-pseudo gap is relabel sensitivity, not noise. `mixed_cpfd_seed42` started 13:24, cleared "=== Task 2" (classifier-property fix verified live), at task 4/6 by 13:48 (After task 2: 1.7 / 0.0 / 41.8 — collapsing as expected). GPU 3.8 GB, RAM 6 GB. Gate-1 draft block updated. Queue 24/30.
- **Tick 12:48 — no new cells; §7 CoLaR-R paragraph drafted (commented):** `mixed_colar_r_pseudo_seed42` at task 4/6 (cord_s0) since 12:14, ~14:00 finish; refresh lines so far 0 tokens [0 candidates] / 0 [0] — pseudo null by mechanism reproduces on Mixed. `mixed_cpfd_seed42` not yet started (Task-2 check carries to next tick). GPU 2.7 GB, RAM 5 GB. Wrote the §7 "stale labels / re-annotatable latents" paragraph as a `%`-comment block after the CoLaR r=128 paragraph in paper_cvpr/main.tex (oracle 26.8/63.5 seed-42, pseudo null + mechanism, sliced_pseudo as the pre-reg test, `\vnum{--}` for three-seed means); rebuilt 10 pp / refs p9 / 0 undefined. Queue 25/30.
- **Tick 11:46 — MIXED ORACLE seed42 DONE = 63.5 AA (BWT −15.1):** +6.1 over plain 57.5; above ER 60.3, DER++ 60.8, best 62.2 and the 63.2 pre-reg bar (by 0.3, n=1). Final row plain [64.3 53.6 64.9 11.2 62.4 88.6] → oracle [65.6 54.2 78.1 **38.6** 54.9 89.8]; refresh 3120 / 2640 / 0 / 316 tokens. The cord_s0 collapse recovers 11.2→38.6 after the cord_s1 refresh, and the funsd_s1 refresh lifts the SROIE-row retention (funsd_s1 52.0→69.9). BWT halves (−33.3→−15.1). **Anomaly:** oracle's cord_s1 diagonal reads 12.8 (plain 71.1) and recovers to 54.9 after the FUNSD revisit — looks like an early-stop/eval quirk or seed noise; check `per_class_f1.json`/log of task 5 before citing. Oracle gate intact (oracle ≫ pseudo on both scenarios). `mixed_colar_r_pseudo_seed42` running since 11:19: first refresh **0 tokens, 0 candidates** — the candidate logging confirms the mechanism (stale-O replay leaves nothing to relabel). Both scenarios' oracle/plain/pseudo seed-42 rows parked in the GATE-1 DRAFT comment in `paper_cvpr/main.tex` (10 pp, body 8, 0 undefined). 25/30 pending. GPU 2.7 GB, RAM 5 GB.
- **Tick 10:49 — mixed oracle still running (task 5/6 since 10:45, ~11:35 finish):** refresh counts 3120 / 2640 / **0** after funsd_s1 / sroie / cord_s0 — the cord_s0 head growth relabels nothing because the stale docs (FUNSD, SROIE) carry no CORD classes in `labels_full`; the CORD s0 docs get their refresh after cord_s1. So on Mixed the oracle can only act on within-dataset CIL steps, and the 11.2 collapse on cord_s0 itself is unreachable by refresh (nothing stale is CORD yet) — expect the gain, if any, on cord_s1 retention of cord_s0. No new cells, no new failures, 26/30 pending. GPU 2.7 GB, RAM 5 GB.
- **Tick 09:47 — mixed oracle refreshing; no new cells:** `mixed_colar_r_oracle_seed42` at task 4/6 (`mixed_cord_s0`, the slot where plain collapsed to 11.2); oracle relabelled 3120 / 2640 tokens after T1 / T2 (head grows at funsd_s1 and sroie). Mixed plain row (57.5, final row, collapse note) added to the GATE-1 DRAFT comment block in `paper_cvpr/main.tex` under Table 1; rebuilt 10 pp, body 8, 0 undefined. 26/30 pending, the two cpfd/is3 failures wait for the retry queue. GPU 2.7 GB, RAM 5 GB.
- **Tick 08:45 — MIXED plain CoLaR seed42 DONE = 57.5 AA (BWT −33.3):** below ER 60.3 / best 62.2 / bar 63.2 (joint 67.5). Final row [64.3 53.6 64.9 **11.2** 62.4 88.6] — the collapse sits on `mixed_cord_s0`, the class-incremental transition inside CORD, i.e. exactly the stale-O slot the refresh targets; the domain transitions (FUNSD/SROIE) hold at 54–65. `mixed_colar_r_oracle_seed42` running since 08:34 (task 2/6, no refresh line yet — first fires after the first head growth). `scripts/stale_label_audit.py` now prints AA/BWT per run beside the per-task relabel counts. No failures; 26/30 pending. GPU 2.7 GB, RAM 6 GB.
- **08:05 — user decision: R2d launched, pseudo cells kept.** `scripts/run_r2d_queue.sh` (sliced_pseudo, cil_cord+mixed × seeds 42/7/123) is running and sleeps until `R2 RETRY COMPLETE`; GPU chain is now R2 → R2b → R2c → retry → R2d, one job at a time. The five remaining R2 pseudo cells stay in the queue as the pre-registered three-seed deployable-arm row.
- **Tick 07:43 — no new cells; gate-1 draft parked in the paper:** `mixed_colar_seed42` at task 4/6 (started 06:29, ~09:00 finish), then `mixed_colar_r_oracle_seed42`. No new failures; the two dead cil_cord cpfd/is3 cells wait in `run_r2_retry.sh`. `paper_cvpr/main.tex`: LaTeX comment block under the CIL-CORD/LayoutLMv3 row of Table 1 holding the seed-42 CoLaR / CoLaR-R(oracle) / CoLaR-R(pseudo) numbers, BWT, final rows, refresh counts and the n=1 caveat (unrendered, one uncomment on PASS); Limitations now states the CIL-CORD single-seed spread (~2 AA between two runs differing by three memory tokens) and that no CIL-CORD claim is made below three seeds. Rebuilt: 10 pp, body 8, 0 undefined. GPU 2.7 GB, RAM 6 GB.
- **Tick 06:40 — PSEUDO seed42 DONE = 23.7 AA (BWT −12.6); CPFD/IS3 cells crashed on a wrapper bug, fixed:** pseudo relabelled 1/0/0/2 tokens yet lands +2.1 over plain 21.7 with a completely different retention row (plain [6.2 54.5 30.4 14.1 3.2], pseudo [1.8 22.1 46.5 45.1 3.2], oracle [45.6 28.5 22.4 34.1 3.2]). Training is near-identical between plain and pseudo, so that spread is **run-to-run noise on CIL-CORD** — the oracle's +5.1 (n=1) is inside it; no gate reading until seeds 7/123. `cil_cord_cpfd_seed42` (after 27 min of T0) and `cil_cord_is3_seed42` (immediately) both died: `AttributeError: 'LayoutLMv3Wrapper' object has no attribute 'classifier'` — CPFD/IS3 were only ever tested on a fake wrapper with `.classifier`. Fix (AGENT FIX): read-only `classifier` property on `LayoutLMv3Wrapper`, `TokenClassificationWrapper`, `BertTokenClassificationWrapper` returning the inner head; CPFD/IS3 smoke-tested on CPU through two CIL tasks (expand 3→5) on the real wrapper, tests 19/19. Remaining R2 cpfd/is3 cells pick the fix up (fresh processes). The two dead cells re-run via `scripts/run_r2_retry.sh` (launched, sleeps until R2c completes; R2d now chains after it). Queue: `mixed_colar_seed42` running since 06:29. GPU 2.7 GB.
- **Tick 06:15 — pseudo arm is null by mechanism; sliced_pseudo arm added (not launched):** pseudo seed42 refresh counts 1/0/0 tokens after T1/T2/T3 (oracle 101/760/1461/1402 after T1–T4; `scripts/stale_label_audit.py` prints the table). With a median-quantile rule, ~0 relabelled ⇒ ~0 *candidates*: replaying a stale O label during the task trains the model to output O on exactly the memory tokens the refresh then inspects, so a looser quantile cannot help (q85 idea dropped). Fix is to stop teaching the stale O while the task runs: new `refresh: sliced_pseudo` in `doccl/methods/colar_r.py` (sliced capture-time softmax during replay + CPFD pseudo refresh after the task; refresh log now also records candidate counts), `configs/method/colar_r_sliced_pseudo.yaml`, test added (11/11 green). Follow-up queue `scripts/run_r2d_queue.sh` (cil_cord+mixed × 3 seeds, waits for R2c) is written but **not started** — ~15 h of GPU; decide after the R2 pseudo cells confirm ≈ plain. Pseudo seed42 at T4, then cpfd/is3 seed42. GPU 2.7 GB, RAM 6 GB.
- **Tick 05:15 — ORACLE seed42 DONE = 26.8 AA (BWT −11.3):** +5.1 over plain CoLaR 21.7, above the 23.0 pre-reg bar, naive 19.0, ER 16.0; joint 33.3. Final row plain [6.2 54.5 30.4 14.1 3.2] → oracle [45.6 28.5 22.4 34.1 3.2]: T0 retention recovered ×7, T1 traded down. n=1, oracle = ceiling arm; gate waits on pseudo×3. **Watch:** pseudo seed42 relabelled only 1 token after T1 (oracle 101) — if the T2/T3 counts stay ≪ oracle's 760/1461, the median-entropy rule is too strict and pseudo will sit at plain; consider a looser quantile arm (refresh_quantile 0.75–0.9) after R2. Overleaf re-uploaded (main.tex, supp.tex, refs.bib overwritten via upload dialog; project compiles to 10 pp). Writing queue empty; next GPU-free item: stale-label audit over refresh counts from the logs (`grep 'colar_r\[' results/logs_vision/*.log`).
- **Tick 04:15 — supplementary split:** appendix (Setup details, Additional results incl. prompt/LoRA table, Pareto note) moved from `paper_cvpr/main.tex` into `paper_cvpr/supp.tex` (same preamble, `\maketitlesupplementary`); main cross-refs now say "supplementary Sec. A/B", supp refs say "Sec. 5 / Table 1 of the main paper". main.pdf 10 pp (8 body + refs), supp.pdf 2 pp, 0 undefined in both. Oracle seed42 at T4 ep2; refresh counts 101/760/1461 tokens after T1/T2/T3. Remaining writing item: Overleaf re-upload (main.tex + supp.tex + refs.bib).
- **Tick 03:15 — F9 inversion paragraph in §7:** ViT-B/16 image CIL (CIFAR-100/ImageNet-R, 3 seeds, recomputed from results/): SLCA 87.5/65.0 > ER@2000 76.2/54.8, DER++ 76.5/52.8, latent bank 80.0/55.1; SLCA-noCA 38.0/25.8; readout refit on adapted trunk 88.6 > RanPAC 85.2. Documents: Gaussian-head idea null (39.4 / 41.9 / 86.2 vs 86.4). `zhang2023slca` added to refs.bib. §7 routing/r128 tails compressed and the Limitations routing paragraph folded into §7 to hold the body at 8 pp. Oracle seed42 at T2 ep5; refresh log fires (50 docs / 101 tokens after T1). Next writing item: supplementary skeleton.
- **Tick 02:15 — first R2 cell DONE:** cil_cord/colar/seed42 AA 21.7 (BWT −15.9) — plain CoLaR already > naive 19.0 / ER 16.0 on CIL-CORD. Paper: HGT/CUBER ledger cell filled (CUBER 39.8 vs own naive 38.7, HGT ≤25.3, frozen-bb 5-ep ablation); app:extra `\todo` replaced by Table 3 prompt/LoRA/C-Flat++ (3-seed, L3+BERT) + per-backbone BWT/AF pointer.
- **Tick 01:10 — CoLaR headline reconciled:** 10-ep protocol 3-seed = 87.75±0.47 (87.58/88.30/87.38, log-sourced); 5-ep control = 86.62±0.25; stray 86.8±0.4 removed from paper/thesis. R2c queue (colar_e10 ×3) chained after R2b to make the headline disk-backed. R2: cil_cord/colar/seed42 at T3 (early stop ⇒ ~1.5 h/run, not 7).

- **Idea tournament closed** (67 candidates, 5 lenses, red-team round): P1 live-labels package
  (CoLaR-R oracle = stream-implementable label ledger since CIL-CORD re-shows 100 % of docs;
  pseudo + new `colar_r_sliced` as label-free variants) > P2 evaluation methodology (headroom
  certification + SLCA-token inversion) > P3 sequence-position anatomy. Verdict + table in
  `docs/RESEARCH_PLAN_2026-10_next_mechanism.md` §0''. If CoLaR-R is null, P2 becomes headline.
- **Ops loop** armed (hourly ticks; `docs/RESEARCH_OPS_LOOP.md`, `scripts/r2_status.py`).
  R2 queue at cil_cord/colar/seed42 T0 ep5 (~9 min/epoch → ~7 h per CoLaR run; 30 cells ≈ Oct 19–20).
  `scripts/run_r2b_queue.sh` (sliced control, 6 cells) chains after R2 — launch it once
  `colar_r_sliced` tests are green.
- Deadlines (sourced): registration **Nov 10**, paper **Nov 16** AoE, supp Nov 23.

## RESUME HERE (2026-10-14 23:00, auto mode): R2 queue armed; CoLaR-R plumbing landed; CVPR pivot

- **Direction (CVPR 2027, deadline ~2026-11-13).** DIL is saturated (Independent 88.2 ≈ Joint 88.7;
  Indep > Joint on LiLT 85.3 / BROS 88.8). Headroom is CIL-CORD (gap 14, ER 16.0 < naive 19.0) and
  Mixed (gap 5). Cause (code-verified): every buffer banks labels in the capture-session head space
  (post-capture classes masked to O). Plan + evidence: `docs/RESEARCH_PLAN_2026-10_next_mechanism.md`
  §0', `docs/RESULTS_SYNTHESIS_CVPR2027_2026-10-14.md`. Head-Gaussian / token-coreset directions
  retracted (pre-falsified by Amendment 9 B2 and D1/D2).
- **CoLaR-R (`doccl/methods/colar_r.py`, configs `colar_r_oracle` / `colar_r_pseudo`).** Label refresh
  of the stored memory after each task: oracle unmasks true labels (`labels_full` now emitted by
  CIL remappers built with `full_head_label_to_id`; cil_cord + mixed train sets wired), pseudo
  relabels O tokens confidently predicted as post-capture classes (CPFD median-entropy rule on
  the memory). Store docs carry `n_labels` (head size at capture). Tests: `tests/methods/test_colar_r.py`;
  fast suite 470 green.
- **R2 queue running** (`scripts/run_r2_queue.sh`, log `results/logs_vision/r2_queue.log`): waits for
  the Independent grid (last BERT cell), then per seed 42/7/123 × {cil_cord, mixed}: colar k4/d50/r128,
  colar_r_oracle, colar_r_pseudo, cpfd (bs1), is3 (bs1). Resume-safe. Pre-reg bars in plan §0'.
- **Independent grid** (`results/logs_vision/indep_queue.log`): LiLT 85.34±0.66, BROS 88.75±0.08, BERT
  pending 1 cell. Then: replace the three `\tbd` cells in `paper_cvpr/main.tex` Table 1 + thesis table.
- **DONE 23:10 — localisation protocol formalised:** `doccl/diag/protocol.py` (`ForgettingLocaliser`
  arm/measure, `dominance_test`, `shared_location_test`, `freeze_groups`), doc
  `docs/PROTOCOL_FORGETTING_LOCALISATION.md`, paper §3 paragraph rewritten as the named protocol
  (body still 8 pp). Overleaf copy of `paper_cvpr/main.tex` NOT yet re-uploaded. Original note: formalise the forgetting-localisation pipeline as a named, model-agnostic
  protocol (inputs: checkpoint pair + `param_groups_by_depth` partition + probe loader; outputs:
  per-group old-task-Fisher displacement, depth-matched CKA, permutation test for shared location,
  protect-and-migrate experiment) — algorithm box in paper §4 + one `doccl` entry point (user ask).

## RESUME HERE (2026-10-13 05:00, auto mode): Amendment 10 KILLED; B2 retry + codec probe running

- **Routing confirmation is in — KILL.** far@d50 vs uniform d50 by seed: 42 +1.8, 7 +0.4
  (87.0 vs 86.6), 123 **−0.6** (86.3 vs 86.9). Sign agreement 2/3, mean +0.5 pp (far 87.2±0.9,
  uniform 86.6±0.2). Pre-registered rule → the lexical gate adds nothing beyond replay; **thesis
  title must revert to the diagnosis/replay framing.** Verdict recorded in
  `docs/IMAGE_SCOPE_PREREG.md` (Amendment 10 verdict). Still to do: revert the title in the thesis
  LaTeX / CVPR port, move the routing rows into the falsification-chain table, update the Reports
  sheet (doc routing block) with the three confirmation cells.
- **Codec probe DONE (05:03).** token F1 through frozen upper layers: exact 97.6; SVD r16/32/64 =
  40.9/65.6/89.6; AE c16/32/64 = 21.2/27.0/31.5 — learned codec loses to per-doc SVD at matched
  bytes → Amendment 9 cell A is a negative, no replay cell. Table in `docs/IMAGE_SCOPE_PREREG.md`.
- (history) Codec probe failed twice (04:51: CPU/GPU device mismatch — stored hidden vs AE on cuda);
  fixed in `scripts/colar_codec_probe.py` (one `.to(m.device)`), rerun launched 05:05 via
  `/tmp/claude-1000/probe_then_b2b.sh` → `results/logs_vision/codec_probe.txt`,
  `results/dil_docD1_seed42/codec_probe.json`.
- **B2b DONE (06:09): AA 86.2 vs uniform d50 86.4 → null.** Drift-comp head alignment is dead on
  documents at both d5 (74.6) and d50. BWT +8.7 is a diagonal artefact (see prereg). GPU idle;
  `wandb sync --sync-all --include-offline` launched 06:46 (`results/wandb_backlog_sync.log`).
- (history) B2 retry (`docB2b`): d50 drift-comp + head-align 10 ep with `method.head_align_batch=8`
  (default 64 stacked 150 docs' latents → 1.1 GB alloc OOM). Runs after the probe (~75 min);
  log `results/logs_vision/dil_docB2b_seed42.log`, status in `doc_dba_queue.log`.

## RESUME HERE (2026-10-13): chain drained; routing `far` is the only live arm; independent bound in

- **Routing (Amendment 10, seed 42, CoLaR k4 r128 5 ep).** d50 vs uniform 86.4: **far 88.2 (+1.8)**,
  task 87.3 (+0.9), near 86.5 (+0.1). d5 vs uniform control `docU_d5` 75.9: far 77.8 (+1.9 < +5),
  task 73.4, near 64.0 (near-routing *hurts* at small buffers — replaying lexically-similar docs
  starves the far task). Only `far@d50` clears the support bar → confirmation seeds 7/123 running
  (`scripts/run_doc_route_confirm.sh`), then uniform d50 seeds 7/123 for sign agreement
  (`/tmp/claude-1000/route_confirm_then_rest.sh`), then the codec probe (head-size bug fixed).
  Kill rule unchanged: no 3/3 sign agreement → thesis title reverts.
- **Doc D/B (seed 42).** D1 labeled-tokens-only 79.6 @ 41.6 MB (SROIE 81→59); D2 +drop visual
  tokens 61.2 @ 23.4 MB (SROIE 12) — position-axis compression breaks whole-doc consistency on
  the dense receipt task; registered negative. B1 drift-comp Gaussians d5 74.6 @ 6 MB (vs uniform
  d5 75.9) — null. B2 (d50) OOM on the 6 GB box — not run.
- **CoLaR on ImageNet-R (3 seeds, k4, 500/task).** raw bank **55.1 ± 1.3**, r128 **55.3 ± 1.4**
  (BWT ≈ −37). Pre-run estimate was 59 / 58 (ranges 54–63 / 53–62): measured inside the range,
  low end — CoLaR ≈ ER@2000 (54.8) < RanPAC 58.6 < SLCA 65.0. Sheet rows 57–58 `measured_*` filled.
- **Independent per-task bound (3 seeds).** DIL **88.8 ± 0.6** (joint 88.7 — no interference
  gain; replay 87.9 is 1 pp short of both); CIFAR-100 Task-IL oracle 98.1; ImageNet-R 90.2
  (CIL joint 89.5 / 77.6 — the task-ID oracle, NOT a CIL bound).
- Reports sheet: Image Summary carries 4 native charts + an "ESTIMATES — NOT MEASURED" block
  (rows 55–67) with its own chart; Image All Runs has a per-run AA chart. The Diagnosis /
  Approaches-by-dataset / Metrics tabs still need the user to import `clue_report_addendum.xlsx`.

## RESUME HERE (2026-10-08): W&B back online; Reports sheet extended; independent bound queued

- **W&B online again.** All runner scripts dropped the forced `WANDB_MODE=offline` (80ae89e);
  the 503 offline runs since 2026-07-02 were synced into `thanh-workspace/doccl-aaai2027`
  (`results/wandb_backlog_sync.log`). Jobs launched before the change (ppI124, docD1) still ran
  offline → `wandb sync --sync-all --include-offline` once more when the chain drains.
- **Google "Reports" sheet** (`1Oa98rqXXs7hRL2hfWA4dwSLKPJJ2G-TR0OCtVpYV_BA`): added `Image All
  Runs`, `Image Summary` (CSV import via the Sheets picker — it lists real files only, not Google
  Sheets). Chart-bearing tabs come from `scripts/report_addendum_xlsx.py` (Diagnosis locus,
  Approaches by dataset, Metrics, cil_cifar100/imagenet_r (vit)) → xlsx handed to the user for
  File→Import→Upload. **Lesson:** binary files cannot go through the Drive MCP base64 path
  (transcription corrupts bytes; verified by download+cmp) — hand the file over instead.
- **Independent per-task upper bound** (one model per task, task-ID oracle) implemented
  (7ce4bec): `scenario.kwargs.only_task=k` → run dir `<scenario>_indep<k>_naive_seed<s>`;
  `analyze_results.py` folds them into an `independent` row under joint. Valid bound on DIL
  (fixed labels); on CIL it is the Task-IL oracle — report as such, never as a CIL bound.
  Queue `scripts/run_indep.sh` (DIL ×3×3; CIFAR-100 + ImageNet-R ×10×3) armed behind the
  routing cells (`/tmp/claude-1000/indep_after.sh`). Chain: docD1 (running) → D2 → B1 → B2 →
  codec probe → routing near/far/task × d50/d5 → independent bound.
- Diagnosis AA/BWT in the sheet come from the pilot JSONs with the `_ord210` reversed-order runs
  excluded (LayoutLMv3 full 28.3/−89.8); `docs/RESULTS_REPORT_2026-10.md` §3.1 quotes 31.6/−81.3
  for the same arm — reconcile which aggregation the report used.

## Latent replay × gradient checkpointing bug (found 2026-10-06; scope NARROWED same day)

HF checkpointing recomputes `layer.__call__` in backward and re-fires the layer-k injection
pre-hook after `_inject` was cleared → layer k recomputed on the dummy input (silent). Fixed in
38648af (`LatentReplay.no_checkpointing()`), test `tests/methods/test_latent_replay_checkpointing.py`.
**Scope: only backbones using HF-native checkpointing (ViT; LiLT/BROS/BERT if ever used with
latent replay).** LayoutLMv3 checkpoints via our own per-layer `forward` wrap
(`layoutlm_wrapper.py:440`), whose recompute does not re-fire the hook → **all document
latent/CoLaR/CA-CoLaR/CoLaSlot/PLaR results are unaffected.** Verified: post-fix rerun of the
document latent replay k8/d5 (identical config) reproduces 65.80 / 63.91 / 69.70 byte-identically
→ the 66.47 non-reproduction of 87.3 stands. Affected and superseded: the pre-fix ViT H3 arms.
AMP: Turing now uses fp16+GradScaler (was emulated bf16); ER@2000 seed 42 fp16 77.11 vs fp32
75.59 (inside the 3-seed spread 76.2±1.2) → fp16 cells are comparable.

**ViT Step 1 (post-fix, fp16, seed 42, CIFAR-100, d500, replay batch 16, task-balanced):**
k=8 77.5, k=11 71.3, k=12 head-only 67.9, **k=4 78.8 → selected** (prereg rule: highest ≥ 76.2).
Latent replay now works on ViT (diagonal ≥ 95 on every task). Step 2 launched
2026-10-06 20:08: `K=4 scripts/run_colar_vision.sh` (bank500, CoLaR r128/r64/r16 @ d500,
CoLaR r16 @ d2000, AGLR; CIFAR-100 then ImageNet-R; 3 seeds; ~2 days).

## RESUME HERE (2026-10-05): ALL image-CL experiments complete; consolidated report written

`docs/RESULTS_REPORT_2026-10.md` = every result in comparable tables (doc grid AA/BWT, extra
baselines, diagnostics, replay ladder, falsification ledger, image grid + verdicts) — the basis
for the write-up. Image verdicts (`docs/IMAGE_SCOPE_VERDICT.md`): gate PASS; **H1 SUPPORTED**
(ViT displacement 97–100 % at head on 3/3 seeds, CKA monotone; slow trunk 100 %); H2 mixed
(CIFAR yes, ImageNet-R no); H2b NOT SUPPORTED; **H3 INCONCLUSIVE** (both latent arms at floor:
real bank d500 12.0 — frozen-trunk latent replay loses plasticity from task 5 on ViT; Gaussian
10.7). Gotcha: `aglr_replay` with `attn_keep=1.0` gets a `_keep100` run-dir suffix — the
after-grid queue logged it FAIL but the runs completed. Optional follow-ups (not needed for the
claims): bs-128 SLCA/Seq-FT replication on ≥16 GB; H3 redesign with a working ViT replay;
H4 RVL-CDIP bridge. Next: paper §7 "what changes when the head is not per-token" + figures
(`doccl-pilot-analyze`, `build_backbone_figures.py` know the ViT condition).

## (2026-10-02): image-CL main grid DONE (60/60); H1/H3/calibration queue running

**Verdicts** (`docs/IMAGE_SCOPE_VERDICT.md`, rules in `docs/IMAGE_SCOPE_PREREG.md` + Amendments
1–3): gate PASS both benchmarks. **H2 SUPPORTED on CIFAR-100** (SLCA 87.5 vs joint 89.5, +11 over
ER@2000) and **NOT SUPPORTED on ImageNet-R** (SLCA 65.0 vs joint 77.6, still +10 over ER@2000) →
"mixed, report as measured". **H2b NOT SUPPORTED**: slow trunk alone 38.0 / 25.8 — old-task rows
extinguished at the head (readout snap reproduced on images); the Gaussian head re-grounding
carries the gain. EWC/LwF at the naive floor on both. Unresolved: published Seq-FT 88.9 vs our
slow-trunk 38 (not schedule — fixed-20-ep arm gives 26.6; likely bs 128 vs 16).
**Running:** `scripts/queue_after_grid.sh` → slca_noca_pub ×3, H3 pair (latent_replay d500 vs
aglr_replay) ×3, H1 pilot cv_vit_fast/slow ×3 (`results/pilot/cv_vit_*.json`). Then: extend
`image_scope_verdict.py` for H1/H3, `doccl-pilot-analyze` + `build_backbone_figures.py` with the
ViT condition, write the paper section "what changes when the head is not per-token".

## (2026-09-25): image-CL grid launched; ICLR full paper due Sep 25 AoE

**Image grid (local 2060, bs 16 + ckpt, document recipe).** Launched 07:18 Sep 25 via
`scripts/run_vision_grid.sh`: CIFAR-100 {naive, ewc, lwf, er, er_b2000, der_pp,
der_pp_b2000, slca, slca_noca, joint} × {42,7,123}, then ImageNet-R same. Resume-safe; log
`results/logs_vision/queue.log`. Seed-42 gate results: joint 89.08 (published 93.22 →
gate made *internal*, see prereg Amendment 1), ER@200 44.15 / BWT −59.8 (200 exemplars =
0.4 % of CIFAR → added 2000-exemplar arm), naive 1-ep smoke 11.75. Lesson: a `setsid bash -c
'while pgrep -f "bash scripts/X"...'` waiter matches ITSELF and deadlocks — cost ~20 h idle
GPU on Sep 24. Use direct sequential launches or match on a unique token.
**Prereg amended** (`docs/IMAGE_SCOPE_PREREG.md` Amendment 1): internal gate, buffer-size
arm, H2b (slow-trunk vs alignment), bug disclosure. **Wired but not yet queued:** H1 pilot
conditions `cv_vit_fast`/`cv_vit_slow` (`doccl-pilot --conditions cv_vit_fast --seeds 42 7 123
--epochs 3 --batch_size 16`), H3 pair (`latent_replay` real bank vs `aglr_replay`, ViT).
Queue them after the main grid (one GPU job at a time). Then `scripts/image_scope_verdict.py`
(to write) → `docs/IMAGE_SCOPE_VERDICT.md`.

## (2026-09-23): ICLR 2027 draft v1 + image-CL scope test implemented

**Paper.** `paper/main.tex` (ICLR 2027 style, anonymised) builds; main text ends p.9 (limit 9).
Abstract registered on OpenReview 2026-09-18; **full paper due 2026-09-25 AoE**. Table 1 = 68/72
grid cells from regenerated aggregates (`analyze_results.py --source local`, 2026-09-23).
Remaining: appendix B extra tables, ledger HGT/CUBER number, trim, anonymisation check.
Plan/story: `docs/ICLR2027_PAPER_PLAN.md`.

**Grid.** 68/72. BROS DER++ 9/9 done with fp16 + replay batch 1 (HF BROS has no gradient
checkpointing): dil 86.8/87.1/87.6-ish (seeds 42/7/123), mixed 63.0±0.4, cil_cord 17.7±1.1.
Still OOM on 6 GB even with fp16: LiLT LwF (dil/mixed/cil_cord ×3 seeds) + LiLT cil_cord EWC ×3
= 12 runs → rented GPU or grey cells (paper currently discloses them as missing).
**Hygiene fix:** 109 locally-run result dirs lacked the `.done` marker `analyze_results.py`
requires → they were silently skipped by the aggregates; marked 2026-09-23.

**CA-CoLaR is a null at 3 seeds.** Treatment A1 [18,5,2]/[64,128,64] vs equal-byte uniform
d10/r64 control, dil k4 5ep — treatment / control AA: seed42 75.98 / 71.93 (+4.04; the
non-gate rerun `results/dil_colar_adaptive_seed42` gives 79.04, +7.11), seed7 71.18 / 79.21
(−8.03), seed123 75.29 / 69.63 (+5.66) → **+0.6 to +1.6 ± 8** depending on which seed-42 run is
used; either way a null. Controls: `results/dil_colar_seed{7,123}_k4_d10`.
Reported as a frontier-not-gain negative in the paper appendix.

**Image-CL scope test (next cycle, pre-registered).** `docs/IMAGE_SCOPE_PREREG.md` (H1–H3 +
decision rules + sanity gate) written BEFORE any run. Implemented: `ViTWrapper`
(`doccl/models/vit_wrapper.py`), `doccl/data/vision.py` (Split CIFAR-100 / ImageNet-R),
scenarios `cil_cifar100` / `cil_imagenet_r`, configs `model/vit_b16`, `training/vision`,
`method/slca` (= `lca` align with `merge: false`, `backbone_lr_scale: 0.01`), accuracy metric
dispatch on `model.task_type`, Fisher 1-D label guard, ViT param grouping. 302 offline tests +
ViT integration test green. Runner `scripts/run_vision_grid.sh`. Smoke run (naive, 1 ep, bs 8,
2026-09-24) PASSED end-to-end: task0 98.6 → after task1 [33.6, 95.6]; ~3 min/task, 1.8 GB VRAM
on the 2060 — CIFAR-100 cells are feasible locally (~2.5 h/run at bs 32). ImageNet-R must be
downloaded manually to `data/vision/imagenet-r/`. Needs a rented GPU (~25 GPU-h for 54 runs).

## GO — Pareto branch (2026-08-21): CA-CoLaR A1 re-gate passes all four guards

Amendment A1 (`docs/CACOLAR_T1_T2_2026-08-20.md`, preregistered before launch): treatment
docs=[18,5,2] rank=[64,128,64], budget-neutral vs T4 (1,920 units), adjudicated against the
already-completed equal-byte uniform d10/r64 control.

**Result: 75.9751 AA / [76.5283, 54.6006, 96.7963]** vs control 71.9307 / [70.2294, 48.3149,
97.2477]. **+4.0444 AA** (guard +0.5 PASS), **+6.2923 old-domain mean** (guard +1.0 PASS),
**min domain −0.4514** (guard −0.5 **PASS** — the CORD failure is removed), runtime −0.43%
(PASS). Banks 6.29 vs 6.41 MB. matrix.npy == metrics.json; rc=0. Evidence:
`results/gates/cacolar_a1/`, `results/gates/cacolar_a1.log`.

Predictions: CORD ≥ control−0.5 **HIT** (96.7963 vs 96.7477, by 0.05 — narrow); old-domain mean
≥ +8.0 **MISS** (+6.29). Per the prereg's own decision rule this is therefore the **Pareto-trade
branch, not the clean-GO branch**: the headline is a frontier, not a point.

The frontier, measured (both budget-neutral, vs the same control):
| schedule | AA | old-dom mean | CORD | guards |
|---|---|---|---|---|
| [20,5,0] (T4) | +7.1052 | +11.1604 | −1.0054 | 3/4 (CORD fails) |
| [18,5,2] (A1) | +4.0444 | +6.2923 | −0.4514 | **4/4** |
Two CORD documents cost 4.87 old-domain points and buy back 0.55 CORD points — a ~9:1 exchange
rate. The allocation effect is real and large in both arms; what the guard selects is where on
the frontier to sit, and the guard-satisfying point is materially weaker than the guard-violating
one. This is evidence that the per-domain guard, written for augmentation gates, prices zero-sum
allocation badly — but the rule was fixed in advance and A1 is the admitted result.

Status: CA-CoLaR v1 is a **GO on the Pareto branch** → proceed to T5 (seeds 7/123, dil orders,
cil_cord). Seed-fragility warning from the shadow chain applies: T5 decides whether this holds.
No third schedule (prereg: escalate-don't-iterate).

# STATE

## ⚠ NON-REPRODUCTION (2026-08-23): latent-replay 87.3 does not replicate at the canonical config

Ran the re-run the ROADMAP's evidence-hygiene item called for: `latent_replay`, **k=8, d=5,
early-stopped** (the config the record identifies for "Exp #5"; STATE:928 — *"that was a
prior-session early-stop run"*). Seed 42:

| | cited | re-run | delta |
|---|---|---|---|
| AA | 87.3 | **65.7981** | **−21.50** |
| BWT | −2.3 | **−30.8380** | −28.54 |
| final row | ≈ER-200 | [67.94, 34.53, 94.92] | SROIE collapses |

**The run is healthy, not truncated.** It converged normally (task 2 ran 12 epochs to a val-F1
plateau of 94.92 with patience-2 early stop; tasks 0/1 likewise), peak VRAM 2,408 MB, 81.5 min,
rc=0, `EPOCHS_CAP=100`. So this is a real converged latent-replay run at the canonical operating
point, and it does not behave like the citation: FUNSD holds (67.94) but SROIE collapses (34.53),
versus the cited near-zero forgetting.

**FINAL, 3 seeds:** 65.7981 / 63.9095 / 69.6993 → **mean 66.47 ± 2.95 AA**, BWT −26.4 to −34.2.
Same signature every time (FUNSD holds 67.9–78.4, SROIE collapses 25.5–35.6). Not seed variance;
**−20.83 vs the cited 87.3**, and **+26.5 vs the falsified marginals** (~40).

### Likely cause: the cited run was **d=50**, not d=5

The coverage ladder, measured, all at converged/5ep in the same family:

| config | AA |
|---|---|
| colar **d5**/r64 (3 seeds) | 60.79 |
| latent_replay **d5** k8 converged (today, 2 seeds) | **63.91 – 65.80** |
| colar **d50**/r64 (seed 42) | 80.61 |
| colar **d50**/r128 (3 seeds) | **86.75** |

**d=5 lands in the 60s for BOTH compressed and uncompressed replay; 87.3 sits exactly where
d=50 lands.** The only run in the entire family on disk above 84 is `dil_colar_seed42_k4_d50_r128`
(86.38). This is independently corroborated by today's CA-CoLaR T1 response surface, which
measured the coverage effect directly (d5→d50 at r64: FUNSD **+37.97**, SROIE +21.08).

⇒ STATE:834's "private **d5**-conv 87.3" is almost certainly a **d50** run mislabelled as d5.
The number is real; the *config attribution* is wrong. Fix the label, not the value — and re-cite
it against a run that exists on disk.

**The qualitative chain claim is unaffected**: real latent replay at 63.9–65.8 still beats every
falsified marginal method (~37–42) by >20 points, so "real latent replay works where marginal
compressions collapse" stands. Only the headline magnitude and its config label need correcting.

## FINDING (2026-08-23): generality holds for the PHENOMENON, not for every REMEDY

Two four-backbone results that must be stated together, because they point different ways.

**(a) The forgetting phenomenon is architecture-invariant.** On `mixed`, the joint→naive
forgetting gap is +33.65 (LayoutLMv3), +31.63 (BERT), +31.11 (LiLT), +31.42 (BROS) — a 2.5-point
spread across a tri-modal encoder, a text-only encoder, and two text+layout encoders, despite
~10 points of difference in absolute capability (BERT sits lowest throughout; the three
layout-aware backbones cluster at 30.7–33.9 naive). *How much is lost to sequential training is
conserved under architectural variation* — a sharper claim than raw AA convergence, and close to
the conservation framing the thesis already uses for locus migration.

**(b) A remedy's effectiveness is NOT.** EWC on `mixed`, benefit vs naive:
LayoutLMv3 **−11.81** (actively harmful), LiLT −2.60, BROS **+3.52**, BERT **+5.27** — a 17-point
spread with a sign flip. EWC harming a strong backbone is consistent with regularization fighting
plasticity, and ch4/ch7 already report EWC as weak off domain-shift; but the *cross-backbone sign
flip* is new and the current write-up does not capture it.

Reading: this strengthens rather than weakens the diagnostic framing — a remedy chosen without
regard to where forgetting lives behaves unpredictably across architectures, which is the
thesis's argument for diagnosis-first design. **Caveat:** whether *replay* (the remedy the
diagnosis predicts) is similarly unstable cannot be answered locally — all LiLT/BROS replay cells
are in the deferred set (`replay_batch_size: 8` OOMs 6 GB). That is the single most valuable
thing rented GPUs would buy.

## FINDING (2026-08-22): the naive forgetting signature is four-backbone general

First BROS run ever completed. `dil` naive, seed 42, across all four backbones:

| backbone | architecture | AA | FUNSD | SROIE | CORD |
|---|---|---|---|---|---|
| LayoutLMv3 | tri-modal (text+layout+vision) | 40.06 | 18.3 | 4.3 | 97.6 |
| BERT | text only | 39.41 | 16.1 | 5.0 | 97.1 |
| LiLT | decoupled text+layout | 40.68 | 20.8 | 3.1 | 98.1 |
| **BROS** | order-robust text+layout | **40.84** | 21.1 | 4.1 | 97.3 |

All four within **1.4 AA**, with near-superimposable final rows: total collapse on the two older
domains, near-perfect retention on the newest. This is the widest architectural span the project
has tested and it directly answers the reviewer objection the ROADMAP flags as bounding the
paper ("a negative result on ONE setup is not a finding"). Same pattern already confirmed on
`cil_cord` naive (LiLT 18.89 vs LayoutLMv3 18.79 vs BERT 19.39) and joint (LiLT 32.44 vs 33.32 /
32.41). BROS is also cheap: 26.8 min/run, 4,617 MB peak (≈1 GB more headroom than LiLT).

## DONE (2026-08-23): generality-grid fill complete — 44/72 → 57/72 method-cells

| backbone | dil | cil_cord | mixed | change |
|---|---|---|---|---|
| LayoutLMv3 | 6/6 | 6/6 | 6/6 | — |
| BERT | 6/6 | 6/6 | 6/6 | — |
| LiLT | 5/6 | 3/6 | 4/6 | was 4 / **0** / 4 |
| BROS | 3/6 | 3/6 | 3/6 | was **0 / 0 / 0** |

**BROS built from nothing**: 27 new runs (naive/joint/ewc × 3 scenarios × 3 seeds). LiLT gained
its entire `cil_cord` column plus the `dil` `joint` triple. All at `EPOCHS_CAP=100`, matching the
existing table's protocol.

**Every one of the 15 remaining cells is the same hardware ceiling**, not a defect or a dead end:
`lwf` everywhere (teacher deepcopy), `er`/`der_pp` on BROS + parts of LiLT (`replay_batch_size: 8`
forwards 8 extra docs/step regardless of `training.batch_size`), `ewc` on LiLT `cil_cord` (Fisher
+ growing head). Full analysis and the job lists: `docs/gridfill/`. **These need a >6 GB GPU.**

The most valuable missing piece: **every LiLT/BROS replay cell**. Replay is the remedy the
head-localization diagnosis predicts should win, so "ER/DER++ reach the joint bound" is the
paper's payoff claim — currently demonstrated on 2 of 4 backbones.

## SUPERSEDED (2026-08-22): BROS/LiLT generality-grid fill — in flight

**Running unattended:** `docs/gridfill/grid_fill.sh` (PID 2203423 this session), 67 local jobs,
resume-safe (skips any run with `metrics.json`), logs to `results/gates/gridfill.log` and
per-run `results/logs_gridfill_<run>.log`. ETA ~1–2 days from LiLT's measured 0.41 h median
(`cil_cord` runs slower, ~1 h). **To resume after any interruption: just re-run the script** —
it skips completed work. Full context, job lists, and traps: `docs/gridfill/README.md`.

Coverage when this started → now: LayoutLMv3 6/6 and BERT 6/6 on all three scenarios (unchanged);
LiLT dil 4/6 → **5/6 (local max)**, LiLT mixed 4/6 (+1 retry pending), LiLT cil_cord 0/6 → filling;
BROS 0/6 everywhere → queued next. Validation so far is clean: LiLT cil_cord naive AA **18.89** vs
LayoutLMv3 18.79 / BERT 19.39, with the textbook CIL collapse row `[0,0,0,0,94.47]` — head
expansion works on LiLT and the naive floor reproduces across backbones.

**Three carry-overs for the next session (none block the queue):**
1. **18 `lwf` jobs need >6 GB VRAM** (`docs/gridfill/lwf_deferred_jobs.txt`). Hard ceiling: `lwf`
   deepcopies the model into a frozen teacher, so two full LiLT/BROS backbones + optimizer state
   coexist from task 1. Failed identically at bs=2 and bs=1. Rent, or disclose the gap.
2. **Joint-oracle inversion, LiLT dil only** (joint 83.79 vs ER 85.65, DER++ 84.35, all 3 seeds).
   LayoutLMv3, BERT, and LiLT mixed are all clean → footnote-scale disclosure or a patience/val
   protocol change, not a crisis. Recheck after the grid lands.
3. **`EPOCHS_CAP=100` is mandatory** for any hand-run job — every method config says `epochs: 10`
   while the whole existing table was built at 100. Silent contamination otherwise (verified:
   `dil_joint_seed7_lilt` 84.27@10ep vs 85.12@100ep).

## PARTIAL PASS (2026-08-21): CA-CoLaR T4 equal-byte allocation gate

First CA-CoLaR gate (`docs/CACOLAR_T1_T2_2026-08-20.md`, preregistered before launch). Treatment
`colar_adaptive` docs=[20,5,0] rank=[64,128,64] vs equal-byte uniform `colar` d10/r64, seed 42,
dil k4/e5. Measured banks 6.29 MB vs 6.41 MB (treatment 1.9% SMALLER — parity holds; the doc·rank
unit model is byte-exact since per-doc bytes (seq+d)·r·2 are linear in r).

**Result: 79.0358 AA / [81.1966, 59.6685, 96.2423]** vs control **71.9307 / [70.2294, 48.3149,
97.2477]**. Deltas **+7.1052 AA** (guard +0.5 PASS), **+11.1604 old-domain mean** (guard +1.0
PASS), runtime **−0.26%** (guard +10% PASS), old micro-precision **+10.90 FUNSD / +15.62 SROIE**
(secondary, both positive). **The per-domain guard FAILS: CORD −1.0054** (guard ≥ −0.5).
All three falsifiable predictions HIT (FUNSD ≥ control+3, SROIE ≥ control, CORD ≥ 96).
matrix.npy == metrics.json on both arms; both exited rc=0. Evidence:
`results/dil_colar_adaptive_seed42/`, `results/dil_colar_seed42_k4_d10/`,
`results/gates/cacolar_t4_{adaptive,uniform_d10}.log`.

Verdict per prereg letter: **not a clean GO** — one guard fails. Substantively the effect is the
largest allocation gain measured in this project (+7.1 AA at equal bytes, ~2× the old-domain
retention of the shadow chain's best) and the CORD loss is the *designed* cost: CORD banks zero
docs, so it keeps only what the frozen trunk plus current-task training provide. The trade is
+22.3 points of old-domain retention for −1.0 on the task that never forgets.

RCA / next step: the guard was written for *augmentation* gates (add capacity, harm nothing),
and is mis-specified for *allocation* gates where spending is necessarily zero-sum. Do NOT relax
it retroactively for this run. Instead the admitted correction is one re-gate at
docs=[20,5,2]/rank=[64,128,64] (2 CORD docs, still inside budget by shifting 2 FUNSD docs), which
tests whether a token CORD bank recovers the −1.0 while keeping the old-domain gain. If it does,
CA-CoLaR v1 passes cleanly and proceeds to T5 (seeds 7/123, dil orders, cil_cord); if the
old-domain gain collapses, the allocation claim is coverage-only and the guard question is moot.

## MIXED (2026-08-17): CoLaSlot-Shadow-Replay d5/r64 multi-seed adjudication

Seeds 7 and 123 ran with matched CoLaR controls (sequential queue, all rc=0). Per-seed deltas vs
matched control: **seed 42 +1.9131 AA / +2.8117 old-mean** (all guards pass), **seed 7 −0.6565 /
−0.9847** (guards FAIL — full inversion, precision deltas also negative), **seed 123 +1.8456 /
+2.7684** (all guards pass, gain concentrated in SROIE +4.985 with precision +5.007). Pooled:
**+1.034 ± 1.196 AA**, **+1.532 ± 1.784 old-mean** (n=3; CI includes 0). Verdict: the d5
replay-shadow effect is **real on average but not seed-robust** — 2/3 GO under the per-seed
guards. Combined with the d50 NO-GO, the shadow branch stays closed as a method; the surviving
claim is the mechanism statement (gradient/read-policy alignment can help only in the
low-coverage regime, and not reliably). Evidence:
`results/dil_colaslot_shadow_replay_seed{7,123}_k4/`, `results/dil_colar_seed{7,123}_k4/`,
`results/gates/shadow_replay_seed_queue.log`.

Note for CA-CoLaR (docs/PLAN_CACOLAR_2026-08-17.md): the plan's independence from the shadow
branch is confirmed as the right call; the seed-7 inversion coincides with its control's high
FUNSD retention (67.9 vs 44.3 at seed 42) — per-seed retention-need heterogeneity is itself
response-surface input for plan task T1.

## GO (2026-08-16): CoLaSlot-Shadow-Replay bounded correction

The replay-only shadow gate **passes every preregistered guard** — the first positive in the
CoLaSlot successor chain. Task-1 continuation gate held: FUNSD **56.129624** (>= 54.594293) with
exact SROIE **80.833333**. Final result **62.702740 AA / 38.235500 AF /
[46.409675, 44.874972, 96.823574]** versus exact CoLaR 60.789595 / [44.321885, 41.339396,
96.707504]: deltas **+1.913145 AA** (>= +0.5), **+2.811683 old-domain mean** (>= +1.0), per-domain
**[+2.087790, +3.535575, +0.116071]** (all >= -0.5), and old micro-precision deltas
**+2.3490 FUNSD / +3.8038 SROIE** (both non-negative; SROIE recall -0.2761, so the gain is
precision-led — the shared-shadow false-positive failure mode is reversed). Runtime 9,395.12 s
(+77.74%), peak VRAM 2,427.27 MB, replay memory 3,265,920 bytes. matrix.npy equals metrics.json.
Evidence: `results/dil_colaslot_shadow_replay_seed42_k4/`,
`results/gates/colaslot_shadow_replay_d5_e5.log`.

RCA of the pass: training the shared rank-16 branch only on replay aligns its gradient exposure
with its read policy (aged owners only), removing the newest-gradient domination (33.2x CE ratio)
that sank the shared-shadow gate. Per prereg, the pass unlocks the scale-up ladder previously
barred: d50/r128 and extra seeds are now admissible; rank/depth/LR/router sweeps remain closed.

## NO-GO (2026-08-17): CoLaSlot-Shadow-Replay d50/r128 scale-up

The admitted d50/r128 seed-42 run finished at **86.569699 AA / [85.244292, 77.495108,
96.969697]**. Against a freshly run matched CoLaR d50/r128 e5 control (**86.382931 AA /
[85.265997, 76.913100, 96.969697]**), deltas are **+0.186768 AA** (< +0.5, FAIL),
**+0.280151 old-domain mean** (< +1.0, FAIL), per-domain **[-0.021705, +0.582008, 0.000000]**
(pass), and old micro-precision **+0.0995 FUNSD / +0.9312 SROIE** (pass). Runtime +76.78%
(10,272.3 s vs 5,810.8 s), peak VRAM 2,414.7 MB, replay memory 61,017,600 bytes. matrix.npy
equals metrics.json for both runs. Evidence:
`results/dil_colaslot_shadow_replay_seed42_k4_d50_r128/`,
`results/dil_colar_seed42_k4_d50_r128/` (new e5 control),
`results/gates/colaslot_shadow_replay_d50_r128.log`, `results/gates/colar_d50_r128_e5_control.log`.

RCA: the replay-shadow gain is a low-coverage phenomenon. At d5 the shadow recovers signal the
sparse replay bank cannot (+1.91 AA, +2.81 old mean); at d50 CoLaR's own replay already saturates
old-domain retention and the shadow adds only +0.19 AA at +77% runtime. The d5 GO stands as a
mechanism finding (gradient/read-policy alignment), not a scalable method: at the canonical
CoLaR operating point the correction is inert-positive and cost-dominated. Remaining admissible
step: extra seeds at d5/r64 to confirm the low-coverage effect; no sweeps, no further scale-up.

## DONE (2026-08-15): thesis results update; CoLaSlot-RF record corrected

**Correction:** CoLaSlot-RF's raw AA 72.42 was previously compared with an older CoLaR run at
60.79. Its saved dual evaluation gives the authoritative same-state slot-free control: AA 72.42
and final row [67.65, 52.67, 96.94], exactly equal to slots-on. RF is inert and fails the GO rule;
it is not a provisional pass. The detailed RF/RA NO-GO record below owns the verdict. Evidence:
`results/gates/colaslot_rf_d5_e5_retry2/`.

**Thesis results-only update DONE (structural reframe task #5 still separate):** new ch6 §6.6
(RCA snap verdict, 5 kill-test nulls, latent-replay falsification ladder + consistency control,
CoLaR control + conservation law + read-side negatives, PLaR, CoLaSlot-R/RF/RA NO-GO);
backbone section rewritten for the completed BERT slice; ch7 limitations aligned (BERT full grid
done — replay ordering reproduces text-only; LiLT partial 23 runs; BROS zero). Builds 128 pp.
**Pipeline fix:** analyze_results.py was renaming ALL bert runs to `bert_textonly`, pooling 14
methods into one cell (DIL 51.8±18.7 garbage) — now naive-only; bert added to SECONDARY_FAMILIES
(backbone tables carry the full BERT grid); ingest_to_thesis.py no longer clobbers the
hand-patched table_main.tex and ships table_backbone_AA → table_backbone_byscenario.tex.
Per-method BERT cells are tight (sd ≤5): dil er_cflat 78.3 / er 77.1 / der_pp 77.4 / joint 77.2 /
naive 39.4 / ewc 46.6; prompt-LoRA 34–39. bert lexslot (standalone soft) 37.96 ≈ naive —
standalone-≈-naive finding reproduces off LayoutLMv3.

## NO-GO (2026-08-15): CoLaSlot-FDP token-pathway gate

The fully RNG-matched FD gate is a decisive NO-GO. It reproduced every archived CoLaR epoch
checkpoint, so the same-state comparison is exact, but slots-on AA is **60.792444** versus
**60.789595** slot-free: **+0.002849 AA**, **+0.004273 old-domain mean**, and final-domain
deltas **[0.000000, +0.008546, 0.000000]**. Runtime rose from 5,286 to 8,224 seconds (+55.6%)
with no useful return. Evidence: `results/dil_colaslot_fd_seed42_k4/` and
`results/gates/colaslot_fd_boundaryfix_d5_e5.log`.

RCA rules out the document router as the old-task bottleneck: at final evaluation it makes zero
confident errors on FUNSD/SROIE with 92.0%/98.85% coverage. FD measures real centered-logit drift
(mean MSE 1.34e-3--2.33e-3), while null losses are only 0.99e-6--3.19e-6. A CPU fixture then
localizes the optimization ceiling: the configured first slot step cancels 0.0017% of a known
drift; 20--2,000x higher learning rates plateau near 4% cumulative cancellation. Removing the
support-null term improves one stable point to 13% but diverges at larger rates, matching RA
false-positive failure. The shared cause is structural: the owner route is document-level and
the existing head residual is only a low-rank linear token map, so entity correction conflicts
with staying zero on O/current support. Another LR/null sweep is closed.

The bounded `method=colaslot_fdp` gate also fails. Its full matrix is exactly equal to FD: AA
**60.792444**, final row **[44.321885, 41.347943, 96.707504]**, and deltas versus pure CoLaR
**[0.000000, +0.008546, 0.000000]**. Old-domain micro-precision changes are 0.000000 FUNSD and
+0.008709 SROIE, so the safety clause passes but the +0.5 AA and +1.0 old-domain efficacy clauses
fail by two orders of magnitude. JSON and `matrix.npy` agree. Runtime is 8,243 seconds (+55.9%
versus pure CoLaR), with unchanged 2,426 MB peak VRAM. Do not run d50/r128 or more seeds.

Mechanism RCA: every owner has eight pathways, so equality is empirical rather than forced by a
single active slot. `values` start random while `proj` starts at exact zero; the first FD update can
train only the projection, and subsequent 5e-5 bilinear updates scarcely make the energy keys
target-aware. Squared activation energy is also class-agnostic. FDP therefore applies random-feature
token weighting to the same tiny linear correction instead of resolving the entity/O support
conflict. FDP and energy-temperature/LR sweeps are closed. Evidence:
`results/dil_colaslot_fdp_seed42_k4/` and `results/gates/colaslot_fdp_d5_e5.log`.

## NO-GO (2026-08-15): CoLaSlot-FDA support-gated analytic drift compensation

The matched DIL/LayoutLMv3 seed-42 k4/d5/r64/5-epoch gate completed with **58.8440 AA /
43.8493 AF** and final row **[32.7904, 47.2666, 96.4751]**, versus the exact slot-free CoLaR
control **60.7896 AA / [44.3219, 41.3394, 96.7075]**. FDA therefore changes AA by **-1.9456**;
it fails the +0.5 AA, +1.0 old-domain, and no-domain-below -0.5 promotion clauses. Evidence:
results/dil_colaslot_fda_seed42_k4/ and results/gates/colaslot_fda_d5_e5.log.

The support gate itself is not the failure: final held-out owner precision/F1 is 100/100 for
owners 0 and 1 (owner 2 correctly abstains at 0/0). The analytic solve cancels 76.25% of owner-0
drift after task 1, 61.48% and 66.19% for owners 0/1 after task 2, yet retention falls FUNSD
54.0943 -> 37.0857 after task 1 and 44.3219 -> 32.7904 at the final boundary. SROIE improves
41.3394 -> 47.2666. This is direct additive-readout overgeneralization from repeated online residual accumulation;
RF keeps slot reads disabled and excludes slot parameters from the CoLaR optimizer, so shared-weight
co-adaptation is not implicated. It is not a router or numerical failure. Runtime is 8,309 s (+57.1% vs pure CoLaR) with
2,426 MB peak VRAM.

The CL4IE literature review sharpens the next direction: stability-gap/NMC, SLCA, RanPAC, LayUP,
ProtoNER, and IS3 all replace or align the drifting head with prototypes/closed-form statistics;
SER adds forward consistency to prevent low-buffer overfit; concept-drift work separates virtual
template shift from real label-boundary shift. The next admissible test is therefore a new,
post-task, route-conditioned prototype readout with an explicit O/NA null and held-out
precision gate. Do not run d50/r128, extra seeds, or FDA ridge/LR sweeps.


## NO-GO (2026-08-15): CoLaSlot-Proto route-conditioned prototype readout

The CL4IE review motivated a frozen route-conditioned prototype-vs-O margin on top of unchanged
CoLaR training. The RNG-corrected matched gate completed at **60.7526 AA** with final row
**[44.7447, 40.8055, 96.7075]**. Its exact slot-free CoLaR control is **60.7896 AA / [44.3219,
41.3394, 96.7075]**: deltas are **-0.0370 AA**, **-0.0555 old-domain mean**, and
**[+0.4229, -0.5339, 0.0000]** by domain. It fails both efficacy clauses and narrowly violates
the -0.5 per-domain safety floor on SROIE. No d50, scale/rank sweep, or extra seeds.

An initial run was invalid because prototype fitting performed dropout-active replay forwards and
advanced RNG before CORD. The preserved contaminated run diverged at CORD epoch 1 (93.2920 vs
94.9140). The fix evaluates prototypes inside a forked RNG context and restores model/slot-read
state; the corrected run reproduced every CoLaR checkpoint exactly. Final base-only scores also
match CoLaR exactly, proving the remaining failure is readout-level owner heterogeneity: owner 0
helps FUNSD while owner 1 harms SROIE despite both final support gates reporting 100% precision.
The next bounded test is owner-level leave-one-document-out prototype-utility gating, not a global
scale sweep. Evidence: results/dil_colaslot_proto_seed42/, results/gates/colaslot_proto_d5_e5.log,
and the preserved RNG-contaminated artifacts.

## NO-GO (2026-08-15): CoLaSlot-Proto-U owner-utility gate

The matched k4/d5/r64/5-epoch gate finished at the exact slot-free CoLaR result:
**60.789595 AA / 41.047186 AF / [44.321885, 41.339396, 96.707504]**. Training reproduced every
CoLaR checkpoint, including CORD **94.9140, 95.4755, 94.2226, 96.7075, 96.0308**. Owner-level
leave-one-document-out replay utility tied base and prototype token F1 at **100.00/100.00** for
both old owners, so both prototype reads were disabled. Runtime was 6,747.80 s, peak VRAM
2,426.20 MB, and replay memory 3,296,760 bytes.

This is a safe but inert NO-GO: replay documents are perfectly memorized even while held-out
old-domain F1 falls to 44.32/41.34, so training-memory utility cannot estimate test-time readout
utility. Do not sweep thresholds, rank, d50, or seeds. Close prototype/readout gating on this
memory. Return to the only large positive integration signal, CoLaSlot-R, and test the minimal
age-aware lifecycle: train slots normally but read only owners older than the current task, so
current-domain acquisition is exact CoLaR while old-domain slots supply retention.

## NO-GO (2026-08-15): CoLaSlot-Age read lifecycle

A matched single-FUNSD preflight falsified the age-only premise before DIL. Newest-owner slots off
and ordinary CoLaSlot-R slots on both score exactly **86.122548**, versus pure CoLaR
**87.293087**. Thus the current slot is already inference-inert; the 1.170539 acquisition loss is
training-time shared-weight co-adaptation. No DIL or sweep. Any successor must separate the exact
CoLaR base gradient from slot learning, keep base-only validation, and read slots only after aging.

## NO-GO (2026-08-15): CoLaSlot-Sidecar separated head slots

The matched k4/d5/r64/5-epoch gate preserved every CoLaR validation checkpoint exactly but
finished at **60.758764 AA / 41.093433 AF / [44.321885, 41.246903, 96.707504]**. Against exact
CoLaR, deltas are **-0.030831 AA**, **-0.046247 old-domain mean**, and
**[0.000000, -0.092493, 0.000000]**. The first aged read was already inert/slightly harmful:
post-SROIE FUNSD 54.0323 versus CoLaR 54.0943. Runtime was 5,762.27 s, peak VRAM 2,423.97 MB,
and replay memory 3,265,920 bytes. No d50/r128 or extra seeds.

RCA: gradient/RNG separation solved acquisition safety, but head-only sidecars became redundant as
the exact CoLaR head learned the same hard labels; their nonzero acquisition loss produced no
held-out retention gain. The original CoLaSlot-R's +14.02 AA therefore depends on a co-adapted
upper-layer branch, not a detachable head correction. Admit one bounded capacity discriminator:
the same separated objective with existing `head_late` representation slots. Current-task reads
remain off and the CoLaR base stays exact; no new router, target, memory, or optimizer is added.

## NO-GO (2026-08-16): CoLaSlot-Sidecar late-capacity discriminator

The matched k4/d5/r64/5-epoch gate preserved every CoLaR validation checkpoint but fell to
**59.050179 AA / 43.714345 AF / [43.456291, 36.870672, 96.823574]**. Against exact CoLaR,
deltas are **-1.739416 AA**, **-2.667159 old-domain mean**, and
**[-0.865594, -4.468725, +0.116071]**. FUNSD was already -2.210145 after SROIE. Runtime was
6,081.79 s, peak VRAM 2,458.46 MB, and replay memory 3,265,920 bytes.

RCA: adding detached representation capacity amplifies owner-read overgeneralization rather than
recovering the coupled signal; head-only is inert and head+late is harmful. Close detached-sidecar
capacity, LR, rank, and depth sweeps. The remaining causal hypothesis is a replay-trained low-rank
shadow branch that can co-adapt while the exact CoLaR base remains the newest/abstained fallback.

## NO-GO (2026-08-16): CoLaSlot-Shadow exact-base branch gate

Implemented a shared rank-16 residual at every plastic layer, trained with owner head slots on the
same current+replay sample while base parameters are frozen. It adds 196,608 parameters at k4
(about 0.16%), preserves Python/Torch RNG and base gradients, and reuses the existing router.
Single-FUNSD preflight matched every control checkpoint and final **87.293087 AA** exactly; the
preregistered seed-42 k4/d5/r64/5-epoch DIL gate is running. Six focused tests, Ruff/Black, Hydra
composition, and diff checks pass. The per-class exporter now enters each method's routed
diagnostic context so future precision evidence reflects the method rather than its base fallback.

Task 1 already gives FUNSD 50.13 versus CoLaR 54.09. The mechanism mismatch is explicit: current
examples dominate training of a shadow that is suppressed for the current owner and later read by
old owners. A minimal replay-only exposure fix is implemented and focused-tested. After the current
gate finishes, it gets an exact single-FUNSD preflight and may continue beyond task 1 only at
FUNSD >=54.594293 with exact SROIE 80.833333; failure closes shared shadows without sweeps.

The final gate is **57.5963 AA / 45.8371 AF / [41.1619, 34.9195, 96.7075]**, versus CoLaR
**60.7896 / [44.3219, 41.3394, 96.7075]**. It fails every efficacy and old-domain precision
guard. SROIE precision falls 6.12 points while recall falls only 0.14, identifying false-positive
overgeneralization. Runtime rises 77.95%. The replay-only single-FUNSD preflight is now running;
no d50, seeds, or shadow hyperparameter sweep.

## NO-GO (2026-08-15): CoLaSlot-RO online hard-label drift tracking

The matched gate completed with **72.4103 AA / 24.5255 AF / [67.5951, 52.6964, 96.9396]**.
Its same-state slot-free control is **72.4198 AA / 24.5113 AF /
[67.6494, 52.6704, 96.9396]**. Delta is **-0.0095 AA**, **-0.0142 old-domain mean**, and
**[-0.0544, +0.0259, 0.0000]** by domain. The only earlier-stage effect is -0.1973 FUNSD.

Owner hard-label losses average 4.47e-5--2.19e-3; final recall is unchanged, while tiny
precision shifts cancel. RO uses the same 3,266,040 bytes and peak VRAM as RF but adds
1,615 seconds (26.9 minutes). Saved JSON and matrix.npy match. The gate fails both improvement
clauses, closing hard-label head residuals regardless of timing. Evidence:
`results/dil_colaslot_ro_seed42_k4/`.

## NO-GO (2026-08-14): retention-only CoLaSlot-RF and acquisition-anchor CoLaSlot-RA

Both decoupled successors completed the matched DIL/LayoutLMv3 seed-42 k4/d5/r64/5-epoch
gate. Their slot-free trajectories are identical, so the comparison isolates the residual:
CoLaSlot-RF = **72.42 AA / 24.51 AF / [67.65, 52.67, 96.94]**; CoLaSlot-RA =
**71.99 AA / 25.15 AF / [67.36, 51.68, 96.94]**. The same-state CoLaR fallback is
**72.42 AA / [67.65, 52.67, 96.94]**.

RF's hard-label refit collapses to positive losses near 1e-5 and its final slot-on scores
equal the fallback exactly. RA fixes that optimization failure: acquisition-logit MSE losses
remain 2.82--5.70, but its residual changes final domains by **[-0.29, -0.99, 0.00]**,
giving **-0.43 AA** and **-0.64 old-domain mean**. SROIE recall is unchanged while
precision falls 1.19 points, directly identifying false-positive residual overgeneralization
from five replay documents. RA also costs 4.23% more memory than RF.

Both fail the preregistered GO rule (AA >= +0.5, old-domain mean >= +1.0, no domain below
-0.5). Do not run d50/r128 or more seeds. Post-task head residuals are closed in their
current form: hard labels provide no drift signal, while dense logit anchors provide signal
without document-level support. Any successor must first pass a held-out replay test that
enables only entity/class corrections with non-negative precision; current, foreign, and
non-entity support must be explicitly driven toward zero. Full RCA:
`docs/reports/colaslot_integration_rca_2026-08-07/report.html`.

## ACTIVE (2026-08-08): CoLaSlot-R cheap gate NO-GO; scalar abstention closed

Implemented `method=colaslot_r`: claim-before-mask owner-only gradients, silent unowned
slots, original token IDs retained in CoLaR replay, attention-mask/special-ID-aware signatures,
hard top-1 task-block routing, and scalar-margin abstention to zero slot contribution. The
historical `method=colaslot` config remains reproducible. Validation: 54 focused tests and
392 fast tests pass; focused Ruff/Black are clean.

The matched DIL/LayoutLMv3 seed-42 k4/d5/r64/5-epoch gate completed. CoLaR = **60.79 AA /
41.05 AF / [44.32, 41.34, 96.71]**; CoLaSlot-R = **74.81 AA / 20.60 AF / [64.31,
64.27, 95.87]**. Delta = **+14.02 AA / -20.45 AF / [+19.99, +22.93, -0.84]**.
The large signal fails the preregistered no-domain-below -0.5 guardrail on CORD.

A test-blind 5-fold out-of-fold calibration over all 1,575 training documents closed the
only permitted scalar-margin follow-up. At margin 0.05, overall coverage is 92.47% and
accepted accuracy 99.22%, but CORD accepted accuracy is 97.71% (16 errors). The zero-error
margin 0.326 leaves 0% FUNSD/SROIE coverage. No threshold on the 0.001 grid simultaneously
gives every seen stage/domain >=50% coverage and >=99% accepted accuracy. Therefore do not
rerun the cheap cell, d50/r128, or more seeds. Any successor needs either a different router or a slot lifecycle that makes wrong routes
non-destructive, plus a new preregistration—not another scalar sweep. Evidence:
`results/gates/colaslot_r_margin_calibration_seed42.json` and
`results/gates/colaslot_r_margin_feasibility_seed42.json`.

## ACTIVE (2026-07-23): Continual DocRE salient-graph replay gate implemented

Added CPU-only `scripts/docre_graph_replay_gate.py`. It emits random and graph-salient
DocRED replay memories under an exact serialized-byte budget after every task. The greedy
selector covers rare relations, relation-conditioned entity-type pairs, cross-sentence
relations, and bridge entities while retaining intact raw documents.
`tests/test_docre_graph_replay_gate.py` pins byte limits, graph coverage, and preservation.

Real Re-DocRED/DREEAM runs wait for the active GPU grids to drain because another dataset
job would violate the one-dataset-builder RAM rule. This remains outside the locked paper.

## ACTIVE (2026-08-05): BERT slice DONE (126/126); LiLT+BROS bootstrap ready for rented box

**BERT slice COMPLETE — 126/126, 0 missing.** Long haul (~9 days local, RTX 2060 6GB):
key lessons baked into ops — DocCL on BERT needs ~5.5G alone so it MUST run 1-slot on a
6GB card (2-slot silently OOM-looped ~16h before caught); er_cflat (SAM) × cil_cord is the
single heaviest combo; 3-slot RAM ceiling is real (each job ~4G RSS vs 14G cap). None of
these apply on a 24GB rental. Backbone tables still stale (Jun 29) — regenerate with
analyze_results.py once LiLT/BROS land.

**User renting the box (08-05).**  is the paste-ready
one-shot: git pull (MUST — grid-script fixes) → uv sync → the scoped LiLT+BROS launch
(252 planned = 126 lilt + 126 bros, DRY_RUN-verified clean, ~23 pre-done skip). On 24GB run
doccl at 2 slots (the 1-slot workaround is 6GB-only). BERT excluded — slices disjoint.
~3 days, ~00. After: pull results → analyze_results.py → ingest_to_thesis.py.

Thesis reframe (#5, CPU-only, examiner-annotated design ready) is the parallel work item.


## DONE (2026-07-18): Thesis reframe underway + B+ grid audited & ready (user gate: rent box)

**(b) Method chapter:** decision executed — chapter becomes *diagnostic-driven design +
pre-registered falsification, CoLaR as constructive control*. Design + drafted prose +
examiner annotations: `docs/THESIS_METHOD_CHAPTER_REFRAME_2026-07.md` (scout found the
LexSlot-as-headline framing threads through title page + ch1/ch3/ch6/ch7 — reframe is
thesis-wide). **DONE now: the ~25-site "buffer-free"→"small-buffer (200 exemplars)"
correctness pass across ch1/3/6/7** (the 87.3 row is the DocCL-hybrid w/ 200 exemplars;
per-seed 88.4/87.2/86.3 now cited; grep-clean; thesis builds 125 pp, no new errors).
REMAINING (session task #5): the structural rewrite per the design doc — resolve every
[EXAMINER] annotation first (esp. the frozen-trunk-causality overclaim in one outline
cell). Note: the "stale early-stopping prose" memory was itself stale — ch4/5/6 already
describe the convergence protocol correctly.
**(c) B+ grid:** audited plan `docs/BPLUS_GRID_PLAN_2026-07.md` — 355 missing cells
(LiLT 103, BROS 126 = zero coverage, BERT 117, LM3-lexslot 9), ≈293 GPU-h ≈ $88–147.
Empty-override bug (`${VAR:-}` on CORE/PROMPT/CURRENCY_METHODS) FIXED in
run_grid_multigpu.sh + setup_remote.sh, DRY_RUN-verified. **Blocked on user renting the
Vast.ai box (task #6); commands are paste-ready.**

**CoLaR-WSVD candidate (2026-07-18):** implemented `method=colar_wsvd` as a strict CoLaR
ablation: same per-document factors and bytes, entity-token rows weighted in the SVD
objective. DIL seed42 k4/d50/r128 result = **87.08 AA / 60.4 MB**, final row
85.90/78.13/97.21. This is below CoLaR 87.58 (89.19/76.30/97.25) and near CoLaR-Bal
87.11. Read: weighting buys SROIE (+1.84 micro; KEY +1.50, VALUE +1.80) but steals FUNSD
retention (−3.30), so it is a negative ablation, not a new headline. Do not spend more on
row reweighting unless paired with a constraint that prevents task-0 capacity theft.

## DONE (2026-07-17 eve): KILL-TESTS COMPLETE + ADJUDICATED — all corrections are nulls; re-exercise is the only lever

**All 5 kill-tests ran and were adjudicated against the amended prereg, every verdict
adversarially verified (5 independent recomputation agents; 2 of my draft verdicts were
corrected by them — full detail in `docs/RCA_FORGETTING_BASELINES_2026-07.md` kill-test
sections):**
- **Readout suite: KILLED as tested, family verdict "deferred — information destroyed
  pre-correction"** (best pooled recovery −0.1 AA; saturation: KEY mass ~7e-6, 2–3 orders
  below the 1e-3 branch threshold). Per-doc EM headline is dead; pre-written fallback
  framing applies. Scope note: funsd-KEY partially recovers (0→16.4) under marginal_match
  but pooled worsens (sroie degenerates to all-O).
- **Frozen-trunk: NOT CONFIRMED as clean causality** (3 of 4 conjunctive clauses fail;
  O-collapse is LESS under frozen trunk (13.3% vs 0.0%) = the escape-hatch direction;
  acquisition-invalid: cord at-learning 68.4). Old-class extinction DOES reproduce with
  trunk fixed; effect-size match to naive is confounded.
- **marginal_kl: provisional NULL** (KEY/HEADER 0.0 → H2 terminal case holds; H4′-direction
  movement negligible: VALUE +8.5, O-recall 0→3.1%; AA 39.1 below sanity window — open:
  KL/CE scale at λ=1, batch-2 MC noise; anchor code verified correct).
- **logit_adjust: no buffer-free signal at τ=1** (survival bar 0.0 vs >20; acquisition
  guard PASSES — at-learning within 5 of naive's actual 87.8/82.5/96.1; prereg's stale
  hard-coded reference corrected in-place with a dated note).
- **Consolidated (n=1, provisional): the snap is a SYMPTOM, not an invertible mechanism.**
  The head collapses predicted mass on under-exercised classes to ~1e-5 (prediction-mass
  statistic; weight geometry unmeasured); no eval-time reweighting recovers it in
  aggregate, marginal-level training objectives don't prevent it, extinction reproduces
  with frozen trunk, features survive (head-refit 55.3). **Only task-consistent
  re-exercise (replay) prevents it.** Paper gain: the "did you try cheap recalibration"
  hole is closed with a REGISTERED NULL.
**Next: fold into thesis/paper (falsification chapter + method-chapter reframe per the
pre-written fallback), then the B+ grid critical path. The method-chapter bet: per-doc EM
is dead as headline; honest options = negative-result chapter around the kill-test suite
+ CoLaR as constructive control, or a replay-adjacent evidence-store method (the only
open readout lever).**

## DONE (2026-07-17 pm): Kill-tests queued — brainstorm done, top-5 implemented, DOUBLY REVIEWED

**Method brainstorm (judged workflow: 4 lenses → novelty audit → 2 adversarial judges →
synthesis) produced a top-5 shortlist; all implemented + PRE-REGISTERED
(`docs/RCA_KILLTESTS_PREREG_2026-07.md`, commit 6d3f019) and queued as
`scripts/run_rca_killtests.sh` behind the read-side chain (~3.5 GPU-h):**
1. **Readout suite** (`rca_readout_fixes.py`): naive retrain + logit dump, then
   marginal-match oracle / prior-ratio one-step / per-doc Saerens–Latinne EM (task-ID-FREE
   — a doc's ~512 tokens are its own calibration batch). SUPPORT bar: pooled AA AND
   AA_old(funsd,sroie) both >15 pts recovery; cord non-regression ≤2; KILL family if <5
   pooled — split by the saturation check (KEY/HEADER prob mass ≥/< 1e-3 distinguishes
   "mechanism dead" from "information destroyed pre-correction").
2. **Frozen-trunk naive** (`--freeze-trunk`): pure-head causality; CONFIRM if extinction/
   O-collapse/snap-cos match full naive ±5 F1 / cos 0.02.
3. **marginal_kl** (KL to PRIOR-task mixture, 9 floats/task): H4′-vs-H2 disentangler;
   adjudicable rule = KEY/HEADER >20 refutes H2 terminal case; AA 45–55 is sanity only.
4. **logit_adjust** (balanced softmax under cumulative prior, buffer-free): survival bar
   KEY/HEADER >20; MUST check at-learning F1 (EWC-style acquisition regression voids it).
Killed by judges: OT recalibration (no mixed eval stream), DoLa layer-contrast (head is the
damaged locus), head-only replay (decoupled-carrier risk — kill reasoning flagged as weak,
revisit next brainstorm). Gated: Bayesian last-layer behind a 2-line head-only-EWC
ablation; NCM head waits for colar_knn results.
Outputs land in `results/rca/killtests/` + `results/rca/dil_{naive_frozen,marginal_kl,logit_adjust}_seed42_rca.json`
(rca_synthesize.py picks them up). Adjudicate strictly by the prereg doc.

**Adversarial reviews BOTH DONE before any kill-test executed (commits f79ac0b + afd26e9):**
- **Plan red-team** (4 attack angles → adjudicator): survives-with-amendments. Caught:
  marginal_kl mixture included the CURRENT task (self-anchor no-op at task 0, 1/(t+1)
  dilution — CRITICAL, silently changed what rule 3 tests; fixed to prior-only); KILL-rule
  blind spot (added saturation split); rule-1 CORD dilution (added AA_old + guard); AA
  45–55 demoted to sanity check; novelty framing corrected (logit_adjust = Menon/Ren
  mechanism in a new fixed-head-DIL regime; per_doc_em = Saerens–Latinne 2002 at document
  granularity — delta is the calibration unit, closest prior TTLSA arXiv:2211.15646);
  per-doc-EM terminal-case fallback framing PRE-WRITTEN in the prereg. Stale premise
  retired: **colar_knn λ=1.0 landed at AA 46.24 / BWT −6.51 — far below CoLaR 87.6; a pure
  kNN head is NOT a hedge** (λ=0.3 + colar_meta still pending).
- **Code review** (4 dimensions, findings adversarially verified): 2 real bugs beyond the
  red-team overlap — **per_doc_em zero-prior fixed point** (π init from CORD marginal with
  exact-zero KEY/HEADER locked those classes at 0 forever regardless of document evidence;
  reproduced in fp64; Laplace-smoothed + regression test) and **no self-duplication guard
  on chain scripts** (flock added to run_rca_killtests.sh + run_readside_gate01.sh via
  mv-replace so running instances keep their old inode). Earlier fp16 hazards (logit dump
  flooring extinct probs; KL underflow-NaN under Turing fp16 autocast) fixed in f79ac0b.
⚠ A duplicate parked `run_readside_gate01.sh` instance exists (kill was permission-blocked);
flock now prevents future duplicates, but the live pair predates it — if two train.py
appear simultaneously, kill the newer.

**KILL-TESTS EXECUTION STATUS (2026-07-17 15:20): chain RUNNING** (readout-suite naive
retrain live on GPU) after a ~5 h deadlock, root-caused and fixed:
- **Postmortem:** a leftover launcher shell's cmdline contained the strings both chains'
  `pgrep -f` guards watch (`scripts/train.py`, `run_readside_gate01`) → the duplicate
  read-side instance never exited, the kill-tests chain waited on it. Killed the zombie;
  chains flowed immediately. **Rule going forward: never echo watched script names into
  long-lived shells; prefer flock guards (already added to both chain scripts) over
  pgrep-cmdline matching for NEW scripts.**
- **m3 output-dir mismatch:** train.py saved colar_meta m3 to the run-name dir (no `_m3`
  suffix — m3 is the config default) while the chain's resume-check watched the hydra dir.
  Verified artifacts copied to `results/dil_colar_meta_seed42_d50_r128_m3/` (train.log in
  that dir proves provenance). Follow-up filed (session task #4; move to bd when unblocked).
- **Monitoring now stall-proof:** completion watcher (fires on COMPLETE marker or chain
  death) + stall detector (fires if chain alive but chain.log frozen > 60 min). Session
  task list tracks: monitor → adjudicate (strictly per amended prereg) → write results +
  push.

**Frozen-trunk kill-test IN (2026-07-17 pm):** old-class extinction repeats (FUNSD
HEADER/KEY = 0.0, SROIE KEY = 0.0), but the preregistered match fails: final AA 28.3 vs
full naive 37.8, driven by poor acquisition under a frozen pretrained trunk (diag
41.4/47.7/68.4). Verdict: acquisition-invalid for pure-head causality; useful only as
"extinction can happen with fixed trunk," not a clean effect-size confirmation.

**Readout kill-test result IN (2026-07-17 pm):** retrained naive smoke-passed (AA
39.78). Eval-time marginal repair is a clean negative: marginal-match/prior-ratio/per-doc
EM give AA 38.58/39.70/39.60 vs uncorrected 39.78; AA_old stays ~11. Saturation split: old
KEY/HEADER prob mass is <1e-3 (~4e-6--7e-6 KEY, <7e-7 HEADER), so the head has already
annihilated recoverable mass. Test-time readout correction is closed; next viable lever
must prevent extinction during training or re-exercise old classes.

**Deep RCA extension implemented (2026-07-17 pm):** `scripts/rca_deep_dive.py` adds the
paper-facing CPU consolidation pass over Tier-B JSONs and writes `results/rca/d_*.csv` +
`d_summary.md`. Focused test added in `tests/test_rca_aggregation.py`. Current summary
keeps the RCA verdict intact: failing methods concentrate final old-task extinctions and
the strongest cells snap to the just-trained marginal; modality-mask table remains a
quantification layer, not a new mechanism.

**READ-SIDE GATE 1+2 RESULTS (2026-07-17, adjudicated against the pre-written bar):**
- **colar_meta m3 = AA 79.48, BWT −5.62, row [84.7, 65.2, 88.6]** vs m1 control 85.85 and
  CoLaR 87.6: metaplastic consolidation is dose-dependently harmful (m1 −1.75 → m3 −8.1).
  **RCA prediction (colar_meta fails) CONFIRMED.** Read-side direction fully closed:
  both tracks (kNN blend, metaplasticity) are clean negatives.
- **colar_knn λ=0.3: AA 87.64, row [89.2, 76.5, 97.2]** vs λ=0 CoLaR 87.6 [89.2, 76.3, 97.2].
  Formal bar (AA ≥ 87.6, FUNSD not below 89.2, SROIE > 76.3) is met — at the boundary:
  +0.04 AA, +0.2 SROIE, n=1 seed. The ROADMAP's "beats λ=0 by REAL margin" clause FAILS.
  **Verdict: NO SIGNAL** — the kNN blend is inert at λ=0.3 and destructive at λ=1.0
  (AA 46.24; pure kNN head ≈ gate0 probe quality 44–68 F1, far below the trained head).
  **Do NOT build R3 (colar_mbpa).** Mechanistic read: blending in a weaker reader cannot
  fix the snap — consistent with the RCA (the fix must repair the readout, not average it
  with a noisier one).
- **colar_meta m1 control: AA 85.85 [87.7, 75.4, 94.4]** — fails to reproduce CoLaR
  (−1.75 AA): the metaplastic machinery costs performance even at m=1. m3 (training now)
  is therefore ambiguous-by-construction; per the pre-registered control rule, read m3
  only as consistent/inconsistent with the RCA's colar_meta-will-fail prediction, not as
  a clean effect size.
- gate0 probe (pretrained-trunk lower bound): best per-task kNN F1 funsd 55.8 / sroie 67.8
  (k8_top20/top5 softmax) — passed the ≥20 go bar, but the ceiling it implies explains the
  λ=1.0 collapse.

## DONE (2026-07-17 am): RCA COMPLETE — readout-marginal snap is the root cause; read-side chain running

**Tier B landed + Tier C adjudicated, then AMENDED after a 16-agent adversarial verification
pass** (`docs/RCA_FORGETTING_BASELINES_2026-07.md`; every load-bearing number independently
recomputed): **H2 SUPPORTED** (KEY extinction mask-uniform — naive/lwf KEY F1 = 0.0 under EVERY
valid mask; replay stratum clean, er KEY 85.8/91.3), **H4 SUPPORTED AS AMENDED (H4′ marginal
snap)** — the as-registered falsifier had a drafting flaw (O is both SROIE's dominant label,
84.8% not 99%, and the falsifier target); the decisive amended test: the head's output marginal
on old tasks snaps to the just-trained task's gold marginal (cos 0.997/0.998 naive/lwf at b1 vs
0.60 to own gold; replay stays 0.99–1.00 to own gold; `results/rca/c_marginal_snap.csv`).
**H3 REFUTED** (trunk displacement FRONT-loaded: naive early 0.70 vs late 0.03), **H1 REFUTED
as mechanism** (raw spread is a baseline-floor artifact — normalized retention spread 4.7pp for
naive = lockstep annihilation; text embed moves LEAST 5/6; layernorm moves most, 262–10,055×
text embed), **QA NOT SUPPORTED** (no receipts-overlap signature).
New verified findings the snap explains: **O-collapse** (naive/lwf O-row acc exactly 0.00% at
final — invisible to seqeval; O dies at b2 not b1 because the b1 snap direction WAS O);
+7.9 "recovery" = snap rotation, not healing; **EWC protection is task-0-biased** (funsd resists,
sroie snaps 0.986 — and ewc's sroie at-learning was crippled 41.9 vs 82.5: acquisition failure);
head/trunk displacement 8–500× (ewc ~1× = Finding-2 migration, still forgets 43 pts).
**Root cause: readout-marginal snap on a class-asymmetric substrate.** Coheres with F1,
head-refit oracle, F3. Implications: attack the readout (read-side premise re-derived
independently); marginal/logit-prior recalibration = mandatory cheap baseline; targeted
minority-class replay (but SROIE's 3.7% KEY share was NOT enough — re-exercise must hit the old
distribution); colar_meta predicted to FAIL (its queued run = the falsification test).
Caveats: Tier B n=1 seed; colar row = canonical k8/d5/r64 not headline.
**3-seed conservation verdict IN:** base 87.8 = colar_bal 87.8 > kcenter 87.0; no lever beats base
at any seed; redistribution softer than seed-42 suggested (bal: SROIE +1.8 for FUNSD −0.5).
**Read-side chain relaunched** (Gate 0 probe → colar_knn λ runs → colar_meta m1/m3),
log `results/readside_gate01.log`.

## DONE (2026-07-16 eve): Deep forgetting RCA of baselines (Tier A DONE — two findings; Tier B queued)

**User pivot:** before more method design, RCA the baselines' forgetting patterns + the
multimodal correlation. Plan: `/home/thanh/.claude/plans/let-s-find-a-way-humming-flurry.md`.

**Tier A (artifact mining, commits b688682) — findings from 223 per-class runs + 308 matrices:**
1. **M5/Q6 ANSWERED — ER is NOT label-degenerate** (reviewer's suspicion refuted): er KEY-F1
   87.2 ≈ AA 86.8; er_cflat KEY 89.3. The degeneracy signature (KEY dead, VALUE carries AA)
   instead marks the FAILING methods: naive & lwf & the whole falsified buffer-free chain show
   **TOTAL KEY+HEADER extinction (F1=0.0)** while VALUE survives ~47. Forgetting is
   class-asymmetric; aggregate AA hides it. LwF protects nothing (KEY 0.0 despite distillation).
   → `results/rca/a1_*.csv`, `a1_summary.md`.
2. **Forgetting is one-boundary collapse, not decay:** naive −75.6 F1 at the first boundary
   then +7.9 RECOVERY; family signatures differ (penalty dampens ×4 to −19.5; replay flattens
   to −1..−4; prompts: small immediate but ongoing leak — immediate_share <0.6).
   → `results/rca/a2_*.csv`, `a2_signature.png`.
3. **A3 null confirmed mechanically:** 0/196 tb dirs have displacement scalars
   (tensorboard.diagnostics never enabled) → per-baseline locus requires Tier B. → `a3_verdict.md`.

**Tier B (commit 2b5645f, PROMOTED 2026-07-16 eve** — read-side gate chain KILLED before it
produced anything; queue is now seed sweep → RCA chain, log `results/rca/chain.log`): core-6
{naive, ewc, lwf, er, der_pp, colar} dil seed42, per-boundary probes: modality-ablated eval
(train FULL, eval {FULL, TEXT_ONLY, TEXT_LAYOUT, IMAGE_LAYOUT}) + per-class F1 + token confusion
(new `doccl/eval/confusion.py`) + Fisher displacement per component/depth. Output
`results/rca/dil_<m>_seed42_rca.json` (n=1 seed, provisional).

**Pre-registration + synthesis shipped (2026-07-16 eve, commits cab6c3a + 5ae2131):**
- `docs/RCA_HYPOTHESES_2026-07.md` — H1 modality-asymmetric drift (a: text / b: layout-vision),
  H2 head/label interference (modality-independent), H3 late-layer integration drift, H4
  recency/logit bias; decision rules against the Tier B JSON schema written BEFORE data exists
  (mask-validity guard ≥20 F1, method strata, H2/H4 disambiguation, QA sroie-vs-cord overlap).
- `scripts/rca_synthesize.py` (+ fixture test) — reads the 6 JSONs, emits
  `results/rca/c_modality_deltas.csv`, `c_confusion_flow.csv`, `c_displacement_vs_signature.csv`
  + 2 figures. Ready to run the moment Tier B lands.
**Tier C next session:** smoke-check naive (FULL AA ≈ 40 ± 3 vs head_refit_oracle 39.56), run
`rca_synthesize.py`, adjudicate H1–H4 per the pre-registered rules in
`docs/RCA_FORGETTING_BASELINES_2026-07.md`, then re-brainstorm methods against the RCA.

## QUEUED (2026-07-16 pm): READ-SIDE memory for CoLaR — kNN readout + metaplastic weights (built, runs queued)

**New direction (user-chosen, brainstormed):** every dead memory attempt (LARM, LexSlot-on-CoLaR,
nullspace_analytic) was a training-time WRITE — the redistribution law governs exactly that lever
class. Untested territory: READ the store at inference. Two assets nobody used: (1) layers <k are
frozen after task 0, so layer-k features are drift-free forever ⇒ CoLaR's SVD store is a stable
token-level labeled DATASTORE (LARM's stale-correction failure cannot occur below k); (2) forgetting
is head-localized (Finding 1) ⇒ replace the forgetting locus with a memory read.

**Shipped (commits dcedb94, 8ee7abd, tests 348 green):**
- `colar_knn` (R1): eval-time blend p=(1−λ)·softmax(head)+λ·kNN-vote over banked layer-k tokens
  (`doccl/methods/latent_datastore.py` shared view, zero extra bytes; λ=0 ≡ CoLaR, λ=1 = pure
  memory head). Training untouched ⇒ any gain is non-redistributive by construction.
- `colar_meta` (M1): Benna–Fusi m-level metaplastic consolidation on the plastic bucket, composed
  with replay (new no-op `_post_optimizer_step` hook in LatentReplay). FALSIFICATION-RISK track
  (EWC precedent); m=1 control must reproduce CoLaR before m=3 is read.
- `scripts/gate0_knn_probe.py`: NO-training layer-k kNN separability probe (pretrained trunk =
  conservative lower bound). Go/no-go for R1.
- R3 (`colar_mbpa`, MbPA-style episodic head adaptation) is designed in the plan but NOT built —
  conditional on R1 Gate 1 signal.

**Queued detached** (`scripts/run_readside_gate01.sh`, PID survives session, log
`results/readside_gate01.log`): waits for the seed sweep → Gate 0 probe → if ≥2/3 tasks best-F1≥20:
colar_knn λ=0.3 + λ=1.0 at k4/d50/r128 seed42 → colar_meta m1/m3 pair at k8/d50/r128.
**Gate 1 bar:** AA ≥ 87.6 with FUNSD not below 89.2 AND SROIE above 76.3 (non-redistributive win);
SROIE↑/FUNSD↓ 1:1 = just another frontier point, log plainly. λ=0 baseline = existing colar r128 run.
Plan file: `/home/thanh/.claude/plans/let-s-find-a-way-humming-flurry.md`.

## PRIOR ACTIVE (2026-07-16): Retention is REDISTRIBUTIVE under compressed replay — CoLaR is on the frontier

**Last decision:** do NOT build CoLaR+LexSlot (the KT doc's plan A). Killed on evidence, not opinion.

Chased CoLaR r128's one hole (SROIE 76.3 vs joint 83.4 = the whole AA gap) with two independent,
already-in-code levers. Both hit the target; neither improves AA; they don't stack. Full table +
reasoning in `EXPLORE.md` §7.

| dil k4/d50/r128 s42 | AA | BWT | [FUNSD, SROIE, CORD] |
|---|---|---|---|
| CoLaR r128 (base) | **87.6** | −1.7 | [89.2, **76.3**, 97.2] |
| + kcenter | 86.8 | −2.6 | [84.5, 78.9, 96.9] |
| CoLaR-Bal (soft) | 87.1 | −2.7 | [84.0, **79.9**, 97.4] |
| CoLaR-Bal + kcenter | 87.3 | −2.5 | [86.0, 78.6, 97.3] |

**The law:** SROIE rises only by spending over-held FUNSD (~1:1). FUNSD's 89.2 is *above* the joint
oracle's 87.9 — surplus that is load-bearing, not headroom. Generalizes E1's selection-redistributes
result to a mechanistically unrelated lever (loss shape) and to their combination.

**Why LexSlot helped DocCL but won't help CoLaR:** DocCL (84.7) sat *below* the frontier — slack on
every task, so LexSlot's buffer-backed capacity lifted all three at once (FUNSD/SROIE +4.7, CORD +2.0).
That row is also buffer-based (200 exemplars + KD + Fisher; standalone LexSlot = 42.2). CoLaR is on
the frontier and already replays. Same tool, different regime, opposite payoff.

**Shipped:** `doccl/methods/colar_bal.py` + config + 4 tests (commit e44e42b) — soft-target replay
for compressed latent replay. Literature gap (nobody reweights the reconstructed gradient of
compressed replay for minority classes); mechanism works (SROIE +3.6), conservation caps it.

**Blocked on / next:** the law is **PROVISIONAL — seed 42 only, dil only, LayoutLMv3 only**. The
86.8–87.6 band sits inside the ±1.1 joint seed sd. Seeds 7/123 running detached
(`scripts/run_seeds_conservation.sh` → `results/run_seeds.log`). Read those 6 runs before writing
the law into the paper. If the band survives 3 seeds → Finding 3c in the diagnostic chain.

## SUPERSEDED (2026-07-10 late): SLR falsified → successor idea "Consistency Law → PLaR/CoLaR"

SLR/AGLR/coreset all FALSIFIED (see GATE-0 FINAL below) but proved the **consistency law**
(whole-doc (feature,position,label) binding is the necessary replay ingredient; +27 AA single-var
control). Successor idea (wiki `ideas/2026-07-10-consistency-law-replay.md`): keep consistency,
remove/shrink private storage —
- **PLaR (proxy latent replay):** bank pseudo-labeled PUBLIC docs (WildReceipt) as whole-doc
  frozen-trunk latents; ZERO private bytes. Implemented (`doccl/methods/proxy_latent_replay.py`,
  3 tests green, registered).
  **First run (dil 5ep, 5 docs, τ=0): AA 45.8 [FUNSD 40.2, SROIE 4.1, CORD 93.1]** — beats every
  falsified marginal (36.7–41.9; FUNSD retention 40.2 vs their 14–29 = the consistency effect is
  real on public docs) but below real-private-doc replay at same count (d4=63.8) and below the
  ≥70 gate. Mid-task SROIE still collapses.
  **d50 RESULT: AA 57.2 [FUNSD 73.3, SROIE 5.1, CORD 93.3]** — the count lever WORKS where
  pseudo-labels are informative: FUNSD 40.2→**73.3** (ABOVE private-doc d4's 60.8!). But SROIE
  stays dead (5.1). **Diagnosis: class density.** FUNSD labels are dense (most form tokens are
  question/answer → proxies carry rich signal); SROIE has 4 sparse entity types → argmax
  pseudo-labels on receipts ≈ 99% "O" → replay carries no SROIE-class gradient.
  **SOFT RESULT (00:52): AA 58.9 [FUNSD 76.1, SROIE 7.0, CORD 93.5]** — incremental (+1.7 AA);
  FUNSD keeps climbing (76.1 = +15 over matched private d4!) but SROIE barely moves. So sparse-class
  starvation is only part of it: either the head puts ~zero SROIE-class mass on WildReceipt tokens
  (now SELF-DIAGNOSED: banking logs a pseudo-label class histogram per boundary), or the mid-task
  needs the CONVERGENCE budget (every PLaR run so far = 5ep diagnostic; the 87.3 reference needed
  10ep early-stop).
  **CONVERGENCE RUN IN FLIGHT overnight** (`_run_plar_d50_soft_conv`, task `by803p7bw`, ~2.5-3h):
  d50+soft at the default 10ep/val-F1-early-stop. ⚠️ writes to the SAME run-name dir
  `dil_proxy_latent_replay_seed42_d50_soft` (epochs not in name) — the 5ep result is archived at
  `..._d50_soft_5ep_archive/`. Morning read: (1) the class histograms at each boundary (does the
  post-SROIE head label ANY proxy tokens with SROIE classes?), (2) the converged matrix. Decision:
  SROIE ≥30 → PLaR fully alive → Vast.ai sweep + CoLaR build. SROIE still ~7 + histogram shows no
  SROIE mass → the proxy pool carries no SROIE signal → mixed/matched pool or class-balanced
  weighting; if those fail, PLaR is a *partial* method (grounds dense-label tasks only) and folds
  into the law paper as the bounded-transfer finding.

  **PLaR ladder so far (dil, zero private bytes):** d5-hard 45.8 → d50-hard 57.2 → d50-soft 58.9
  → d50-soft-CONV pending. References: private d4-5ep 63.8, private d5-conv 87.3, best falsified
  marginal 41.9, naive ~41.

  **DIAGNOSIS v1 (01:30) — RETRACTED at 02:15.** First reading of the histograms ("head assigns
  zero SROIE-class labels; task 0 captures the pool's LABELS") was label-centric and WRONG for
  dil: `build_dil` gives every task the SAME `DIL_UNIFIED_LABELS` (9 shared BIO tags — there are
  no task-specific classes to capture; histogram ids 0-6 are 7/9 unified tags, ids 7-8 = the
  OTHER tags simply never predicted on proxies). Consequently the built `task_masked_labels` fix
  is a **NO-OP on dil** (kept in the code — tested, and genuinely relevant for CIL later); the
  planned tmask dil run was NOT launched (would prove nothing).

  **DIAGNOSIS v2 (02:15, fits ALL of tonight's data): FEATURE-REGION COVERAGE.** Proxy replay
  anchors the head's conditional only on the PROXY POOL'S OWN feature region (as encoded by the
  trunk that was FUNSD-tuned before freezing). FUNSD's region is covered → held (76.1 at 5ep);
  SROIE's region (different OCR/layout stats, encoded through a FUNSD-shaped frozen trunk) is NOT
  covered by WildReceipt latents → drifts freely (5.5) no matter the label format (hard/soft
  identical) or count (d5→d50 no help on SROIE). Also explains why CONVERGENCE HURT proxy replay
  (58.9@5ep → 50.3@conv) while it helps real replay (63.8→87.3): longer training = more drift in
  the uncovered region while the anchor holds only elsewhere — a noisy/offset anchor amplifies.

  **MORNING PLAN (in order, cheap→decisive):**
  1. **Coverage probe (no training, ~1 min):** mean/quantile L2 (or CKA) distances at layer 8
     between each task's latents and (a) the WildReceipt pool, (b) the other tasks. Prediction:
     d(SROIE, pool) ≫ d(FUNSD, pool). Confirms/kills v2 before any run.
  2. **Coverage-targeted proxy selection (~20 lines):** at boundary t, RETRIEVE from the pool the
     `docs_per_task` docs nearest to task t's latent centroid (CIL-QUD-style) instead of random.
     Run dil 5ep d50+soft. If SROIE recovers → PLaR alive + the coverage law is the paper's second
     mechanism. If the pool simply contains NO docs near SROIE's region (probe shows a floor) →
     enlarge/mix pools (XFUND + more corpora) or accept bounded-transfer.
  3. Optional: pool-mix ablation; CoLaR build (per-doc SVD, independent of the proxy question).
- **CoLaR (compressed latent replay):** per-DOC SVD is low-rank (r64=87.7%, r128=95.2% measured —
  unlike the full-rank pooled space) → 6–12× private-latent compression preserving whole-doc
  binding. NOT built yet; next after PLaR verdict.

## SUPERSEDED same-day (kept for the record): DIRECTION PIVOT → method-led ICML paper "SLR"

**This supersedes the 2026-07-04 "diagnostic + falsification, NOT a method paper" lock recorded
lower in this file.** Trigger: Exp #5 latent replay (dil AA 87.3 / BWT −2.3, ≈ ER-200, −1.4 from
joint oracle, **zero raw documents stored**) is a *method*, not a tombstone — and a new NTK theorem
("Catastrophic Forgetting is Low-Rank", arXiv 2606.18024, ICML'26 wksp) formalizes our migration law.

**SLR — Spectral Latent Replay (one line):** generative latent replay, rank-budgeted to the
forgetting eigenmodes, injected as gradients confined to the low-rank old-task subspace, on
structured head-growing document IE (BIO CIL → Re-DocRED relations).

**User decisions locked (2026-07-10):** (1) method-led spine; (2) include the Re-DocRED relational
extension as a headline; (3) feasibility on local RTX 2060 first, headline sweep on Vast.ai.

**Design + pre-registered gates** live in the wiki:
`CL4IE/wiki/ideas/2026-07-10-spectral-latent-replay-icml.md` (+ 4 new `CL4IE/wiki/sources/` pages:
forgetting-is-low-rank, aglr-cl-generative-latent-replay, replay-can-increase-forgetting, gpm; and
provable-effects-data-replay). Gate order:

- **Gate 0 (existential, LOCAL, do first):** beat a faithful AGLR-CL port (GMM whole-latent replay,
  the closest prior art). No delta → no paper.
- **Gate 1:** rank-`r` spectral memory vs raw-latent buffer at equal bytes (target AA ≥ 84 @ ≤25% bytes).
- **Gate 2:** subspace-projected replay vs full-gradient replay.
- **Gate 3 (Vast.ai):** AA-vs-memory-bytes Pareto sweep over `r` = the headline figure.
- **Gate 4:** cross-backbone (LiLT/BROS/BERT).
- **Gate 5:** Re-DocRED continual-relation stream (relational headline extension).

**Code base:** existing `latent_replay` method (commit 93aa144, 13 tests green) — do NOT rebuild.
**bd is write-blocked** (schema-migration fork, see memory [[clue-bd-dolt-lock]]) → SLR tasks tracked
here + in ROADMAP, not in beads, until the user reconciles bd.

**Gate-0 implementation progress (2026-07-10 session):**
- Built `SpectralMemory` (`doccl/methods/spectral_memory.py`, SLR) + `AGLRReplay`
  (`doccl/methods/aglr_replay.py`, the AGLR-CL comparator), both subclassing `LatentReplay`
  and sharing its layer-`k` replay hook. Wired into `train.py` (registry + `_STD_FORWARD` +
  run-name suffixes), configs added, 12 unit tests green (incl. 2 GPU-free regressions for the
  LayoutLMv3 image-patch width bug: layer-`k` hidden is text+patches = 709, wider than the
  512 attention_mask; capture slices to text, replay reconstructs at full width).
- **Bug found + fixed in first smoke:** initial SpectralMemory synthesised replay features
  from the *global* task distribution, decoupled from each token's label → replayed
  head-gradient taught noise → AA 39.3 / BWT −56 (worse than naive). Rewrote as
  **class-conditional inside the shared low-rank basis**: per-class Gaussians in the top-`r`
  forgetting subspace, so feature↔label stay coupled AND the "index by forgetting subspace"
  identity holds. Both methods vectorised (per-token loop → gather+matmul, 57→25 ms/call).
- **Operating-point finding (important):** 1-epoch is NOT a valid judging budget. Reference
  `latent_replay` at 1 epoch = **AA 64.9** (not the converged 87.3) — under-converged by ~22
  pts. So Gate 0 MUST run at convergence (grid default: ~10ep, val-F1 early stop), not on
  1-epoch smokes. Corrected class-conditional `spectral_memory` convergence run is IN FLIGHT.
- **Memory razor confirmed in bytes:** latent_replay banks 5.4–16 MB (raw docs);
  spectral_memory banks ~0.15 MB (≈36–70× smaller). Whether it holds AA is the pending answer.
- **⚠️ Clobber to flag:** my 1-epoch reference run wrote metrics to the run-name dir
  `results/dil_latent_replay_seed42/` (hydra.run.dir only redirects logs, not metrics),
  overwriting a prior result there (now AA 64.9, 1-epoch). `results/` is not git-tracked so the
  prior value is unrecoverable, but it is regenerable by re-running at convergence and STATE
  records the canonical 87.3. Future smokes must use a method/seed that doesn't collide.

**Compute reality (2026-07-10):** a converged dil run WITH active replay is ~90–120 min on the
local RTX 2060 (CORD alone ~5 min/epoch × up to 10ep; a 50-min timeout died mid-CORD at ep4/10,
no matrix). The local box has now served its purpose — it proved the *mechanism* (methods run
e2e on real LayoutLMv3, replay fires, class-Gaussians bank, 36× memory compression, the
class-conditional fix is correct). **Recommendation: move the actual Gate-0 comparison + the
whole gate sequence to Vast.ai** (user pre-approved), where a dil run is minutes not hours. A
capped `method.epochs=5` local run is in flight ONLY to get a first read on the retention matrix
(row 3 = does task-0/task-1 survive); it is a feasibility confirmation, not the headline number.

**Observability gap (follow-up):** replay loss is only visible via tqdm postfix (`\r`, not
flushed to piped logs) and `train/loss` is logged once per task-end, so replay *magnitude*
can't be watched live. Mechanism is verified 3 ways (unit test injection→plastic-grad, fit log
shows banked class-Gaussians, `_sample_replay` returns non-None when populated), but before the
Gate-1 rank sweep, add a per-step `train/replay_loss` TB scalar to `LatentReplay.train_task`
(or override in SpectralMemory) so replay activity is directly auditable.

**GATE-0 RESULT (2026-07-10, spectral_memory dil 5ep, class-conditional, rank-16): NEGATIVE.**
AA 41.9 / BWT −66.3, matrix row3 [FUNSD 29.3, SROIE **3.2**, CORD 93.1] — the LexMem-v5 mid-task
collapse signature. Only the current task survives; all prior tasks destroyed. Memory 0.454 MB.

**Diagnosis (the failure is isolated and mechanistic):** the difference vs latent_replay (which
grounds the head to AA 87.3 at the SAME frozen `k`=8, replaying REAL activations) is *real vs
synthesized* features. A rank-16 per-class Gaussian is too crude an approximation of the layer-`k`
feature distribution — the head trained on that thin surrogate can't hold its boundary against
the trunk's rich real features for the new task. NOT the label-coupling (fixed, still collapses),
NOT the frozen boundary (real replay at same k works). **The compression itself is the failure:
forgetting may be low-rank in OUTPUT space (the theorem), but reconstructing head-INPUT features
from a low-rank model loses what the head needs.** The razor's premise ("compress to the
forgetting rank and it still works") is FALSE at rank 16 on doc-IE.

**RANK-FIDELITY PROBE (2026-07-10, cheap GPU probe, `scratchpad/rank_probe.py`): the razor's
premise is broken.** Layer-8 LayoutLMv3 features are NOT low-rank — cumulative variance explained:
r16=42.9%, r32=55.2%, r64=68.3%, r128=81.0%, r256=91.6%, r512=98.3%. So rank-16 discarded 57% of
the feature signal (why the head collapsed), and you need r≈256 (1/3 of d=768) for 90% — which is
NOT compression. **The theorem's low-rank is the OUTPUT/NTK space; the head-INPUT features are
near-full-rank. SLR conflated the two.** A rank that works ≈ a rank that doesn't compress → the
accuracy-vs-memory Pareto curve the paper rests on almost certainly does NOT dominate raw replay.

**FORK RESOLVED (2026-07-10): AGLR-CL full-`d` ALSO COLLAPSES.** dil 5ep: AA 39.4 / BWT −69.8,
row3 [FUNSD 22.4, SROIE **3.0**, CORD 92.9], mem 0.387 MB. Full-`d` per-class Gaussians (NO rank
truncation) are *worse* than SLR rank-16 (41.9). So rank was never the issue.

**DECISIVE VERDICT: generative *feature-space* latent replay is dead for doc-IE.** Both variants
(rank-`r` subspace AND full-`d` class Gaussians) reproduce the LexMem-v5 mid-task collapse; only
REAL banked activations ground the head (latent_replay 87.3 @ same frozen k=8). The three-way at
matched budget/frozen-boundary isolates it to the *synthesis*:
| method (dil, k=8) | replay features | AA | SROIE final F1 |
| latent_replay | REAL activations | 87.3 (conv) / 64.9 (1ep) | survives |
| SpectralMemory (SLR) | synth, rank-16 subspace | 41.9 | 3.2 |
| AGLR-CL | synth, full-`d` per-class | 39.4 | 3.0 |
**Mechanism:** a per-class *marginal* Gaussian (mean+var, even full-`d`) discards the joint
feature structure real activations carry; the head trained on that marginal can't hold its
boundary against the trunk's real new-task features. This is a NOVEL, sharp addition to the
falsification chain — not just "buffer-free memory fails" but "the fabricated features must carry
joint structure a per-class Gaussian can't, and doc-IE features are too high-rank to summarise."

**RECOMMENDATION → revert to the diagnostic+falsification framing (the pre-2026-07-04-pivot
plan).** The method-led pivot was worth testing (Exp #5's 87.3 real-latent-replay looked like a
method) but the buffer-free version hits the wall the program already mapped. SLR/AGLR become the
strongest buffer-free tombstone yet, with a crisp mechanism. This is an ICML-viable *diagnostic*
paper, not a method paper — matches STATE history's original venue read (CoLLAs/TMLR/ACL-Findings
for pure-diagnostic; the migration law + this convergent negative could carry an ICML main-track
diagnostic).

**DECISION TAKEN (user, 2026-07-10): option (b) — test the coreset variant.**
Built `CoresetMemory` (`doccl/methods/coreset_memory.py`, subclasses AGLRReplay to reuse its
capture): stores **k-means centroids of REAL layer-k activations** per (task,class) — real feature
points (joint structure preserved), not a fitted marginal. 5 unit tests green incl. a check that
every replayed vector EQUALS a stored real centroid. Wired (registry/`_STD_FORWARD`/run-name/
config). **dil epochs=5 run IN FLIGHT** (`_run_coreset_c5`). This is the decisive test:
- Coreset survives (task-0/1 F1 hold, AA ≫ 42) → real feature modes DO ground the head → a genuine
  buffer-free-ish method; the failure was *synthesis*, not feature-replay per se. Pivot back toward
  a method paper (coreset = the method), run rank/centroid sweep + AGLR/latent baselines on Vast.ai.
- Coreset ALSO collapses → feature-space replay (synthetic AND real-coreset) is dead for doc-IE;
  falsification chain complete and airtight → diagnostic paper. Either way, publishable.

**Bench (dil, k=8, 5ep unless noted), AA / SROIE-final-F1 / mem:**
latent_replay(real, full) 87.3(conv)/survive/16MB · SpectralMemory 41.9/3.2/0.45MB ·
AGLR-CL 39.4/3.0/0.39MB · **CoresetMemory (real centroids) 36.7/2.9/0.48MB — ALSO COLLAPSED.**

**CORESET RESULT (2026-07-10): real centroids collapse TOO (AA 36.7, worst of the three).** So the
failure is NOT synthesis (real feature points fail identically) — my marginal-vs-joint hypothesis
was WRONG. New root-cause hypothesis, sharper: **the CARRIER design is the bottleneck.** All three
failing methods drape features onto a few (4) fixed carrier skeletons, whereas latent_replay
replays 5 WHOLE REAL documents/task. Two possible culprits, being tested:
1. **Carrier diversity**: 4 layouts/task can't represent the task (latent_replay has 5+ varied).
2. **Feature↔position decoupling** (deeper): a carrier pastes a class-`c` feature onto position `t`
   whose bbox/mask came from a DIFFERENT original token — so feature[t], bbox[t], label[t] are no
   longer mutually consistent, unlike a real replayed doc where they co-occur. This may be fatal
   regardless of carrier count.

**CARRIER-DIVERSITY TEST RESULT: still collapses.** 50 carriers + 16 centroids/class → AA 39.7,
SROIE 3.1, at 4.05 MB (8× the memory). Diversity is NOT the fix. So the carrier-scaffold approach
is dead: pasting features (real or synthetic) onto layout skeletons cannot ground the head at ANY
carrier count.

**DATA-HYGIENE (fix before write-up):** (1) the ladder table must have ALL methods at epochs=5 for
a fair compare — the `dil_latent_replay_seed42` on disk is the stale 1-epoch REFERENCE (AA 64.9),
NOT the converged 87.3 (that was a prior-session early-stop run) and NOT 5ep. Need latent_replay
@ 5ep for the honest headline; the docs=4 control gives one 5ep latent point, may also want docs=5
@ 5ep. (2) `dil_coreset_memory_seed42` (4-carrier, AA 36.7) artifact was rm'd launching carrier-50;
numbers recorded here but re-run to regenerate the artifact if needed for the paper table.
(3) `replay_memory_bytes` only persists for runs AFTER the instrumentation (spectral/aglr/coreset
have it; the old latent ref doesn't) — the 5ep latent re-run will capture it.

## LARM (fuse CoLaR replay + lexical routing) — GATE 0 PARTIAL FAIL (2026-07-13)

Built `LARM(CoLaR)` (`doccl/methods/larm.py`): a lexically-keyed low-rank feature-correction memory
M, READ additively at layer k (routed by OCR sig), WRITTEN by replay CE. 6 unit tests green,
vectorized, committed `e3d7efa`/`f21456e`.

**Gate 0 ADD result (dil k4/d5 grid): AA 66.3 / BWT −34.4, row3 [60.5, 40.7, 97.7] — WORSE than its
own CoLaR/latent_replay baseline (76.0 / −19.3 / [72.6, 57.6, 97.9]).** So the "add, not replace"
safety claim is FALSE: the memory is not a helpful-or-no-op; it actively HURTS retention.

**Diagnosis (localized, mechanism clear):** LARM LEARNS each task fine (diag [87.3,82.8,97.7] ≈
CoLaR) — the damage is to RETENTION (row3). Root cause: **the additive correction is trained against
the plastic layers' state AT the time each task was learned, but applied at eval through plastic
layers that have since drifted.** Replay refreshes the correction relative to the FROZEN trunk
(below k); the drift is in the PLASTIC layers (above k) that consume the corrected features — the
correction and its consumer sit on opposite sides of the plasticity boundary. **LARM inherited
exactly LexSlot's staleness weakness, one layer up.** (Ruled out: cross-task routing leak — on dil's
disjoint vocab the softmax router correctly puts ~99.9% mass on same-task cells, verified.)

**GATE 0 COMPLETE: LARM REPLACE = AA 32.6 (collapsed below naive), vs ADD 66.3 vs CoLaR 76.0.**
Proven: (1) add > replace (consistency law holds directionally — replacing whole-doc features is
catastrophic); (2) but add < CoLaR (the layer-k fusion is a net negative). Mechanism NOT yet fully
pinned — see below (two wrong confident calls in a row; STOP asserting, TEST).

**BUG-FIX ATTEMPT (2026-07-14): the softmax→clamped-cosine gate fix did NOT recover LARM.**
Fixed run (dil k4/d5 grid): **AA 62.7 / BWT −39.4, row3 [63.5, SROIE 27.1, 97.6]** — still far
below CoLaR 76.0, ~same as buggy 66.3. So the softmax-null-option bug was NOT the whole cause
(my "fix confirmed" prediction was wrong — 2nd confident-wrong call). What the fix DID do (verified
on realistic dense keys): cross-task foreign leakage dropped to ~0.03 mass. But total gate mass
stays ~1.0 because a self-matching doc has cosine ~0.96 to its OWN cell → `clamp(min=1)` never
triggers → the in-domain correction fires at ~full strength on every eval doc. So the damage is
NOT cross-task routing; it's that the **in-domain learned correction itself is wrong at final-eval
time** (candidate: stale vs the drifted head — the drift hypothesis, now with SOME evidence but
STILL UNPROVEN). diag [87.3,82.0,97.6] = learns each task fine; row3 destroyed = retention only.

**DECISIVE DISCRIMINATOR NEEDED (do not guess again):** run LARM with memory ON at train, OFF at
eval (`mem_eval_off` flag). If eval-off → ~76: the learned correction is wrong at eval (drift) →
fixable. If eval-off still ~63: damage is during TRAINING (memory corrupted plastic weights/replay)
→ different fix. This one run splits the remaining hypotheses. GPU is free; fix is committed
(ca4e19e); do NOT conclude LARM's fate until this run decides.

**Fix hypotheses (if LARM is worth continuing):** (a) gate the memory OFF at eval / detach its
contribution to old-task eval once the head has moved; (b) put the rewrite where the head reads it
consistently — i.e. correct at the classifier-INPUT (post-plastic) not layer-k (pre-plastic), so no
plastic drift sits between correction and head; (c) make the correction itself replay-refit on the
CURRENT plastic state (re-derive up-factors each task from replayed activations run through the
current head). Option (b) is the cleanest — it moves the rewrite above the drift. Decide with user;
this may reduce LARM to "CoLaR + a head-input lexical adapter," a different (simpler) method.

## ==== THREAD CLOSED (2026-07-12 pm): CoLaR r128 = LOSSLESS 2.7× — d50 accuracy at 60 MB ====

**CoLaR k4/d50/r128 (grid): AA 87.6 / BWT −1.7, row3 [89.2, 76.3, 97.2], 60.4 MB** — MATCHES raw
d50 (87.3 / −2.2 / 163 MB) on every cell; BWT slightly better. The per-doc-variance prediction
held exactly (r128 = 95.2% var → full AA; r64 = 87.7% → −6.7 AA). **Final Pareto (dil, k4):**
raw-d50 87.3@163MB · **CoLaR-r128 87.6@60MB** · CoLaR-r64 80.6@32MB · d5 76-78@16MB · naive 41@0.
The d50→d5/d0 answer: docs can't be cut (E1: selection ≈ random; E3: public-proxy d0 caps ~57);
**bytes CAN — losslessly to 2.7×, lossy dial to 5×.** Untested cheap multiplier: int8 on the
factors (~2× more → r128 @ ~30 MB). For the paper: single-seed/single-scenario — needs seeds 7/123
+ cil_cord on Vast.ai. CoLaR = the constructive leg of the consistency-law finding (whole-doc unit
compression preserves the binding; the falsified marginals are the control group).

**LEXSLOT VERIFICATION — MY 42.4 RESULT WAS THE WRONG CONFIG (corrected 2026-07-13, user caught it).**
The archived 87.3/88.4 row is **LexSlot slots layered on the FULL DocCL machinery**: git shows at
the run date (Jun 27, commit c1c6f19) `class LexSlot(DocCL)` inheriting CE + KD + reservoir-replay
+ depth-Fisher; the archived hparams confirm **use_replay=True, buffer_size=200, kd_alpha=1.0,
lambda_=2000**. After the 2026-07-02 RCA the class was refactored to `LexSlot(NaiveFineTune)` —
stripped to STANDALONE (no buffer/KD/Fisher) to isolate the slots' own effect. My "verification"
ran that standalone class (`method=lexslot`, current config use_replay-absent) → 42.4, which only
re-confirms the already-known "slots ALONE ≈ naive" RCA. It says NOTHING about the real hybrid row.
**Corrections to my prior claims:** (a) 87.3 is NOT standalone LexSlot; (b) 87.3 is NOT buffer-free
— it uses a 200-exemplar replay buffer (so it's DocCL-family, buffer-based); (c) whether it's a
pre-fix unnormalized-gate artifact is STILL OPEN — untested, not falsified.

**RESOLVED — the 87.3 row is REAL (buffer-based hybrid); my "artifact" claim was wrong.** The three
`_off` seed runs on disk are the pre-fix DocCL-HYBRID: seed42=88.4 [87.2,80.7,97.3], seed123=86.3
[85.6,76.7,96.8], seed7=87.2 [85.3,81.7,94.8] — all use_replay=True, buffer=200. Mean =
**87.3 ± 1.1**, exactly reproducing table_main. I had OVERWRITTEN seed42 with a wrong-config
standalone re-run (42.4); **RESTORED from archive** (parked the bad run at
`dil_lexslot_seed42_off_STANDALONE_wrongconfig`). The 3-seed evidence is intact again.

**Corrected status of LexSlot:** (a) 87.3 is a genuine near-joint result of DocCL+slots; (b) it is
**buffer-based** (200 exemplars), NOT buffer-free — so the chapter-7 "without storing any past data"
PROSE is factually wrong and must be corrected to "small-buffer (200 exemplars)"; the NUMBER stays;
(c) still-open (low priority): re-run the hybrid on POST-gate-fix code to confirm the normalized gate
doesn't move ~87 — plausibly it doesn't (buffer carries retention), but untested. (d) standalone
LexSlot ≈ naive was already the RCA finding; that is separate from this row and unchanged.
See [[clue-thesis-lexslot-row-source]].

## SUPERSEDED — E2 RESULT (2026-07-12): CoLaR r64 = AA 80.6 @ 32 MB — compression is a real but lossy dial

CoLaR k4/d50/r64 (grid): **AA 80.6 / BWT −12.3, row3 [82.3, SROIE 62.4, 97.1], 32 MB** — sits
between raw-d50 (87.3 @ 163 MB, −6.7 AA) and random-d5 (76-78 @ 16 MB, +3-5 AA at 2× bytes).
Compression loss concentrates on the mid-task (SROIE 76.7→62.4), consistent with everything else
in the program. Exactly on the per-doc-variance prediction (r64 = 87.7% var). **r128 point IN
FLIGHT** (`_run_e2_r128`, 95.2% var, ~64 MB): ≥85 → Pareto reads "d50 coverage at 2.5× less
memory, tunable to 5×"; else per-doc SVD is a shallow dial and int8-on-raw is the better lever.
The d50→d5/d0 thread's final ledger so far: selection NO (E1 77.6), d0-public NO (E3 57.3),
compression PARTIAL (E2 80.6@r64, r128 pending).

## SUPERSEDED VERDICTS (2026-07-12 am): E3 and E1 falsified their hypotheses; E2 re-running

- **E3 (PLaR@k4, d0-private): FALSIFIED depth hypothesis.** AA 57.3 [68.7, SROIE 5.9, 97.4] ≈ k8's
  58.9. k4 improves ACQUISITION (SROIE at-learning 82.0, CORD 97.4) but retention structure is
  identical → PLaR's wall is the POOL CONTENT, not interface depth. d0-at-parity via WildReceipt
  proxies is bounded ~57-59 across every knob tried (k, count, hard/soft).
- **E1 (kcenter d5, coverage-not-count): NOT SUPPORTED.** kcenter 77.6 [71.2, 64.5, 97.2] vs
  random 76.0 [72.6, 57.6, 97.9] (s42) — within random-d5's 3-seed noise (78.2±2.8). Nuance:
  SROIE retention +6.9 (64.5 vs 57.6) but FUNSD −1.4; selection redistributes, doesn't add.
  **Count (coverage MASS), not curation, drives the d5→d50 gap.** → The d50→d5 reduction lives
  or dies on E2 (compression).
- **E2 (CoLaR) crash ROOT-CAUSED (not OOM):** parent banking-log summed `d["hidden"]` over the
  whole store, but CoLaR pops "hidden"→us/v for earlier tasks → KeyError at task-1 after_task.
  Both crashes, same bug; first was misdiagnosed because chain stderr was piped through tail -2.
  Fixed (format-agnostic count) + 2-task regression test; pushed. **E2 re-running**
  (`_run_e2_colar_fix`, task bcpeq2ajd, ~3.5h): colar k4/d50/r64, grid → target ≥ ~85 @ ~32 MB.
  Note from the crashed run's log: replay loss during T1 was ~0.000-0.007 — plausibly normal
  (task-0 docs still well-classified early), but if E2's AA lands LOW with near-zero replay loss
  throughout, suspect reconstruction-too-easy/fp16-fidelity and check r128.

## SUPERSEDED (2026-07-11 pm): "reduce d50 → d5/d0" for latent replay k4 — 3-run chain

User directive: shrink k4/d50 (87.3, ~163 MB) toward d5 (78.2, ~16 MB) or d0, possibly via
lexical/latent memory. Critical constraint from our own chain: marginal memories are falsified —
only selection / compression / augmentation / public-substitution / isolation+replay doors remain.
Three hypotheses → three runs chained (task `bpmugeger`, ~8.5h, grid budget unless noted):
1. **E1 — H1 coverage-not-count:** `latent_replay` + NEW `doc_selection: kcenter` knob (greedy
   farthest-point over per-doc mean latents, pool=100; default `random` byte-identical). k4/d5
   → `dil_latent_replay_seed42_k4_kc` vs random-d5 76.0 (s42). If ≥ ~83, selection closes half
   the gap for free.
2. **E2 — H2 bytes-not-docs:** NEW method `colar` (CoLaR): per-DOC rank-r SVD of banked latents
   (whole-doc binding preserved; per-doc r64 = 87.7% var measured). k4/d50/r64 →
   `dil_colar_seed42_k4_d50` vs raw d50 87.3 at ~28 MB (below raw-d5 bytes ~16MB? factors ≈
   0.19 MB/doc × 150 = 28.4 MB — 5.7×; +int8 later → ~14 MB). Target: ≥ ~85.
3. **E3 — H3 d0-private:** PLaR@k4 relaunch (killed at final eval last night; banking had
   completed, histograms still 7/9 tags). 5ep → vs k8's 58.9. Tests the generic-interface
   coverage fix.
Both prior threads' pending runs (lexslot-off verification) queue AFTER this chain — the killed
`bldaw8zmb` chain never reached it. `docs/RESULTS_LEDGER_DIL.md` is the reference table.

**Chain incidents (2026-07-11 pm), both recovered:**
- E1 died at launch: `method.doc_selection=` isn't in the immutable latent_replay.yaml → Hydra
  struct error → fixed with NEW option file `configs/method/latent_replay_kc.yaml` (committed
  144a806); E1 re-queued behind E3 (waiter task `bp56bzp6b`).
- **E2 (colar) died ABRUPTLY** 3s after SROIE's early-stop (ep7 STOP 14:11:03; wandb EOF
  14:11:06): no Python traceback in the job log → hard SIGKILL signature (likely host OOM-killer;
  journal not readable). Task-0 leg had worked (banked 50 + **compressed to 10.7 MB vs 54.5 raw,
  5.1×** — the CoLaR mechanic is proven live). Rerun queued THIRD with full stderr capture
  (waiter `b7ddipxom` → `scratchpad/e2_full.log`, session 82a55d2e) — my chain's `tail -2` had
  swallowed the crash output (lesson: never pipe run stderr through tail in chains).
- Execution order now: **E3 (running, PLaR@k4 5ep) → E1 (kcenter d5, grid) → E2-retry (colar
  k4/d50, grid, full logs)**. Host RAM at E3-task-2: 6G used / 9G avail.

## SUPERSEDED (2026-07-11 am): deep dive on the two 87.3 rows — chain was KILLED (task bldaw8zmb)

User picked latent-replay(k4/d50) + LexSlot-hybrid as directions. Deep dive (full write-up:
`CL4IE/wiki/analyses/2026-07-11-two-87point3-rows-deep-dive.md`) found:
1. **latent k4/d50 is genuine**: final FUNSD 88.8 > at-learning 86.3 (positive backward transfer
   — the head re-carves the joint optimum); residual gap = SROIE −7 only. Dose-response d1/d5/d50
   = 61/78/87; depth k2≈k4≫k8 → generic-interface + plasticity effects confounded (k8_d50 queued).
2. **LexSlot `_off` 87.3 is UNVERIFIED**: `_off` = slot_sharing=off (task-PRIVATE slots at
   head+late, lexical inference gate, no buffer/teacher/Fisher; shared-slots variant = 42.2). BUT
   the 3 `_off` runs (Jun 28) PREDATE the gate-normalization fix 90f5a15 (Jun 30) — the
   unnormalized gate can act as a covert task oracle (multi-head DIL, inflated). The thesis
   table_main row rests on these pre-fix runs. Pre-fix dir archived (`..._off_prefix_archive`).
**Overnight chain:** (a) PLaR k4/d50/soft 5ep — does the generic k4 interface fix SROIE coverage
(vs k8's 58.9/SROIE 7)? → `dil_proxy_latent_replay_seed42_k4_d50_soft`; then (b) **lexslot-off
VERIFICATION on post-fix code**, grid budget → `dil_lexslot_seed42_off`. Morning: read both; if
lexslot-off collapses → correct the thesis row + strongest tombstone; if it reproduces → real
buffer-free positive (parameter-isolation face of the law) → multi-seed immediately.
Queued next: k8_d50, k4_d200, k4_d50 seeds 7/123, CoLaR r64 on k4/d50, slots+replay combo cell.

**Full results ledger (all dil approaches/baselines, disk-verified): `docs/RESULTS_LEDGER_DIL.md`.**
Note a labeling correction recorded there: the latent-replay 87.3 headline is the **k=4/50-doc**
run; the matched-count 5ep anchor used for this week's comparisons is k8/d4 = 63.8 (unchanged).

## ==== GATE-0 FINAL: buffer-free feature replay FALSIFIED for doc-IE (2026-07-10) ====

Complete ladder (dil, k=8, 5ep), AA / SROIE-final-F1 / mem:
| latent_replay (WHOLE real docs)      | 87.3 (conv) | survive | 16 MB |
| SpectralMemory (synth, rank-16)      | 41.9 | 3.2 | 0.45 MB |
| AGLR-CL (synth, full-`d` Gaussian)   | 39.4 | 3.0 | 0.39 MB |
| CoresetMemory (REAL cents, 4 car)    | 36.7 | 2.9 | 0.48 MB |
| CoresetMemory (REAL, 50 car/16 cent) | 39.7 | 3.1 | 4.05 MB |

**Proven (4 independent falsifications):** buffer-free feature replay fails for doc-IE, and it is
NOT (a) synthesis quality — real centroids fail like synthetic; NOT (b) rank — full-`d` fails like
rank-16; NOT (c) carrier diversity — 50 carriers fail like 4. **Only replaying WHOLE REAL
documents grounds the head.** Every collapse is the LexMem-v5 signature (current task ~93,
all prior ~3).

**Mechanism (why whole-doc replay is special) — NOW PROVEN by controlled ablation:** latent_replay
stores each doc as an intact {hidden[709], bbox, mask, labels} bundle and replays it unchanged —
feature[t], bbox[t], label[t] all from the SAME real token (consistent co-occurrence). The carrier
scaffold breaks this: it pools features across the task, then pastes a class-`c` feature onto
position `t` whose bbox/label came from a DIFFERENT token. The head trained on those inconsistent
triples learns nothing transferable.

**CONTROL RESULT (matched count = 4, the clean isolation):**
| 4 WHOLE real docs (consistent)          | AA 63.8 | row3 [60.8, 37.1, 93.6] | survives |
| 4 carriers + real centroids (decoupled) | AA 36.7 | row3 [14.5,  2.9, 92.7] | collapses |
**+27 AA / +34 SROIE-F1 from consistency ALONE** (same count, same boundary, same real features).
=> **feature↔position↔label CO-OCCURRENCE, not count/diversity/synthesis, is the necessary
ingredient.** This is the key figure for the paper — a single-variable ablation isolating the
mechanism. (Note docs=4 gives AA 63.8, below the docs=5 converged 87.3 — count/epochs still help,
but consistency is the *binary* enabler: without it the head collapses regardless.)

**This closes the method-led pivot. Verdict: diagnostic + falsification paper** (reverts to the
2026-07-04 framing, now MUCH sharper). SLR/AGLR/Coreset extend the existing chain
(`docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`) with a new terminal level: "even a real-activation
coreset — the strongest possible buffer-free feature memory — fails; only whole-document replay
works, because the head needs consistent (feature, position, label) co-occurrence that no
per-class/per-token summary preserves." Novel, mechanistic, airtight.

**Next action:** DECIDE (user) — run the rank sweep on Vast.ai (the local box is too slow, ~90
min/run), OR accept the negative result and fold SLR into the falsification chain. If sweeping,
also run AGLR-CL (does class-first full-d Gaussian beat basis-first rank-r? if AGLR works and SLR
doesn't, that's informative about what the head needs). See wiki idea page Gate 0/1.

---

## Experiment #6 RESULT — the law BENDS at parameter granularity, doesn't break (2026-07-07)

Seed-42 dil, masks verified at exact target fractions (50.0/80.0/95.0% of 125.3M):

| arm | AA | BWT | final row [t0/t1/t2] |
|-----|----|-----|----------------------|
| p=0.50 | 45.0 | −66.0 | 30.6 / 6.9 / 97.5 |
| p=0.80 | 49.6 | −59.0 | 44.5 / 6.2 / 97.9 |
| p=0.95 | 54.8 | −46.3 | 62.4 / 5.8 / 96.0 |

Reading vs pre-registered gates: neither strict conservation (54.8 > naive+3) nor a
real opening (< v3b 66.1) — the "interesting" band. Refinements:
1. GRANULARITY MODULATES THE LAW: layer-level protection conserves (freeze≈free,
   exp #4); per-tensor interleaved protection buys up to +13.5 AA over naive with
   plasticity fully intact (new-task 96–98 even at p=95). Hard masking also beats
   soft penalty at the same importance signal (fisher_mask 54.8 vs EWC 41.2).
2. PROTECTION IS TASK-SELECTIVE, NOT GENERAL: task-0 (forms, lexically distinct)
   retention rises monotonically with p (31→45→62), but the mid task (receipts,
   competing with task 2 for the same parameters) collapses to ~6 at EVERY p — the
   drift signature survives protection wherever tasks compete for capacity.
   Same scenario-boundedness pattern as lexical routing (works across domain gaps,
   fails within them).
3. CURVE COMPLETE (2026-07-07 15:55, both exit 0) — 7 points, seed 42:

   | p | 0 (naive) | 0.50 | 0.80 | 0.90 | 0.95 | 0.99 | 1.0 (freeze-all) |
   |---|-----------|------|------|------|------|------|------------------|
   | AA | 41.3 | 45.0 | 49.6 | 51.1 | **54.8** | 52.5 | 41.1 |

   Peak at p=0.95. At p=0.99 plasticity finally degrades (new-task 89.5 vs 96–98
   elsewhere) — the turn toward the freeze-all collapse. Mid-task stays 5–9 at
   EVERY p: within-domain capacity competition is never rescued by protection.

Verdict for the paper: the lottery-ticket family gets a diagnosed ceiling — masking
protects only lexically-separable tasks and never recovers within-domain competition;
still 27 pts below ER-5. Consistent with Finding 3: no protection scheme substitutes
for real past-task gradients.

## Experiment #6 launch record (2026-07-07)

The migration law was established at LAYER granularity (exp #4); this closes the
granularity attack surface ("maybe fine-grained interleaved masks share capacity where
whole-layer freezing cannot" — the PackNet/HAT/WSN lottery-ticket argument). Method
`fisher_mask` (commit 37ff257): per-tensor top-p Fisher-important entries hard-frozen
(exact freeze: grad mask + snapshot/restore across AdamW step), rest plastic, mask
rebuilt after every task from the running Fisher sum (same estimator as EWC).

**PRE-REGISTERED READINGS (set before any result):**
- Conservation holds at every p (AA ≈ naive 41.3 ± ~3, or pure-stability collapse at
  high p) → law is GRANULARITY-INDEPENDENT — stronger lead result, and explains away
  the PackNet/WSN family for task-ID-free doc-IE.
- Some p gives AA well above floor with real retention (≥ ~50 interesting, ≥ 66 = v3b
  → real opening) → law breaks at fine grain → mask-based method justified by mechanism.

3 arms queued behind exp #5 chain (bg task bi1cmr7r8, polls for free GPU):
`dil_fisher_mask_seed42` (p=0.8 canonical) / `_p50` / `_p95`.
Decided vs alternatives (2026-07-07 session): directional optimization (GPM-family)
already falsified in-house (HGT/CUBER); RanPAC premise contradicted by frozen-encoder
45.3 arm but worth one baseline run later; activation-based masking rejected in favor
of Fisher to keep one importance language.

## Experiment #5 RESULT — P1: THE MIGRATION LAW IS BOUNDED (read 2026-07-07)

Seed-42 dil results against the pre-registered gates (set before results existed):

| arm | AA | BWT | final row [funsd/receipts-1/receipts-2] | reading |
|-----|----|-----|------------------------------------------|---------|
| k=4 (8 plastic) | **76.0** | −19.3 | 72.6 / **57.6** / 97.9 | **P1** (gate ≥75) |
| k=8 (4 plastic) | 65.8 | −30.8 | 67.9 / 34.5 / 94.9 | ≈ v3b (66.1) |
| ctrl (freeze-only k=8) | 37.3 | −74.1 | 15.2 / 2.4 / 94.4 | conservation confirmed |

Reading: real past-task gradients into the plastic remainder BREAK conservation, and
monotonically in plastic depth (76.0 @ 8 plastic > 65.8 @ 4 plastic; control = law).
The mid-task drift signature (~13–17 for every buffer-free method) is broken: 57.6.
Law statement upgrades to: "forgetting is conserved under architectural constraint
*unless* the plastic remainder receives real past-task gradients — and raw inputs are
not the only sufficient carrier: frozen-space activations (no raw documents stored)
recover most of the replay effect (76.0 vs ER-5 82.1 vs best buffer-free 66.1)."
The 2×2 (real gradients × feature validity) is complete: it is the GRADIENTS, not the
storage format, that matter — but the features must not be stale.

Extension COMPLETE (2026-07-07 13:27, all exit 0):

| run | AA | BWT | final [t0/t1/t2] | note |
|-----|----|-----|------------------|------|
| k=4, 50 docs/task | **87.3** | **−2.3** | 88.8 / 76.7 / 96.6 | ≈ ER-200 (87.9), 1.4 below joint (88.7); BEATS raw ER-50 (85.6) at equal doc count |
| k=2, 5 docs | 79.7 | −13.9 | 73.1 / 68.5 / 97.4 | depth dose–response still monotone: 65.8 (k8) → 76.0 (k4) → 79.7 (k2) |
| k=4, 5 docs, seed 7 | 77.3 | −17.4 | 81.0 / 53.3 / 97.4 | |
| k=4, 5 docs, seed 123 | 81.4 | −11.7 | 79.4 / 67.9 / 97.0 | |

Headline k=4/5-docs across seeds: **78.2 ± 2.8** (n=3) — all seeds clear the P1 gate (75).
Doc-count crossover vs raw ER at k=4: 1 doc 61.3 vs 57.6 (latent wins), 5 docs 76.0–81.4
vs 82.1 (ER slightly ahead), 50 docs 87.3 vs 85.6 (latent wins). At scale, activation
replay on a drift-immunized trunk ≈ joint oracle WITHOUT raw documents. Honest caveats
for the paper: activation storage ≈1.1 MB/doc (bytes larger than raw docs — claim is
"no raw documents stored", not compression) and activations are partially invertible.

## Experiment #5 launch record (2026-07-07)

Chain level 6: frozen-trunk activation replay (Pellegrini-style), the untested cell of
the 2x2 (real gradients x feature validity). Freeze layers <k after task 0; bank
layer-k hidden (fp16) for 5 docs/task; replay via pre-hook injection → real past-task
gradients into exactly the plastic layers the migration law says absorb the drift.
Method `latent_replay` (commit 93aa144), 13 tests green, smoke OK (single_funsd 70.5@1ep).

**PRE-REGISTERED GATES (set before any result existed):**
- P1 success: dil AA ≥ 75 → law bounded ("conserved *unless* plastic part gets real
  past-task gradients"); raw-data-free method on par with ER exists.
- P2 falsified: AA ≤ 68 (v3b 66.1+2σ) → chain closes at level 6 with predicted mechanism.
- P3 68–75: boundary result; dose-response on k tells where it breaks.
- Signature to watch: mid-task row (buffer-free collapses ~67→~17; ER holds ~67+).

3 sequential runs on local box (bg task b8rnl3zyp, logs in scratchpad/latent_runs/):
1. `dil_latent_replay_seed42` (k=8, 4 plastic — matches map8) — primary
2. `dil_latent_replay_seed42_k4` (8 plastic) — dose-response
3. `dil_latent_replay_seed42_ctrl` (freeze-only, no replay) — control
References (no new runs): naive 41.3 / v3b 66.1±3.0 / ER-5 82.1 / joint 88.7.
If P1/P3: extend best k × docs {1,50} + seeds 7,123. Spec + rationale:
`~/.claude/plans/1-yes-my-motivation-foamy-boot.md`. bd still degraded — issue NOT
filed in bd; this section is the tracking record.

## Experiment #4 COMPLETE — migration robustness (2026-07-05). PROGRAM DONE -> WRITING

8 runs (lexmem_ctrl freeze vs free, seed42, drift-probed):
| arm | AA | BWT | t0-final | t0-CKA trace |
| o2_freeze / o2_free | 46.6 / 47.6 | -63 / -62 | 4.3 / 4.7 | .08,.22 / .14,.25 |
| o3_freeze / o3_free | 30.6 / 32.0 | -86 / -85 | 5.8 / 7.9 | .40,.19 / .48,.20 |
| map8 (4 plastic) / map1 (11 plastic) | 40.8 / 39.7 | -73 / -75 | 20.7 / 17.7 | ~.24 endpoint both |
| long_freeze / long_free (10-session CIL) | 9.3 / 9.5 | -83 / -80 | 0 / 0 | ~.1-.2 across 9 boundaries |

**LAW (strongest form): total forgetting is CONSERVED under architectural
constraint — freeze-vs-free AA within 1-2 pts in every order, map, and horizon;
only the locus moves.** Freeze arms prove localization structurally (head+late
frozen => endpoint CKA collapse can only originate early/mid). Dose-response:
even 4 plastic layers absorb the full drift (map8 ~= map1). Long horizon: the
effect compounds to t0-F1=0 over 10 boundaries regardless of map.

All 4 program experiments done: oracle (55.3 low branch) / gauss-replay
falsified (63.6) / retention curve (5 raw docs 82 > all buffer-free) /
migration law. results/migration/*. NEXT: thesis/paper writing.
## SESSION HANDOFF (2026-07-04 close)

- **Done this session:** v5 graph-as-memory gate run to completion → FALSIFIED
  (both arms AA 63/SROIE 13, byte-identical). Found+fixed the val-restore no-op bug.
  Then the KEY pivot (user): v5's mechanism = solved literature (FeCAM/FeTrIL/PASS) →
  **reframed the whole program as a diagnostic+falsification paper, v5 = tombstone.**
  Docs `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` + STATE updated & pushed (7514cf3).
- **Quality gates:** 287 fast tests green, ruff clean on v5+train.py.
- **NEXT SESSION = WRITE THE PAPER, not more methods.** Lead = migration finding +
  convergent negative result. Concrete next steps (cheapest first, per program review):
  (1) head-refit probe oracle already done (AA 55.3 — representation forgetting is REAL);
  (2) exp #4 migration robustness (2 freeze maps × 2 task orders + cil_cord_long) — the
  one remaining experiment; (3) relabel thesis "LexSlot 87.3" row (it's the DocCL hybrid).
- **KNOWN BLOCKER:** `bd` is in a degraded re-clone-warning state (stray dolt server
  killed but local DB still complaining). Paper-reframe follow-up issue NOT filed in bd —
  captured here + in FINDINGS doc instead. Fix bd (or `bd export`/re-clone carefully)
  before relying on it next session. Don't block on it.

## FRAMING LOCKED: diagnostic+falsification paper, NOT a method paper (2026-07-04)

- **Decision (user):** v5 (feature-Gaussian graph replay) is NOT a contribution.
  Its core mechanism = PASS/FeTrIL/FeCAM (solved 2021-2023) → un-submittable as novelty.
  v5 enters the paper ONLY as the **terminal tombstone** of the falsification chain:
  even the field's strongest buffer-free tool (+ a novel relational variant) can't beat
  representation drift (edges-on AA 63.2, SROIE→13.0).
- **The paper's actual novelty:** (1) the diagnosis — forgetting is head-localized,
  architecture-general, and the locus MIGRATES under naive protection; (2) a rigorous
  convergent negative result: FIVE buffer-free families fail for one identified reason
  (representation drift), so "only replay grounds the head." NOT any method.
- **Venue:** CoLLAs/TMLR/ACL-Findings for the negative result; AAAI/ICML main-track only
  if the LEAD is the migration finding, never a method.
- **Docs updated:** `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` (v5 = falsification level 4;
  "is NOT a method paper" section). v5 code stays (evidence), no further method work on it.
- **v5 gate COMPLETE (both arms):** edges-on = edges-off = AA 63.17, SROIE→13.0,
  BYTE-IDENTICAL (ΔAA exactly 0.00) despite differing recon-loss streams (1.42 vs 0.013).
  → do NOT claim an edges ablation; the head-side signal is swamped by trunk drift (recon
  grad reaches head but doesn't move the evaluated model). Uninformative ablation is itself
  consistent with Finding 3 (bottleneck = trunk representation drift, not head). v5 = tombstone.
  Runs: results/dil_lexmem_v5_seed42{,_bank}. Earlier byte-identical pair (pre-fix, from
  the val-restore bug) archived at results/_invalid_v5_restore_bug/.

## Analysis program: experiments 1-3 COMPLETE (2026-07-03)

**#1 Head-refit oracle (LOW branch):** pooled probe on naive trunk AA 55.3
(FUNSD per-task probe caps at 41.9). Representation forgetting is REAL in
doc-IE — Davari's "representations survive" does NOT transfer. All head-only
methods on a naive trunk ceiling-bounded at ~55. results/head_refit_oracle.json

**#2 Gaussian head-replay (FALSIFIED, 4th chain level):** AA 63.6 ≈ v3b 66.1±3.0.
Failure signature IDENTICAL across head mechanisms (slots vs replay: SROIE
~67→~17 despite balanced head gradients) => mid-task bottleneck is per-task
REPRESENTATION drift, not head misalignment. results/dil_gauss_replay_seed42

**#3 Retention-information curve (dil seed42):**
| stored | AA |
| naive (nothing) | 41.3 |
| input counts (ledger) | 33.8 |
| naive-trunk probe ceiling | 55.3 |
| ER 1 doc/task | 57.6 |
| feature Gaussians | 63.6 |
| slot memory (v3b, 3 seeds) | 66.1 ± 3.0 |
| **ER 5 docs/task** | **82.1** |
| ER 10 docs/task | 84.8 |
| ER 50/task | 85.6 | ER-200 87.9 | joint 88.7 |
=> FIVE raw documents per task beat every buffer-free memory we built by 16 AA.
Sharpened claim: raw inputs are the only storage that survives representation
drift (encoder re-renders current features); all derived quantities go stale.
Runs: results/retention_curve/er_buf{3,15,30,150}.

**Remaining: #4 migration robustness** (2 freeze maps x 2 task orders +
cil_cord_long) — needs dil task-order scenario variants. Then thesis writing.
v3b 3-seed: AA 66.1±3.0. gauss_replay code committed (bec95b4).

## LexSlot-FM: FM refit bug FIXED — AA +5.25/+5.68 on both gate modes (2026-07-03)

- **Bug found + fixed:** `_refit_old_fm_slots` ran `self.model.eval()` → forward hook
  used `_replacement_blend` (gate-routed, uniform blend) instead of `_additive_blend`
  (single-slot, `_cur_fm_idx`). Gradients dispersed to all FM slots + base head, never
  reaching only the target slot. Fixed by switching to `self.model.train()` so
  `_additive_blend` fires with `_cur_fm_idx` set to the old slot.
- **Results comparison (seed=7, dil FUNSD→SROIE→CORD):**

  | Variant | AA | T0 | T1 | T2 |
  |---------|-------|-------|-------|-------|
  | Frozen encoder (s=42, no refit) | 45.31 | 20.50 | 46.79 | 68.64 |
  | Layout+EWC (no refit) | 44.09 | 31.12 | 27.59 | 73.56 |
  | Layout+EWC+refit (**FIXED**) | **48.67** | **40.78** | 20.35 | 84.87 |
  | TF-IDF+EWC (no refit) | 39.05 | 19.53 | 8.80 | 88.81 |
  | TF-IDF+EWC+refit (**FIXED**) | **44.73** | **26.10** | **19.45** | 88.64 |

  **Key findings:**
  1. FM refit (train-mode fix) improves both gates — Layout AA +5.25, TF-IDF AA +5.68
  2. T0 jumps +11.09 (Layout) and +6.57 (TF-IDF) — FM_0 refit now actually works
  3. T2 stays high on both gates: 84.87 (Layout) / 88.64 (TF-IDF)
  4. T1 (SROIE) still degraded by encoder drift — 50-sample FM_1 replay buffer insufficient
  5. **Best AA so far: 48.67** (Layout gate + EWC λ=1000 + FM refit 3ep/50samples)
- **Remaining issue:** T1 collapses from 70.98 (post-training) to 20.35 (after CORD training).
  FM_1 refit helps (+4.82 vs buggy before fix) but the 50-sample buffer can't fully recover
  SROIE features after the encoder shifts during CORD training. Possible fixes: larger replay
  buffer (200+ samples), more refit epochs, or adaptive refit learning rate.
- **Files changed:** `scripts/train.py` (added `gate_mode` suffix to lexslot_fm run names);
  `doccl/methods/lexslot_fm.py` (train mode in refit). bd: **CLUE-85g**.
- **Blocked on:** user decision for next direction (increase refit budget, conditional gate,
  more seeds, or pivot).

## Ledger gate falsified standalone; strategy fork OPEN (2026-07-02 23:30)

- Ledger gate v2 (relational context keys): CORD OWN 76.1 (!) but cumulative
  FUNSD 21 / SROIE 23 << 60 bar; key-collision interference -18 on CORD.
  Memory-as-retention falsified at all 3 levels (parametric / +drift-ctrl /
  input-keyed). Script: scripts/lexical_ledger_gate.py (commit 12dfba2);
  results/ledger_gate{_v1_unigram,}.json.
- **USER DECISION PENDING** (asked, away): (a) analysis-paper + v3b spine
  [recommended], (b) push v3b vs replay, (c) xlingual ledger gate.
- Meanwhile: v3b seeds 7+123 running overnight (serves all forks).


## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Older history
Superseded/completed sections (June + early-July: LexMem pilot, baseline porting,
backbone-agnostic audit, HGT/DocMERGE/LCA/HRP branches, bug sagas) moved to
`STATE_ARCHIVE.md` (2026-07-04) to keep this ledger lean. All DONE/FALSIFIED — kept for provenance.
