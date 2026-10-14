# Forgetting-localisation protocol (model-agnostic)

Code: `doccl/diag/protocol.py` (`from doccl.diag import ForgettingLocaliser, dominance_test,
shared_location_test, freeze_groups`). Tests: `tests/eval/test_forgetting_protocol.py`.
Paper: §3 "Metrics and the localisation protocol"; results §4 Findings 1, 1′, 2.

## Contract (what a backbone must expose)

| Member | Used for | Already on |
|---|---|---|
| `param_groups_by_depth` / `param_groups` (properties → `{group: [Parameter]}`) | the partition $g$ (depth buckets input/early/mid/late/head, or component groups) | LayoutLMv3, LiLT, BROS, BERT, ViT (`param_grouping.py` reads `encoder.layer.<i>` + `classifier`) |
| `cka_layers` (module names) | one depth-matched activation probe per bucket | all wrappers |
| `forward(**batch) → .logits` | Fisher and activations | all wrappers |

Any other model works by passing `groups=callable(model) -> {name: params}` and `cka_layers=[...]`.

## Procedure

```
loc = ForgettingLocaliser(model, device, groups="depth", fisher_n=200, cka_n=2000)
loc.arm(old_task_probe_loader)          # before training task t: F^{(t-1)}, θ^{(t-1)}, old activations
...train task t...
rep = loc.measure(old_task_probe_loader, boundary=f"{t-1}_to_{t}")
```

1. **Localise** — `rep.displacement[g]` = $D_g=\sum_{\phi\in g}F^{(t-1)}_\phi(\theta^{(t)}_\phi-\theta^{(t-1)}_\phi)^2$
   (per-parameter mean; `displacement_sum` for the raw form), `rep.cka[layer]` = per-token linear
   CKA between the two checkpoints. `rep.fisher_old[g]` is importance, **not** forgetting — the head
   has the largest Fisher for any token classifier by construction; never read it as the locus.
2. **Test** — `dominance_test([rep_seed42, rep_seed7, rep_seed123])`: dominant group, its L1
   share, Mann–Whitney vs each rival (Bonferroni). `shared_location_test(reports_A, reports_B)`:
   permutation test on the L1-normalised $D_g$ profiles of two backbones/conditions;
   `shared_location=True` when the null of a common profile is not rejected.
3. **Perturb** — `freeze_groups(model, [rep.dominant(), ...])`, retrain the stream, re-run 1–2.
   Prediction from the anatomy: the profile *relocates* to what stayed plastic (early/mid CKA
   1.0 → 0.19 on DIL), total forgetting is conserved. A buffer-free fix at the locus is ruled
   out when this holds.

## Reporting checklist

- seeds ≥ 3 per backbone; report `dominant`, `share`, Bonferroni p; for cross-backbone
  claims report the `shared_location_test` p and claim *location only* (magnitude comparison is
  underpowered at three seeds).
- always pair $D_g$ with CKA at the same boundary; always print `fisher_old` next to $D_g$ so a
  reader can see importance ≠ forgetting.
- the perturb step is a *prediction test*, not an ablation: state the predicted relocation
  before running it.

## Provenance

The pilot (`doccl/pilot/run_pilot.py`) and the TensorBoard deep-diagnostic hook in
`scripts/train.py` compute the same quantities inline; the 54 validated LayoutLMv3 pilot runs are
left untouched. The module is the public, reusable form of that procedure.
