"""CPU probe: does classif_adaptive_factor crush the CE on new-entity tokens in a sparse batch?

Context: cil_cord_cpfd_seed7 collapsed to AA 0.0 from task 1 (T0 was 94.0). Hypothesis: the
per-sample factor (fraction of O tokens the teacher pseudo-labels) scales the CE of the few
new-class tokens too. Replays the exact `train_task` CE path on a synthetic 200-token batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812

from tests.methods.test_cpfd import _make_cpfd

N_OLD, N_TOTAL, N_O = 13, 25, 190
N_NEW = N_TOTAL - N_OLD  # one token per new class (13..24)


def _scaled_new_token_ce(teacher_logit_peak: float) -> float:
    m = _make_cpfd()
    m._n_old = N_OLD
    m._thresholds = torch.full((N_OLD,), m.threshold_floor)  # 0.001 floor
    labels = torch.cat([torch.zeros(N_O), torch.arange(N_OLD, N_TOTAL, dtype=torch.float)])
    labels = labels.long().view(1, -1)  # 190 O + 12 new-class tokens (13..24)
    t_logits = torch.zeros(1, N_O + N_NEW, N_OLD)
    t_logits[..., 0] = teacher_logit_peak  # peak → entropy ~1e-4 (confident) or 0 → uniform
    s_logits = torch.zeros(1, N_O + N_NEW, N_TOTAL)  # uniform student → CE = log(25)
    relabeled, factor = m._apply_pseudo_labels(labels, t_logits)
    ce_per_tok = F.cross_entropy(
        s_logits.permute(0, 2, 1), relabeled, ignore_index=-100, reduction="none"
    )
    return float((factor * ce_per_tok)[0, N_O:].mean())


def test_confident_teacher_keeps_new_entity_ce_intact():
    ce = _scaled_new_token_ce(teacher_logit_peak=13.0)
    assert ce > 1.0 and math.isclose(ce, math.log(N_TOTAL), rel_tol=1e-3), ce


def test_uncertain_teacher_zeroes_new_entity_ce():
    # Documents the mechanism's ceiling: adaptive_ce_min=0.0 → a batch whose O tokens the
    # teacher is unsure about contributes NO CE on its new entities.
    assert _scaled_new_token_ce(teacher_logit_peak=0.0) == 0.0
