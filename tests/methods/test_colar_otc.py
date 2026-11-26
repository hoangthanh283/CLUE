"""CoLaR-OTC assignment rules on synthetic tensors (fast, GPU-free)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F  # noqa: N812

from doccl.methods.colar_otc import CoLaROTC, quota_assign, sinkhorn_assign, threshold_assign


def _scores(n: int = 400, k: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g).softmax(-1)


def test_sinkhorn_meets_both_marginals_when_balanced():
    probs = _scores()
    col_mass = torch.tensor([280.0, 60.0, 40.0, 20.0])  # sums to N
    plan = sinkhorn_assign(probs.log(), col_mass, n_iter=200)
    assert torch.allclose(plan.sum(1), torch.ones(400), atol=1e-4)
    assert torch.allclose(plan.sum(0), col_mass, rtol=2e-2)


def test_sinkhorn_fixes_the_mass_of_a_teacher_biased_to_background():
    """The failure this method targets: the teacher ranks the hidden entities first but keeps
    O dominant on them, so argmax assigns nothing. The plan assigns the prior's amount."""
    n, n_hidden = 200, 30
    probs = torch.full((n, 2), 0.0)
    probs[:, 1] = torch.linspace(0.40, 0.01, n)  # ranking correct, never above 0.5
    probs[:, 0] = 1 - probs[:, 1]
    truth = (torch.arange(n) < n_hidden).long()
    assert threshold_assign(probs, 0.5).sum() == 0
    plan = sinkhorn_assign(probs.log(), torch.tensor([n - n_hidden, n_hidden]).float())
    assert (plan.argmax(-1) == truth).float().mean() > 0.95


def test_sinkhorn_gives_nothing_to_a_zero_mass_column():
    plan = sinkhorn_assign(_scores().log(), torch.tensor([300.0, 100.0, 0.0, 0.0]))
    assert plan[:, 2:].sum() == 0
    assert torch.isfinite(plan).all()


def test_relaxed_columns_assign_less_than_an_inflated_prior_demands():
    probs = torch.full((100, 2), 0.0)
    probs[:, 1] = torch.cat([torch.full((10,), 0.9), torch.linspace(0.2, 0.01, 90)])
    probs[:, 0] = 1 - probs[:, 1]
    inflated = torch.tensor([60.0, 40.0])  # the prior claims 40 entities; 10 exist
    hard = sinkhorn_assign(probs.log(), inflated).argmax(-1).sum()
    soft = sinkhorn_assign(probs.log(), inflated, col_relax=0.5).argmax(-1).sum()
    assert abs(int(hard) - 40) <= 2  # the balanced plan obeys the wrong prior (argmax: +-1)
    assert 10 <= soft < hard


def test_quota_assign_takes_top_k_per_class_and_resolves_conflicts_by_score():
    probs = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.5, 0.4],
            [0.2, 0.2, 0.6],
            [0.9, 0.05, 0.05],
        ]
    )
    out = quota_assign(probs, torch.tensor([1.0, 2.0, 2.0]))
    # class 1 claims rows 0,1; class 2 claims rows 2,1; row 1 goes to the higher score (1)
    assert out.tolist() == [1, 1, 2, 0]


def test_threshold_assign_keeps_background_and_low_confidence_tokens():
    probs = torch.tensor([[0.05, 0.95], [0.4, 0.6], [0.8, 0.2]])
    assert threshold_assign(probs, 0.9).tolist() == [1, 0, 0]


def test_current_oracle_reads_true_labels_of_head_classes_only():
    logits = torch.randn(1, 4, 3)  # head: O, class 1, class 2; class 3 is a later session
    labels = torch.tensor([[0, 2, 0, -100]])  # CIL-masked: the class-1 token reads O
    labels_full = torch.tensor([[1, 2, 3, 1]])
    m = SimpleNamespace(config={"current_oracle": True})
    loss = CoLaROTC._current_ce(
        m, SimpleNamespace(logits=logits), {"labels": labels, "labels_full": labels_full}
    )
    want = F.cross_entropy(logits[0], torch.tensor([1, 2, 0, -100]), ignore_index=-100)
    assert torch.allclose(loss, want)


def _bare(rule: str) -> CoLaROTC:
    m = CoLaROTC.__new__(CoLaROTC)
    m.completion, m.config = rule, {}
    m._completed, m._completion_miss = {}, 0
    return m


def test_current_completion_writes_old_classes_and_the_loss_reads_them():
    m = _bare("transport")
    m._class_freq = {1: 0.25, 2: 0.25}  # two old classes, a quarter of the tokens each
    ids = torch.arange(8).view(2, 4)
    rows = [torch.tensor([0, 3, 0, -100]), torch.tensor([0, 0, 3, -100])]  # class 3 = new
    probs = [  # O-labelled tokens only, columns O / class 1 / class 2; O dominant everywhere
        torch.tensor([[0.6, 0.39, 0.01], [0.6, 0.01, 0.39]]),
        torch.tensor([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]]),
    ]
    keys = [ids[0].numpy().tobytes(), ids[1].numpy().tobytes()]  # no bbox: keyed on input_ids
    m._complete_current(keys, [r.clone() for r in rows], probs, [], n_valid=4)
    assert m._completed[keys[0]].tolist() == [1, 3, 2, -100]  # argmax would have kept O
    assert m._completed[keys[1]].tolist() == [0, 0, 3, -100]

    logits = torch.randn(2, 4, 4)
    loss = m._current_ce(
        SimpleNamespace(logits=logits), {"labels": torch.stack(rows), "input_ids": ids}
    )
    want = F.cross_entropy(
        logits.reshape(-1, 4), torch.tensor([1, 3, 2, -100, 0, 0, 3, -100]), ignore_index=-100
    )
    assert torch.allclose(loss, want) and m._completion_miss == 0


def test_stored_completion_assigns_only_the_sessions_classes_to_o_tokens():
    m = _bare("quota")
    m._class_freq = {3: 0.25}  # head: O, 1, 2 at capture; class 3 arrives now
    doc = {"labels": torch.tensor([0, 1, 0, 0, -100]), "n_labels": torch.tensor(3)}
    scores = torch.tensor(  # full softmax of the trained model, (seq, 4)
        [
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.1],
            [0.1, 0.0, 0.8, 0.1],  # an older class the page hides: not this pass's business
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    m._memory_probs = lambda docs: [scores]
    changed, considered = m._pseudo_pass([doc], n_cur=4)
    assert doc["labels"].tolist() == [3, 1, 0, 0, -100]
    assert (changed, considered) == (1, 3)


def test_mask_stale_o_hides_background_of_uncompleted_documents_only():
    from unittest.mock import patch

    from doccl.methods.colar_r import CoLaRRefresh

    m = _bare("transport")
    m.config = {"mask_stale_o": True}
    m.model = SimpleNamespace(label_to_id=dict.fromkeys(range(4)))  # head size 4
    stacked = {
        "labels": torch.tensor([[0, 1, -100], [0, 3, -100]]),
        "n_labels": torch.tensor([3, 4]),  # first document predates the running session
    }
    with patch.object(CoLaRRefresh, "_stack_replay", lambda self, docs: dict(stacked)):
        out = m._stack_replay([])
    assert out["labels"].tolist() == [[-100, 1, -100], [0, 3, -100]]


def test_capture_banks_the_completed_labels_when_asked():
    from unittest.mock import patch

    from doccl.methods.colar_r import CoLaRRefresh

    m = _bare("transport")
    m.config = {"bank_completed": True}
    m.model = SimpleNamespace(label_to_id=dict.fromkeys(range(4)))
    bbox = torch.arange(12).view(3, 4)
    page = {"labels": torch.tensor([0, 3, -100]), "bbox": bbox}
    m.store = []
    m._completed = {bbox.numpy().tobytes(): torch.tensor([2, 3, -100])}
    with patch.object(CoLaRRefresh, "_capture_task", lambda self, loader: self.store.append(page)):
        m._capture_task(None)
    assert m.store[0]["labels"].tolist() == [2, 3, -100]
