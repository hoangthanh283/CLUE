"""CoLaR-MiB unbiased CE — hand-computed values, both directions; synthetic tensors only."""

import math

import torch
import torch.nn.functional as F  # noqa: N812

from doccl.methods.colar_mib import CoLaRMiB, unbiased_ce
from tests.methods.test_latent_replay import _batch, _Wrapper

# 4-class head: 0 = O, 1 = old (session 0), 2-3 = new (session 1); one doc, 3 tokens
LOGITS = torch.tensor([[[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 0.0, 1.0]]])
LABELS = torch.tensor([[0, 2, -100]])


def _nll(p_event):
    return -math.log(p_event)


def test_current_direction_matches_hand_computation():
    p = LOGITS.softmax(-1)[0]
    merge = torch.tensor([True, True, False, False]).view(1, 1, -1)  # O + old class 1
    want = (_nll(p[0, 0] + p[0, 1]) + _nll(p[1, 2])) / 2
    assert math.isclose(unbiased_ce(LOGITS, LABELS, merge).item(), want, rel_tol=1e-6)


def test_replay_direction_matches_hand_computation():
    p = LOGITS.softmax(-1)[0]
    merge = torch.tensor([True, False, True, True]).view(1, 1, -1)  # O + classes after storage
    want = (_nll(p[0, 0] + p[0, 2] + p[0, 3]) + _nll(p[1, 2])) / 2
    assert math.isclose(unbiased_ce(LOGITS, LABELS, merge).item(), want, rel_tol=1e-6)


def test_one_session_reduces_to_plain_ce():
    merge = torch.tensor([True, False, False, False]).view(1, 1, -1)
    plain = F.cross_entropy(LOGITS.reshape(-1, 4), LABELS.reshape(-1), ignore_index=-100)
    assert torch.allclose(unbiased_ce(LOGITS, LABELS, merge), plain)


def test_method_uses_session_boundaries():
    torch.manual_seed(0)
    m = CoLaRMiB(_Wrapper(), {"split_layer_k": 2, "docs_per_task": 2, "replay_batch_size": 2})
    batch = _batch(2)
    out = m.model(**batch)
    plain = F.cross_entropy(out.logits.reshape(-1, 4), batch["labels"].reshape(-1))
    assert torch.allclose(m._current_ce(out, batch), plain)  # session 0: n_old unset
    m.state.custom["n_old"] = 2  # session 1: head ids 0, 1 are O + old
    want = unbiased_ce(out.logits, batch["labels"], torch.tensor([1, 1, 0, 0]).bool())
    assert torch.allclose(m._current_ce(out, batch), want)

    m._capture_task([batch])
    for d, n in zip(m.store, (2, 4), strict=True):
        d["n_labels"] = torch.tensor(n)  # doc 0 stored before classes 2-3, doc 1 after
    replay = m._stack_replay(m.store)
    rout = m._replay_forward(replay)
    cls = torch.arange(4)
    merge = torch.stack([(cls >= 2) | (cls == 0), cls == 0]).unsqueeze(1)
    assert torch.allclose(rout.loss, unbiased_ce(rout.logits, replay["labels"], merge))
