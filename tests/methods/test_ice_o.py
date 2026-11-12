"""AGENT IMPL 2026-11-12 (D18): ICE-O loss/inference and SBR constant-background, synthetic."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from doccl.methods.colar_sbr import CoLaRSBR
from doccl.methods.ice_o import ice_logits, ice_o_loss


def test_ice_o_loss_is_ce_over_constant_other_and_session_tags():
    # head ids: 0=O, 1..2 = session-0 tags, 3 = a session-1 tag (absent from session-0 data)
    logits_k = torch.tensor([[[2.0, -1.0], [0.5, 0.5]]])  # 2 tokens, session-1 head (2 tags)
    labels = torch.tensor([[3, 0]])
    idx = torch.tensor([3, 4])
    got = ice_o_loss(logits_k, labels, idx, n_head=5, delta=0.0)
    t1 = -math.log(math.exp(2) / (1 + math.exp(2) + math.exp(-1)))  # tag 3 -> class 1
    t2 = -math.log(1 / (1 + 2 * math.exp(0.5)))  # O -> constant Other (class 0)
    assert abs(got.item() - (t1 + t2) / 2) < 1e-6


def test_ice_o_inference_takes_largest_tag_logit_across_heads_if_above_other():
    feat = torch.zeros(1, 3, 1)  # heads ignore it; each returns fixed logits per token
    h0 = lambda f: torch.tensor([[[1.0], [-2.0], [-0.5]]])  # noqa: E731  session 0: tag 1
    h1 = lambda f: torch.tensor([[[3.0], [-1.0], [-0.1]]])  # noqa: E731  session 1: tag 2
    out = ice_logits(feat, [h0, h1], [torch.tensor([1]), torch.tensor([2])], n_head=4, delta=0.0)
    assert out[0, 0].tolist() == [0.0, 1.0, 3.0, -1e4]
    # token 0: max tag logit 3 > 0 -> tag 2; tokens 1-2: every tag logit < 0 -> O; col 3 unowned
    assert out.argmax(-1).tolist() == [[2, 0, 0]]


def _sbr(const_o: bool, o_head: nn.Linear) -> CoLaRSBR:
    m = object.__new__(CoLaRSBR)  # only the readout pieces; no model or dataset
    m.model = SimpleNamespace(sbr_o=nn.ModuleList([o_head]))
    m.blocks, m.const_o = [torch.tensor([1, 2])], const_o
    return m


def test_constant_background_gets_no_gradient_and_equals_sbr_with_a_zero_learned_logit():
    torch.manual_seed(0)
    feat, logits = torch.randn(1, 4, 6), torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, -100]])
    head = nn.Linear(6, 1)
    const = _sbr(True, head)
    const._block_ce(logits, feat, labels, 0).backward()
    assert head.weight.grad is None and head.bias.grad is None
    assert logits.grad is not None
    zero = nn.Linear(6, 1)
    nn.init.zeros_(zero.weight)
    nn.init.zeros_(zero.bias)
    learned = _sbr(False, zero)
    a = const._block_logits(logits.detach(), feat, 0)
    b = learned._block_logits(logits.detach(), feat, 0)
    assert torch.equal(a, b) and a[..., 0].abs().max() == 0


class _Wrap(nn.Module):
    """Tiny stand-in for a token-classification wrapper (encoder -> plain Linear head)."""

    def __init__(self):
        super().__init__()
        self.enc, self.hidden_size = nn.Linear(3, 4), 4
        self.model = nn.Module()
        self.model.classifier = nn.Linear(4, 1)
        self.label_to_id, self.id_to_label = {"O": 0}, {0: "O"}

    def freeze_backbone(self):
        self.enc.requires_grad_(False)

    def expand_classifier(self, new):
        for lb in new:
            self.id_to_label[len(self.label_to_id)] = lb
            self.label_to_id[lb] = len(self.label_to_id)
        self.model.classifier = nn.Linear(4, len(self.label_to_id))

    def forward(self, x, **_):
        return SimpleNamespace(logits=self.model.classifier(self.enc(x)))


def test_ice_o_end_to_end_two_sessions_freezes_encoder_and_old_heads():
    from doccl.methods.ice_o import ICEO
    from doccl.types import TaskInfo

    torch.manual_seed(0)
    a, b, o = [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]
    w = _Wrap()
    m = ICEO(w, {"lr": 0.1, "epochs": 60})
    enc0 = w.enc.weight.clone()
    sessions = [(["O", "B-a"], [a, o, b], [1, 0, 0]), (["O", "B-b"], [b, o, a], [2, 0, 0])]
    for t, (labels, xs, ys) in enumerate(sessions):  # session data label only their own tag
        w.expand_classifier([lb for lb in labels if lb not in w.label_to_id])
        task = TaskInfo(task_id=t, task_name=f"s{t}", label_set=labels)
        batch = {"x": torch.tensor([xs]), "labels": torch.tensor([ys])}
        m.before_task(task, None)
        if t == 1:
            head0 = w.ice_heads[0].weight.clone()
        m.train_task(task, [batch])
    assert torch.equal(w.enc.weight, enc0)  # encoder never trained
    assert torch.equal(w.ice_heads[0].weight, head0)  # session-0 head frozen in session 1
    test = {"x": torch.tensor([[a, b, o]]), "labels": torch.tensor([[1, 2, 0]])}
    w.eval()
    preds = w(**test).logits.argmax(-1)  # the hook swaps in the ICE-O logits
    assert preds.tolist() == [[1, 2, 0]]
    assert m.evaluate({1: [test]})[1].f1 == 100.0
