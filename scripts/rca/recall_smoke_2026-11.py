"""GPU smoke: 2-task CIL lifecycle for colar_sbr_recall (base frozen after task 0, one LoRA
delta for session 1, blocks read under their own delta). Tiny synthetic docs. Exit 1 on failure."""

import sys

import torch
import yaml

sys.path.insert(0, "tests/methods")
import test_all_methods_e2e as t  # noqa: E402

from doccl.methods.colar_sbr_recall import CoLaRSBRRecall  # noqa: E402
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper  # noqa: E402
from doccl.types import TaskInfo  # noqa: E402

CFG = {
    **yaml.safe_load(open("configs/method/colar_sbr_recall.yaml")),
    "epochs": 1,
    "split_layer_k": 4,
    "docs_per_task": 2,
    "rank_r": 4,
    "replay_batch_size": 2,
}
torch.manual_seed(0)
model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=5)
model.label_to_id = {lab: i for i, lab in enumerate(t._T0_LABELS)}
model.id_to_label = {i: lab for lab, i in model.label_to_id.items()}
model = model.cuda()
model.enable_gradient_checkpointing()
m = CoLaRSBRRecall(model, CFG)
t0 = TaskInfo(task_id=0, task_name="cord_t0", label_set=t._T0_LABELS)
l0 = t._run_one_task(m, t0, t._loader(n_classes=5))
base_after0 = {n: p.detach().clone() for n, p in model.model.layoutlmv3.named_parameters()}
model.expand_classifier(t._T1_NEW)
model.cuda()
t1 = TaskInfo(task_id=1, task_name="cord_t1", label_set=["O"] + t._T1_NEW)
l1 = t._run_one_task(m, t1, t._loader(n_classes=7))
res = m.evaluate({0: t._loader(n_classes=7, n=2), 1: t._loader(n_classes=7, n=2)})
moved = [
    n
    for n, p in model.model.layoutlmv3.named_parameters()
    if ".A." not in n and ".B." not in n and not torch.equal(p.detach(), base_after0[n])
]
delta_moved = any(
    p.abs().sum() > 0 for lo in m.loras for p in lo.B
)  # B starts at 0; training session 1 must move it
ok = (
    all(torch.isfinite(torch.tensor(x)) for x in (l0, l1))
    and set(res) == {0, 1}
    and m.block_delta == [None, 0]
    and not moved
    and delta_moved
    and set(m.eval_dump) == {0, 1}
    and m.eval_dump[1]["e"].shape[1] == 2
)
print(
    f"recall {'OK ' if ok else 'BAD'} loss {l0:.3f}/{l1:.3f} block_delta={m.block_delta} "
    f"base_moved={len(moved)} delta_moved={delta_moved} sites={len(m.loras)} "
    f"mem={m.memory_bytes() / 2**20:.2f}MB peak={torch.cuda.max_memory_allocated() / 2**30:.2f}G"
)
print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
