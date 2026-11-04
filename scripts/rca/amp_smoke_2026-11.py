"""GPU smoke: the 2-task CIL lifecycle under CUDA AMP (fp16 + GradScaler on Turing) for the
baselines whose loops were just wired to the AMP helpers, plus CoLaR-SBR's eval dump.
Tiny synthetic docs (no dataset build) — safe to run between grid jobs. Exit 1 on failure."""

import sys

import torch
import yaml

sys.path.insert(0, "tests/methods")
import test_all_methods_e2e as t  # noqa: E402

from doccl.methods.colar_sbr import CoLaRSBR  # noqa: E402
from doccl.methods.cpfd import CPFD  # noqa: E402
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper  # noqa: E402
from doccl.types import TaskInfo  # noqa: E402

SBR_CFG = {
    **yaml.safe_load(open("configs/method/colar_sbr.yaml")),
    "epochs": 1,
    "split_layer_k": 4,
    "docs_per_task": 2,
    "rank_r": 4,
    "replay_batch_size": 2,
}
CASES = {
    n: (t._METHOD_CLS[n], {**t._BASE_CFG, **t._METHOD_CFG.get(n, {})})
    for n in ("er", "der_pp", "lwf", "coda_prompt", "cl_lora")
}
CASES["cpfd"] = (CPFD, dict(t._BASE_CFG))
CASES["colar_sbr"] = (CoLaRSBR, SBR_CFG)
fails = []
for name, (cls, cfg) in CASES.items():
    model = method = None
    try:
        torch.manual_seed(0)
        model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=5)
        model.label_to_id = {lab: i for i, lab in enumerate(t._T0_LABELS)}
        model.id_to_label = {i: lab for lab, i in model.label_to_id.items()}
        model = model.cuda()
        method = cls(model, cfg)
        method.amp_enabled = True
        t0 = TaskInfo(task_id=0, task_name="cord_t0", label_set=t._T0_LABELS)
        l0 = t._run_one_task(method, t0, t._loader(n_classes=5))
        model.expand_classifier(t._T1_NEW)
        model.cuda()
        t1 = TaskInfo(task_id=1, task_name="cord_t1", label_set=["O"] + t._T1_NEW)
        l1 = t._run_one_task(method, t1, t._loader(n_classes=7))
        res = method.evaluate({0: t._loader(n_classes=7, n=2), 1: t._loader(n_classes=7, n=2)})
        ok = all(torch.isfinite(torch.tensor(x)) for x in (l0, l1)) and set(res) == {0, 1}
        if name == "colar_sbr":
            ok = ok and set(method.eval_dump) == {0, 1} and method.eval_dump[1]["e"].shape[1] == 2
        scaler = getattr(method, "_amp_scaler", None)
        print(
            f"{name:12s} {'OK ' if ok else 'BAD'} loss {l0:.3f}/{l1:.3f} "
            f"scaler={'on' if scaler is not None else 'off'} peak={torch.cuda.max_memory_allocated() / 2**30:.2f}G"
        )
        if not ok or scaler is None:
            fails.append(name)
    except Exception as e:  # report every method, then fail
        print(f"{name:12s} FAIL {type(e).__name__}: {e}")
        fails.append(name)
    finally:
        del model, method
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
print("SMOKE", "FAIL " + " ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
