"""Class-frequency table of a CIL stream: share of labelled tokens of class c in the training
set of the session that annotates c. The only statistic the transport readout needs.

    uv run python scripts/rca/class_freq_table.py cil_funsd results/gates/ot_gate0

Builds one dataset (CPU, no model). Do not run beside a training job on the local box.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
from omegaconf import OmegaConf

from doccl.data.encoders import build_encoder
from doccl.data.scenarios import get_scenario


def main(scenario_name: str, out_dir: str) -> None:
    model_cfg = OmegaConf.load("configs/model/layoutlmv3_base.yaml")
    scenario = get_scenario(scenario_name, encoder=build_encoder(model_cfg))
    freq: dict[int, float] = {}
    n_prev = 1
    for t, ds in enumerate(scenario.train_datasets):
        counts = torch.zeros(512, dtype=torch.long)
        for i in range(len(ds)):
            labels = ds[i]["labels"]
            counts += torch.bincount(labels[labels != -100], minlength=512)
        n_valid = int(counts.sum())
        n_cur = int(counts.nonzero().max()) + 1
        for c in range(n_prev, n_cur):
            freq[c] = counts[c].item() / max(n_valid, 1)
        print(f"session {t}: {len(ds)} documents, {n_valid} tokens, classes {n_prev}..{n_cur - 1}")
        n_prev = max(n_prev, n_cur)
    out = pathlib.Path(out_dir) / f"class_freq_{scenario_name}.json"
    out.write_text(json.dumps(freq, indent=1))
    print(f"sum of class shares {sum(freq.values()):.3f}; wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
