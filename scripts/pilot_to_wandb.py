#!/usr/bin/env python3
"""Log pilot diagnostic JSONs to Weights & Biases for later review.

The pilot runner (``doccl.pilot.run_pilot``) writes one JSON per condition×seed
to ``results/pilot/``; it does NOT touch W&B by design. This backfill pushes each
completed JSON to W&B as its own run, so the diagnostics (per-task F1, CKA per
layer at each task boundary, Fisher per component group, AA/BWT/AF/FWT) are
browsable later.

Idempotent: a run is identified by name ``pilot_<condition>_seed<seed>[_ord...]``;
existing W&B runs with that name are skipped unless --force. Safe to re-run as new
JSONs land (e.g. from cron / the overnight orchestrator).

Creds come from the environment (WANDB_API_KEY/ENTITY/PROJECT) — source .env first.

Usage:
    set -a; source .env; set +a
    python scripts/pilot_to_wandb.py --pilot_dir results/pilot --project CL4IE
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run_name_for(summary: dict) -> str:
    cond = summary["condition"]
    seed = summary["seed"]
    order = summary.get("task_order", [0, 1, 2])
    suffix = "" if order == [0, 1, 2] else "_ord" + "".join(str(i) for i in order)
    return f"pilot_{cond}_seed{seed}{suffix}"


def existing_run_names(api, entity: str, project: str) -> set[str]:
    try:
        return {r.name for r in api.runs(f"{entity}/{project}")}
    except Exception:
        return set()


def log_one(summary: dict, project: str, entity: str | None, group: str) -> None:
    import wandb

    name = run_name_for(summary)
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type="pilot",
        tags=["pilot", summary["condition"], f"seed{summary['seed']}"],
        config={
            "condition": summary["condition"],
            "seed": summary["seed"],
            "task_order": summary.get("task_order", [0, 1, 2]),
            "phase": "pilot",
        },
        reinit=True,
    )

    # ── Per-task accuracy after each task (forgetting curve) ──
    for rec in summary.get("accuracy_records", []):
        step = rec["task_idx"]
        payload = {"task_step": step}
        for tid, m in rec["results"].items():
            payload[f"f1/task_{tid}"] = m["f1"] if isinstance(m, dict) else m
        wandb.log(payload, step=step)

    # ── CKA per layer at each task boundary ──
    for rec in summary.get("cka_records", []):
        boundary = rec["task_boundary"]  # e.g. "0_to_1"
        for layer, val in rec["cka"].items():
            short = layer.replace("model.layoutlmv3.", "").replace("model.", "")
            wandb.log({f"cka/{boundary}/{short}": val})

    # ── Fisher per component group per task ──
    for rec in summary.get("fisher_records", []):
        step = rec["task_idx"]
        for grp, val in rec["fisher_per_group"].items():
            wandb.log({f"fisher/{grp}": val}, step=step)

    # ── Final CL metrics + accuracy matrix as a table ──
    clm = summary.get("cl_metrics", {})
    wandb.summary.update({f"final/{k}": v for k, v in clm.items()})

    matrix = summary.get("matrix")
    if matrix:
        cols = [f"eval_t{j}" for j in range(len(matrix[0]))]
        tbl = wandb.Table(columns=["after_task"] + cols)
        for i, row in enumerate(matrix):
            tbl.add_data(f"t{i}", *row)
        wandb.log({"accuracy_matrix": tbl})

    run.finish()
    print(f"  logged {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", type=Path, default=Path("results/pilot"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "CL4IE"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--group", default="pilot-study")
    ap.add_argument("--force", action="store_true", help="re-log even if run name exists")
    args = ap.parse_args()

    jsons = sorted(args.pilot_dir.glob("*.json"))
    if not jsons:
        print(f"No pilot JSONs in {args.pilot_dir}")
        return

    import wandb

    api = wandb.Api()
    seen = set() if args.force else existing_run_names(api, args.entity, args.project)
    print(f"{len(jsons)} pilot JSONs; {len(seen)} runs already in {args.entity}/{args.project}")

    for jp in jsons:
        summary = json.loads(jp.read_text())
        name = run_name_for(summary)
        if name in seen:
            print(f"  skip {name} (already in W&B)")
            continue
        log_one(summary, args.project, args.entity, args.group)


if __name__ == "__main__":
    main()
