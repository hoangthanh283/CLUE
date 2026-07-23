from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_gate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "docre_graph_replay_gate.py"
    spec = importlib.util.spec_from_file_location("docre_graph_replay_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _document(title, relation, head_type="PER", tail_type="ORG", bridge=False):
    vertex_set = [
        [{"name": "head", "sent_id": 0, "pos": [0, 1], "type": head_type}],
        [{"name": "tail", "sent_id": 1, "pos": [0, 1], "type": tail_type}],
    ]
    labels = [{"h": 0, "t": 1, "r": relation, "evidence": [0, 1]}]
    if bridge:
        vertex_set.append([{"name": "other", "sent_id": 1, "pos": [1, 2], "type": "LOC"}])
        labels.append({"h": 1, "t": 2, "r": "bridge_relation", "evidence": [1]})
    return {
        "title": title,
        "sents": [["head"], ["tail", "other"]],
        "vertexSet": vertex_set,
        "labels": labels,
    }


def test_graph_salient_memory_covers_relations_and_preserves_documents(tmp_path):
    common = _document("common", "common_relation")
    duplicate = _document("duplicate", "common_relation")
    rare_bridge = _document("rare", "rare_relation", bridge=True)
    budget = gate.memory_bytes([common, rare_bridge])

    task0, task1 = tmp_path / "task0.json", tmp_path / "task1.json"
    task0.write_text(json.dumps([common, duplicate]))
    task1.write_text(json.dumps([rare_bridge]))
    output_dir = tmp_path / "output"
    report = gate.run([task0, task1], budget, output_dir, seed=42)
    selected = json.loads((output_dir / "task1_graph_salient.json").read_text())

    assert gate.memory_bytes(selected) <= budget
    assert {document["title"] for document in selected} == {"common", "rare"}
    assert {label["r"] for document in selected for label in document["labels"]} == {
        "common_relation",
        "rare_relation",
        "bridge_relation",
    }
    assert selected[[document["title"] for document in selected].index("rare")] == rare_bridge
    assert report["tasks"][-1]["graph_salient"]["relation_coverage"] == 1.0
