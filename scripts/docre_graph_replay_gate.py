"""Build random and graph-salient replay memories from DocRED-format task files.

This is the cheapest falsification gate for salient-entity graph replay:

* memory stores intact documents, preserving text/entity/evidence/label consistency;
* graph salience only changes which documents survive a fixed serialized-byte budget;
* each input JSON file is one continual-learning task;
* after every task, old memory plus current documents are reselected;
* outputs remain ordinary DocRED JSON, so an existing DocRE trainer can consume them.

Run:
    uv run python scripts/docre_graph_replay_gate.py \
        task0.json task1.json --budget-bytes 1000000 \
        --output-dir results/docre_graph_replay_gate
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _encoded(document: dict) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()


def memory_bytes(documents: list[dict]) -> int:
    """Exact compact-JSON size of a list of documents."""
    return 2 + sum(len(_encoded(document)) for document in documents) + max(len(documents) - 1, 0)


def _entity_type(document: dict, entity_id: int) -> str:
    mentions = document["vertexSet"][entity_id]
    return str(mentions[0].get("type", "UNKNOWN")) if mentions else "UNKNOWN"


def _mention_sentences(document: dict, entity_id: int) -> set[int]:
    return {int(mention["sent_id"]) for mention in document["vertexSet"][entity_id]}


def graph_features(document: dict) -> set[str]:
    """Return interpretable relation/entity/evidence features for greedy coverage."""
    features: set[str] = set()
    degree: Counter[int] = Counter()
    relations_by_entity: dict[int, set[str]] = {}

    for label in document.get("labels", []):
        head, tail, relation = int(label["h"]), int(label["t"]), str(label["r"])
        head_type, tail_type = _entity_type(document, head), _entity_type(document, tail)
        features.add(f"relation:{relation}")
        features.add(f"type_pair:{relation}:{head_type}:{tail_type}")
        if _mention_sentences(document, head).isdisjoint(_mention_sentences(document, tail)):
            features.add(f"cross_sentence:{relation}")
        degree.update((head, tail))
        relations_by_entity.setdefault(head, set()).add(relation)
        relations_by_entity.setdefault(tail, set()).add(relation)

    for entity_id, entity_degree in degree.items():
        if entity_degree >= 2:
            for relation in relations_by_entity[entity_id]:
                features.add(f"bridge:{relation}")
    return features


def validate_document(document: dict) -> None:
    for key in ("title", "sents", "vertexSet"):
        if key not in document:
            raise ValueError(f"DocRED document missing {key!r}")
    for label in document.get("labels", []):
        for key in ("h", "t", "r"):
            if key not in label:
                raise ValueError(f"DocRED label missing {key!r}")
        if not 0 <= int(label["h"]) < len(document["vertexSet"]):
            raise ValueError("head entity index is out of range")
        if not 0 <= int(label["t"]) < len(document["vertexSet"]):
            raise ValueError("tail entity index is out of range")


def _deduplicate(documents: list[dict]) -> list[dict]:
    by_title = {}
    for document in documents:
        validate_document(document)
        by_title[str(document["title"])] = document
    return list(by_title.values())


def select_random(documents: list[dict], budget_bytes: int, seed: int) -> list[dict]:
    candidates = _deduplicate(documents)
    random.Random(seed).shuffle(candidates)
    selected = []
    for document in candidates:
        if memory_bytes([*selected, document]) <= budget_bytes:
            selected.append(document)
    return selected


def select_graph_salient(documents: list[dict], budget_bytes: int) -> list[dict]:
    """Greedy weighted set cover over relation-graph features per serialized byte."""
    candidates = _deduplicate(documents)
    feature_counts = Counter(
        feature for document in candidates for feature in graph_features(document)
    )
    selected: list[dict] = []
    covered: set[str] = set()

    while candidates:
        fitting = [
            document
            for document in candidates
            if memory_bytes([*selected, document]) <= budget_bytes
        ]
        if not fitting:
            break

        def score(document: dict) -> tuple[float, int, str]:
            features = graph_features(document)
            gain = sum(
                (
                    4.0
                    if feature.startswith("relation:")
                    else 2.0 if feature.startswith(("type_pair:", "cross_sentence:")) else 1.0
                )
                / feature_counts[feature]
                for feature in features - covered
            )
            cost = len(_encoded(document)) + (1 if selected else 0)
            return gain / cost, len(features), str(document["title"])

        chosen = max(fitting, key=score)
        selected.append(chosen)
        covered.update(graph_features(chosen))
        candidates.remove(chosen)
    return selected


def summarize(documents: list[dict], universe: set[str]) -> dict:
    features = set().union(*(graph_features(document) for document in documents))
    relations = {feature for feature in features if feature.startswith("relation:")}
    universe_relations = {feature for feature in universe if feature.startswith("relation:")}
    return {
        "documents": len(documents),
        "bytes": memory_bytes(documents),
        "relations": len(relations),
        "relation_coverage": round(len(relations) / max(len(universe_relations), 1), 4),
        "graph_feature_coverage": round(len(features) / max(len(universe), 1), 4),
        "positive_edges": sum(len(document.get("labels", [])) for document in documents),
    }


def run(task_paths: list[Path], budget_bytes: int, output_dir: Path, seed: int) -> dict:
    if budget_bytes < 2:
        raise ValueError("budget_bytes must fit at least an empty JSON list")
    output_dir.mkdir(parents=True, exist_ok=True)
    memories = {"random": [], "graph_salient": []}
    universe: set[str] = set()
    report = {"budget_bytes": budget_bytes, "seed": seed, "tasks": []}

    for task_id, task_path in enumerate(task_paths):
        current = json.loads(task_path.read_text())
        if not isinstance(current, list):
            raise ValueError(f"{task_path} must contain a JSON list")
        for document in current:
            validate_document(document)
            universe.update(graph_features(document))

        memories["random"] = select_random(
            [*memories["random"], *current], budget_bytes, seed + task_id
        )
        memories["graph_salient"] = select_graph_salient(
            [*memories["graph_salient"], *current], budget_bytes
        )

        task_report = {"task": task_id, "source": str(task_path)}
        for strategy, documents in memories.items():
            target = output_dir / f"task{task_id}_{strategy}.json"
            target.write_text(json.dumps(documents, ensure_ascii=False, separators=(",", ":")))
            task_report[strategy] = summarize(documents, universe)
        report["tasks"].append(task_report)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path, help="DocRED JSON files in task order")
    parser.add_argument("--budget-bytes", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(args.tasks, args.budget_bytes, args.output_dir, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
