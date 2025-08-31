"""
Continual Learning evaluation metrics.

Computes standard CL metrics from the accuracy matrix R, where
R[i][j] is the evaluation accuracy on task j after finishing training task i
(0-based indexing). Also uses R0[j] which is the accuracy on task j before
training any task.
"""

from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")


def compute_cl_metrics(
    R: List[List[float]],
    R0: List[float],
    task_names: List[str],
) -> Dict[str, Any]:
    """Compute ACC, BWT, FWT, AAA, and Forgetting measures.

    Args:
        R: T x T matrix where R[i][j] is accuracy on task j after task i.
        R0: length-T vector of accuracies before any training on task j.
        task_names: names of tasks in order.

    Returns:
        A dictionary with aggregated metrics and helpful breakdowns.
    """
    T = len(task_names)
    assert len(R) == T and all(len(row) == T for row in R), "R must be T x T"
    assert len(R0) == T, "R0 must be length T"

    # ACC: final average accuracy after last task, across all tasks
    final_row = R[T - 1]
    ACC = sum(final_row) / T if T > 0 else 0.0

    # BWT: average change on old tasks after learning all tasks
    # BWT = (1/(T-1)) * sum_{j=0..T-2} (R[T-1, j] - R[j, j])
    if T > 1:
        BWT = sum(R[T - 1][j] - R[j][j] for j in range(T - 1)) / (T - 1)
    else:
        BWT = 0.0

    # FWT: performance on new tasks before training them, relative to baseline R0
    # FWT = (1/(T-1)) * sum_{j=1..T-1} (R[j-1, j] - R0[j])
    if T > 1:
        FWT = sum(R[j - 1][j] - R0[j] for j in range(1, T)) / (T - 1)
    else:
        FWT = 0.0

    # AAA: Average Any-time Accuracy across the sequence
    # After each task i, average accuracy over seen tasks {0..i}
    AAA_curve = []
    for i in range(T):
        avg_i = sum(R[i][j] for j in range(i + 1)) / (i + 1)
        AAA_curve.append(avg_i)
    AAA = sum(AAA_curve) / T if T > 0 else 0.0

    # Forgetting per task: max drop from best accuracy achieved on that task to final accuracy
    # For each task j: F_j = max_{i in j..T-1} R[i, j] - R[T-1, j]
    forgetting_per_task = {}
    for j in range(T):
        best_j = max(R[i][j] for i in range(j, T))
        forgetting_per_task[task_names[j]] = max(0.0, best_j - R[T - 1][j])
    forgetting = sum(forgetting_per_task.values()) / T if T > 0 else 0.0

    return {
        "ACC": ACC,
        "BWT": BWT,
        "FWT": FWT,
        "AAA": AAA,
        "AAA_curve": AAA_curve,
        "Forgetting": forgetting,
        "Forgetting_per_task": forgetting_per_task,
    }


def save_aaa_curve_plot(aaa_curve: List[float], task_names: List[str], out_path: str) -> None:
    """Save a line plot for the AAA curve.

    Args:
        aaa_curve: list where index i is AAA after finishing task i.
        task_names: names of tasks for x-axis labels.
        out_path: path to write the image (PNG recommended).
    """
    try:
        sns.set_context("talk")
        sns.set_style("whitegrid")
    except Exception:
        pass

    xs = list(range(1, len(aaa_curve) + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, aaa_curve, marker="o")
    plt.xticks(xs, task_names, rotation=30, ha="right")
    plt.xlabel("After Task")
    plt.ylabel("AAA")
    plt.title("Average Any-time Accuracy (AAA)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
