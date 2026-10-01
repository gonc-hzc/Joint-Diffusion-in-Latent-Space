"""Recompute stage-end metrics from the archived seed-12 data, using only CPU.

Per-task metrics retain the original logger's batch aggregation approximation.
Task1-5 is not part of this four-stage snapshot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent
STAGES = [
    "CL_CIFAR10_SUP_TASK1_SEED12",
    "CL_CIFAR10_SUP_TASK1-2_SEED12",
    "CL_CIFAR10_SUP_TASK1-3_SEED12",
    "CL_CIFAR10_SUP_TASK1-4_SEED12_RERUN_T1_4",
]


def main():
    data = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in data["runs"]}
    matrix = []
    seen_accuracies = []
    rows = []
    for index, name in enumerate(STAGES):
        run = runs[name]
        if not run["reached_target_steps"] or run["checkpoint"]["global_step"] != run["max_steps_target"]:
            raise ValueError(f"Required archived stage is incomplete: {name}")
        accuracy = run["last_seen_accuracy_pct"]
        seen_accuracies.append(accuracy)
        task_values = [accuracy] if index == 0 else [
            run["last_validation"][f"cl_tasks_val/task{task}_accuracy_ema"] * 100
            for task in range(index + 1)
        ]
        matrix.append(task_values)
        forgetting = [max(matrix[stage][task] for stage in range(task, index)) - task_values[task] for task in range(index)]
        backward_transfer = [task_values[task] - matrix[task][task] for task in range(index)]
        rows.append({
            "stage": index + 1,
            "name": name,
            "all10_accuracy_pct": run["last_accuracy_all10_pct"],
            "seen_accuracy_pct": accuracy,
            "logged_task_macro_accuracy_pct_approx": mean(task_values),
            "old_tasks_macro_accuracy_pct_approx": mean(task_values[:-1]) if index else None,
            "new_task_accuracy_pct_approx": task_values[-1],
            "forgetting_pp_approx": mean(forgetting) if index else None,
            "backward_transfer_pp_approx": mean(backward_transfer) if index else None,
            "average_incremental_seen_accuracy_pct": mean(seen_accuracies),
            "forgetting_by_old_task_pp_approx": forgetting,
            "backward_transfer_by_old_task_pp_approx": backward_transfer,
        })
    fields = [key for key in rows[0] if not key.endswith("by_old_task_pp_approx")]
    with (OUT / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    result = {
        "source": "results.json",
        "scope": "Completed Task1-4 stage-end snapshot; Task1-5 excluded",
        "selection": "last validation, EMA weights, global stage ends",
        "accuracy_unit": "percent",
        "change_unit": "percentage points",
        "per_task_aggregation": "Approximate: original per-batch task accuracy aggregated by Lightning; Task1 uses audited seen-class conversion",
        "aia_definition": "Unweighted average of stage-end seen-class overall accuracies",
        "task_matrix_pct": matrix,
        "stages": rows,
    }
    (OUT / "metrics_explained.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    final = rows[-1]
    print(f"Task1-4: seen accuracy {final['seen_accuracy_pct']:.2f}%")
    print(f"Approximate F4 {final['forgetting_pp_approx']:.2f} pp; BWT4 {final['backward_transfer_pp_approx']:.2f} pp")
    print(f"New task {final['new_task_accuracy_pct_approx']:.2f}%; old task macro average {final['old_tasks_macro_accuracy_pct_approx']:.2f}% (approximate logged metrics)")
    print(f"Stage-average seen accuracy AIA4 {final['average_incremental_seen_accuracy_pct']:.2f}%")
    print(f"Saved: {OUT / 'stage_metrics.csv'} and {OUT / 'metrics_explained.json'}")


if __name__ == "__main__":
    main()
