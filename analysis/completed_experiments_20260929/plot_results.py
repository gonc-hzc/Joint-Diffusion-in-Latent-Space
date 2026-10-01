"""Render publication-style figures from the extracted local experiment data."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = json.loads((OUT / "results.json").read_text())
RUNS = {run["name"]: run for run in DATA["runs"]}
PREFIX = "CL_CIFAR10_SUP_"
COLORS = ["#26749e", "#c55343", "#28865b", "#8b64ac"]
TASK_NAMES = ["T1: airplane / automobile", "T2: bird / cat", "T3: deer / dog", "T4: frog / horse"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 180})


def rows(name):
    with (OUT / "history" / f"{name}.csv").open() as handle:
        return [
            {key: float(value) for key, value in row.items() if value}
            for row in csv.DictReader(handle) if row.get("val/accuracy_ema")
        ]


def save(figure, name):
    figure.savefig(OUT / f"{name}.png", bbox_inches="tight")
    figure.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def consolidation_curves():
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    stages = ["TASK1-2_SEED12", "TASK1-3_SEED12", "TASK1-4_SEED12_RERUN_T1_4"]
    for index, stage in enumerate(stages):
        axis = axes.flat[index]
        name = PREFIX + stage
        history = rows(name)
        x = [(row["trainer/global_step"] + 1) / 1000 for row in history]
        axis.axvspan(0, 10, color="#eeeeee", label="Classification warm-up")
        axis.plot(x, [row["val/accuracy_ema"] * 100 for row in history], color="#26749e", lw=2.3, label="EMA")
        axis.plot(x, [row["val/accuracy"] * 100 for row in history], color="#adadad", lw=1.3, alpha=0.8, label="Normal weights")
        best = RUNS[name]["best_validation"]
        axis.scatter([(best["trainer/global_step"] + 1) / 1000], [best["val/accuracy_ema"] * 100], color="#26749e", s=50, zorder=5)
        n = RUNS[name]["class_count"]
        axis.axhline(n * 10, ls=":", color="#555555", label=f"Seen-class ceiling: {n * 10}%")
        if index == 2:
            original = rows(PREFIX + "TASK1-4_SEED12")
            axis.plot([(row["trainer/global_step"] + 1) / 1000 for row in original], [row["val/accuracy_ema"] * 100 for row in original], color="#bd7942", ls="--", lw=1.6, label="Original run (interrupted)")
        axis.set(title=["Consolidation T1-2", "Consolidation T1-3", "Consolidation T1-4 (rerun)"][index], xlabel="Optimizer steps (thousands)", ylabel="Accuracy on all 10 classes (%)", xlim=(0, 30), ylim=(0, n * 10 + 4))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    axis = axes.flat[3]
    best = [RUNS[PREFIX + stage]["best_seen_accuracy_pct"] for stage in stages]
    last = [RUNS[PREFIX + stage]["last_seen_accuracy_pct"] for stage in stages]
    x = np.arange(3)
    for offset, values, color, label in [(-0.19, best, "#26749e", "Best EMA checkpoint"), (0.19, last, "#c55343", "Final EMA checkpoint")]:
        bars = axis.bar(x + offset, values, 0.36, color=color, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    axis.set(xticks=x, xticklabels=["T1-2", "T1-3", "T1-4 rerun"], ylabel="Accuracy on seen classes (%)", title="Best vs final checkpoint", ylim=(0, 105))
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Completed CIFAR-10 consolidation runs, seed 12", fontsize=15)
    save(figure, "consolidation_curves")


def per_task_curves():
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for axis, stage, count in zip(axes, ["TASK1-3_SEED12", "TASK1-4_SEED12_RERUN_T1_4"], [3, 4]):
        name = PREFIX + stage
        history = rows(name)
        x = [(row["trainer/global_step"] + 1) / 1000 for row in history]
        axis.axvspan(0, 10, color="#eeeeee")
        for task in range(count):
            key = f"cl_tasks_val/task{task}_accuracy_ema"
            axis.plot(x, [row[key] * 100 for row in history], color=COLORS[task], lw=2.2, label=TASK_NAMES[task])
        best = RUNS[name]["best_validation"]
        axis.axvline((best["trainer/global_step"] + 1) / 1000, ls=":", color="#555555", label="Best overall EMA checkpoint")
        axis.set(title="T1-3 consolidation" if count == 3 else "T1-4 consolidation (rerun)", xlabel="Optimizer steps (thousands)", ylabel="Per-task EMA accuracy (%)", xlim=(0, 30), ylim=(0, 104))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8.5, loc="lower right")
    figure.suptitle("Task 2 declines while the new task improves", fontsize=15)
    save(figure, "per_task_curves")


def accuracy_matrix():
    names = [PREFIX + stage for stage in ["TASK1_SEED12", "TASK1-2_SEED12", "TASK1-3_SEED12", "TASK1-4_SEED12_RERUN_T1_4"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, selection in zip(axes, ["best", "last"]):
        matrix = np.full((4, 4), np.nan)
        for row_index, name in enumerate(names):
            run = RUNS[name]
            row = run[f"{selection}_validation"]
            if row_index == 0:
                matrix[0, 0] = run[f"{selection}_seen_accuracy_pct"]
            else:
                for task in range(row_index + 1):
                    matrix[row_index, task] = row[f"cl_tasks_val/task{task}_accuracy_ema"] * 100
        color = plt.get_cmap("YlGnBu").copy()
        color.set_bad("#f1f1f1")
        image = axis.imshow(matrix, cmap=color, vmin=0, vmax=100)
        for row in range(4):
            for col in range(4):
                if np.isfinite(matrix[row, col]):
                    axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white" if matrix[row, col] > 75 else "black")
        axis.set(xticks=range(4), xticklabels=["T1", "T2", "T3", "T4"], yticks=range(4), yticklabels=["T1 local", "T1-2", "T1-3", "T1-4 rerun"], xlabel="Evaluated task", ylabel="Training stage", title="Best checkpoint in each stage" if selection == "best" else "Final checkpoint in each stage")
    figure.colorbar(image, ax=axes, label="Per-task EMA accuracy (%)", shrink=0.85)
    save(figure, "continual_accuracy_matrix")


def replay_examples():
    import torch

    torch.set_num_threads(2)
    target = ROOT / "data" / "cl" / "CL_CIFAR10_SUP_TASK1-4_SEED12_RERUN_T1_4_imgs.pt"
    images = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
    labels = torch.load(target.with_name(target.name.replace("_imgs.pt", "_labels.pt")), map_location="cpu", weights_only=True)
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse"]
    generator = np.random.default_rng(42)
    figure, axes = plt.subplots(8, 6, figsize=(8, 10), constrained_layout=True)
    for label in range(8):
        candidates = np.flatnonzero(labels.numpy() == label)
        selected = generator.choice(candidates, size=6, replace=False)
        for col, index in enumerate(selected):
            axis = axes[label, col]
            axis.imshow(images[index].permute(1, 2, 0).numpy(), interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if col == 0:
                axis.set_ylabel(class_names[label], fontsize=9)
    figure.suptitle("Existing T1-4 replay images\nRows show teacher-assigned labels; fixed random selection", fontsize=12)
    figure.savefig(OUT / "replay_examples.png", bbox_inches="tight", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    consolidation_curves()
    per_task_curves()
    accuracy_matrix()
    replay_examples()
