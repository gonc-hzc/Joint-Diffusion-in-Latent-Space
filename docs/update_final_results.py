"""Finalize the current seed-12 study guide after Task1-5 finishes.

Reads original artifacts on CPU, retains the earlier analysis snapshot, and
updates only the marked result block in the guide. Does not launch training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "analysis" / "completed_experiments_20260929"
OUT = ROOT / "analysis" / "full_experiment_results"
GUIDE = ROOT / "docs" / "JDCL自学精讲与复现指南.md"
PREFIX = "CL_CIFAR10_SUP_"
START = "<!-- FINAL_RESULTS_START -->"
END = "<!-- FINAL_RESULTS_END -->"


def load_extractor():
    target = SNAPSHOT / "extract_results.py"
    spec = importlib.util.spec_from_file_location("jdcl_completed_experiment_extractor", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.torch.set_num_threads(2)
    return module


def get_argument(arguments, name):
    if name not in arguments:
        return None
    return arguments[arguments.index(name) + 1]


def task_matrix(stage_runs, selection):
    matrix = [[None] * 5 for _ in range(5)]
    for stage, run in enumerate(stage_runs):
        if stage == 0:
            matrix[stage][0] = run[f"{selection}_seen_accuracy_pct"]
        else:
            row = run[f"{selection}_validation"]
            for task in range(stage + 1):
                matrix[stage][task] = row[f"cl_tasks_val/task{task}_accuracy_ema"] * 100
    return matrix


def write_summary(runs):
    fields = ["name", "reached_target_steps", "class_count", "max_steps_target", "best_completed_steps", "last_validation_completed_steps", "best_accuracy_all10_pct", "last_accuracy_all10_pct", "best_seen_accuracy_pct", "last_seen_accuracy_pct"]
    with (OUT / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: run[key] for key in fields} for run in runs)


def write_matrix(matrix):
    with (OUT / "task_accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "task1_pct", "task2_pct", "task3_pct", "task4_pct", "task5_pct"])
        for index, row in enumerate(matrix):
            writer.writerow([index + 1, *row])


def audit_final_replay(extractor, run):
    arguments = run.get("args", [])
    images_argument = get_argument(arguments, "--saved-samples")
    labels_argument = get_argument(arguments, "--saved-labels")
    images_path = ROOT / images_argument if images_argument else ROOT / "data" / "cl" / f"{run['name']}_imgs.pt"
    labels_path = ROOT / labels_argument if labels_argument else ROOT / "data" / "cl" / f"{run['name']}_labels.pt"
    if not images_path.is_file() or not labels_path.is_file():
        return None
    images = extractor.torch.load(images_path, map_location="cpu", weights_only=True, mmap=True)
    labels = extractor.torch.load(labels_path, map_location="cpu", weights_only=True)
    values, counts = labels.unique(return_counts=True)
    if len(images) != len(labels) or labels.ndim != 1:
        raise ValueError("Replay images and hard labels have inconsistent shapes")
    array = images.numpy()
    digests = {hashlib.sha256(image.tobytes()).hexdigest() for image in array}
    result = {
        "name": run["name"],
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "counts": dict(zip(values.tolist(), counts.tolist())),
        "unique_images": len(digests),
        "total_images": len(images),
        "sha256_tensor": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        "sha256_labels": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "range": [float(images.min()), float(images.max())],
    }
    del images, labels
    return result


def make_plots(final_run, matrix):
    temporary_dependencies = Path("/tmp/jdcl_analysis_plot_deps")
    if temporary_dependencies.is_dir():
        sys.path.insert(0, str(temporary_dependencies))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return [], "绘图依赖不可用，数值表已生成；安装独立 Matplotlib 依赖后重新执行可补齐图表。"

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with (OUT / "history" / f"{final_run['name']}.csv").open() as handle:
        history = [
            {key: float(value) for key, value in row.items() if value}
            for row in csv.DictReader(handle) if row.get("val/accuracy_ema")
        ]
    x = [(row["trainer/global_step"] + 1) / 1000 for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["val/accuracy_ema"] * 100 for row in history], label="EMA", lw=2)
    axes[0].plot(x, [row["val/accuracy"] * 100 for row in history], label="Normal", color="#999999", lw=1)
    axes[0].set(title="Task1-5 overall accuracy", xlabel="Optimizer steps (thousands)", ylabel="Accuracy (%)")
    task_names = ["T1 airplane/automobile", "T2 bird/cat", "T3 deer/dog", "T4 frog/horse", "T5 ship/truck"]
    for task, label in enumerate(task_names):
        key = f"cl_tasks_val/task{task}_accuracy_ema"
        axes[1].plot(x, [row[key] * 100 for row in history], label=label, lw=1.8)
    axes[1].set(title="Task1-5 per-task EMA accuracy", xlabel="Optimizer steps (thousands)", ylabel="Per-task accuracy (%)", ylim=(0, 104))
    for axis in axes:
        axis.axvspan(0, 10, color="#eeeeee", zorder=-1)
        axis.axvline(final_run["best_completed_steps"] / 1000, ls=":", color="#555555")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=7.5)
    for suffix in ["png", "svg"]:
        figure.savefig(OUT / f"final_task_curves.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(figure)

    array = np.array([[float("nan") if value is None else value for value in row] for row in matrix])
    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    color = plt.get_cmap("YlGnBu").copy()
    color.set_bad("#eeeeee")
    plot = axis.imshow(array, cmap=color, vmin=0, vmax=100)
    for row in range(5):
        for col in range(5):
            if np.isfinite(array[row, col]):
                axis.text(col, row, f"{array[row,col]:.2f}", ha="center", va="center", color="white" if array[row,col] > 75 else "black")
    axis.set(xticks=range(5), xticklabels=["T1", "T2", "T3", "T4", "T5"], yticks=range(5), yticklabels=["G1", "G2", "G3", "G4 rerun", "G5"], xlabel="Evaluated task", ylabel="Final checkpoint of each stage", title="Complete five-task accuracy matrix")
    figure.colorbar(plot, ax=axis, label="EMA accuracy (%)")
    for suffix in ["png", "svg"]:
        figure.savefig(OUT / f"five_task_accuracy_matrix.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(figure)
    return ["final_task_curves.png", "five_task_accuracy_matrix.png"], None


def render_result(stage_runs, final_run, matrix, figures, plot_note):
    previous_maxima = [max(matrix[stage][task] for stage in range(task, 4)) for task in range(4)]
    forgetting = [previous_maxima[task] - matrix[4][task] for task in range(4)]
    backward_transfer = [matrix[4][task] - matrix[task][task] for task in range(4)]
    f5 = sum(forgetting) / 4
    bwt5 = sum(backward_transfer) / 4
    metrics = {
        "final_all10_accuracy_pct": final_run["last_accuracy_all10_pct"],
        "best_all10_accuracy_pct": final_run["best_accuracy_all10_pct"],
        "best_completed_steps": final_run["best_completed_steps"],
        "final_completed_steps": final_run["checkpoint"]["global_step"],
        "forgetting_per_old_task_pp": forgetting,
        "mean_forgetting_f5_pp": f5,
        "mean_backward_transfer_bwt5_pp": bwt5,
        "per_task_values_are_batch_aggregated": True,
    }
    (OUT / "final_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"Task1–5 已完成 {final_run['checkpoint']['global_step']:,} 个优化步骤。核对时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。checkpoint 步数、训练完成消息以及最佳分数与日志一致。",
        "",
        "| 全局阶段 | 全十类最佳 EMA | 全十类结束 EMA | 已学类别结束 EMA | 最佳点步数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, run in enumerate(stage_runs):
        lines.append(f"| G{index+1}{' 重跑' if index == 3 else ''} | {run['best_accuracy_all10_pct']:.2f}% | {run['last_accuracy_all10_pct']:.2f}% | {run['last_seen_accuracy_pct']:.2f}% | {run['best_completed_steps']:,} |")
    lines += ["", "五个任务全部学满时，已学类别准确率与完整十类准确率相同。下面使用各全局阶段结束点组成任务矩阵，任务指标保留原日志的批次聚合口径。", "", "| 全局阶段 | Task1 | Task2 | Task3 | Task4 | Task5 |", "|---|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(matrix):
        cells = ["—" if value is None else f"{value:.2f}%" for value in row]
        lines.append(f"| G{index+1}{' 重跑' if index == 3 else ''} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"最终结束准确率为 **{final_run['last_accuracy_all10_pct']:.2f}%**；最佳诊断准确率为 **{final_run['best_accuracy_all10_pct']:.2f}%**，出现在 **{final_run['best_completed_steps']:,} 步**。最佳值仍受现有测试集选择规则影响。",
        "",
        f"旧四个任务的平均遗忘率 $F_5$ 约 **{f5:.2f} 个百分点**，后向迁移 $BWT_5$ 约 **{bwt5:.2f} 个百分点**。这两个值从全局阶段结束矩阵计算，任务指标是日志聚合的近似值；没有混入独立局部专家的成绩。",
        "",
        "| 旧任务 | 此前全局阶段最好成绩 | G5 结束成绩 | 遗忘幅度 |",
        "|---|---:|---:|---:|",
    ]
    for task in range(4):
        lines.append(f"| Task{task+1} | {previous_maxima[task]:.2f}% | {matrix[4][task]:.2f}% | {forgetting[task]:.2f} 个百分点 |")
    strongest_loss = max(range(4), key=lambda task: forgetting[task])
    if forgetting[strongest_loss] > 0:
        lines += ["", f"按这些结束点，旧任务中 Task{strongest_loss+1} 的遗忘幅度最大，为约 {forgetting[strongest_loss]:.2f} 个百分点。判断原因时还需要检查回放、教师预测和对照实验。"]
    gap = final_run["best_accuracy_all10_pct"] - final_run["last_accuracy_all10_pct"]
    lines += ["", f"最佳点与结束点相差 {gap:.2f} 个百分点。可按曲线检查哪些任务在这段期间改善或退化，再评价阶段结束点的选择。"]
    replay = final_run.get("replay_audit")
    if replay:
        count_text = "、".join(f"类别{label}：{count}" for label, count in replay["counts"].items())
        lines += ["", f"最终阶段回放共 {replay['total_images']:,} 张，完全重复图像去重后为 {replay['unique_images']:,} 张，像素范围为 {replay['range']}。各类配额为：{count_text}。这个检查反映数量与完全重复情况，生成语义质量和伪标签正确性仍需另外评价。"]
    for figure in figures:
        label = "最终阶段的整体与任务曲线" if figure.startswith("final_task") else "完整五任务的准确率矩阵"
        lines += ["", f"![{label}](../analysis/full_experiment_results/{figure})"]
    if plot_note:
        lines += ["", plot_note]
    lines += [
        "",
        "原始提取数据：[完整 JSON](../analysis/full_experiment_results/results.json)、[阶段汇总 CSV](../analysis/full_experiment_results/run_summary.csv)、[任务矩阵 CSV](../analysis/full_experiment_results/task_accuracy_matrix.csv)、[最终指标 JSON](../analysis/full_experiment_results/final_metrics.json)。",
        "",
        "本次仍只有 seed 12，没有组件消融和跨 seed 重复；完整训练结束使五任务结果可报告，但不能据此单独推断某个组件的贡献或统计稳定性。",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=PREFIX + "TASK1-5_SEED12", help="Final run name; its teachers must match the archived experiment chain")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    checkpoint_path = ROOT / "logs" / args.run / "checkpoints" / "last.ckpt"
    if not checkpoint_path.is_file():
        print("Task1–5 尚未生成最终 checkpoint；请在训练结束后执行。", file=sys.stderr)
        return 2
    extractor = load_extractor()
    metadata = extractor.checkpoint_metadata(ROOT / "logs" / args.run)
    if metadata.get("global_step") != 30000:
        print(f"Task1–5 checkpoint 当前为 {metadata.get('global_step')} 步，目标为 30,000 步；结果区块保持不变。", file=sys.stderr)
        return 2
    candidates = sorted(path for path in (ROOT / "wandb").glob("offline-run-*") if path.name.endswith("-" + args.run))
    if not candidates:
        print("缺少对应的离线 W&B 日志；结果区块保持不变。", file=sys.stderr)
        return 2
    raw_metadata = json.loads((candidates[-1] / "files" / "wandb-metadata.json").read_text())
    arguments = raw_metadata.get("args", [])
    expected_paths = {
        "-o": "logs/CL_CIFAR10_SUP_TASK1-4_SEED12_RERUN_T1_4/checkpoints/last.ckpt",
        "-c": "logs/CL_CIFAR10_SUP_TASK1-4_SEED12_RERUN_T1_4/checkpoints/last.ckpt",
        "-n": "logs/CL_CIFAR10_SUP_TASK5_SEED12_FULL/checkpoints/last.ckpt",
    }
    for argument, expected in expected_paths.items():
        observed = get_argument(arguments, argument)
        if observed is None or (ROOT / observed).resolve() != (ROOT / expected).resolve():
            print(f"{argument} 的权重来源与已有阶段快照不同；应先核对实验链，结果区块保持不变。", file=sys.stderr)
            return 2
    guide_text = GUIDE.read_text(encoding="utf-8")
    if guide_text.count(START) != 1 or guide_text.count(END) != 1:
        raise ValueError("The guide must contain exactly one final-result block")
    archived = json.loads((SNAPSHOT / "results.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    final_run = extractor.extract(candidates[-1])
    if not final_run["reached_target_steps"]:
        print("最终日志尚未包含完整训练结束消息；结果区块保持不变。", file=sys.stderr)
        return 2
    if abs(final_run["checkpoint"]["best_model_score"] - final_run["best_validation"]["val/accuracy_ema"]) > 1e-7:
        raise ValueError("Checkpoint best score does not match W&B history")
    runs = [run for run in archived["runs"] if run["name"] != args.run] + [final_run]
    by_name = {run["name"]: run for run in runs}
    stage_names = [PREFIX + name for name in ["TASK1_SEED12", "TASK1-2_SEED12", "TASK1-3_SEED12", "TASK1-4_SEED12_RERUN_T1_4"]] + [args.run]
    stage_runs = [by_name[name] for name in stage_names]
    for run in stage_runs:
        if not run["reached_target_steps"] or run["checkpoint"]["global_step"] != run["max_steps_target"]:
            raise ValueError(f"Incomplete required stage: {run['name']}")
    if any(not math.isfinite(run["last_accuracy_all10_pct"]) for run in stage_runs):
        raise ValueError("Non-finite final accuracy")
    matrix = task_matrix(stage_runs, "last")
    replay = audit_final_replay(extractor, final_run)
    final_run["replay_audit"] = replay
    write_summary(runs)
    write_matrix(matrix)
    replay_rows = list(archived["replay"])
    if replay:
        replay_rows.append(replay)
    (OUT / "results.json").write_text(json.dumps({"runs": runs, "replay": replay_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    figures, note = ([], "本次选择只生成数值表。") if args.skip_plots else make_plots(final_run, matrix)
    block = render_result(stage_runs, final_run, matrix, figures, note)
    before, remainder = guide_text.split(START, 1)
    _, after = remainder.split(END, 1)
    updated = before + START + "\n" + block + "\n" + END + after
    GUIDE.write_text(updated, encoding="utf-8")
    (OUT / "final_report.md").write_text(block, encoding="utf-8")
    print(f"已核对并更新：{GUIDE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
