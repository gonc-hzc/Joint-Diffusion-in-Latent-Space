"""Read completed local W&B logs and checkpoints without using the GPU.

Run with the existing jdcl Python environment. Task1-5 consolidation is excluded.
Original logs, checkpoints, and replay tensors are opened read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def unpack_items(items):
    result = {}
    for item in items:
        key = item.key or "/".join(item.nested_key)
        try:
            result[key] = json.loads(item.value_json)
        except (ValueError, TypeError):
            result[key] = item.value_json
    return result


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def checkpoint_metadata(directory):
    target = directory / "checkpoints" / "last.ckpt"
    if not target.exists():
        return {}
    checkpoint = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
    result = {key: checkpoint.get(key) for key in ["epoch", "global_step"]}
    for name, state in checkpoint.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(name):
            for key in ["best_model_score", "current_score", "best_model_path", "last_model_path"]:
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    value = value.item()
                result[key] = value
    del checkpoint
    return result


def confusion_audit(source, classes, row):
    tables = sorted(
        (source.parent / "files" / "media" / "table" / "val").glob("*.table.json"),
        key=lambda path: int(re.search(r"table_(\d+)_", path.name).group(1)),
    )
    candidates = [path for path in tables if int(re.search(r"table_(\d+)_", path.name).group(1)) <= row["_step"]]
    if not candidates:
        return {}
    path = candidates[-1]
    entries = json.loads(path.read_text())["data"]
    count = sum(entry[2] for entry in entries)
    correct = sum(n for actual, predicted, n in entries if actual == predicted)
    unseen_correct = sum(
        n for actual, predicted, n in entries
        if actual == predicted and int(actual.split("_")[-1]) - 1 not in classes
    )
    # The existing logger pools normal and EMA predictions into one confusion
    # table. Its zero unseen diagonal proves both have zero unseen successes.
    expected_combined = (row["val/accuracy"] + row["val/accuracy_ema"]) * 50
    observed_combined = correct / count * 100
    if abs(expected_combined - observed_combined) > 0.001:
        raise ValueError(f"Confusion table does not match validation: {path}")
    return {
        "source": str(path.relative_to(ROOT)),
        "count": count,
        "unseen_correct_combined": unseen_correct,
        "accuracy_combined_pct": observed_combined,
    }


def extract(run_directory):
    source = next(run_directory.glob("run-*.wandb"))
    name = source.stem.removeprefix("run-")
    metadata_path = run_directory / "files" / "wandb-metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    store = DataStore()
    store._fname = str(source)
    store._fp = source.open("rb")
    store._index = 0
    store._size_bytes = source.stat().st_size
    store._opened_for_scan = True
    store._read_header()
    records = Counter()
    history = []
    summary = {}
    finish_messages = []
    try:
        while True:
            raw = store.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            kind = record.WhichOneof("record_type")
            records[kind] += 1
            if kind == "history":
                row = unpack_items(record.history.item)
                numeric = {key: value for key, value in row.items() if finite(value)}
                history.append(numeric)
            elif kind == "summary":
                summary.update(unpack_items(record.summary.update))
            elif kind == "output_raw":
                value = record.output_raw.line
                if any(word in value for word in ["max_steps=", "KeyboardInterrupt", "Traceback", "SAMPLING TIME"]):
                    finish_messages.append(value.strip()[-500:])
    finally:
        store.close()

    validations = [row for row in history if finite(row.get("val/accuracy_ema"))]
    best = max(validations, key=lambda row: row["val/accuracy_ema"]) if validations else {}
    last = validations[-1] if validations else {}
    epoch_rows = [row for row in history if any(key.startswith("train/") and key.endswith("_epoch") for key in row)]
    config_path = next((ROOT / "logs" / name / "configs").glob("*project.yaml"))
    project = yaml.safe_load(config_path.read_text())
    params = project["model"]["params"]
    classes = params.get("old_classes", []) + params.get("new_classes", [])
    if not classes:
        task_number = int(metadata["args"][metadata["args"].index("-t") + 1])
        classes = [2 * task_number, 2 * task_number + 1]
    args = metadata.get("args", [])
    result = {
        "name": name,
        "source": str(source.relative_to(ROOT)),
        "started_at": metadata.get("startedAt"),
        "args": args,
        "gpu": metadata.get("gpu"),
        "record_counts": dict(records),
        "checkpoint": checkpoint_metadata(ROOT / "logs" / name),
        "classes": classes,
        "class_count": len(classes),
        "history_count": len(history),
        "validation_count": len(validations),
        "max_logged_step": max((row.get("trainer/global_step", row.get("global_step", 0)) for row in history), default=0),
        "max_runtime_seconds": max((row.get("_runtime", 0) for row in history), default=0),
        "best_validation": best,
        "last_validation": last,
        "last_train_epoch": epoch_rows[-1] if epoch_rows else {},
        "finish_messages": finish_messages[-5:],
        "hyperparameters": {key: params.get(key) for key in ["classification_loss_scale", "classification_start", "kl_classification_weight", "kd_loss_weight", "l_simple_weight", "ckpt_path"]},
    }
    for label, row in [("best", best), ("last", last)]:
        if row:
            result[f"{label}_accuracy_all10_pct"] = row["val/accuracy_ema"] * 100
            audit = confusion_audit(source, classes, row)
            result.setdefault("confusion_audit", {})[f"{label}_validation"] = audit
            if not audit or audit["unseen_correct_combined"]:
                raise ValueError(f"Cannot infer exact seen-class accuracy for {name}")
            result[f"{label}_seen_accuracy_pct"] = row["val/accuracy_ema"] * 100 * 10 / len(classes)
    result["reached_target_steps"] = any("reached" in line for line in finish_messages)
    result["max_steps_target"] = 50000 if len(classes) == 2 else 30000
    result["best_completed_steps"] = int(best["trainer/global_step"]) + 1
    result["last_validation_completed_steps"] = int(last["trainer/global_step"]) + 1
    (OUT / "history").mkdir(exist_ok=True)
    columns = sorted({key for row in validations + epoch_rows for key in row})
    with (OUT / "history" / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([
            row for row in history
            if finite(row.get("val/accuracy_ema"))
            or any(key.startswith("train/") and key.endswith("_epoch") for key in row)
        ])
    print(json.dumps({key: result[key] for key in ["name", "checkpoint", "validation_count", "max_logged_step", "best_accuracy_all10_pct", "last_accuracy_all10_pct"]}, ensure_ascii=False), flush=True)
    return result


def main():
    torch.set_num_threads(2)
    results = []
    for directory in sorted((ROOT / "wandb").glob("offline-run-*")):
        if "CL_CIFAR10_SUP_TASK" not in directory.name:
            continue
        if "TASK1-5" in directory.name:
            continue
        results.append(extract(directory))
    replay = []
    replay_details = []
    for target in sorted((ROOT / "data" / "cl").glob("*labels.pt")):
        if "TASK1-5" in target.name:
            continue
        labels = torch.load(target, map_location="cpu", weights_only=True)
        values, counts = labels.unique(return_counts=True)
        image_path = target.with_name(target.name.replace("_labels.pt", "_imgs.pt"))
        images = torch.load(image_path, map_location="cpu", weights_only=True, mmap=True)
        replay.append({
            "name": target.stem.removesuffix("_labels"),
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "counts": dict(zip(values.tolist(), counts.tolist())),
            "image_file_bytes": image_path.stat().st_size,
        })
        array = images.numpy()
        digests = [hashlib.sha256(image.tobytes()).hexdigest() for image in array]
        replay_details.append({
            "name": target.stem.removesuffix("_labels"),
            "sha256_tensor": hashlib.sha256(array.tobytes()).hexdigest(),
            "sha256_labels": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
            "unique_images": len(set(digests)),
            "total_images": len(images),
            "unique_per_class": {
                str(label): len({digest for digest, value in zip(digests, labels.tolist()) if value == label})
                for label in values.tolist()
            },
            "range": [float(images.min()), float(images.max())],
        })
        del labels, images
    (OUT / "results.json").write_text(json.dumps({"runs": results, "replay": replay}, ensure_ascii=False, indent=2, allow_nan=True))
    (OUT / "replay_audit.json").write_text(json.dumps(replay_details, ensure_ascii=False, indent=2))
    fields = ["name", "reached_target_steps", "class_count", "max_steps_target", "best_completed_steps", "last_validation_completed_steps", "best_accuracy_all10_pct", "last_accuracy_all10_pct", "best_seen_accuracy_pct", "last_seen_accuracy_pct", "max_runtime_seconds"]
    with (OUT / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fields})
    fields = ["name", "selection", "completed_steps", "task", "ema_accuracy_pct"]
    with (OUT / "per_task_accuracy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if result["class_count"] == 2:
                continue
            for selection in ["best", "last"]:
                row = result[f"{selection}_validation"]
                for key, value in sorted(row.items()):
                    if key.startswith("cl_tasks_val/") and key.endswith("_ema"):
                        task = int(re.search(r"task(\d+)", key).group(1)) + 1
                        writer.writerow({"name": result["name"], "selection": selection, "completed_steps": int(row["trainer/global_step"]) + 1, "task": task, "ema_accuracy_pct": value * 100})
    print("REPLAY", json.dumps(replay, ensure_ascii=False))


if __name__ == "__main__":
    main()
