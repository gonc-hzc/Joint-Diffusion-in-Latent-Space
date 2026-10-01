"""Archive verified JDCL teaching materials and experimental artifacts.

Does not launch training, read checkpoint tensors, or copy an active run.
Checkpoint binaries remain in the workspace and are indexed in the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "analysis" / "completed_experiments_20260929"
FINAL = ROOT / "analysis" / "full_experiment_results"
GUIDE = ROOT / "docs" / "JDCL自学精讲与复现指南.md"


def git_commit(directory):
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def eligible(path):
    return path.is_file() and not path.is_symlink() and not any(
        part in {".git", "__pycache__", ".ipynb_checkpoints"} or part.endswith(".egg-info")
        for part in path.relative_to(ROOT).parts
    ) and path.suffix not in {".pyc", ".pyo"}


def collect_files(results, include_final):
    selected = {}

    def add(path):
        if not eligible(path):
            raise ValueError(f"缺少预期文件或不是普通文件：{path}")
        selected[path.relative_to(ROOT).as_posix()] = path

    def add_tree(directory):
        for path in directory.rglob("*"):
            if eligible(path):
                add(path)

    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in {".py", ".sh"} or path.name in {
            "README.md", "requirements.txt", "environment.yml", ".gitmodules", ".gitignore"
        }):
            add(path)
    for name in ["models", "cl_methods", "dataloading", "configs", "docs"]:
        add_tree(ROOT / name)
    add_tree(SNAPSHOT)
    if include_final:
        add_tree(FINAL)
    # Preserve executable dependency sources and licenses, excluding unrelated
    # datasets, showcase assets, notebooks, and Git internals.
    for dependency, package in [
        ("latent-diffusion", "ldm"),
        ("src/latent-diffusion", "ldm"),
        ("src/taming-transformers", "taming"),
        ("src/clip", "clip"),
    ]:
        base = ROOT / dependency
        for name in [package, "configs"]:
            if (base / name).is_dir():
                add_tree(base / name)
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".txt", ".yml", ".yaml"}:
                add(path)
            elif path.is_file() and path.name.lower().startswith(("license", "copying")):
                add(path)
        for path in (base / "scripts").rglob("*.py"):
            if eligible(path):
                add(path)
    for run in results["runs"]:
        source = ROOT / run["source"]
        add(source)
        files_root = source.parent / "files"
        for name in ["wandb-metadata.json", "wandb-summary.json", "config.yaml", "requirements.txt", "output.log"]:
            if (files_root / name).is_file():
                add(files_root / name)
        if (files_root / "media").is_dir():
            add_tree(files_root / "media")
        configs = ROOT / "logs" / run["name"] / "configs"
        if not configs.is_dir():
            raise ValueError(f"缺少实际运行配置：{configs}")
        add_tree(configs)
    for replay in results["replay"]:
        for suffix in ["_imgs.pt", "_labels.pt"]:
            add(ROOT / "data" / "cl" / (replay["name"] + suffix))
    add(ROOT / "data" / "cifar-10-python.tar.gz")
    return selected


def checkpoint_index(results):
    entries = []
    for run in results["runs"]:
        paths = sorted((ROOT / "logs" / run["name"] / "checkpoints").glob("*.ckpt"))
        entries.append({
            "run": run["name"],
            "reached_target_steps": run["reached_target_steps"],
            "last_checkpoint_metadata": run["checkpoint"],
            "files": [{
                "relative_path": path.relative_to(ROOT).as_posix(),
                "original_absolute_path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "included_in_zip": False,
            } for path in paths],
        })
    return {
        "description": "模型权重不在本 ZIP 中；以下文件保留在原工作区。元数据来自已提取的阶段结果。",
        "original_workspace": str(ROOT),
        "total_checkpoint_bytes": sum(item["bytes"] for run in entries for item in run["files"]),
        "runs": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output ZIP; defaults to dist/ with a date suffix")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    results = json.loads((SNAPSHOT / "results.json").read_text(encoding="utf-8"))
    include_final = False
    final_status = "Task1–5 全局合并尚未补录；当前包不含它的最终结果。"
    if (FINAL / "results.json").is_file() and (FINAL / "final_report.md").is_file():
        final_results = json.loads((FINAL / "results.json").read_text(encoding="utf-8"))
        final_runs = [run for run in final_results["runs"] if "TASK1-5" in run["name"]]
        if len(final_runs) == 1 and final_runs[0]["reached_target_steps"] and final_runs[0]["checkpoint"]["global_step"] == 30000:
            guide_text = GUIDE.read_text(encoding="utf-8")
            final_block = guide_text.split("<!-- FINAL_RESULTS_START -->", 1)[-1].split("<!-- FINAL_RESULTS_END -->", 1)[0].strip()
            if final_block == (FINAL / "final_report.md").read_text(encoding="utf-8").strip():
                include_final = True
                results = final_results
                final_status = "Task1–5 最终结果已核实并补录；早期阶段快照同时保留。"
    selected = collect_files(results, include_final)
    checkpoints = checkpoint_index(results)
    completed = sum(run["reached_target_steps"] for run in results["runs"])
    archive_name = f"JDCL_精讲与实验资料_{now:%Y%m%d}"
    output = args.output.resolve() if args.output else ROOT / "dist" / (archive_name + ".zip")
    if output.suffix.lower() != ".zip":
        raise ValueError("输出文件名必须以 .zip 结尾")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    total_source_bytes = sum(path.stat().st_size for path in selected.values())
    print(f"准备打包：{len(selected)} 个原始文件，{total_source_bytes / 2**20:.1f} MiB", flush=True)
    readme = f"""# JDCL 精讲与实验资料

打包时间：{now.isoformat(timespec='seconds')}。

先阅读 [十课自学精讲](docs/JDCL自学精讲与复现指南.md)，再阅读 [实验分析](analysis/completed_experiments_20260929/实验讲解与分析.md)。

[评价指标精讲](docs/评价指标精讲.md) 解释指标为何能反映性能，包含当前实验的计算实例、评价规则与局限。

[资料包使用说明](docs/资料包使用说明.md) 解释每类文件、准备数据的方法、绘图和校验命令。

当前纳入 {len(results['runs'])} 次运行：{completed} 个阶段完成，{len(results['runs']) - completed} 次历史运行未完成。{final_status}

包内包含精讲、CSV/JSON、PNG/SVG、原始 W&B 日志和混淆表、运行配置、回放张量、CIFAR-10 数据压缩包，以及训练和分析源码。

模型检查点原文件共 {checkpoints['total_checkpoint_bytes'] / 2**30:.2f} GiB，保留在原工作区。位置、大小及训练状态见 [CHECKPOINT_INDEX.json](CHECKPOINT_INDEX.json)；本包不含这些权重二进制。权重推理或从已有模型恢复需要原始检查点。

解压后先运行 `python3 docs/verify_bundle.py` 校验，准备数据用 `python3 docs/prepare_cifar10.py`。浏览教学内容和已有图表不需要启动训练。

文件级 SHA-256 见 [SHA256SUMS.txt](SHA256SUMS.txt)，载荷范围及版本信息见 [MANIFEST.json](MANIFEST.json)。清单和校验表本身不列入载荷哈希；外置 ZIP SHA-256 校验文件覆盖整个压缩包。
"""
    generated = {
        "先读我.md": readme.encode("utf-8"),
        "CHECKPOINT_INDEX.json": json.dumps(checkpoints, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    manifest = {
        "created_at": now.isoformat(timespec="seconds"),
        "original_workspace": str(ROOT),
        "code_git_commit": git_commit(ROOT),
        "dependency_git_commits": {name: git_commit(ROOT / name) for name in ["latent-diffusion", "src/latent-diffusion", "src/taming-transformers", "src/clip"]},
        "scope": final_status,
        "run_count": len(results["runs"]),
        "completed_stage_count": completed,
        "incomplete_historical_run_count": len(results["runs"]) - completed,
        "contains_final_task1_5_results": include_final,
        "checkpoint_binaries_included": False,
        "excluded": [".ckpt 模型权重二进制", "当前未核实的 Task1–5 运行产物", "第三方展示素材和无关数据集", "历史论文展示 notebook", "Git 内部目录和 Python 环境二进制", "重复的 CIFAR-10 压缩包和可重新解压的数据副本"],
        "files": [],
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for index, (relative, path) in enumerate(sorted(selected.items()), 1):
            before = path.stat()
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            compression = zipfile.ZIP_STORED if path.suffix.lower() in {".gz", ".png", ".jpg", ".jpeg"} else zipfile.ZIP_DEFLATED
            archive.write(path, f"{archive_name}/{relative}", compress_type=compression)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"文件在打包期间改变，已中止：{relative}")
            manifest["files"].append({"path": relative, "bytes": before.st_size, "sha256": checksum})
            if before.st_size > 10 * 2**20:
                print(f"已归档 {index}/{len(selected)}：{relative}", flush=True)
        for relative, content in generated.items():
            archive.writestr(f"{archive_name}/{relative}", content)
            manifest["files"].append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        manifest["files"].sort(key=lambda item: item["path"])
        archive.writestr(f"{archive_name}/MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr(f"{archive_name}/SHA256SUMS.txt", "".join(f"{item['sha256']}  {item['path']}\n" for item in manifest["files"]))
    print("压缩完成，正在读取 ZIP 并校验全部载荷哈希。", flush=True)
    validation = subprocess.run([sys.executable, str(ROOT / "docs" / "verify_bundle.py"), "--archive", str(temporary)], check=False)
    if validation.returncode:
        raise RuntimeError("压缩包校验未通过；未替换正式输出")
    temporary.replace(output)
    with output.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    output.with_name(output.stem + "_文件清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{output}", flush=True)
    print(f"大小：{output.stat().st_size / 2**20:.1f} MiB；{len(manifest['files'])} 个载荷文件", flush=True)
    print(f"SHA-256：{checksum}", flush=True)


if __name__ == "__main__":
    main()
