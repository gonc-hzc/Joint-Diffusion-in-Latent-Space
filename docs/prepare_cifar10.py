"""Prepare both CIFAR-10 data roots from the archive in the study bundle."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"batches.meta", "test_batch", "readme.html"} | {
    f"data_batch_{index}" for index in range(1, 6)
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    source = ROOT / "data" / "cifar-10-python.tar.gz"
    if not source.is_file():
        raise SystemExit(f"找不到包内数据：{source}")
    destination = ROOT / "data" / "cifar-10-batches-py"
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(source, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 2 or parts[0] != "cifar-10-batches-py":
                continue
            if parts[1] not in NAMES:
                continue
            target = destination / parts[1]
            temporary = target.with_name(target.name + ".preparing")
            with archive.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if target.exists():
                same = target.stat().st_size == temporary.stat().st_size and digest(target) == digest(temporary)
                temporary.unlink()
                if not same:
                    raise SystemExit(f"已有数据内容不同，请核对：{target}")
            else:
                temporary.replace(target)
            count += 1
    if count != len(NAMES):
        raise SystemExit(f"数据压缩包不完整：找到 {count}/{len(NAMES)} 个预期文件")
    replay_root = ROOT / "data" / "cl" / "cifar-10-batches-py"
    replay_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(NAMES):
        source_file = destination / name
        target = replay_root / name
        if target.exists():
            if target.stat().st_size != source_file.stat().st_size or digest(target) != digest(source_file):
                raise SystemExit(f"已有回放数据目录内容不同，请核对：{target}")
            continue
        try:
            os.link(source_file, target)
        except OSError:
            shutil.copy2(source_file, target)
    print(f"数据已准备：{destination}")
    print(f"回放数据根目录已准备：{replay_root}")


if __name__ == "__main__":
    main()
