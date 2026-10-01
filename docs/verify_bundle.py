"""Verify the study bundle's payload hashes, with or without extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="ZIP to verify; omit to verify the extracted bundle")
    args = parser.parse_args()
    failures = []
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            manifests = [name for name in archive.namelist() if name.endswith("/MANIFEST.json")]
            if len(manifests) != 1:
                raise SystemExit("ZIP 中应有且仅有一份根目录 MANIFEST.json")
            manifest_name = manifests[0]
            prefix = manifest_name.removesuffix("MANIFEST.json")
            manifest = json.loads(archive.read(manifest_name))
            expected = {prefix + item["path"] for item in manifest["files"]} | {
                prefix + "MANIFEST.json", prefix + "SHA256SUMS.txt"
            }
            actual = {info.filename for info in archive.infolist() if not info.is_dir()}
            if expected != actual:
                raise SystemExit("ZIP 内容与清单范围不一致")
            for item in manifest["files"]:
                with archive.open(prefix + item["path"]) as handle:
                    checksum = hashlib.file_digest(handle, "sha256").hexdigest()
                if checksum != item["sha256"]:
                    failures.append(item["path"])
    else:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = root / item["path"]
            if not path.is_file():
                failures.append(item["path"])
                continue
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            if checksum != item["sha256"]:
                failures.append(item["path"])
    if failures:
        raise SystemExit("校验失败：\n" + "\n".join(failures))
    print(f"校验通过：{len(manifest['files'])} 个载荷文件，SHA-256 全部一致。")


if __name__ == "__main__":
    main()
