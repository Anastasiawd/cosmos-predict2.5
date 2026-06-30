# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recursively extract all .tar.gz archives under a directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm


DEFAULT_STATE_FILE_NAME = ".extract_tar_gz_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively extract .tar.gz archives with resume support")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to scan for .tar.gz files")
    parser.add_argument(
        "--state_file",
        type=str,
        default=None,
        help=f"Optional resume state file path (default: <input_dir>/{DEFAULT_STATE_FILE_NAME})",
    )
    return parser.parse_args()


def _load_state(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        with state_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        completed = data.get("completed", [])
        if not isinstance(completed, list):
            return set()
        return {str(item) for item in completed}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return set()


def _save_state(state_file: Path, completed: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed": sorted(completed)}
    tmp_file = state_file.with_suffix(state_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write(chr(10))
    os.replace(tmp_file, state_file)


def _find_archives(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.name.endswith(".tar.gz"))


def _extract_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=archive_path.parent)



def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    state_file = Path(args.state_file).expanduser().resolve() if args.state_file else input_dir / DEFAULT_STATE_FILE_NAME
    completed = _load_state(state_file)

    archives = _find_archives(input_dir)
    pending = [archive for archive in archives if str(archive.relative_to(input_dir)) not in completed]

    if not pending:
        print("No pending .tar.gz archives found.")
        return

    completed_before = len(completed)
    failed = 0

    try:
        for archive_path in tqdm(pending, desc="Extracting", unit="archive"):
            rel_path = str(archive_path.relative_to(input_dir))
            try:
                _extract_archive(archive_path)
            except (EOFError, OSError, tarfile.ReadError) as exc:
                failed += 1
                print(f"Warning: failed to extract {archive_path}: {exc}", file=sys.stderr)
                continue

            archive_path.unlink()
            completed.add(rel_path)
            _save_state(state_file, completed)
    except KeyboardInterrupt:
        print()
        print("Interrupted. Progress has been saved; rerun the same command to continue.")
        _save_state(state_file, completed)
        raise

    extracted = len(completed) - completed_before
    if failed:
        print(f"Finished extracting {extracted} archive(s); {failed} failed.")
    else:
        print(f"Finished extracting {extracted} archive(s).")
    _save_state(state_file, completed)


if __name__ == "__main__":
    main(parse_args())
