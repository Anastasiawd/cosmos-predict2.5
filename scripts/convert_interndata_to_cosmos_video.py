# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Convert InternData trajectory videos into Cosmos Predict2 VideoDataset format.

The output directory is compatible with
``cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video.VideoDataset``:

    output_root/
    ├── videos/
    │   └── <dataset>__<scene>__episode_000000.mp4
    └── metas/
        └── <dataset>__<scene>__episode_000000.txt

Only RGB episode videos are exported. Missing, incomplete, or still-compressed
scene data is skipped and summarized in a JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_INPUT_ROOT = Path("/home/csevolunt/dongzhih/dataset/InternData")
DEFAULT_OUTPUT_ROOT = Path("datasets/vln_cosmos")
DEFAULT_REPORT_PATH = Path("outputs/vln_cosmos_conversion_report.json")
DEFAULT_CHUNKS_SIZE = 1000
RGB_VIDEO_KEY = "observation.video.rgb"


@dataclass
class ConvertedItem:
    dataset: str
    scene: str
    episode_index: int
    video: str
    meta: str
    source_video: str
    prompt_source: str


@dataclass
class SkippedItem:
    dataset: str
    scene: str
    episode_index: int | None
    reason: str
    source_path: str
    detail: str = ""


@dataclass
class Summary:
    datasets_scanned: int = 0
    scenes_scanned: int = 0
    episodes_scanned: int = 0
    converted: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert InternData trajectory episodes to Cosmos VideoDataset mp4 + txt format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root directory of InternData.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root containing videos/ and metas/.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to place source mp4 files in the output videos directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output video/meta files.")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not create or modify files.")
    parser.add_argument("--limit-scenes", type=int, default=None, help="Only process the first N scene directories.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="JSON report output path.")
    return parser.parse_args()


def safe_scene_name(scene_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", scene_name).strip("_")
    return safe or "scene"


def scene_label(input_root: Path, scene_dir: Path) -> tuple[str, str]:
    try:
        relative_parts = scene_dir.relative_to(input_root).parts
    except ValueError:
        relative_parts = scene_dir.parts[-2:]

    if len(relative_parts) >= 2:
        return relative_parts[0], "__".join(relative_parts)
    if relative_parts:
        return "unknown_dataset", relative_parts[0]
    return "unknown_dataset", scene_dir.name


def discover_scene_dirs(input_root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in input_root.rglob("meta/episodes.jsonl") if path.is_file())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def load_info(scene_dir: Path) -> dict[str, Any]:
    info_path = scene_dir / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def instruction_span(task: dict[str, Any], key: str) -> tuple[int, int]:
    index_key = key.replace("instruction", "indexes")
    indexes = task.get(index_key)
    if isinstance(indexes, list) and len(indexes) >= 2:
        try:
            return int(indexes[0]), int(indexes[1])
        except (TypeError, ValueError):
            pass
    return 0, 0


def choose_prompt(tasks: Any) -> tuple[str | None, str | None]:
    if not isinstance(tasks, list):
        return None, None

    for task in tasks:
        if isinstance(task, dict) and task.get("sum_instruction"):
            return normalize_text(task["sum_instruction"]), "sum_instruction"

    revised: list[tuple[tuple[int, int], str]] = []
    for task in tasks:
        if isinstance(task, dict) and task.get("revised_sub_instruction"):
            revised.append((instruction_span(task, "sub_instruction"), normalize_text(task["revised_sub_instruction"])))
    if revised:
        revised.sort(key=lambda item: item[0])
        return " ".join(text for _, text in revised), "revised_sub_instruction"

    original: list[tuple[tuple[int, int], str]] = []
    for task in tasks:
        if isinstance(task, dict) and task.get("sub_instruction"):
            original.append((instruction_span(task, "sub_instruction"), normalize_text(task["sub_instruction"])))
    if original:
        original.sort(key=lambda item: item[0])
        return " ".join(text for _, text in original), "sub_instruction"

    return None, None


def get_chunks_size(info: dict[str, Any]) -> int:
    chunks_size = info.get("chunks_size", DEFAULT_CHUNKS_SIZE)
    try:
        chunks_size = int(chunks_size)
    except (TypeError, ValueError):
        chunks_size = DEFAULT_CHUNKS_SIZE
    return chunks_size if chunks_size > 0 else DEFAULT_CHUNKS_SIZE


def resolve_source_video(scene_dir: Path, episode_index: int, info: dict[str, Any]) -> Path:
    chunks_size = get_chunks_size(info)
    episode_chunk = episode_index // chunks_size

    video_template = info.get("video_path")
    if isinstance(video_template, str) and video_template:
        try:
            rel_path = video_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=RGB_VIDEO_KEY,
            )
            return scene_dir / rel_path
        except Exception:
            pass

    return (
        scene_dir
        / "videos"
        / f"chunk-{episode_chunk:03d}"
        / RGB_VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )


def link_or_copy(source: Path, destination: Path, mode: Literal["symlink", "copy", "hardlink"]) -> None:
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def write_manifest(manifest_path: Path, converted_items: list[ConvertedItem]) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in converted_items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def record_skip(
    skipped_items: list[SkippedItem],
    summary: Summary,
    dataset: str,
    scene: str,
    episode_index: int | None,
    reason: str,
    source_path: Path,
    detail: str = "",
) -> None:
    skipped_items.append(
        SkippedItem(
            dataset=dataset,
            scene=scene,
            episode_index=episode_index,
            reason=reason,
            source_path=str(source_path),
            detail=detail,
        )
    )
    summary.skipped += 1


def convert_scene(
    scene_dir: Path,
    dataset_name: str,
    scene_name: str,
    safe_scene: str,
    output_root: Path,
    link_mode: Literal["symlink", "copy", "hardlink"],
    overwrite: bool,
    dry_run: bool,
    summary: Summary,
    converted_items: list[ConvertedItem],
    skipped_items: list[SkippedItem],
) -> None:
    episodes_path = scene_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        record_skip(
            skipped_items,
            summary,
            dataset_name,
            scene_name,
            None,
            "missing_episodes_jsonl",
            episodes_path,
        )
        return

    info = load_info(scene_dir)
    try:
        episodes = list(read_jsonl(episodes_path))
    except ValueError as exc:
        record_skip(
            skipped_items,
            summary,
            dataset_name,
            scene_name,
            None,
            "invalid_episodes_jsonl",
            episodes_path,
            str(exc),
        )
        return

    videos_dir = output_root / "videos"
    metas_dir = output_root / "metas"

    for episode in episodes:
        episode_index_raw = episode.get("episode_index")
        try:
            episode_index = int(episode_index_raw)
        except (TypeError, ValueError):
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                None,
                "invalid_episode_index",
                episodes_path,
                f"episode_index={episode_index_raw!r}",
            )
            continue

        summary.episodes_scanned += 1
        basename = f"{safe_scene}__episode_{episode_index:06d}"
        output_video = videos_dir / f"{basename}.mp4"
        output_meta = metas_dir / f"{basename}.txt"
        source_video = resolve_source_video(scene_dir, episode_index, info)

        prompt, prompt_source = choose_prompt(episode.get("tasks"))
        if not prompt:
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                "missing_prompt",
                episodes_path,
            )
            continue

        if not source_video.exists():
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                "missing_video",
                source_video,
            )
            continue

        if source_video.stat().st_size == 0:
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                "empty_video",
                source_video,
            )
            continue

        if not overwrite and (output_video.exists() or output_video.is_symlink() or output_meta.exists()):
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                "already_exists",
                output_video if output_video.exists() or output_video.is_symlink() else output_meta,
            )
            continue

        if not dry_run:
            try:
                remove_if_exists(output_video)
                remove_if_exists(output_meta)
                link_or_copy(source_video, output_video, link_mode)
                output_meta.write_text(prompt + "\n", encoding="utf-8")
            except Exception as exc:
                remove_if_exists(output_video)
                remove_if_exists(output_meta)
                record_skip(
                    skipped_items,
                    summary,
                    dataset_name,
                    scene_name,
                    episode_index,
                    "write_error",
                    source_video,
                    str(exc),
                )
                continue

        converted_items.append(
            ConvertedItem(
                dataset=dataset_name,
                scene=scene_name,
                episode_index=episode_index,
                video=str(output_video),
                meta=str(output_meta),
                source_video=str(source_video),
                prompt_source=prompt_source or "unknown",
            )
        )
        summary.converted += 1


def print_summary(summary: Summary, dry_run: bool, output_root: Path, report_path: Path) -> None:
    mode = "DRY RUN" if dry_run else "CONVERT"
    print(f"\n=== InternData Cosmos Conversion Summary ({mode}) ===")
    print(f"Datasets scanned: {summary.datasets_scanned}")
    print(f"Scenes scanned:   {summary.scenes_scanned}")
    print(f"Episodes scanned: {summary.episodes_scanned}")
    converted_label = "Would convert" if dry_run else "Converted"
    print(f"{converted_label}:        {summary.converted}")
    print(f"Skipped:          {summary.skipped}")
    if dry_run:
        print("Dry run: no files were written.")
    else:
        print(f"Output root:      {output_root}")
        print(f"Report:           {report_path}")
        print(f"Manifest:         {output_root / 'manifest.jsonl'}")
    print("===================================================\n")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser()
    output_root = args.output_root.expanduser()
    report_path = args.report_path.expanduser()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    scene_dirs = discover_scene_dirs(input_root)
    if args.limit_scenes is not None:
        scene_dirs = scene_dirs[: args.limit_scenes]

    if not args.dry_run:
        (output_root / "videos").mkdir(parents=True, exist_ok=True)
        (output_root / "metas").mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = Summary()
    converted_items: list[ConvertedItem] = []
    skipped_items: list[SkippedItem] = []
    safe_scene_by_original: dict[str, str] = {}
    original_by_safe_scene: dict[str, str] = {}
    dataset_names: set[str] = set()

    for scene_dir in scene_dirs:
        summary.scenes_scanned += 1
        dataset_name, original_scene = scene_label(input_root, scene_dir)
        dataset_names.add(dataset_name)
        safe_scene = safe_scene_name(original_scene)
        if safe_scene in original_by_safe_scene and original_by_safe_scene[safe_scene] != original_scene:
            suffix = hashlib.sha1(original_scene.encode("utf-8")).hexdigest()[:8]
            safe_scene = f"{safe_scene}_{suffix}"
        safe_scene_by_original[original_scene] = safe_scene
        original_by_safe_scene[safe_scene] = original_scene

        convert_scene(
            scene_dir=scene_dir,
            dataset_name=dataset_name,
            scene_name=original_scene,
            safe_scene=safe_scene,
            output_root=output_root,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            summary=summary,
            converted_items=converted_items,
            skipped_items=skipped_items,
        )

    summary.datasets_scanned = len(dataset_names)

    if not args.dry_run:
        write_manifest(output_root / "manifest.jsonl", converted_items)
        report = {
            "summary": asdict(summary),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "link_mode": args.link_mode,
            "overwrite": args.overwrite,
            "safe_scene_by_original": safe_scene_by_original,
            "skipped": [asdict(item) for item in skipped_items],
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(summary, args.dry_run, output_root, report_path)


if __name__ == "__main__":
    main()
