# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build index-only Cosmos VLN subclip manifests from InternData.

The script does not copy, symlink, or write video files. Each manifest row points
at an existing source mp4 and stores the exact frame_indices used for training.
Only lightweight text/index artifacts are written:

    output_root/
    ├── manifest.jsonl
    └── metas/
        └── <scene>__episode_000000__sub_000.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from decord import VideoReader, cpu


DEFAULT_INPUT_ROOT = Path("/home/csevolunt/dongzhih/dataset/InternData")
DEFAULT_OUTPUT_ROOT = Path("datasets/vln_cosmos_subclips_index")
DEFAULT_REPORT_PATH = Path("outputs/vln_cosmos_subclip_index_report.json")
DEFAULT_CHUNKS_SIZE = 1000
RGB_VIDEO_KEY = "observation.video.rgb"


@dataclass
class SkippedItem:
    dataset: str
    scene: str
    episode_index: int | None
    sub_instruction_index: int | None
    reason: str
    source_path: str
    detail: str = ""


@dataclass
class Summary:
    datasets_scanned: int = 0
    scenes_scanned: int = 0
    episodes_scanned: int = 0
    subinstructions_scanned: int = 0
    converted: int = 0
    skipped: int = 0
    static_dropped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an index-only Cosmos VideoDataset manifest aligned to InternData sub_instructions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root directory of InternData.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for manifest/metas.")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Training FPS represented by frame_indices.")
    parser.add_argument("--num-frames", type=int, default=29, help="Number of frames per indexed clip.")
    parser.add_argument(
        "--index-units",
        choices=("frame", "second"),
        default="frame",
        help="Units used by sub_instruction_indexes in InternData tasks.",
    )
    parser.add_argument(
        "--static-mse-threshold",
        type=float,
        default=1.0,
        help="Drop clips whose adjacent-frame grayscale MSE is below this value. Use 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing meta files.")
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


def normalize_text(text: Any) -> str:
    return " ".join(str(text).split())


def instruction_span(task: dict[str, Any], key: str = "sub_instruction") -> tuple[int, int] | None:
    index_key = key.replace("instruction", "indexes")
    indexes = task.get(index_key)
    if isinstance(indexes, list) and len(indexes) >= 2:
        try:
            start = int(indexes[0])
            end = int(indexes[1])
        except (TypeError, ValueError):
            return None
        if end > start:
            return start, end
    return None


def global_instruction(tasks: Any) -> str:
    if not isinstance(tasks, list):
        return ""
    for task in tasks:
        if isinstance(task, dict) and task.get("sum_instruction"):
            return normalize_text(task["sum_instruction"])
    return ""


def iter_subinstructions(tasks: Any) -> Iterable[tuple[int, int, int, str, str]]:
    if not isinstance(tasks, list):
        return
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        local = task.get("revised_sub_instruction") or task.get("sub_instruction")
        if not local:
            continue
        span = instruction_span(task, "sub_instruction")
        if span is None:
            continue
        prompt_source = "revised_sub_instruction" if task.get("revised_sub_instruction") else "sub_instruction"
        yield idx, span[0], span[1], normalize_text(local), prompt_source


def build_prompt(local: str, global_text: str) -> str:
    if global_text and global_text != local:
        return f"Local instruction: {local}\nGlobal route: {global_text}"
    return f"Local instruction: {local}"


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


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_label(delta_yaw: float) -> str:
    abs_yaw = abs(delta_yaw)
    if abs_yaw < math.radians(10):
        return "straight"
    prefix = "left" if delta_yaw > 0 else "right"
    if abs_yaw < math.radians(35):
        return f"slight_{prefix}"
    return prefix


def pose_to_xy_yaw(pose: Any) -> tuple[float, float, float] | None:
    if isinstance(pose, dict):
        if all(key in pose for key in ("x", "y", "yaw")):
            return float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        if all(key in pose for key in ("x", "y", "theta")):
            return float(pose["x"]), float(pose["y"]), float(pose["theta"])
        position = pose.get("position") or pose.get("translation")
        yaw = pose.get("yaw", pose.get("heading", pose.get("theta")))
        if isinstance(position, (list, tuple)) and len(position) >= 2 and yaw is not None:
            return float(position[0]), float(position[1]), float(yaw)
    if isinstance(pose, (list, tuple)) and len(pose) >= 3:
        return float(pose[0]), float(pose[1]), float(pose[2])
    return None


def find_pose_sequence(episode: dict[str, Any]) -> list[Any] | None:
    for key in ("poses", "pose", "trajectory", "traj", "base_poses", "agent_poses", "robot_poses"):
        value = episode.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def ego_delta_from_episode(episode: dict[str, Any], start_frame: int, end_frame: int) -> dict[str, Any]:
    poses = find_pose_sequence(episode)
    if not poses:
        return {}
    start_pose = pose_to_xy_yaw(poses[min(max(start_frame, 0), len(poses) - 1)])
    end_pose = pose_to_xy_yaw(poses[min(max(end_frame, 0), len(poses) - 1)])
    if start_pose is None or end_pose is None:
        return {}
    delta_x = end_pose[0] - start_pose[0]
    delta_y = end_pose[1] - start_pose[1]
    delta_yaw = normalize_angle(end_pose[2] - start_pose[2])
    return {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_yaw": delta_yaw,
        "yaw_label": yaw_label(delta_yaw),
        "forward_distance": math.hypot(delta_x, delta_y),
    }


def span_to_frames(start: int, end: int, fps: float, index_units: str) -> tuple[int, int]:
    if index_units == "second":
        return int(round(start * fps)), int(round(end * fps))
    return start, end


def frame_indices_for_span(
    sub_start: int,
    sub_end: int,
    total_frames: int,
    source_fps: float,
    target_fps: float,
    num_frames: int,
) -> list[int]:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    step = source_fps / target_fps
    window_span = step * (num_frames - 1)
    center = (sub_start + sub_end - 1) / 2.0
    window_start = center - window_span / 2.0
    max_start = max(0.0, (total_frames - 1) - window_span)
    window_start = min(max(window_start, 0.0), max_start)
    return [min(max(int(round(window_start + idx * step)), 0), total_frames - 1) for idx in range(num_frames)]


def motion_score(vr: VideoReader, frame_indices: list[int]) -> float:
    frames = vr.get_batch(frame_indices).asnumpy().astype(np.float32)
    gray = frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
    diffs = np.diff(gray, axis=0)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs * diffs))


def record_skip(
    skipped_items: list[SkippedItem],
    summary: Summary,
    dataset: str,
    scene: str,
    episode_index: int | None,
    sub_instruction_index: int | None,
    reason: str,
    source_path: Path,
    detail: str = "",
) -> None:
    skipped_items.append(
        SkippedItem(
            dataset=dataset,
            scene=scene,
            episode_index=episode_index,
            sub_instruction_index=sub_instruction_index,
            reason=reason,
            source_path=str(source_path),
            detail=detail,
        )
    )
    summary.skipped += 1
    if reason == "static_clip":
        summary.static_dropped += 1


def convert_scene(
    scene_dir: Path,
    dataset_name: str,
    scene_name: str,
    safe_scene: str,
    output_root: Path,
    target_fps: float,
    num_frames: int,
    index_units: str,
    static_mse_threshold: float,
    overwrite: bool,
    dry_run: bool,
    summary: Summary,
    manifest_items: list[dict[str, Any]],
    skipped_items: list[SkippedItem],
) -> None:
    episodes_path = scene_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        record_skip(skipped_items, summary, dataset_name, scene_name, None, None, "missing_episodes_jsonl", episodes_path)
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
            None,
            "invalid_episodes_jsonl",
            episodes_path,
            str(exc),
        )
        return

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
                None,
                "invalid_episode_index",
                episodes_path,
                f"episode_index={episode_index_raw!r}",
            )
            continue

        summary.episodes_scanned += 1
        source_video = resolve_source_video(scene_dir, episode_index, info)
        if not source_video.exists():
            record_skip(skipped_items, summary, dataset_name, scene_name, episode_index, None, "missing_video", source_video)
            continue
        if source_video.stat().st_size == 0:
            record_skip(skipped_items, summary, dataset_name, scene_name, episode_index, None, "empty_video", source_video)
            continue

        try:
            vr = VideoReader(str(source_video), ctx=cpu(0), num_threads=2)
            total_frames = len(vr)
            source_fps = float(vr.get_avg_fps())
        except Exception as exc:
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                None,
                "video_read_error",
                source_video,
                str(exc),
            )
            continue

        global_text = global_instruction(episode.get("tasks"))
        found_subinstruction = False
        for sub_idx, sub_start_raw, sub_end_raw, local_text, prompt_source in iter_subinstructions(episode.get("tasks")):
            found_subinstruction = True
            summary.subinstructions_scanned += 1
            sub_start, sub_end = span_to_frames(sub_start_raw, sub_end_raw, source_fps, index_units)
            sub_start = min(max(sub_start, 0), max(total_frames - 1, 0))
            sub_end = min(max(sub_end, sub_start + 1), total_frames)
            if total_frames < num_frames:
                record_skip(
                    skipped_items,
                    summary,
                    dataset_name,
                    scene_name,
                    episode_index,
                    sub_idx,
                    "short_video",
                    source_video,
                    f"total_frames={total_frames}, num_frames={num_frames}",
                )
                continue

            frame_indices = frame_indices_for_span(
                sub_start=sub_start,
                sub_end=sub_end,
                total_frames=total_frames,
                source_fps=source_fps,
                target_fps=target_fps,
                num_frames=num_frames,
            )
            try:
                score = motion_score(vr, frame_indices)
            except Exception as exc:
                record_skip(
                    skipped_items,
                    summary,
                    dataset_name,
                    scene_name,
                    episode_index,
                    sub_idx,
                    "motion_score_error",
                    source_video,
                    str(exc),
                )
                continue

            if static_mse_threshold > 0 and score < static_mse_threshold:
                record_skip(
                    skipped_items,
                    summary,
                    dataset_name,
                    scene_name,
                    episode_index,
                    sub_idx,
                    "static_clip",
                    source_video,
                    f"motion_score={score:.6f}, threshold={static_mse_threshold:.6f}",
                )
                continue

            basename = f"{safe_scene}__episode_{episode_index:06d}__sub_{sub_idx:03d}"
            meta_rel = Path("metas") / f"{basename}.txt"
            meta_path = output_root / meta_rel
            prompt = build_prompt(local_text, global_text)
            if not overwrite and meta_path.exists():
                record_skip(
                    skipped_items,
                    summary,
                    dataset_name,
                    scene_name,
                    episode_index,
                    sub_idx,
                    "already_exists",
                    meta_path,
                )
                continue

            item = {
                "dataset": dataset_name,
                "scene": scene_name,
                "episode_index": episode_index,
                "sub_instruction_index": sub_idx,
                "basename": basename,
                "meta": str(meta_rel),
                "source_video": str(source_video.resolve()),
                "frame_indices": frame_indices,
                "start_frame": frame_indices[0],
                "end_frame": frame_indices[-1] + 1,
                "sub_start_frame": sub_start,
                "sub_end_frame": sub_end,
                "source_fps": source_fps,
                "target_fps": target_fps,
                "num_frames": num_frames,
                "local_instruction": local_text,
                "global_instruction": global_text,
                "training_prompt": prompt,
                "prompt_source": prompt_source,
                "motion_score": score,
            }
            item.update(ego_delta_from_episode(episode, frame_indices[0], frame_indices[-1]))

            if not dry_run:
                meta_path.write_text(prompt + "\n", encoding="utf-8")
            manifest_items.append(item)
            summary.converted += 1

        del vr
        if not found_subinstruction:
            record_skip(
                skipped_items,
                summary,
                dataset_name,
                scene_name,
                episode_index,
                None,
                "missing_sub_instruction",
                episodes_path,
            )


def write_manifest(manifest_path: Path, manifest_items: list[dict[str, Any]]) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in manifest_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_summary(summary: Summary, dry_run: bool, output_root: Path, report_path: Path) -> None:
    mode = "DRY RUN" if dry_run else "CONVERT"
    print(f"\n=== InternData Cosmos Subclip Index Summary ({mode}) ===")
    print(f"Datasets scanned:       {summary.datasets_scanned}")
    print(f"Scenes scanned:         {summary.scenes_scanned}")
    print(f"Episodes scanned:       {summary.episodes_scanned}")
    print(f"Subinstructions scanned:{summary.subinstructions_scanned}")
    converted_label = "Would index" if dry_run else "Indexed"
    print(f"{converted_label}:              {summary.converted}")
    print(f"Static dropped:         {summary.static_dropped}")
    print(f"Skipped:                {summary.skipped}")
    if dry_run:
        print("Dry run: no files were written.")
    else:
        print(f"Output root:            {output_root}")
        print(f"Manifest:               {output_root / 'manifest.jsonl'}")
        print(f"Report:                 {report_path}")
    print("====================================================\n")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser()
    output_root = args.output_root.expanduser()
    report_path = args.report_path.expanduser()

    if args.num_frames != 29:
        print(f"Warning: --num-frames is {args.num_frames}; your current Cosmos experiment expects 29 frames.")
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    scene_dirs = discover_scene_dirs(input_root)
    if args.limit_scenes is not None:
        scene_dirs = scene_dirs[: args.limit_scenes]

    if not args.dry_run:
        (output_root / "metas").mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = Summary()
    manifest_items: list[dict[str, Any]] = []
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
            target_fps=args.target_fps,
            num_frames=args.num_frames,
            index_units=args.index_units,
            static_mse_threshold=args.static_mse_threshold,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            summary=summary,
            manifest_items=manifest_items,
            skipped_items=skipped_items,
        )

    summary.datasets_scanned = len(dataset_names)

    if not args.dry_run:
        write_manifest(output_root / "manifest.jsonl", manifest_items)
        report = {
            "summary": asdict(summary),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "target_fps": args.target_fps,
            "num_frames": args.num_frames,
            "index_units": args.index_units,
            "static_mse_threshold": args.static_mse_threshold,
            "safe_scene_by_original": safe_scene_by_original,
            "skipped": [asdict(item) for item in skipped_items],
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_summary(summary, args.dry_run, output_root, report_path)


if __name__ == "__main__":
    main()
