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

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


_DIRECTIONAL_NEGATIVE_RULES: tuple[tuple[str, str], ...] = (
    (r"\bcounter-clockwise\b", "clockwise"),
    (r"\bcounterclockwise\b", "clockwise"),
    (r"\bclockwise\b", "counterclockwise"),
    (r"\bleft\b", "right"),
    (r"\bright\b", "left"),
    (r"\bforward\b", "backward"),
    (r"\bbackward\b", "forward"),
    (r"\bbackwards\b", "forward"),
    (r"\bgo straight\b", "turn left"),
    (r"\bmove straight\b", "turn left"),
    (r"\bwalk straight\b", "turn left"),
    (r"\bstraight\b", "left"),
)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", "manifest", or "auto"
        video_paths: Optional[list[str]] = None,
        manifest_path: Optional[str] = None,
        sample_mode: str = "random",
        negative_caption_strategy: str = "none",
        directional_negative_prob: float = 0.0,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        The default mode reads ``dataset_dir/videos/*.mp4`` plus ``metas`` or
        ``captions`` exactly as before. If ``manifest_path`` is provided, each
        JSONL row is treated as one sample and may point to an existing source
        video plus explicit ``frame_indices``. This supports subclip training
        without materializing new mp4 files.
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format
        self.sample_mode = sample_mode
        self.negative_caption_strategy = negative_caption_strategy
        self.directional_negative_prob = directional_negative_prob

        if self.sample_mode not in {"random", "center", "start"}:
            raise ValueError(f"Invalid sample_mode: {self.sample_mode}. Must be 'random', 'center', or 'start'")
        if self.negative_caption_strategy not in {"none", "directional", "batch_shuffle"}:
            raise ValueError(
                f"Invalid negative_caption_strategy: {self.negative_caption_strategy}. "
                "Must be 'none', 'directional', or 'batch_shuffle'"
            )

        self.manifest_path = self._resolve_manifest_path(manifest_path)
        self.manifest_items = self._load_manifest(self.manifest_path) if self.manifest_path else None

        # Determine caption format and directory.
        self._setup_caption_format()

        if self.manifest_items is not None:
            self.video_paths = [str(self._resolve_manifest_video(item)) for item in self.manifest_items]
        else:
            video_dir = os.path.join(self.dataset_dir, "videos")
            if video_paths is None:
                self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
                self.video_paths = sorted(self.video_paths)
            else:
                self.video_paths = video_paths
        log.info(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _resolve_manifest_path(self, manifest_path: Optional[str]) -> Optional[Path]:
        if manifest_path:
            path = Path(manifest_path).expanduser()
            return path if path.is_absolute() else Path(self.dataset_dir) / path if not path.exists() else path
        candidate = Path(self.dataset_dir) / "manifest.jsonl"
        return candidate if candidate.exists() else None

    def _load_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}: {exc}") from exc
                item["_manifest_dir"] = str(manifest_path.parent)
                items.append(item)
        if not items:
            raise ValueError(f"Manifest contains no samples: {manifest_path}")
        return items

    def _resolve_path_from_manifest(self, value: str, manifest_dir: str | None = None) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        candidates = []
        if manifest_dir:
            candidates.append(Path(manifest_dir) / path)
        candidates.append(Path(self.dataset_dir) / path)
        candidates.append(path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_manifest_video(self, item: dict[str, Any]) -> Path:
        for key in ("source_video", "video", "video_path"):
            value = item.get(key)
            if value:
                return self._resolve_path_from_manifest(str(value), item.get("_manifest_dir"))
        raise ValueError(f"Manifest item is missing source_video/video/video_path: {item}")

    def _load_video(self, video_path: str, manifest_item: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        frame_ids = self._select_frame_ids(total_frames, manifest_item)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _select_frame_ids(self, total_frames: int, manifest_item: Optional[dict[str, Any]]) -> list[int]:
        if manifest_item is not None and manifest_item.get("frame_indices") is not None:
            frame_ids = [int(idx) for idx in manifest_item["frame_indices"]]
            if len(frame_ids) != self.sequence_length:
                raise ValueError(
                    f"Manifest frame_indices length {len(frame_ids)} does not match num_frames {self.sequence_length}"
                )
            return [min(max(idx, 0), total_frames - 1) for idx in frame_ids]

        if manifest_item is not None and manifest_item.get("start_frame") is not None:
            start_frame = int(manifest_item["start_frame"])
            end_frame = int(manifest_item.get("end_frame", start_frame + self.sequence_length))
            available = max(0, end_frame - start_frame)
            if available >= self.sequence_length:
                if self.sample_mode == "start":
                    selected_start = start_frame
                elif self.sample_mode == "center":
                    selected_start = start_frame + (available - self.sequence_length) // 2
                else:
                    selected_start = np.random.randint(start_frame, end_frame - self.sequence_length + 1)
            else:
                center = (start_frame + end_frame) // 2
                selected_start = center - self.sequence_length // 2
            selected_start = min(max(selected_start, 0), total_frames - self.sequence_length)
            return np.arange(selected_start, selected_start + self.sequence_length).tolist()

        max_start_idx = total_frames - self.sequence_length
        if self.sample_mode == "start":
            start_frame = 0
        elif self.sample_mode == "center":
            start_frame = max_start_idx // 2
        else:
            start_frame = np.random.randint(0, max_start_idx + 1)
        end_frame = start_frame + self.sequence_length
        return np.arange(start_frame, end_frame).tolist()

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            elif self.manifest_items is not None:
                self.caption_format = "manifest"
                self.caption_dir = None
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        elif self.caption_format == "manifest":
            if self.manifest_items is None:
                raise ValueError("caption_format='manifest' requires manifest_path or dataset_dir/manifest.jsonl")
            self.caption_dir = None
        else:
            raise ValueError(
                f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', 'manifest', or 'auto'"
            )

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text(encoding="utf-8").strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str, manifest_item: Optional[dict[str, Any]] = None) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path, manifest_item)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def _load_manifest_caption(self, item: dict[str, Any], video_basename: str) -> str:
        meta_value = item.get("meta")
        if meta_value:
            meta_path = self._resolve_path_from_manifest(str(meta_value), item.get("_manifest_dir"))
            caption = self._load_text(meta_path)
            if caption:
                return caption
        for key in ("training_prompt", "prompt", "caption", "ai_caption", "local_instruction"):
            if item.get(key):
                return str(item[key]).strip()
        if self.caption_format == "text" and self.caption_dir is not None:
            return self._load_text(Path(self.caption_dir) / f"{video_basename}.txt")
        return ""

    def _load_caption(self, item: Optional[dict[str, Any]], video_path: str) -> tuple[str, str]:
        video_basename = os.path.basename(video_path).replace(".mp4", "")
        if item is not None:
            video_basename = str(item.get("basename") or video_basename)
            if self.caption_format == "manifest":
                return self._load_manifest_caption(item, video_basename), video_basename

        if self.caption_format == "json":
            caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
            return self._load_json_caption(Path(caption_path)), video_basename
        if self.caption_format == "text":
            if item is not None and item.get("meta"):
                return self._load_manifest_caption(item, video_basename), video_basename
            caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
            return self._load_text(Path(caption_path)), video_basename
        return self._load_manifest_caption(item or {}, video_basename), video_basename

    def _make_directional_negative(self, caption: str) -> str:
        if self.directional_negative_prob <= 0 or random.random() > self.directional_negative_prob:
            return ""
        candidates = [rule for rule in _DIRECTIONAL_NEGATIVE_RULES if re.search(rule[0], caption, flags=re.IGNORECASE)]
        if not candidates:
            return ""
        pattern, replacement = random.choice(candidates)
        return re.sub(
            pattern,
            lambda match: _match_case(match.group(0), replacement),
            caption,
            count=1,
            flags=re.IGNORECASE,
        )

    def _add_manifest_fields(self, data: dict[str, Any], item: Optional[dict[str, Any]], caption: str) -> None:
        negative_caption = ""
        if item is not None:
            for key in ("negative_ai_caption", "negative_caption"):
                if item.get(key):
                    negative_caption = str(item[key]).strip()
                    break
            data["local_instruction"] = str(item.get("local_instruction", ""))
            data["global_instruction"] = str(item.get("global_instruction", ""))
            data["motion_score"] = torch.tensor(float(item.get("motion_score", 0.0)), dtype=torch.float32)
            ego_values = [item.get("delta_x"), item.get("delta_y"), item.get("delta_yaw")]
            has_ego = all(value is not None for value in ego_values)
            data["ego_delta"] = torch.tensor(
                [float(value) if value is not None else 0.0 for value in ego_values], dtype=torch.float32
            )
            data["ego_delta_mask"] = torch.tensor(1.0 if has_ego else 0.0, dtype=torch.float32)
            data["yaw_label"] = str(item.get("yaw_label", ""))
            if item.get("forward_distance") is not None:
                data["forward_distance"] = torch.tensor(float(item["forward_distance"]), dtype=torch.float32)
        else:
            data["motion_score"] = torch.tensor(0.0, dtype=torch.float32)
            data["ego_delta"] = torch.zeros(3, dtype=torch.float32)
            data["ego_delta_mask"] = torch.tensor(0.0, dtype=torch.float32)

        if not negative_caption and self.negative_caption_strategy == "directional":
            negative_caption = self._make_directional_negative(caption)
        data["negative_ai_caption"] = negative_caption

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            item = self.manifest_items[index] if self.manifest_items is not None else None
            video_path = self.video_paths[index]
            video, fps = self._get_frames(video_path, item)
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            caption, _ = self._load_caption(item, video_path)

            data["video"] = video
            data["ai_caption"] = caption

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)
            self._add_manifest_fields(data, item, caption)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        log.debug(f"Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=93,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
