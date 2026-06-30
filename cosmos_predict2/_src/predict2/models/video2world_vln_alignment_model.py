# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video2World VLN language-alignment training model."""

from __future__ import annotations

from typing import Any

import attrs
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)


@attrs.define(slots=False)
class Video2WorldVLNAlignmentModelConfig(Video2WorldModelRectifiedFlowConfig):
    language_alignment_weight: float = 0.0
    language_alignment_margin: float = 0.05
    language_alignment_t_min: float = 0.2
    language_alignment_t_max: float = 0.8
    pose_aux_weight: float = 0.0
    pose_aux_hidden_dim: int = 512

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert 0.0 <= self.language_alignment_t_min <= self.language_alignment_t_max <= 1.0
        assert self.language_alignment_weight >= 0.0
        assert self.pose_aux_weight >= 0.0


class Video2WorldVLNAlignmentModel(Video2WorldModelRectifiedFlow):
    """Adds noise-aware caption contrast and optional egomotion supervision."""

    def __init__(self, config: Video2WorldVLNAlignmentModelConfig):
        super().__init__(config)
        self.pose_head = None
        if config.pose_aux_weight > 0:
            self.pose_head = nn.Sequential(
                nn.LayerNorm(config.state_ch),
                nn.Linear(config.state_ch, config.pose_aux_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.pose_aux_hidden_dim, 3),
            ).to(device=self.tensor_kwargs["device"])

    def _compute_online_text_embeddings(self, data_batch: dict[str, Any]) -> None:
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

    @staticmethod
    def _caption_list(captions: Any) -> list[str]:
        if isinstance(captions, str):
            return [captions]
        if isinstance(captions, (list, tuple)):
            return [str(caption) for caption in captions]
        return [str(captions)]

    def _build_negative_batch(self, data_batch: dict[str, Any]) -> dict[str, Any] | None:
        captions = self._caption_list(data_batch.get(self.input_caption_key, []))
        if not captions:
            return None

        batch_size = len(captions)
        shuffled = captions[1:] + captions[:1] if batch_size > 1 else captions[:]
        provided = data_batch.get("negative_ai_caption", None)
        provided_negatives = self._caption_list(provided) if provided is not None else [""] * batch_size
        if len(provided_negatives) != batch_size:
            provided_negatives = [""] * batch_size

        neg_captions: list[str] = []
        has_real_negative = False
        for idx, caption in enumerate(captions):
            negative = provided_negatives[idx].strip()
            if not negative:
                negative = shuffled[idx]
            neg_captions.append(negative)
            has_real_negative = has_real_negative or negative != caption

        if not has_real_negative:
            return None

        neg_batch = dict(data_batch)
        neg_batch[self.input_caption_key] = neg_captions
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self._compute_online_text_embeddings(neg_batch)
        elif "t5_text_embeddings" in data_batch and batch_size > 1:
            neg_batch["t5_text_embeddings"] = torch.roll(data_batch["t5_text_embeddings"], shifts=1, dims=0)
            if "t5_text_mask" in data_batch:
                neg_batch["t5_text_mask"] = torch.roll(data_batch["t5_text_mask"], shifts=1, dims=0)
        else:
            return None
        return neg_batch

    def _make_negative_condition(
        self,
        neg_batch: dict[str, Any],
        data_batch: dict[str, Any],
        x0_B_C_T_H_W: torch.Tensor,
        condition,
    ):
        is_image_batch = self.is_image_batch(data_batch)
        neg_condition = self.conditioner(neg_batch)
        neg_condition = neg_condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        num_conditional_frames = data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, condition.num_conditional_frames_B)
        neg_condition = neg_condition.set_video_condition(
            gt_frames=x0_B_C_T_H_W.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        return neg_condition

    def _assert_aux_losses_support_current_parallelism(self) -> None:
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            raise NotImplementedError(
                "Video2WorldVLNAlignmentModel alignment/pose losses currently require context_parallel_size=1"
            )

    def _pose_loss(self, vt_pred_B_C_T_H_W: torch.Tensor, data_batch: dict[str, Any]) -> tuple[torch.Tensor, Any]:
        if self.pose_head is None or "ego_delta" not in data_batch:
            return vt_pred_B_C_T_H_W.new_zeros(()), None
        pooled = vt_pred_B_C_T_H_W.mean(dim=(2, 3, 4)).float()
        pred_delta = self.pose_head(pooled)
        target = data_batch["ego_delta"].to(device=pred_delta.device, dtype=pred_delta.dtype)
        mask = data_batch.get("ego_delta_mask", torch.ones(target.shape[0], device=target.device))
        mask = mask.to(device=pred_delta.device, dtype=pred_delta.dtype).reshape(-1)
        per_sample = F.smooth_l1_loss(pred_delta, target, reduction="none").mean(dim=1)
        if torch.any(mask > 0):
            loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = pred_delta.new_zeros(())
        return loss, pred_delta

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self._compute_online_text_embeddings(data_batch)

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        train_t_B = t_B
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)

        if self.config.use_high_sigma_strategy:
            raise NotImplementedError("High sigma strategy is buggy when using CP")

        sigmas = self.rectified_flow.get_sigmas(
            timesteps,
            self.tensor_kwargs_fp32,
        )

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )

        base_loss = torch.mean(time_weights_B * per_instance_loss)
        total_loss = base_loss
        alignment_loss = base_loss.new_zeros(())
        neg_per_instance_loss = None
        alignment_mask_ratio = base_loss.new_zeros(())

        if self.config.language_alignment_weight > 0:
            self._assert_aux_losses_support_current_parallelism()
            neg_batch = self._build_negative_batch(data_batch)
            align_mask = (
                (train_t_B.squeeze(-1) >= self.config.language_alignment_t_min)
                & (train_t_B.squeeze(-1) <= self.config.language_alignment_t_max)
            )
            alignment_mask_ratio = align_mask.float().mean()
            if neg_batch is not None and torch.any(align_mask):
                neg_condition = self._make_negative_condition(neg_batch, data_batch, x0_B_C_T_H_W, condition)
                vt_pred_neg_B_C_T_H_W = self.denoise(
                    noise=epsilon_B_C_T_H_W,
                    xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
                    timesteps_B_T=timesteps,
                    condition=neg_condition,
                )
                neg_per_instance_loss = torch.mean(
                    (vt_pred_neg_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
                    dim=list(range(1, vt_pred_neg_B_C_T_H_W.dim())),
                )
                alignment_per_sample = F.relu(
                    self.config.language_alignment_margin + per_instance_loss - neg_per_instance_loss
                )
                alignment_loss = alignment_per_sample[align_mask].mean()
                total_loss = total_loss + self.config.language_alignment_weight * alignment_loss

        pose_loss = base_loss.new_zeros(())
        pred_ego_delta = None
        if self.config.pose_aux_weight > 0:
            self._assert_aux_losses_support_current_parallelism()
            pose_loss, pred_ego_delta = self._pose_loss(vt_pred_B_C_T_H_W, data_batch)
            total_loss = total_loss + self.config.pose_aux_weight * pose_loss

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": base_loss,
            "total_loss": total_loss,
            "language_alignment_loss": alignment_loss,
            "language_alignment_mask_ratio": alignment_mask_ratio,
            "pose_aux_loss": pose_loss,
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
        }
        if neg_per_instance_loss is not None:
            output_batch["negative_per_instance_loss"] = neg_per_instance_loss
        if pred_ego_delta is not None:
            output_batch["pred_ego_delta"] = pred_ego_delta

        return output_batch, total_loss
