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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


vln_cosmos_video_dataset_29f_272_480 = L(VideoDataset)(
    dataset_dir="datasets/vln_cosmos",
    num_frames=29,
    video_size=(272, 480),
    caption_format="text",
)

vln_cosmos_dataloader_train_29f_272_480 = L(get_generic_dataloader)(
    dataset=vln_cosmos_video_dataset_29f_272_480,
    sampler=L(get_sampler)(dataset=vln_cosmos_video_dataset_29f_272_480),
    batch_size=28,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)


predict2_video2world_training_2b_vln_cosmos_29f_10fps_272_480 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=vln_cosmos_dataloader_train_29f_272_480,
    checkpoint=dict(
        save_iter=2000,
        load_path='',
        # load_path="/home/csevolunt/dongzhih/cosmos-predict2.5/runs/cosmos_predict_v2p5/video2world/2b_vln_cosmos_vam_49f_10fps_272x480_h200x4_b24/checkpoints/iter_000020000",
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_vln_cosmos_vam_29f_10fps_272x480_h200x4_gb112",
        wandb_mode="disabled",
    ),
    optimizer=dict(
        lr=1.5e-4,
        weight_decay=0.0001,
    ),
    scheduler=dict(
        f_max=[1.0],
        f_min=[0.01],
        warm_up_steps=[1000],
        cycle_lengths=[50_000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=50_000,
        straggler_detection=dict(
            enabled=False,
            max_diff=1.5,
        ),
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=0,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=0,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    model=dict(
        config=dict(
            state_t=8,
            fsdp_shard_size=4,
            min_num_conditional_frames=2,
            max_num_conditional_frames=2,
            conditional_frames_probs=None,
        ),
    ),
)


cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_vln_cosmos_29f_10fps_272_480,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
