# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-dataset initialisation for robot video training.

Provides :class:`MultiDataset` (a thin wrapper that concatenates multiple
:class:`RobotVideoDataset` instances into a single indexable dataset) and
:func:`init_datasets` (a factory that builds the combined dataset, weighted
sampler, and data-loader in one call).

Typical usage::

    from app.robo_jepa.datasets.init_data import init_datasets
    from app.robo_jepa.transforms import make_transforms

    loader, sampler = init_datasets(
        data_paths=[None, None],
        dataset_names=['Droid', 'Roboset'],
        dataset_weights=[0.6, 0.4],
        transform=make_transforms(crop_height=256, crop_width=256),
        common_kwargs=dict(
            s3_profile='<aws_profile>',
            s3_prefix='s3://<bucket>/data/robotics/',
            pad_size=35,
            frames_per_clip=16,
        ),
        dataset_kwargs={
            'Droid':   dict(embodiment_name='franka', s3_prefix='droid/1.0.1/',
                            raw_fps=60, fps=5, pose_dim=6, ...),
            'Roboset': dict(embodiment_name='franka', s3_prefix='roboset/',
                            raw_fps=5, fps=5, pose_dim=6, ...),
        },
        batch_size=4,
        num_workers=10,
    )

``common_kwargs`` are shared across all datasets. ``dataset_kwargs`` override
or extend them per-dataset. The per-dataset ``s3_prefix`` is joined with the
common ``s3_prefix`` (i.e. ``os.path.join(common, per_dataset)``).
"""

import os
from copy import deepcopy as copy
from logging import getLogger

import numpy as np
import torch
import torch.utils.data

from app.robo_jepa.datasets.libero_dataset import LiberoDataset
from src.datasets.utils.weighted_sampler import MemoryEfficientDistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


class MultiDataset(torch.utils.data.Dataset):
    """Concatenation of multiple :class:`RobotVideoDataset` instances.

    Indexes are contiguous: if dataset A has 100 samples and B has 200,
    indices 0–99 map to A and 100–299 map to B.  Use together with
    :class:`MemoryEfficientDistributedWeightedSampler` for weighted sampling.

    Args:
        *datasets: Variable number of :class:`RobotVideoDataset` instances.
        dataset_weights (list[float], optional): Sampling weights per dataset.
            Normalised internally to sum to 1. If None, weights are
            proportional to dataset length.
    """

    def __init__(self, *datasets, dataset_weights=None):
        self.datasets = datasets
        if dataset_weights is None:
            dataset_weights = [len(d) for d in datasets]
        total = sum(dataset_weights)
        dataset_weights = [w / total for w in dataset_weights]
        self.dataset_weights = dataset_weights
        self.dataset_sizes = np.array([len(d) for d in datasets])
        self.dataset_start_ids = np.cumsum(self.dataset_sizes) - self.dataset_sizes
        self.total = sum(len(d) for d in datasets)

    def __getitem__(self, index):
        dataset_mask = self.dataset_start_ids <= index
        # find the last maximum in dataset_mask i.e. the last dataset start_id that is lower than our index
        dataset_bin = len(dataset_mask) - 1 - np.argmax(dataset_mask[::-1])
        return self.datasets[dataset_bin][index - self.dataset_start_ids[dataset_bin]]

    def __len__(self):
        return self.total


def init_datasets(
    data_paths,
    dataset_names,
    dataset_weights,
    common_kwargs,
    dataset_kwargs,
    rank=0,
    world_size=1,
    batch_size=1,
    collator=None,
    transform=None,
    drop_last=False,
    num_workers=0,
    pin_mem=True,
    shuffle=True,
    persistent_workers=True,
    num_replicas_override=None,
    rank_override=None,
):
    """Build a multi-dataset data-loader with weighted distributed sampling.

    Creates one dataset per entry in ``dataset_names``: ``dataset_type``
    in ``dataset_kwargs[name]`` selects ``'robot_video'`` (default) or
    ``'libero'``. Wraps the datasets in :class:`MultiDataset` and attaches
    :class:`MemoryEfficientDistributedWeightedSampler`.

    Configuration format
    --------------------
    ``data_paths``, ``dataset_names``, and ``dataset_weights`` are nonempty
    lists of equal length, with corresponding entries at each index.
    Every name must have an entry in ``dataset_kwargs``. Per-dataset options
    override ``common_kwargs``; the merged options must be accepted by the
    selected dataset constructor. ``dataset_type`` is a factory option and
    belongs in ``dataset_kwargs[name]``, not ``common_kwargs``.

    RobotVideoDataset storage (local filesystem or S3)
    ------------------------------------------------
    Pass ``None`` as its ``data_paths`` entry. Both common and per-dataset
    ``s3_prefix`` keys are required, even for local storage; they are joined::

        effective_s3_prefix = os.path.join(common_kwargs['s3_prefix'],
                                           dataset_kwargs[name]['s3_prefix'])

    The resulting root contains::

        dataset_index.csv              # "path" column: episode directories
        aggregated_metadata.json       # when required by metadata filters
        <episode>/data.h5
        <episode>/<camera>/video.mp4

    Each ``data.h5`` contains this default Cartesian layout::

        cameras/<camera_name>          # string: relative MP4 filename
        trajectory/observation/robot_state/
            cartesian_position         # [N, pose_dim], typically xyz + Euler XYZ
            gripper_position           # [N, gripper_dim]
        trajectory/action/<field>      # needed for action_source='action_key'

    ``action_source='state_delta'`` derives actions from consecutive states.
    ``action_source='action_key'`` reads recorded commands using the configured
    ``action_key_layout``, chunks them over each sampled interval, and supplies
    zero proprioceptive states. Other state layouts are configured through
    RobotVideoDataset options such as ``state_keys`` or ``use_joint_state``.

    Native LIBERO storage
    ---------------------
    Set ``dataset_type='libero'``. Its ``data_paths`` entry is a local HDF5
    filename or a list of filenames, not an episode directory or CSV. Each
    file contains::

        data/demo_<id>/
            actions                    # [N, 7], native controller commands
            obs/
                agentview_rgb          # uint8 [N, H, W, 3]
                eye_in_hand_rgb         # uint8 [N, H, W, 3]
                ee_pos                 # [N, 3]
                ee_ori                 # [N, 3], axis-angle rotation vector
                gripper_states         # [N, 2], finger joint positions

    All fields in a demonstration must have the same N. ``cameras`` selects
    image dataset names under ``obs``; only selected cameras are required.
    ``view_map`` must map each selected name to a semantic camera ID understood
    by the model. ``demos`` optionally selects demonstration names in each file.
    Only complete, contiguous clips are indexed; short demonstrations are
    skipped with a warning. ``fps``, if supplied, must equal ``raw_fps``.
    No state/action convention conversion is performed: states concatenate
    ``ee_pos``, ``ee_ori``, and ``gripper_states`` into 8 values, and recorded
    7-D actions are retained. Both are zero-padded to ``pad_size >= 8``.

    Example LIBERO loader configuration::

        loader, sampler = init_datasets(
            data_paths=[['/data/task_a.hdf5', '/data/task_b.hdf5']],
            dataset_names=['Libero'],
            dataset_weights=[1.0],
            common_kwargs={'frames_per_clip': 16, 'pad_size': 64},
            dataset_kwargs={'Libero': {
                'dataset_type': 'libero',
                'embodiment_name': 'libero',
                'cameras': ['eye_in_hand_rgb'],
                'view_map': {'eye_in_hand_rgb': 2},
                'raw_fps': 20,
            }},
            batch_size=4,
        )

    The model must separately register ``embodiment_name`` with compatible
    learned projections; the loader does not add or adapt model parameters.
    Do not include S3-only options in LIBERO's merged constructor options.

    Training batch format
    ---------------------
    With default collation and ordinary predictor clips, batches contain::

        (video, raw_video, actions, states, extrinsics, padding_mask,
         view_ids, dt, start_frame, trajectory_name, embodiment_name, index)

        video          [B, 3, V, T, H, W]  normalized encoder input
        raw_video      [B, T, V, H, W, 3]  unnormalized RGB, 0-255 float
        actions        [B, T, P]          padded action vectors
        states         [B, T, P]          padded proprioceptive states
        extrinsics     [B, T, V, 6]       NaN when unavailable
        padding_mask   [B, T]             1=valid, 0=padding
        view_ids       [B, V]             integer camera IDs
        dt             [B, T]             seconds per sampled interval
        start_frame    [B]                clip start indices
        trajectory_name, embodiment_name  sequences of B strings
        index          [B]                per-dataset sample indices

    B is batch size, V views, T clip length, and P ``pad_size``. Numeric
    video/action/state tensors are float32 before training-device conversion.
    Spatial sizes follow the transform; ``resize`` can give normalized and
    raw video different sizes. RobotVideoDataset's intermediate-frame decoder
    mode has different video time lengths; see its class documentation.

    Args:
        data_paths (list): One entry per dataset. For LIBERO, each entry is
            an HDF5 filename or list of filenames; for legacy datasets, use None.
        dataset_names (list[str]): Dataset identifiers matching keys in
            *dataset_kwargs*.
        dataset_weights (list[float]): Sampling weights (unnormalised).
        common_kwargs (dict): Keyword arguments shared across all datasets
            (e.g. ``s3_profile``, ``pad_size``, ``frames_per_clip``).
        dataset_kwargs (dict[str, dict]): Per-dataset keyword arguments
            keyed by dataset name. Overrides/extends *common_kwargs*.
            Set ``dataset_type='libero'`` for native LIBERO HDF5 clips.
            LIBERO requires ``embodiment_name`` and ``view_map`` and accepts
            ``frames_per_clip``, ``pad_size``, ``cameras``, ``raw_fps``, and
            ``demos``. No S3 options are needed for a LIBERO-only loader.
        rank (int): Current process rank for distributed training.
        world_size (int): Total number of processes.
        batch_size (int): Batch size per rank.
        collator (callable, optional): Custom collate function.
        transform (callable, optional): Video transform passed to every
            dataset (see :func:`transforms.make_transforms`). Maps
            ``[T, H, W, 3]`` arrays to normalized and raw ``[3, T, H, W]``
            tensors. LIBERO uses ``make_transforms()`` when omitted.
        drop_last (bool): Drop the last incomplete batch.
        num_workers (int): DataLoader worker processes.
        pin_mem (bool): Pin memory for GPU transfer.
        shuffle (bool): Enable random sampling.
        persistent_workers (bool): Keep workers alive between epochs.
        num_replicas_override (int, optional): Override sampler world size
            for a data-parallel/view group.
        rank_override (int, optional): Override sampler rank within that group.

    Returns:
        tuple: ``(data_loader, dist_sampler)``.
    """
    if not dataset_names or not (len(data_paths) == len(dataset_names) == len(dataset_weights)):
        raise ValueError("data_paths, dataset_names, and dataset_weights must have the same nonzero length")
    datasets = []
    for data_path, dataset_name in zip(data_paths, dataset_names):
        current_kwargs = copy(common_kwargs)
        new_kwargs = copy(dataset_kwargs[dataset_name])
        dataset_type = new_kwargs.pop("dataset_type", "robot_video")
        if dataset_type == "robot_video":
            from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset

            print("current_kwargs s3 prefix:", current_kwargs.get("s3_prefix"))
            print("new_kwargs s3 prefix:", new_kwargs.get("s3_prefix"))
            new_kwargs["s3_prefix"] = os.path.join(current_kwargs["s3_prefix"], new_kwargs["s3_prefix"])
            dataset_class = RobotVideoDataset
        elif dataset_type == "libero":
            dataset_class = LiberoDataset
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        current_kwargs.update(new_kwargs)
        dataset = dataset_class(
            dataset_name=dataset_name,
            data_path=data_path,
            transform=transform,
            **current_kwargs,
        )
        datasets.append(dataset)
    datasets = MultiDataset(*datasets, dataset_weights=dataset_weights)
    # Per-group sharding: when num_replicas_override / rank_override are given
    # (e.g. for view-group routing across DDP ranks), the sampler shuffles the
    # dataset only within the group so that each group sees an independent
    # shuffling/weighting universe even though DDP itself remains a single
    # global process group.
    sampler_num_replicas = num_replicas_override if num_replicas_override is not None else world_size
    sampler_rank = rank_override if rank_override is not None else rank
    dist_sampler = MemoryEfficientDistributedWeightedSampler(
        datasets, num_replicas=sampler_num_replicas, rank=sampler_rank, shuffle=shuffle
    )
    # dist_sampler = DistributedSampler(
    #     datasets, rank=rank, num_replicas=world_size, shuffle=True
    # )

    data_loader = torch.utils.data.DataLoader(
        datasets,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("VideoDataset unsupervised data loader created")
    return data_loader, dist_sampler
