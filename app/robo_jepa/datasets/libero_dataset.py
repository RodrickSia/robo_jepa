"""Native LIBERO HDF5 clips using the RoboJEPA training batch contract."""

import logging
from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from app.robo_jepa.transforms import make_transforms

logger = logging.getLogger(__name__)


class LiberoDataset(Dataset):
    """Read contiguous clips without converting native states or controller actions.

    Expected HDF5 input
    -------------------
    ``data_path`` is a filename or list of filenames with this layout::

        data/demo_<id>/
            actions                    [N, 7]
            obs/
                agentview_rgb          uint8 [N, H_in, W_in, 3]
                eye_in_hand_rgb         uint8 [N, H_in, W_in, 3]
                ee_pos                 [N, 3]
                ee_ori                 [N, 3], axis-angle rotation vector
                gripper_states         [N, 2], finger joint positions

    Only images named in ``cameras`` are required. All fields must have the
    same N within a demonstration. ``demos`` optionally selects demonstration
    names in each file; otherwise all names under ``data`` are used.
    Each sample is a complete slice ``s = start:start + frames_per_clip``
    from one demonstration. Every possible full window is indexed, without
    crossing demonstration boundaries. Short demonstrations are skipped with
    a warning; there is no temporal padding or frame subsampling.

    LIBERO-to-RoboJEPA mapping
    -------------------------
    Let T = ``frames_per_clip``, V = ``len(cameras)``, and P = ``pad_size``.
    ``__getitem__`` returns these 12 fields in order, WITHOUT a batch axis:

    1. ``video``: float32 [3, V, T, H, W]. For each camera in ``cameras``
       order, read ``obs/<camera>[s]`` as [T, H_in, W_in, 3] and apply the
       RoboJEPA video transform. It returns normalized and unnormalized
       tensors, both [3, T, H, W]. Stack the normalized tensors on axis 1.
       The default transform spatially crops/resizes and uses ImageNet
       normalization: (pixel / 255 - mean) / std.
    2. ``raw_video``: float32 [T, V, H_raw, W_raw, 3]. Stack the SAME
       transform calls' unnormalized outputs on axis 1 and permute
       [3, V, T, H_raw, W_raw] to [T, V, H_raw, W_raw, 3]. These are
       spatially transformed 0-255 pixel targets, not untouched HDF5 images.
       A transform's optional extra ``resize`` affects only ``video``, so
       its spatial size may differ from ``raw_video``.
    3. ``actions``: float32 [T, P]. Copy ``actions[s]`` to columns 0:7;
       columns 7:P are zeros. These remain recorded controller commands:
       no state differencing, normalization, or command chunking is applied.
    4. ``states``: float32 [T, P]. Concatenate native observation fields::

           states[:, 0:3] = obs/ee_pos[s]
           states[:, 3:6] = obs/ee_ori[s]
           states[:, 6:8] = obs/gripper_states[s]
           states[:, 8:P] = 0

       No coordinate-frame transform, rotation conversion, or gripper
       calibration is applied. ``joint_states``, simulator ``states``, and
       ``robot_states`` are not read.
    5. ``extrinsics``: float32 [T, V, 6], all NaN. Camera calibration is
       NOT loaded or inferred. The six slots follow RoboJEPA's camera-pose
       convention (xyz + Euler XYZ); NaN denotes unavailable values.
    6. ``padding_mask``: float32 [T], all ones because clips are complete.
    7. ``view_ids``: int64 [V], ``[view_map[c] for c in cameras]``. IDs
       are supplied explicitly, not inferred from image content. Their order
       matches the view axis in both video tensors.
    8. ``dt``: float32 [T], all ``1 / raw_fps`` seconds. ``raw_fps`` is
       supplied by configuration (default 20), not read from HDF5.
       ``fps``, if supplied, must equal ``raw_fps`` to preserve every command.
    9. ``start_frame``: Python int, the slice's start within the demonstration.
    10. ``trajectory_name``: string ``"<HDF5 path>:<demo name>"``.
    11. ``embodiment_name``: the configured string, unchanged.
    12. ``index``: Python int, the requested clip index within this dataset.

    Default DataLoader collation adds a leading B axis to all tensors,
    converts the integer fields to [B] tensors, and groups strings into
    sequences of length B. In particular, ``video`` becomes
    [B, 3, V, T, H, W] and ``raw_video`` becomes
    [B, T, V, H_raw, W_raw, 3].

    Configuration and limitations
    -----------------------------
    ``frames_per_clip >= 2`` and ``pad_size >= 8`` are required.
    ``transform`` defaults to RoboJEPA's ``make_transforms()``; custom
    transforms must preserve temporal order and return the two layouts
    described above. Each camera is transformed separately, with temporally
    shared spatial augmentation under the default transform. No image flip
    or correction of LIBERO's camera convention is performed by this adapter.

    ``embodiment_name`` must separately be registered in the model with
    compatible action/state projections. Padding native values does not make
    them compatible with a pretrained DROID projection. This adapter does
    not modify model parameters or synthesize future robot states.
    """

    def __init__(
        self,
        data_path,
        dataset_name,
        *,
        embodiment_name,
        view_map,
        frames_per_clip=16,
        pad_size=64,
        cameras=("agentview_rgb", "eye_in_hand_rgb"),
        raw_fps=20,
        fps=None,
        demos=None,
        transform=None,
    ):
        if frames_per_clip < 2:
            raise ValueError("frames_per_clip must be at least 2 for next-frame training")
        if pad_size < 8:
            raise ValueError("pad_size must accommodate the native 8-D LIBERO state")
        if not np.isfinite(raw_fps) or raw_fps <= 0:
            raise ValueError("raw_fps must be finite and positive")
        if fps is not None and fps != raw_fps:
            raise ValueError("Native LIBERO commands require contiguous clips: fps must equal raw_fps")
        if not cameras or any(camera not in view_map for camera in cameras):
            raise ValueError("Provide at least one camera and a view_map entry for every camera")
        if any(not isinstance(view_map[camera], int) or view_map[camera] < 0 for camera in cameras):
            raise ValueError("Camera IDs must be nonnegative integers")

        self.dataset_name = dataset_name
        self.embodiment_name = embodiment_name
        self.frames_per_clip = frames_per_clip
        self.pad_size = pad_size
        self.cameras = tuple(cameras)
        self.view_ids = torch.tensor([view_map[camera] for camera in cameras], dtype=torch.long)
        self.raw_fps = raw_fps
        self.transform = transform if transform is not None else make_transforms()
        self.episodes = []
        self.clip_ends = []
        paths = [data_path] if isinstance(data_path, (str, Path)) else list(data_path)
        for path in paths:
            path = str(Path(path))
            with h5py.File(path, "r") as file:
                data = self._group(file, "data")
                names = sorted(data.keys()) if demos is None else demos
                for name in names:
                    demo = self._group(data, name)
                    actions = self._dataset(demo, "actions")
                    if actions.ndim != 2 or actions.shape[1] != 7:
                        raise ValueError(f"{actions.name} must have shape [N, 7]")
                    length = len(actions)
                    obs = self._group(demo, "obs")
                    for key, width in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
                        field = self._dataset(obs, key)
                        if field.shape != (length, width):
                            raise ValueError(f"{field.name} must have shape [{length}, {width}]")
                    for camera in self.cameras:
                        images = self._dataset(obs, camera)
                        if images.ndim != 4 or images.shape[0] != length or images.shape[-1] != 3:
                            raise ValueError(f"{images.name} must have shape [{length}, H, W, 3]")
                        if images.dtype != np.uint8:
                            raise ValueError(f"{images.name} must contain uint8 RGB images")
                    count = length - frames_per_clip + 1
                    if count <= 0:
                        logger.warning("Skipping %s:%s: too short for a complete clip", path, name)
                        continue
                    self.episodes.append((path, name))
                    self.clip_ends.append((self.clip_ends[-1] if self.clip_ends else 0) + count)
        if not self.episodes:
            raise ValueError("No complete LIBERO clips found in data_path")

    @staticmethod
    def _group(parent, name):
        group = parent[name]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"{group.name} must be an HDF5 group")
        return group

    @staticmethod
    def _dataset(parent, name):
        dataset = parent[name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{dataset.name} must be an HDF5 dataset")
        return dataset

    def __len__(self):
        return self.clip_ends[-1]

    def __getitem__(self, index):
        """Return one native LIBERO clip as the 12-field training tuple above."""
        if not 0 <= index < len(self):
            raise IndexError(f"Clip index {index} is outside dataset {self.dataset_name}")
        episode = bisect_right(self.clip_ends, index)
        start = index - (self.clip_ends[episode - 1] if episode else 0)
        path, name = self.episodes[episode]
        selection = slice(start, start + self.frames_per_clip)
        with h5py.File(path, "r") as file:
            demo = self._group(self._group(file, "data"), name)
            obs = self._group(demo, "obs")
            actions = torch.from_numpy(
                np.asarray(self._dataset(demo, "actions")[selection], dtype=np.float32)
            )
            states = torch.from_numpy(
                np.concatenate(
                    [self._dataset(obs, key)[selection] for key in ("ee_pos", "ee_ori", "gripper_states")],
                    axis=-1,
                ).astype(np.float32)
            )
            videos = [LiberoDataset
                self.transform(np.asarray(self._dataset(obs, camera)[selection]))
                for camera in self.cameras
            ]
        video = torch.stack([pair[0] for pair in videos], dim=1)
        raw_video = torch.stack([pair[1] for pair in videos], dim=1).permute(2, 1, 3, 4, 0)
        return (
            video,
            raw_video,
            F.pad(actions, (0, self.pad_size - actions.shape[-1])),
            F.pad(states, (0, self.pad_size - states.shape[-1])),
            torch.full((self.frames_per_clip, len(self.cameras), 6), float("nan")),
            torch.ones(self.frames_per_clip),
            self.view_ids.clone(),
            torch.full((self.frames_per_clip,), 1.0 / self.raw_fps),
            start,
            f"{path}:{name}",
            self.embodiment_name,
            index,
        )
