import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .tof_sim import sample_point_from_hist_parallel, make_tof_config
from .dataloader import patch_info_from_rect_data


def _scale_rectangles_for_resize(fr: np.ndarray, orig_h: int, orig_w: int, image_size: int) -> np.ndarray:
    scale_y = image_size / max(orig_h, 1)
    scale_x = image_size / max(orig_w, 1)
    fr_before = fr.astype(np.float32, copy=True)
    fr_scaled = fr_before.copy()
    fr_scaled[:, 0] *= scale_y
    fr_scaled[:, 2] *= scale_y
    fr_scaled[:, 1] *= scale_x
    fr_scaled[:, 3] *= scale_x

    # Validate that normalized rectangle ranges stay consistent before/after resize.
    before_norm = np.stack(
        [
            fr_before[:, 0] / max(orig_h, 1),
            fr_before[:, 2] / max(orig_h, 1),
            fr_before[:, 1] / max(orig_w, 1),
            fr_before[:, 3] / max(orig_w, 1),
        ],
        axis=1,
    )
    after_norm = np.stack(
        [
            fr_scaled[:, 0] / max(image_size, 1),
            fr_scaled[:, 2] / max(image_size, 1),
            fr_scaled[:, 1] / max(image_size, 1),
            fr_scaled[:, 3] / max(image_size, 1),
        ],
        axis=1,
    )
    assert np.allclose(after_norm, before_norm, atol=1e-5), "Rectangle ranges are inconsistent after resize"
    return fr_scaled


class ZJUL5Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        json_file: str,
        split: str = "train",
        image_size: int = 256,
        depth_min: float = 0.0,
        depth_max: float = 10.0,
        normalize_rgb: bool = True,
        zone_sample_num: int = 16,
        sample_uniform: bool = False,
    ):
        self.data_root = data_root
        self.json_file = json_file
        self.split = split
        self.image_size = image_size
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.normalize_rgb = normalize_rgb
        self.K_list = torch.Tensor([
            611.2,
            609.6,
            323.4,
            244.9
        ])

        with open(json_file, "r") as f:
            meta = json.load(f)
        if split not in meta:
            raise ValueError(f"split {split} not found in {json_file}")
        self.sample_list = meta[split]

        self.cfg = make_tof_config(
            mode="train" if split == "train" else "test",
            zone_sample_num=zone_sample_num,
            sample_uniform=sample_uniform,
        )

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.sample_list)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        depth = (depth - self.depth_min) / max(self.depth_max - self.depth_min, 1e-6)
        depth = depth * 2.0 - 1.0
        return depth

    def __getitem__(self, idx: int):
        focal = self.K_list[0].item()
        rel_path = self.sample_list[idx]["filename"]
        path_file = os.path.join(self.data_root, rel_path)
        with h5py.File(path_file, "r") as f:
            rgb = f["rgb"][:]
            if "depth" in f:
                depth = f["depth"][:]
            else:
                depth = np.zeros_like(rgb[:, :, 0])
            hist_data = f["hist_data"][:]
            fr = f["fr"][:]
            mask = f["mask"][:]

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        if self.image_size is not None:
            orig_h, orig_w = rgb.shape[1], rgb.shape[2]
            rgb = F.interpolate(rgb.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            depth = F.interpolate(depth.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            fr = _scale_rectangles_for_resize(fr, orig_h, orig_w, self.image_size)

        if self.normalize_rgb:
            rgb = (rgb - self.rgb_mean) / self.rgb_std

        depth = self._normalize_depth(depth)

        hist_data = torch.from_numpy(hist_data)
        fr = torch.from_numpy(fr)
        mask = torch.from_numpy(mask).bool()
        fh = sample_point_from_hist_parallel(hist_data, mask, self.cfg)
        patch_info = patch_info_from_rect_data(fr)
        sample = {
            "image": rgb,
            "depth": depth,
            "hist_data": fh.to(torch.float),
            "rect_data": fr.to(torch.float),
            "mask": mask.to(torch.bool),
            "patch_info": patch_info,
        }
        return sample
