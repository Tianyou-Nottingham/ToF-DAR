import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def make_tof_config(
    mode="train",
    input_height=416,
    input_width=544,
    train_zone_num=8,
    train_zone_random_offset=0,
    zone_sample_num=16,
    sample_uniform=False,
    simu_max_distance=4.0,
    drop_hist=0.0,
    noise_mean=0.0,
    noise_sigma=0.0,
    noise_prob=0.0,
    do_random_rotate=True,
    degree=2.5,
    transform=True,

):
    return SimpleNamespace(
        mode=mode,
        input_height=input_height,
        input_width=input_width,
        train_zone_num=train_zone_num,
        train_zone_random_offset=train_zone_random_offset,
        zone_sample_num=zone_sample_num,
        sample_uniform=sample_uniform,
        simu_max_distance=simu_max_distance,
        drop_hist=drop_hist,
        noise_mean=noise_mean,
        noise_sigma=noise_sigma,
        noise_prob=noise_prob,
        random_simu_max_d=False,
        simu_max_d=simu_max_distance,
        simu_min_d=0.0,
        do_random_rotate=do_random_rotate,
        degree=degree,
        transform=transform,
    )


def tensor_linspace(start, end, steps=10):
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def sample_point_from_hist_parallel(hist_data, mask, config):
    znum = int(math.sqrt(mask.numel()))
    fh = torch.zeros([znum**2, config.zone_sample_num], dtype=torch.float32)
    zone_sample_num = config.zone_sample_num
    if not config.sample_uniform:
        delta = 1e-3
        sample_ppf = torch.Tensor(np.arange(delta, 1, (1-2*delta)/(zone_sample_num-1)).tolist()).unsqueeze(0)
        d = torch.distributions.Normal(hist_data[mask, 0:1].clamp(min=1e-6), hist_data[mask, 1:2].clamp(min=1e-6))
        fh[mask] = d.icdf(sample_ppf).to(torch.float32)
    else:
        sigma = hist_data[mask, 1]
        start = hist_data[mask, 0] - 3.0*sigma
        end = hist_data[mask, 0] + 3.0*sigma
        depth = tensor_linspace(start, end, steps=config.zone_sample_num)
        fh[mask] = depth.to(torch.float32)
    return fh


def get_hist_parallel(rgb, dep, config):
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    if isinstance(dep, np.ndarray):
        dep = torch.from_numpy(dep)
    if rgb.dim() == 3 and rgb.shape[0] != 3:
        rgb = rgb.permute(2, 0, 1)
    if dep.dim() == 2:
        dep = dep.unsqueeze(0)
    elif dep.dim() == 3 and dep.shape[0] != 1:
        dep = dep.permute(2, 0, 1)
    height, width = rgb.shape[1], rgb.shape[2]
    max_distance = config.simu_max_distance
    range_margin = list(np.arange(0, max_distance+1e-9, 0.04))
    patch_height = max(1, height // config.train_zone_num)
    patch_width = max(1, width // config.train_zone_num)
    offset = 0
    if config.train_zone_random_offset > 0:
        offset = random.randint(-config.train_zone_random_offset, config.train_zone_random_offset)
    train_zone_num = config.train_zone_num if config.mode == 'train' else 8
    sy = int((height - patch_height*train_zone_num) / 2) + offset
    sx = int((width - patch_width*train_zone_num) / 2) + offset
    dep_extracted = dep[:, sy:sy+patch_height*train_zone_num, sx:sx+patch_width*train_zone_num]
    dep_patches = dep_extracted.unfold(2, patch_width, patch_width).unfold(1, patch_height, patch_height)
    dep_patches = dep_patches.contiguous().view(-1, patch_height, patch_width)
    hist = torch.stack([torch.histc(x, bins=int(max_distance/0.04), min=0, max=max_distance) for x in dep_patches], 0)

    hist[:, 0] = 0
    hist = torch.clip(hist-20, 0, None)
    for i, bin_data in enumerate(hist):
        idx = np.where(bin_data != 0)[0]
        if idx.size == 0:
            continue
        idx_split = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        bin_data_split = np.split(bin_data[idx], np.where(np.diff(idx) != 1)[0] + 1)
        signal = int(np.argmax([torch.sum(b) for b in bin_data_split]))
        hist[i, :] = 0
        hist[i, idx_split[signal]] = bin_data_split[signal]

    dist = ((torch.Tensor(range_margin[1:]) + np.array(range_margin[:-1]))/2).unsqueeze(0)
    sy = torch.Tensor(list(range(sy, sy+patch_height*train_zone_num, patch_height)) * train_zone_num).view([train_zone_num, -1]).T.reshape([-1])
    sx = torch.Tensor(list(range(sx, sx+patch_width*train_zone_num, patch_width)) * train_zone_num)
    fr = torch.stack([sy, sx, sy+patch_height, sx+patch_width], dim=1)
        
    mask = torch.zeros([train_zone_num, train_zone_num], dtype=torch.bool)
    n = torch.sum(hist, dim=1)
    mask = n > 0
    mask = mask.reshape([-1])
    mu = torch.sum(dist * hist, dim=1) / (n+1e-9)
    std = torch.sqrt(torch.sum(hist * torch.pow(dist-mu.unsqueeze(-1), 2), dim=1)/(n+1e-9)) + 1e-9
    fh = torch.stack([mu, std], axis=1).reshape([train_zone_num,train_zone_num,2])
    fh = fh.reshape([-1, 2]) # train_zone_num*train_zone_num, 2

    return fh, fr, mask
