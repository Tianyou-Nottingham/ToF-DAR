import argparse
import os
import os.path as osp
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.tof_var import ToFDepthVAR
from utils.zjul5_dataset import ZJUL5Dataset
from utils.nyu_dataset import NYUv2ToFDataset


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--json_file", type=str, required=True)
    p.add_argument("--dataset", type=str, default="nyu", choices=["zjul5", "nyu"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--depth_min", type=float, default=0.0)
    p.add_argument("--depth_max", type=float, default=10.0)
    p.add_argument("--input_height", type=int, default=416)
    p.add_argument("--input_width", type=int, default=544)
    p.add_argument("--train_zone_num", type=int, default=8)
    p.add_argument("--train_zone_random_offset", type=int, default=0)
    p.add_argument("--zone_sample_num", type=int, default=16)
    p.add_argument("--simu_max_distance", type=float, default=4.0)
    p.add_argument("--sample_uniform", action="store_true")
    p.add_argument("--drop_hist", type=float, default=0.0)
    p.add_argument("--noise_mean", type=float, default=0.0)
    p.add_argument("--noise_sigma", type=float, default=0.0)
    p.add_argument("--noise_prob", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--var_ckpt", type=str, default=None)
    return p.parse_args()


def denorm_depth(x, dmin, dmax):
    return (x * 0.5 + 0.5) * (dmax - dmin) + dmin


def main():
    args = build_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ToFDepthVAR().to(device)
    expected_image = model.var.patch_nums[-1] * model.vae.downsample
    if args.image_size != expected_image:
        print(f"[warn] image_size={args.image_size} not matching VAE/VAR expected {expected_image}. Override to {expected_image}.")
        args.image_size = expected_image

    if args.dataset == "nyu":
        ds = NYUv2ToFDataset(
            data_root=args.data_root,
            json_file=args.json_file,
            split=args.split,
            image_size=args.image_size,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            input_height=args.input_height,
            input_width=args.input_width,
            train_zone_num=args.train_zone_num,
            train_zone_random_offset=args.train_zone_random_offset,
            zone_sample_num=args.zone_sample_num,
            sample_uniform=args.sample_uniform,
            simu_max_distance=args.simu_max_distance,
            drop_hist=args.drop_hist,
            noise_mean=args.noise_mean,
            noise_sigma=args.noise_sigma,
            noise_prob=args.noise_prob,
        )
    else:
        ds = ZJUL5Dataset(
            data_root=args.data_root,
            json_file=args.json_file,
            split=args.split,
            image_size=args.image_size,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt_path = args.vae_ckpt or args.var_ckpt
    if vae_ckpt_path:
        if not osp.exists(vae_ckpt_path) and '://' not in vae_ckpt_path:
            os.system(f"wget {hf_home}/{vae_ckpt_path}")
        ckpt = torch.load(vae_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt:
                ckpt = ckpt["model"]
        missing, unexpected = model.vae.load_state_dict(ckpt, strict=False)
        print(f"[load vae] missing={len(missing)} unexpected={len(unexpected)}")

    model.vae.eval()
    for p in model.vae.parameters():
        p.requires_grad_(False)

    patch_nums = model.var.patch_nums
    seg_lens = [pn * pn for pn in patch_nums]
    seg_offsets = [0]
    for ln in seg_lens[:-1]:
        seg_offsets.append(seg_offsets[-1] + ln)

    def decode_pred_depth(logits):
        pred_BL = logits.argmax(dim=-1)
        pred_ms = []
        for pn, off, ln in zip(patch_nums, seg_offsets, seg_lens):
            pred_ms.append(pred_BL[:, off:off+ln].view(pred_BL.shape[0], pn * pn))
        pred_rec = model.vae.idxBl_to_img(pred_ms, same_shape=True, last_one=True)
        pred_img = pred_rec[:, 0:1].clamp_(-1, 1)
        return denorm_depth(pred_img, args.depth_min, args.depth_max)

    ce_loss = nn.CrossEntropyLoss()

    model.eval()
    tot_loss = 0.0
    tot = 0
    sum_abs_rel = 0.0
    sum_sq_rel = 0.0
    sum_rmse = 0.0
    sum_log10 = 0.0
    sum_d1 = 0.0
    sum_d2 = 0.0
    sum_d3 = 0.0
    with torch.no_grad():
        for batch in dl:
            if "depth" not in batch:
                raise KeyError(f"eval batch missing 'depth'. keys={list(batch.keys())}")
            rgb = (batch.get("rgb") or batch.get("image")).to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            hist = (batch.get("hist") or batch.get("hist_data")).to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            fr = (batch.get("fr") or batch.get("rect_data")).to(device, non_blocking=True)
            logits, gt = model(rgb, hist, mask, depth, fr_BZ4=fr)
            loss = ce_loss(logits.view(-1, logits.shape[-1]), gt.view(-1))
            tot_loss += loss.item() * rgb.shape[0]
            pred_depth = decode_pred_depth(logits)
            gt_depth = denorm_depth(depth[:, 0:1], args.depth_min, args.depth_max)
            if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
                gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)
            valid = gt_depth > 1e-6
            if valid.any():
                diff = pred_depth - gt_depth
                abs_rel = (diff.abs() / gt_depth).masked_select(valid).mean().item()
                sq_rel = ((diff ** 2) / gt_depth).masked_select(valid).mean().item()
                rmse = torch.sqrt((diff ** 2).masked_select(valid).mean()).item()
                log10 = (torch.log10(pred_depth.clamp_min(1e-6)) - torch.log10(gt_depth)).abs().masked_select(valid).mean().item()
                ratio = torch.max(pred_depth / gt_depth, gt_depth / pred_depth)
                d1 = (ratio < 1.25).masked_select(valid).float().mean().item()
                d2 = (ratio < 1.25 ** 2).masked_select(valid).float().mean().item()
                d3 = (ratio < 1.25 ** 3).masked_select(valid).float().mean().item()
                sum_abs_rel += abs_rel * rgb.shape[0]
                sum_sq_rel += sq_rel * rgb.shape[0]
                sum_rmse += rmse * rgb.shape[0]
                sum_log10 += log10 * rgb.shape[0]
                sum_d1 += d1 * rgb.shape[0]
                sum_d2 += d2 * rgb.shape[0]
                sum_d3 += d3 * rgb.shape[0]
            tot += rgb.shape[0]

    avg = tot_loss / max(tot, 1)
    abs_rel_avg = sum_abs_rel / max(tot, 1)
    sq_rel_avg = sum_sq_rel / max(tot, 1)
    rmse_avg = sum_rmse / max(tot, 1)
    log10_avg = sum_log10 / max(tot, 1)
    d1_avg = sum_d1 / max(tot, 1)
    d2_avg = sum_d2 / max(tot, 1)
    d3_avg = sum_d3 / max(tot, 1)
    print(
        f"[eval] loss={avg:.6f} abs_rel={abs_rel_avg:.4f} sq_rel={sq_rel_avg:.4f} "
        f"rmse={rmse_avg:.4f} log10={log10_avg:.4f} d1={d1_avg:.4f} d2={d2_avg:.4f} d3={d3_avg:.4f} "
        f"samples={tot}"
    )


if __name__ == "__main__":
    main()
