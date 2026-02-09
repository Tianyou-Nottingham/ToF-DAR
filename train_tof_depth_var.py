import argparse
import os
import os.path as osp
import time
import json
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from models.tof_var import ToFDepthVAR
from utils.zjul5_dataset import ZJUL5Dataset
from utils.nyu_dataset import NYUv2ToFDataset
from utils.model_io import load_checkpoint, save_checkpoint, save_weights


import wandb


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--dataset", type=str, default="nyu", choices=["zjul5", "nyu"])
    p.add_argument(
        "--dataset_eval",
        type=str,
        default=None,
        choices=[None, "zjul5", "nyu"],
        help="evaluation dataset; defaults to --dataset when not set",
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--json_file", type=str, required=True)
    p.add_argument("--data_root_eval", type=str, default=None)
    p.add_argument("--json_file_eval", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")

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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--weight_decay", default=0.1, type=float)  # backward compat

    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--var_ckpt", type=str, default=None)  # backward compat

    p.add_argument("--vis_every", type=int, default=0)
    p.add_argument("--vis_dir", type=str, default="vis_depth")
    p.add_argument("--log_every", type=int, default=100)

    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--best_metric", type=str, default="rmse",
                   choices=["rmse", "abs_rel", "sq_rel", "log10", "d1", "d2", "d3"])

    p.add_argument("--zju_data_root", type=str, default=None)
    p.add_argument("--zju_json_file", type=str, default=None)
    p.add_argument("--zju_eval_split", type=str, default="test")

    p.add_argument("--logging", action="store_true", default=True)
    return p.parse_args()


def build_dataset(args, split, dataset):
    if dataset == "nyu":
        return NYUv2ToFDataset(
            data_root=args.data_root if split == "train" else args.data_root_eval ,
            json_file=args.json_file if split == "train" else args.json_file_eval ,
            split=split,
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
            drop_hist=args.drop_hist if split == args.split else 0.0,
            noise_mean=args.noise_mean if split == args.split else 0.0,
            noise_sigma=args.noise_sigma if split == args.split else 0.0,
            noise_prob=args.noise_prob if split == args.split else 0.0,
        )
    if dataset == "zjul5":
        return ZJUL5Dataset(
            data_root=args.data_root if split == "train" else args.data_root_eval ,
            json_file=args.json_file if split == "train" else args.json_file_eval,
            split=split,
            image_size=args.image_size,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )
    raise ValueError(f"Unknown dataset: {dataset}. Expected one of ['nyu', 'zjul5']")


def build_model(args, device):
    model = ToFDepthVAR().to(device)
    expected_image = model.var.patch_nums[-1] * model.vae.downsample
    if args.image_size != expected_image:
        print(f"[warn] image_size={args.image_size} not matching VAE/VAR expected {expected_image}. Override to {expected_image}.")
        args.image_size = expected_image
    return model


def load_vae(model, args):
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt_path = args.vae_ckpt or args.var_ckpt
    if not vae_ckpt_path:
        return
    if not osp.exists(vae_ckpt_path) and "://" not in vae_ckpt_path:
        os.system(f"wget {hf_home}/{vae_ckpt_path}")
    ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
    missing, unexpected = model.vae.load_state_dict(ckpt, strict=False)
    print(f"[load vae] missing={len(missing)} unexpected={len(unexpected)}")


def freeze_vae(model):
    model.vae.eval()
    for p in model.vae.parameters():
        p.requires_grad_(False)


def denorm_depth(x, dmin, dmax):
    return (x * 0.5 + 0.5) * (dmax - dmin) + dmin


def decode_pred_depth(model, logits, seg_offsets, seg_lens):
    pred_BL = logits.argmax(dim=-1)
    pred_ms = []
    for off, ln, pn in zip(seg_offsets, seg_lens, model.var.patch_nums):
        pred_ms.append(pred_BL[:, off:off+ln].view(pred_BL.shape[0], pn * pn))
    pred_rec = model.vae.idxBl_to_img(pred_ms, same_shape=True, last_one=True)
    pred_img = pred_rec[:, 0:1].clamp_(-1, 1)
    return pred_img


def compute_metrics(pred_depth, gt_depth):
    valid = gt_depth > 1e-6
    if not valid.any():
        return None
    diff = pred_depth - gt_depth
    abs_rel = (diff.abs() / gt_depth).masked_select(valid).mean().item()
    sq_rel = ((diff ** 2) / gt_depth).masked_select(valid).mean().item()
    rmse = torch.sqrt((diff ** 2).masked_select(valid).mean()).item()
    log10 = (torch.log10(pred_depth.clamp_min(1e-6)) - torch.log10(gt_depth)).abs().masked_select(valid).mean().item()
    ratio = torch.max(pred_depth / gt_depth, gt_depth / pred_depth)
    d1 = (ratio < 1.25).masked_select(valid).float().mean().item()
    d2 = (ratio < 1.25 ** 2).masked_select(valid).float().mean().item()
    d3 = (ratio < 1.25 ** 3).masked_select(valid).float().mean().item()
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "log10": log10,
        "d1": d1,
        "d2": d2,
        "d3": d3,
    }


def evaluate(model, loader, args, seg_offsets, seg_lens, device, tag="eval"):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    tot_loss = 0.0
    tot = 0
    sums = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "log10", "d1", "d2", "d3"]}
    with torch.no_grad():
        for batch in loader:
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

            pred_depth = decode_pred_depth(model, logits, seg_offsets, seg_lens)
            pred_depth = denorm_depth(pred_depth, args.depth_min, args.depth_max)
            gt_depth = denorm_depth(depth[:, 0:1], args.depth_min, args.depth_max)
            if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
                gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False)

            metrics = compute_metrics(pred_depth, gt_depth)
            if metrics is not None:
                for k in sums:
                    sums[k] += metrics[k] * rgb.shape[0]
            tot += rgb.shape[0]

    
    avg = tot_loss / max(tot, 1)
    metrics_avg = {k: sums[k] / max(tot, 1) for k in sums}
    print(
        f"[{tag}] loss={avg:.6f} abs_rel={metrics_avg['abs_rel']:.4f} sq_rel={metrics_avg['sq_rel']:.4f} "
        f"rmse={metrics_avg['rmse']:.4f} log10={metrics_avg['log10']:.4f} d1={metrics_avg['d1']:.4f} "
        f"d2={metrics_avg['d2']:.4f} d3={metrics_avg['d3']:.4f} samples={tot}"
    )
    
    return avg, metrics_avg



def save_best(model, optimizer, epoch, global_step, args, tag):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_best_{tag}.pth")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"[ckpt] saved best {ckpt_path}")


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.batch_size}-tep{args.epochs}-lr{args.lr}-wd{args.weight_decay}-{uuid.uuid4()}"

    if args.logging:
        wandb.init(project="tof_depth_var", name=f"run-{int(time.time())}", config=args)
        with open(f"{wandb.run.dir}/run_args.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    model = build_model(args, device)

    train_ds = build_dataset(args, args.split, args.dataset)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    eval_dataset = args.dataset_eval or args.dataset
    test_ds = build_dataset(args, args.eval_split, eval_dataset)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    load_vae(model, args)
    freeze_vae(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95))
    ce_loss = nn.CrossEntropyLoss()

    best_loss = np.inf

    patch_nums = model.var.patch_nums
    seg_lens = [pn * pn for pn in patch_nums]
    seg_offsets = [0]
    for ln in seg_lens[:-1]:
        seg_offsets.append(seg_offsets[-1] + ln)

    if args.vis_every and args.vis_every > 0:
        os.makedirs(args.vis_dir, exist_ok=True)
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, fpath=args.resume, optimizer=optimizer)
    else:
        start_epoch = 0
    global_step = start_epoch + args.epochs

    total_steps = args.epochs * len(train_dl)
    start_time = time.time()

    model.train()
    for ep in range(start_epoch, args.epochs):
        ep_start = time.time()
        ep_loss = 0.0
        ep_n = 0
        for i, batch in tqdm(enumerate(train_dl), desc=f"Epoch: {ep + 1}/{args.epochs}. Loop: Train", total=len(train_dl)):
            rgb = (batch.get("rgb") or batch.get("image")).to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            hist = (batch.get("hist") or batch.get("hist_data")).to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            fr = (batch.get("fr") or batch.get("rect_data")).to(device, non_blocking=True)

            logits, gt = model(rgb, hist, mask, depth, fr_BZ4=fr)
            loss = ce_loss(logits.view(-1, logits.shape[-1]), gt.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item() * rgb.shape[0]
            ep_n += rgb.shape[0]

            if args.log_every > 0 and (global_step % args.log_every == 0):
                elapsed = time.time() - start_time
                steps_done = max(global_step + 1, 1)
                step_time = elapsed / steps_done
                remain = max(total_steps - steps_done, 0)
                eta_min = (remain * step_time) / 60.0
                print(f"\n[train] step={global_step} loss={loss.item():.4f} avg_step={step_time:.3f}s ETA={eta_min:.1f}m")

            if args.logging and global_step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": ep}, step=global_step)

            if args.vis_every and global_step % args.vis_every == 0:
                with torch.no_grad():
                    pred_depth = decode_pred_depth(model, logits, seg_offsets, seg_lens)
                    gt_depth = depth[:, 0:1].clamp_(-1, 1)
                    pred_img = (pred_depth[:, 0:1].clamp_(-1, 1) + 1.0) * 0.5
                    gt_img = (gt_depth + 1.0) * 0.5
                    for b in range(min(pred_img.shape[0], 2)):
                        p = (pred_img[b, 0].cpu().numpy() * 255.0).astype(np.uint8)
                        g = (gt_img[b, 0].cpu().numpy() * 255.0).astype(np.uint8)
                        Image.fromarray(p).save(os.path.join(args.vis_dir, f"ep{ep:03d}_step{global_step:06d}_b{b}_pred.png"))
                        Image.fromarray(g).save(os.path.join(args.vis_dir, f"ep{ep:03d}_step{global_step:06d}_b{b}_gt.png"))

            global_step += 1

        ep_time = time.time() - ep_start
        avg_step = ep_time / max(len(train_dl), 1)
        remain_epochs = args.epochs - (ep + 1)
        eta_min = (remain_epochs * ep_time) / 60.0
        print(
            f"[train] epoch {ep/args.epochs:.3f} loss={ep_loss / max(ep_n,1):.6f} samples={ep_n} "
            f"ep_time={ep_time:.1f}s avg_step={avg_step:.3f}s ETA={eta_min:.1f}m"
        )

        if test_dl is not None and (ep + 1) % args.eval_every == 0:
            avg, metrics = evaluate(model, test_dl, args, seg_offsets, seg_lens, device, tag=f"{eval_dataset}:{args.eval_split}")
            if args.logging:  
                tag=f"{eval_dataset}:{args.eval_split}"  
                wandb.log({f"{tag}_loss": avg, **{f"{tag}_{k}": v for k, v in metrics.items()}})
                unwrap_model = model.module if hasattr(model, "module") else model
                save_checkpoint(unwrap_model, optimizer, ep, fpath=f"checkpoints/{run_id}_latest.pt")
                save_weights(unwrap_model, fpath=f"weights/{run_id}_latest.pt")
                if metrics['rmse'] < best_loss:
                    save_checkpoint(unwrap_model, optimizer, ep, fpath=f"checkpoints/{run_id}_best.pt")
                    save_weights(unwrap_model, fpath=f"weights/{run_id}_best.pt")
                    best_loss = metrics['rmse']
        model.train()
        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            unwrap_model = model.module if hasattr(model, "module") else model
            save_checkpoint(unwrap_model, optimizer, ep + 1, fpath=f"checkpoints/{run_id}_ep{ep+1:03d}.pt")
            save_weights(unwrap_model, fpath=f"weights/{run_id}_ep{ep+1:03d}.pt")
        

    if args.zju_data_root and args.zju_json_file:
        zju_args = argparse.Namespace(**vars(args))
        zju_args.data_root = args.zju_data_root
        zju_args.json_file = args.zju_json_file
        zju_ds = ZJUL5Dataset(
            data_root=zju_args.data_root,
            json_file=zju_args.json_file,
            split=args.zju_eval_split,
            image_size=args.image_size,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )
        zju_dl = DataLoader(zju_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        evaluate(model, zju_dl, args, seg_offsets, seg_lens, device, tag=f"zjul5:{args.zju_eval_split}")


def main():
    args = build_args()
    train(args)


if __name__ == "__main__":
    main()
