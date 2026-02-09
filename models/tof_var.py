from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .var import VAR
from .vqvae import VQVAE


class ToFInitEncoder(nn.Module):
    def __init__(self, z_channels=32, base_channels=64):
        super().__init__()
        self.tof_proj = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=1, stride=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, z_channels, kernel_size=3, stride=1, padding=1),
        )

    def _build_tof_map(
        self,
        hist_BZ2: torch.Tensor,
        mask_BZ: torch.Tensor,
        fr_BZ4: Optional[torch.Tensor],
        H: int,
        W: int,
    ) -> torch.Tensor:
        B, Z, _ = hist_BZ2.shape
        tof_mean = hist_BZ2[..., 0]
        tof_var = hist_BZ2[..., 1]
        if mask_BZ is not None:
            tof_mean = tof_mean * mask_BZ.to(tof_mean.dtype)
            tof_var = tof_var * mask_BZ.to(tof_var.dtype)
        if fr_BZ4 is None:
            tof_map = torch.stack([tof_mean, tof_var], dim=1).view(B, 2, 8, 8)
            return tof_map
        tof_dense = torch.zeros((B, 2, H, W), dtype=tof_mean.dtype, device=tof_mean.device)
        for b in range(B):
            for i in range(Z):
                if mask_BZ is not None and not mask_BZ[b, i]:
                    continue
                sy, sx, ey, ex = fr_BZ4[b, i].to(torch.int64).tolist()
                sy = max(sy, 0)
                sx = max(sx, 0)
                ey = min(ey, H)
                ex = min(ex, W)
                if ey <= sy or ex <= sx:
                    continue
                tof_dense[b, 0, sy:ey, sx:ex] = tof_mean[b, i]
                tof_dense[b, 1, sy:ey, sx:ex] = tof_var[b, i]
        tof_map = F.interpolate(tof_dense, size=(8, 8), mode="area")
        return tof_map

    def forward(
        self,
        hist_BZ2: torch.Tensor,
        mask_BZ: torch.Tensor,
        fr_BZ4: Optional[torch.Tensor],
        H: int,
        W: int,
    ) -> torch.Tensor:
        # hist_BZ2: (B, 64, 2) mean/variance, mask_BZ: (B, 64)
        tof_map = self._build_tof_map(hist_BZ2, mask_BZ, fr_BZ4, H, W)
        tof_map = torch.log1p(torch.clamp(tof_map, min=0))
        return self.tof_proj(tof_map)


class ToFDepthVAR(nn.Module):
    def __init__(
        self,
        patch_nums=(8, 10, 13, 16),
        vocab_size=4096,
        z_channels=32,
        ch=160,
        share_quant_resi=4,
        var_depth=16,
        var_shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=True,
        fused_if_available=True,
    ):
        super().__init__()
        self.patch_nums = patch_nums
        self.vae = VQVAE(
            vocab_size=vocab_size,
            z_channels=z_channels,
            ch=ch,
            in_channels=3,
            test_mode=False,
            share_quant_resi=share_quant_resi,
            v_patch_nums=patch_nums,
        )
        width = var_depth * 64
        self.var = VAR(
            vae_local=self.vae,
            num_classes=1,
            depth=var_depth,
            embed_dim=width,
            num_heads=var_depth,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1 * var_depth / 24,
            norm_eps=1e-6,
            shared_aln=var_shared_aln,
            cond_drop_rate=0.0,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        )
        self.var.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)
        self.tof_init = ToFInitEncoder(z_channels=z_channels)
        self.cond_proj = nn.Linear(z_channels, width)

    def _expected_image_size(self) -> int:
        return self.patch_nums[-1] * self.vae.downsample

    def _resize_to_expected(self, x: torch.Tensor) -> torch.Tensor:
        exp = self._expected_image_size()
        if x.shape[-2:] != (exp, exp):
            x = F.interpolate(x, size=(exp, exp), mode="bilinear", align_corners=False)
        return x

    def encode_depth_tokens(self, depth_B1HW: torch.Tensor):
        with torch.no_grad():
            depth_B1HW = self._resize_to_expected(depth_B1HW)
            depth_B3HW = depth_B1HW.repeat(1, 3, 1, 1)
            gt_idx_Bl = self.vae.img_to_idxBl(depth_B3HW)
            x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
        return gt_BL, x_BLCv_wo_first_l

    def encode_rgb_condition(self, rgb_B3HW: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            rgb_B3HW = self._resize_to_expected(rgb_B3HW)
            rgb_idx_Bl = self.vae.img_to_idxBl(rgb_B3HW)
            idx_all = torch.cat(rgb_idx_Bl, dim=1)  # (B, L)
            emb = self.vae.quantize.embedding(idx_all)  # (B, L, Cvae)
            cond = emb.mean(dim=1)  # (B, Cvae)
        return cond

    def forward(
        self,
        rgb_B3HW: torch.Tensor,
        hist_BZ2: torch.Tensor,
        mask_BZ: torch.Tensor,
        depth_B1HW: torch.Tensor,
        fr_BZ4: Optional[torch.Tensor] = None,
    ):
        B, _, H, W = rgb_B3HW.shape
        init_h_BChw = self.tof_init(hist_BZ2, mask_BZ, fr_BZ4, H, W)
        cond_BD = self.encode_rgb_condition(rgb_B3HW)
        cond_BD = self.cond_proj(cond_BD)
        gt_BL, x_BLCv_wo_first_l = self.encode_depth_tokens(depth_B1HW)
        label_B = torch.zeros(rgb_B3HW.shape[0], dtype=torch.long, device=rgb_B3HW.device)
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l, cond_BD=cond_BD, init_h_BChw=init_h_BChw)
        return logits_BLV, gt_BL

    @torch.no_grad()
    def infer(
        self,
        rgb_B3HW: torch.Tensor,
        hist_BZ2: torch.Tensor,
        mask_BZ: torch.Tensor,
        fr_BZ4: Optional[torch.Tensor] = None,
        top_k=0,
        top_p=0.0,
    ):
        B, _, H, W = rgb_B3HW.shape
        init_h_BChw = self.tof_init(hist_BZ2, mask_BZ, fr_BZ4, H, W)
        cond_BD = self.encode_rgb_condition(rgb_B3HW)
        cond_BD = self.cond_proj(cond_BD)
        return self.var.autoregressive_infer_with_prior(init_h_BChw, cond_BD, top_k=top_k, top_p=top_p)

    def get_config(self) -> Dict[str, object]:
        return {
            "patch_nums": self.patch_nums,
            "vocab_size": self.vae.vocab_size,
            "z_channels": self.vae.Cvae,
        }
