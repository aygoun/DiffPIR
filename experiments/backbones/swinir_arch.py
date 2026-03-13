"""
SwinIR architecture, compatible with DiffBIR Stage-1 weights.

Adapted from:
  - SwinIR (Liang et al., 2021): https://github.com/JingyunLiang/SwinIR (Apache-2.0)
  - DiffBIR (Lin et al., 2023):  https://github.com/XPixelGroup/DiffBIR  (Apache-2.0)

Code from Github: https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split (B, H, W, C) → (num_windows*B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Merge (num_windows*B, window_size, window_size, C) → (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ---------------------------------------------------------------------------
# Window Attention
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        rpe = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpe = (
            rpe.view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )
        attn = attn + rpe

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer Block
# ---------------------------------------------------------------------------


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # attn_mask is computed dynamically in forward() for whatever x_size arrives;
        # we keep a small dict cache so we only build it once per unique resolution.
        self._mask_cache: dict[tuple[int, int], Optional[torch.Tensor]] = {}

    def _get_attn_mask(
        self, x_size: tuple[int, int], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Return the shifted-window attention mask for *x_size*, cached per resolution."""
        if self.shift_size == 0:
            return None
        if x_size not in self._mask_cache:
            H, W = x_size
            img_mask = torch.zeros(1, H, W, 1, device=device)
            for h_i, h in enumerate(
                (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
            ):
                for w_i, w in enumerate(
                    (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None),
                    )
                ):
                    img_mask[:, h, w, :] = h_i * 3 + w_i
            mask_windows = window_partition(img_mask, self.window_size).view(
                -1, self.window_size**2
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
            self._mask_cache[x_size] = attn_mask
        return self._mask_cache[x_size]

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, _L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        attn_mask = self._get_attn_mask(x_size, x.device)
        shifted_x = (
            torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.shift_size > 0
            else x
        )
        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size**2, C
        )
        attn_windows = self.attn(x_windows, mask=attn_mask).view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        x = (
            torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if self.shift_size > 0
            else shifted_x
        )
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# Patch Embed / UnEmbed
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[type] = None,
    ) -> None:
        super().__init__()
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int = 96) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


# ---------------------------------------------------------------------------
# RSTB – Residual Swin Transformer Block
# ---------------------------------------------------------------------------


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.LayerNorm,
        img_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.residual_group = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim
        )
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        res = x
        for blk in self.residual_group:
            x = blk(x, x_size)
        x = self.patch_embed(self.conv(self.patch_unembed(x, x_size)))
        return x + res


# ---------------------------------------------------------------------------
# Upsamplers
# ---------------------------------------------------------------------------


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        layers: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                layers += [
                    nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            layers += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*layers)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale: int, num_feat: int, num_out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


# ---------------------------------------------------------------------------
# SwinIR (DiffBIR-compatible)
# ---------------------------------------------------------------------------


class SwinIR(nn.Module):
    """
    SwinIR with optional pixel-unshuffle preprocessing (DiffBIR Stage-1).

    DiffBIR configuration (general_swinir_v1.ckpt):
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6,6,6,6,6,6,6,6], num_heads=[6,6,6,6,6,6,6,6],
        window_size=8, mlp_ratio=2, sf=8, img_range=1.0,
        upsampler="nearest+conv", resi_connection="1conv",
        unshuffle=True, unshuffle_scale=8

    With unshuffle=True the model's effective spatial scale is 1:1 (same HxW in/out).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Optional[list[int]] = None,
        num_heads: Optional[list[int]] = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
        # DiffBIR-specific
        sf: int = 1,
        unshuffle: bool = False,
        unshuffle_scale: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if depths is None:
            depths = [6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6]

        num_feat = 64
        num_out_ch = in_chans
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.sf = sf
        self.unshuffle = unshuffle
        self.unshuffle_scale = unshuffle_scale

        self.mean = (
            torch.zeros(1, 1, 1, 1)
            if img_range == 1.0
            else torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        )

        # Shallow feature extraction (with optional pixel-unshuffle)
        if unshuffle:
            self.unshuffle_op = nn.PixelUnshuffle(unshuffle_scale)
            self.conv_first = nn.Conv2d(
                in_chans * unshuffle_scale**2, embed_dim, 3, 1, 1
            )
        else:
            self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Transformer backbone
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList(
            [
                RSTB(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    img_size=img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    resi_connection=resi_connection,
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = norm_layer(self.num_features)

        # Residual conv after backbone (always present)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Output head / upsampler
        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        elif upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # Build the nearest+conv upsample chain for sf=2,4,8
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if sf >= 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if sf >= 8:
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            # Classic denoising/JPEG artefact removal (no upsampling)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(x)
        x = (x - self.mean) * self.img_range

        # Pixel-unshuffle preprocessing (DiffBIR)
        if self.unshuffle:
            x = self.unshuffle_op(x)

        _, _, H, W = x.shape  # H, W after unshuffle (or original for no-unshuffle)
        x = self.check_image_size(x)  # pad so H, W divisible by window_size

        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first

        if self.upsampler == "pixelshuffle":
            x = self.conv_last(self.upsample(self.conv_before_upsample(res)))
        elif self.upsampler == "pixelshuffledirect":
            x = self.upsample(res)
        elif self.upsampler == "nearest+conv":
            res = self.conv_before_upsample(res)
            res = self.lrelu(
                self.conv_up1(F.interpolate(res, scale_factor=2, mode="nearest"))
            )
            if self.sf >= 4:
                res = self.lrelu(
                    self.conv_up2(F.interpolate(res, scale_factor=2, mode="nearest"))
                )
            if self.sf >= 8:
                res = self.lrelu(
                    self.conv_up3(F.interpolate(res, scale_factor=2, mode="nearest"))
                )
            x = self.conv_last(self.lrelu(self.conv_hr(res)))
        else:
            x = x + self.conv_last(res)

        # Crop away padding.
        # For unshuffle=True: H and W are already the pixel-unshuffled dims;
        # the sf-fold upsampler restores exactly H*sf = H_orig. (sf == unshuffle_scale in DiffBIR)
        # For plain SR / denoising: H*upscale or H*1.
        if self.unshuffle:
            x = x[:, :, : H * self.sf, : W * self.sf]
        else:
            x = x[:, :, : H * self.upscale, : W * self.upscale]

        x = x / self.img_range + self.mean
        return x
