"""
Plug-and-play denoiser priors for PnP restoration.

- GaussianDenoiser: simple baseline.
- DRUNetDenoiser: official DPIR DRUNet; loads from Hugging Face (perckle/DPIR) by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from .drunet_arch import UNetRes


class Denoiser(Protocol):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        ...


@dataclass
class GaussianDenoiser:
    """Gaussian-smoothing baseline."""

    kernel_size: int = 5
    sigma: float = 1.0

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        radius = self.kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


# Hugging Face DRUNet (perckle/DPIR) – used when no local weights path is given.
HF_DRUNET_REPO = "perckle/DPIR"
HF_DRUNET_COLOR_FILE = "drunet_color.safetensors"


def _load_drunet_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    path_lower = path.lower()
    if path_lower.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state = load_file(path, device=str(device))
        except ImportError:
            raise ImportError("Loading .safetensors requires: pip install safetensors")
    else:
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state, strict=True)


@dataclass
class DRUNetDenoiser:
    """
    DPIR DRUNet denoiser. Loads pretrained weights from Hugging Face (perckle/DPIR)
    by default. Set weights_path to a local .safetensors or .pth file to override.
    """

    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        # Match DPIR exactly (see cszn/DPIR main_dpir_deblur.py):
        # UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64,128,256,512], nb=4, act_mode='R', ...)
        self.model = UNetRes(
            in_nc=4,  # RGB + sigma map
            out_nc=3,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        ).to(self.device)

        if self.weights_path:
            _load_drunet_weights(self.model, self.weights_path, torch.device(self.device))
        else:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=HF_DRUNET_REPO, filename=HF_DRUNET_COLOR_FILE)
                _load_drunet_weights(self.model, path, torch.device(self.device))
            except Exception as e:
                raise RuntimeError(
                    f"DRUNet: could not load default weights from {HF_DRUNET_REPO}. "
                    f"Install: pip install huggingface_hub safetensors. Error: {e}"
                ) from e

        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Denoise at noise level sigma where sigma is in [0, 1] for image range [0, 1].

        DPIR/DRUNet expects the sigma map concatenated as an extra channel, also
        expressed in [0, 1] (not multiplied by 255).
        """
        orig_device = x.device
        x = x.to(self.device)
        b = x.shape[0]
        # Noise level map as extra channel (DPIR convention: sigma in [0, 1])
        sigma_val = float(sigma)
        sigma_map = torch.full(
            (b, 1, x.shape[2], x.shape[3]),
            sigma_val,
            dtype=x.dtype,
            device=x.device,
        )
        x_in = torch.cat([x, sigma_map], dim=1)
        out = self.model(x_in)
        return out.to(orig_device)
