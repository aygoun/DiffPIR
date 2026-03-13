"""
Plug-and-play denoiser priors for PnP restoration.

- GaussianDenoiser: simple baseline.
- DRUNetDenoiser: official DPIR DRUNet; loads from Hugging Face (perckle/DPIR) by default.
- DiffBIRDenoiser: DiffBIR Stage-1 SwinIR blind denoiser; loads from Hugging Face
  (lxq007/DiffBIR) by default.  Because the SwinIR is a *blind* restorer (no noise-level
  conditioning), the sigma argument is accepted but not forwarded to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from .backbones.drunet_arch import UNetRes
from .backbones.swinir_arch import SwinIR


class Denoiser(Protocol):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor: ...


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
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            x.shape[1], 1, 1, 1
        )
        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


# Hugging Face DRUNet (perckle/DPIR) – used when no local weights path is given.
HF_DRUNET_REPO = "perckle/DPIR"
HF_DRUNET_COLOR_FILE = "drunet_color.safetensors"


def _load_drunet_weights(
    model: torch.nn.Module, path: str, device: torch.device
) -> None:
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
            _load_drunet_weights(
                self.model, self.weights_path, torch.device(self.device)
            )
        else:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=HF_DRUNET_REPO, filename=HF_DRUNET_COLOR_FILE
                )
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


# ---------------------------------------------------------------------------
# DiffBIR Stage-1 SwinIR denoiser
# ---------------------------------------------------------------------------

HF_DIFFBIR_REPO = "lxq007/DiffBIR"
HF_DIFFBIR_SWINIR_FILE = "general_swinir_v1.ckpt"

# DiffBIR Stage-1 architecture (matches general_swinir_v1.ckpt exactly)
_DIFFBIR_SWINIR_KWARGS: dict = dict(
    img_size=64,
    patch_size=1,
    in_chans=3,
    embed_dim=180,
    depths=[6, 6, 6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.0,
    sf=8,
    img_range=1.0,
    upsampler="nearest+conv",
    resi_connection="1conv",
    unshuffle=True,
    unshuffle_scale=8,
)


def _remap_diffbir_keys(state: dict) -> dict:
    """
    Remap keys from the DiffBIR checkpoint format to our SwinIR layout.

    The official checkpoint (general_swinir_v1.ckpt) was saved with
    DataParallel and a slightly different module layout:
      - All keys are prefixed with "module."
      - conv_first is a Sequential([PixelUnshuffle, Conv2d]), so weights sit
        at "conv_first.1.*"; our model uses a bare Conv2d at "conv_first.*".
      - RSTB stores transformer blocks under "residual_group.blocks.N.*";
        our ModuleList names them "residual_group.N.*" directly.
      - MLP layers are named "mlp.fc1" / "mlp.fc2"; our nn.Sequential uses
        integer indices "mlp.0" / "mlp.3".
      - "attn_mask" buffers are stored in the checkpoint but our implementation
        computes them dynamically at runtime, so we skip them.
    """
    import re

    new_state: dict = {}
    for k, v in state.items():
        # 1. Strip DataParallel wrapper prefix
        if k.startswith("module."):
            k = k[len("module.") :]

        # 2. conv_first: Sequential[PixelUnshuffle(no params), Conv2d] → bare Conv2d
        k = k.replace("conv_first.1.", "conv_first.")

        # 3. RSTB residual_group.blocks.N.* → residual_group.N.*
        k = re.sub(r"(layers\.\d+\.residual_group\.)blocks\.(\d+)\.", r"\1\2.", k)

        # 4. MLP fc1/fc2 → Sequential indices 0/3
        k = k.replace(".mlp.fc1.", ".mlp.0.")
        k = k.replace(".mlp.fc2.", ".mlp.3.")

        # 5. Drop attn_mask buffers — we compute them dynamically per x_size
        if k.endswith(".attn_mask"):
            continue

        new_state[k] = v
    return new_state


def _load_swinir_weights(
    model: torch.nn.Module, path: str, device: torch.device
) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    # DiffBIR / BasicSR checkpoints wrap weights under a 'params' key
    if isinstance(state, dict):
        for key in ("params", "params_ema", "state_dict"):
            if key in state:
                state = state[key]
                break
    state = _remap_diffbir_keys(state)
    model.load_state_dict(state, strict=True)


@dataclass
class DiffBIRDenoiser:
    """
    DiffBIR Stage-1 SwinIR blind denoiser.

    Loads pretrained weights from Hugging Face (lxq007/DiffBIR,
    general_swinir_v1.ckpt) by default.  Set ``weights_path`` to a local
    .ckpt / .pth file to override.

    The SwinIR inside DiffBIR is a *blind* restorer — it does not receive a
    noise-level signal.  The ``sigma`` argument in ``__call__`` is accepted for
    API compatibility with the other denoisers but is not used.

    The model maps an image [B, 3, H, W] ∈ [0, 1] to a cleaned image of the
    same shape via an internal pixel-unshuffle → transformer → pixel-shuffle
    pipeline (net scale 1:1).
    """

    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.model = SwinIR(**_DIFFBIR_SWINIR_KWARGS).to(self.device)

        if self.weights_path:
            _load_swinir_weights(
                self.model, self.weights_path, torch.device(self.device)
            )
        else:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id=HF_DIFFBIR_REPO, filename=HF_DIFFBIR_SWINIR_FILE
                )
                _load_swinir_weights(self.model, path, torch.device(self.device))
            except Exception as e:
                raise RuntimeError(
                    f"DiffBIRDenoiser: could not load weights from {HF_DIFFBIR_REPO}. "
                    f"Install: pip install huggingface_hub. Error: {e}"
                ) from e

        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:  # noqa: ARG002
        """
        Apply DiffBIR Stage-1 restoration.

        Args:
            x:     Input tensor [B, 3, H, W] in [0, 1].
            sigma: Noise level (accepted for API compatibility; not used by the
                   blind SwinIR — pass any value).

        Returns:
            Restored tensor [B, 3, H, W] in [0, 1].
        """
        orig_device = x.device
        # Clamp before the model: FFT data steps can push values outside [0,1]
        # and the SwinIR produces garbage when given out-of-distribution inputs.
        out = self.model(x.clamp(0.0, 1.0).to(self.device))
        return out.clamp(0.0, 1.0).to(orig_device)
