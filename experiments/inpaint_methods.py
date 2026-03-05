"""
Inpainting method helpers for experiments.

These functions are intended to mirror the SR helpers from `sr_methods.py`,
but for the inpainting task.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import ImageResult, MethodConfig


@dataclass
class InpaintPlaceholder:
    """Placeholder to keep the API surface for future extensions."""

    name: str = "inpaint_placeholder"


def run_diffpir_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR inpainting runner (to be implemented by you).

    It should:
    - Load the ground-truth image from `img_path`.
    - Generate a mask and apply it according to `cfg`.
    - Run the DiffPIR sampling loop for inpainting.
    - Return an `ImageResult` with PSNR / LPIPS metrics.
    """

    raise NotImplementedError("run_diffpir_inpaint is not implemented yet.")


def run_dps_inpaint(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult:
    """
    Single-image DPS inpainting runner (to be implemented by you).

    `mode` should typically be either \"DPS_y0\" or \"DPS_yt\" and should
    control how the data-consistency term is applied in the loop.
    """

    raise NotImplementedError("run_dps_inpaint is not implemented yet.")


def run_pnp_drunet_inpaint(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play inpainting runner using DRUNet or another `Denoiser` prior.

    Recommended design:
    - Construct a DRUNetDenoiser (or other denoiser) instance.
    - Implement an iterative PnP scheme:
      * data step enforcing consistency on known pixels
      * prior step using the denoiser
    - Return an `ImageResult` with metrics.
    """

    raise NotImplementedError("run_pnp_drunet_inpaint is not implemented yet.")


