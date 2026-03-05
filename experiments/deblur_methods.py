"""
Deblurring method helpers for experiments.

These functions are intended to mirror the SR helpers from `sr_methods.py`,
but for the deblurring task.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import ImageResult, MethodConfig


@dataclass
class DeblurPlaceholder:
    """Placeholder to keep the API surface for future extensions."""

    name: str = "deblur_placeholder"


def run_diffpir_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR deblurring runner (to be implemented by you).

    It should:
    - Load the sharp ground-truth image from `img_path`.
    - Apply the blur kernel and noise according to `cfg`.
    - Run the DiffPIR sampling loop for deblurring.
    - Return an `ImageResult` with PSNR / LPIPS metrics.
    """

    raise NotImplementedError("run_diffpir_deblur is not implemented yet.")


def run_dps_deblur(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult:
    """
    Single-image DPS deblurring runner (to be implemented by you).

    `mode` should typically be either \"DPS_y0\" or \"DPS_yt\" and should
    control how the data-consistency term is applied in the loop.
    """

    raise NotImplementedError("run_dps_deblur is not implemented yet.")


def run_pnp_drunet_deblur(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play deblurring runner using DRUNet or another `Denoiser` prior.

    Recommended design:
    - Construct a DRUNetDenoiser (or other denoiser) instance.
    - Implement an iterative PnP scheme:
      * data step using the blur operator
      * prior step using the denoiser
    - Return an `ImageResult` with metrics.
    """

    raise NotImplementedError("run_pnp_drunet_deblur is not implemented yet.")


