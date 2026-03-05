from __future__ import annotations

"""
Super-resolution method helpers for experiments.

Currently the main comparison entrypoints are the CLI scripts which call
`main_ddpir` with different configurations. This module is reserved for
future finer-grained, per-image APIs.
"""

from dataclasses import dataclass

from .common import ImageResult, MethodConfig


@dataclass
class SRPlaceholder:
    """Placeholder to keep the API surface for future extensions."""

    name: str = "sr_placeholder"


def run_diffpir_sr(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Single-image DiffPIR SR runner (to be implemented by you).

    Once implemented, it should:
    - Load the HR image from `img_path`.
    - Generate the LR observation according to `cfg`.
    - Run the DiffPIR sampling loop for SR.
    - Return an `ImageResult` with PSNR / PSNR-Y / LPIPS.
    """

    raise NotImplementedError("run_diffpir_sr is not implemented yet.")


def run_dps_sr(img_path: str, cfg: MethodConfig, mode: str = "DPS_y0") -> ImageResult:
    """
    Single-image DPS SR runner (to be implemented by you).

    `mode` should typically be either \"DPS_y0\" or \"DPS_yt\" and should
    control how the data-consistency term is applied in the loop.
    """

    raise NotImplementedError("run_dps_sr is not implemented yet.")


def run_pnp_drunet_sr(img_path: str, cfg: MethodConfig) -> ImageResult:
    """
    Plug-and-play SR runner using DRUNet or another `Denoiser` prior.

    Recommended design:
    - Construct a DRUNetDenoiser (or other denoiser) instance.
    - Implement an iterative PnP scheme:
      * data step using the SR forward model
      * prior step using the denoiser
    - Return an `ImageResult` with metrics.
    """

    raise NotImplementedError("run_pnp_drunet_sr is not implemented yet.")


