"""
Simple plug-and-play baselines using an abstract denoiser prior.

This module is intentionally lightweight: it defines a generic `Denoiser`
interface and a very simple Gaussian-smoothing denoiser as a placeholder.

To use a real DRUNet prior, implement `DRUNetDenoiser` that follows the same
API and loads weights from an external source (e.g. DPIR/USRNet repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class Denoiser(Protocol):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        ...


@dataclass
class GaussianDenoiser:
    """Very simple Gaussian-smoothing baseline."""

    kernel_size: int = 5
    sigma: float = 1.0

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        # x: (B, C, H, W)
        if self.kernel_size <= 1:
            return x

        radius = self.kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)

        padding = radius
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


# Placeholder hooks for DRUNet-style priors.

@dataclass
class DRUNetDenoiser:
    """
    Placeholder for a DRUNet prior.

    Implementors should:
    - load the DRUNet architecture and weights in `__post_init__`.
    - implement `__call__` to run the denoiser at noise level `sigma`.
    """

    weights_path: str = ""

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "DRUNetDenoiser is a placeholder. "
            "Please plug in a concrete DRUNet implementation and weights."
        )

