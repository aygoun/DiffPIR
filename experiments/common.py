from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence
import os

import numpy as np
import torch

from utils import utils_image as util


@dataclass
class MethodConfig:
    """Configuration for a single restoration method run."""

    name: str
    task: str  # "sr", "deblur", "inpaint"
    generate_mode: str  # "DiffPIR", "DPS_y0", "DPS_yt", etc.
    lambda_: float
    zeta: float
    sf: int = 4
    # Optional extras are kept in a free-form dict to avoid a huge signature
    extra: Dict[str, object] | None = None


@dataclass
class ImageResult:
    """Per-image metrics for a restoration method."""

    psnr: float
    psnr_y: float | None = None
    lpips: float | None = None


@dataclass
class RunResult:
    """Aggregated metrics over a dataset for one method."""

    method: str
    task: str
    sf: int
    image_results: Dict[str, ImageResult]

    @property
    def average_psnr(self) -> float:
        values = [r.psnr for r in self.image_results.values()]
        return float(np.mean(values)) if values else float("nan")

    @property
    def average_psnr_y(self) -> float | None:
        values = [r.psnr_y for r in self.image_results.values() if r.psnr_y is not None]
        return float(np.mean(values)) if values else None

    @property
    def average_lpips(self) -> float | None:
        values = [r.lpips for r in self.image_results.values() if r.lpips is not None]
        return float(np.mean(values)) if values else None


def load_image_paths(root: str) -> List[str]:
    """Return sorted image paths from a dataset root."""

    paths = util.get_image_paths(root)
    return sorted(paths)


def ensure_output_dir(root: str) -> None:
    """Create an output directory if it does not already exist."""

    util.mkdir(root)


def run_experiment(
    *,
    method_config: MethodConfig,
    image_paths: Sequence[str],
    method_fn: Callable[[str, MethodConfig], ImageResult],
    output_root: str | None = None,
) -> RunResult:
    """
    Run a single method over all images in `image_paths`.

    The `method_fn` is responsible for:
    - Loading the image.
    - Running the restoration method.
    - Saving any visual outputs to disk (if desired).
    - Returning per-image metrics.
    """

    if output_root is not None:
        ensure_output_dir(output_root)

    image_results: Dict[str, ImageResult] = {}

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        image_results[img_name] = method_fn(img_path, method_config)

    return RunResult(
        method=method_config.name,
        task=method_config.task,
        sf=method_config.sf,
        image_results=image_results,
    )

