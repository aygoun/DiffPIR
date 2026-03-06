from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence
import os
import yaml
import numpy as np
import torch

from utils import utils_image as util


@dataclass
class MethodConfig:
    """Configuration for a single restoration method run."""

    task: str  # "sr", "deblur", "inpaint"
    generate_mode: str  # "DiffPIR", "DPS_y0", "DPS_yt", etc.
    lambda_: float
    zeta: float
    sf: int = 1
    # Optional extras are kept in a free-form dict to avoid a huge signature
    extra: Dict[str, object] | None = None

    @staticmethod
    def load_from_yaml(yaml_path: str) -> MethodConfig:
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        return MethodConfig(
            task=cfg["task"],
            generate_mode=cfg["generate_mode"],
            lambda_=cfg["lambda_"],
            zeta=cfg["zeta"],
            sf=cfg["sf"],
            # Extract all keys not part of the core MethodConfig and aggregate in extra
            extra={
                k: v
                for k, v in cfg.items()
                if k not in {"task", "generate_mode", "lambda_", "zeta", "sf"}
            },
        )


@dataclass
class ImageResult:
    """Per-image metrics for a restoration method."""

    psnr: float
    psnr_y: float | None = None
    lpips: float | None = None


@dataclass
class RunResult:
    """Aggregated metrics over a dataset for one method."""

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
        task=method_config.task,
        sf=method_config.sf,
        image_results=image_results,
    )
