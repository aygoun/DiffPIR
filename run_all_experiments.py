"""
Run every (task, method) combination on up to N images and write per-image
and per-combo average metrics to a CSV file.

Usage:
    python run_all_experiments.py \\
        --testset  testsets/demo_test \\
        --max_images 50 \\
        --output   results/experiments.csv \\
        --workers  1
"""

from __future__ import annotations

import kagglehub
import argparse
import csv
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple
import shutil

# ── ensure project root is on sys.path when invoked directly ────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from experiments.common import (
    DegradedInput,
    ImageResult,
    MethodConfig,
    RunResult,
    load_image_paths,
)

# Download latest version
path = kagglehub.dataset_download("ashwingupta3012/human-faces")
print("Path to dataset files:", path)
# Move dataset to datasets folder
os.makedirs(os.path.join(_HERE, "datasets"), exist_ok=True)
shutil.move(path, os.path.join(_HERE, "datasets", "human-faces"))

# ═══════════════════════════════════════════════════════════════════════════
# Registry  –  (task, method_key) → (yaml_path, runner_callable)
# ═══════════════════════════════════════════════════════════════════════════

#: All supported tasks.
ALL_TASKS: List[str] = ["inpaint", "deblur", "sr"]

#: All supported method keys (pnp_diffbir is inpaint-only).
ALL_METHODS: List[str] = [
    "diffpir",
    "dps_y0",
    "dps_yt",
    "pnp_gaussian",
    "pnp_drunet",
    "pnp_diffbir",
]

#: Methods available for each task (keeps the inpaint-only pnp_diffbir scoped).
TASK_METHODS: Dict[str, List[str]] = {
    "inpaint": [
        "diffpir",
        "dps_y0",
        "dps_yt",
        "pnp_gaussian",
        "pnp_drunet",
        "pnp_diffbir",
    ],
    "deblur": ["diffpir", "dps_y0", "dps_yt", "pnp_gaussian", "pnp_drunet"],
    "sr": ["diffpir", "dps_y0", "dps_yt", "pnp_gaussian", "pnp_drunet"],
}

#: YAML config file for each task.
TASK_CONFIG: Dict[str, str] = {
    "inpaint": os.path.join(_HERE, "configs", "inpaint.yaml"),
    "deblur": os.path.join(_HERE, "configs", "deblur.yaml"),
    "sr": os.path.join(_HERE, "configs", "sr.yaml"),
}


def _build_runner(task: str, method: str):
    """Return the runner callable for a (task, method) pair.

    Imports are deferred so that the module can be imported cheaply and so
    that subprocesses only pay the import cost for the modules they need.
    """
    if task == "inpaint":
        from experiments import inpaint_methods

        if method == "diffpir":
            return inpaint_methods.run_diffpir_inpaint
        if method == "dps_y0":
            return partial(inpaint_methods.run_dps_inpaint, mode="DPS_y0")
        if method == "dps_yt":
            return partial(inpaint_methods.run_dps_inpaint, mode="DPS_yt")
        return (
            inpaint_methods.run_pnp_inpaint
        )  # pnp_gaussian / pnp_drunet / pnp_diffbir

    if task == "deblur":
        from experiments import deblur_methods

        if method == "diffpir":
            return deblur_methods.run_diffpir_deblur
        if method == "dps_y0":
            return partial(deblur_methods.run_dps_deblur, mode="DPS_y0")
        if method == "dps_yt":
            return partial(deblur_methods.run_dps_deblur, mode="DPS_yt")
        return deblur_methods.run_pnp_deblur  # pnp_gaussian / pnp_drunet

    if task == "sr":
        from experiments import sr_methods

        if method == "diffpir":
            return sr_methods.run_diffpir_sr
        if method == "dps_y0":
            return partial(sr_methods.run_dps_sr, mode="DPS_y0")
        if method == "dps_yt":
            return partial(sr_methods.run_dps_sr, mode="DPS_yt")
        return sr_methods.run_pnp_sr  # pnp_gaussian / pnp_drunet

    raise ValueError(f"Unknown task {task!r}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════

CSV_FIELDNAMES = ["task", "method", "image_name", "psnr", "psnr_y", "lpips"]


def _fmt(val) -> str:
    """Format a float metric for CSV, returning '' for None / NaN."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return f"{val:.4f}"


def _run_result_to_rows(
    task: str,
    method: str,
    run_result: RunResult,
) -> List[Dict[str, str]]:
    """Expand a RunResult into one CSV row per image plus one AVERAGE row."""
    rows = []
    for img_name, img_result in run_result.image_results.items():
        rows.append(
            {
                "task": task,
                "method": method,
                "image_name": img_name,
                "psnr": _fmt(img_result.psnr),
                "psnr_y": _fmt(img_result.psnr_y),
                "lpips": _fmt(img_result.lpips),
            }
        )

    rows.append(
        {
            "task": task,
            "method": method,
            "image_name": "AVERAGE",
            "psnr": _fmt(run_result.average_psnr),
            "psnr_y": _fmt(run_result.average_psnr_y),
            "lpips": _fmt(run_result.average_lpips),
        }
    )
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Per-combo worker  (also used directly in single-process mode)
# ═══════════════════════════════════════════════════════════════════════════


def _run_combo(
    task: str,
    method: str,
    image_paths: List[str],
) -> Tuple[str, str, RunResult]:
    """Run a single (task, method) combination over *image_paths*.

    Returns (task, method, RunResult).  Individual image failures are caught
    and stored as NaN so that the combo still produces a CSV row.
    """
    yaml_path = TASK_CONFIG[task]
    cfg = MethodConfig.load_from_yaml(yaml_path, method)
    runner_fn = _build_runner(task, method)

    image_results: Dict[str, ImageResult] = {}
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        try:
            result = runner_fn(img_path, cfg, degraded_input=None)
            image_results[img_name] = result
        except Exception as exc:
            logging.warning("[%s/%s] %s failed: %s", task, method, img_name, exc)
            image_results[img_name] = ImageResult(
                psnr=float("nan"),
                image_path=img_path,
            )

    run_result = RunResult(task=task, sf=cfg.sf, image_results=image_results)
    return task, method, run_result


# ═══════════════════════════════════════════════════════════════════════════
# Image-path loading
# ═══════════════════════════════════════════════════════════════════════════


def _collect_image_paths(testset_root: str, max_images: int) -> List[str]:
    """Return up to *max_images* image paths from *testset_root*."""
    paths = load_image_paths(testset_root)

    if not paths:
        # Try a companion .txt list file (filenames only, one per line)
        for fname in sorted(os.listdir(testset_root)):
            if fname.endswith(".txt"):
                list_path = os.path.join(testset_root, fname)
                with open(list_path) as fh:
                    names = [ln.strip() for ln in fh if ln.strip()]
                candidates = [os.path.join(testset_root, n) for n in names]
                paths = [p for p in candidates if os.path.isfile(p)]
                if paths:
                    break

    return paths[:max_images]


# ═══════════════════════════════════════════════════════════════════════════
# Existing-results reader  (used by --skip_existing)
# ═══════════════════════════════════════════════════════════════════════════


def _read_completed_combos(csv_path: str) -> set:
    """Return the set of (task, method) pairs that already have an AVERAGE row."""
    completed = set()
    if not os.path.isfile(csv_path):
        return completed
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("image_name") == "AVERAGE":
                completed.add((row["task"], row["method"]))
    return completed


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all (task, method) experiments and write results to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--testset",
        default="testsets/demo_test",
        help="Root directory of the testset.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Maximum number of images to use per (task, method) combination.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        choices=ALL_TASKS,
        help="Tasks to run.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="Methods to run.  pnp_diffbir is silently ignored for non-inpaint tasks.",
    )
    p.add_argument(
        "--output",
        default="results/experiments.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel (task, method) workers.  "
            "Keep at 1 on a single GPU to avoid VRAM exhaustion."
        ),
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip (task, method) combos that already have an AVERAGE row in --output.",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── collect image paths ────────────────────────────────────────────────
    image_paths = _collect_image_paths(args.testset, args.max_images)
    if not image_paths:
        logging.error("No images found under %r", args.testset)
        sys.exit(1)

    logging.info(
        "Using %d image(s) from %r (max_images=%d)",
        len(image_paths),
        args.testset,
        args.max_images,
    )

    # ── build combo list ───────────────────────────────────────────────────
    combos: List[Tuple[str, str]] = []
    for task in args.tasks:
        valid_methods = TASK_METHODS[task]
        for method in args.methods:
            if method in valid_methods:
                combos.append((task, method))

    # optionally skip already-done combos
    if args.skip_existing:
        completed = _read_completed_combos(args.output)
        if completed:
            logging.info("Skipping %d already-completed combo(s)", len(completed))
        combos = [(t, m) for t, m in combos if (t, m) not in completed]

    if not combos:
        logging.info("Nothing to run – all combos already completed.")
        return

    logging.info(
        "Running %d combo(s) with %d worker(s): %s",
        len(combos),
        args.workers,
        ", ".join(f"{t}/{m}" for t, m in combos),
    )

    if args.workers > 1:
        logging.warning(
            "workers=%d: each worker loads model weights independently. "
            "Make sure you have enough VRAM (one GPU per worker recommended).",
            args.workers,
        )

    # ── prepare CSV writer ─────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    append_mode = args.skip_existing and os.path.isfile(args.output)
    csv_fh = open(args.output, "a" if append_mode else "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDNAMES)
    if not append_mode:
        writer.writeheader()

    # ── run experiments ────────────────────────────────────────────────────
    def _handle_result(task: str, method: str, run_result: RunResult) -> None:
        rows = _run_result_to_rows(task, method, run_result)
        writer.writerows(rows)
        csv_fh.flush()
        avg_psnr = run_result.average_psnr
        avg_lpips = run_result.average_lpips
        logging.info(
            "  ✓ %s / %s  avg PSNR=%.2f dB%s",
            task,
            method,
            avg_psnr,
            f"  avg LPIPS={avg_lpips:.4f}" if avg_lpips is not None else "",
        )

    t_start = time.time()

    if args.workers <= 1:
        # ── sequential ──────────────────────────────────────────────────────
        for task, method in combos:
            logging.info("→ %s / %s …", task, method)
            t0 = time.time()
            try:
                _, _, run_result = _run_combo(task, method, image_paths)
                _handle_result(task, method, run_result)
            except Exception as exc:
                logging.error(
                    "[%s/%s] Fatal error: %s", task, method, exc, exc_info=True
                )
            logging.info("  elapsed %.1f s", time.time() - t0)
    else:
        # ── parallel via ProcessPoolExecutor ────────────────────────────────
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_combo, task, method, image_paths): (task, method)
                for task, method in combos
            }
            for fut in as_completed(futures):
                task, method = futures[fut]
                try:
                    _, _, run_result = fut.result()
                    _handle_result(task, method, run_result)
                except Exception as exc:
                    logging.error(
                        "[%s/%s] Fatal error: %s", task, method, exc, exc_info=True
                    )

    csv_fh.close()
    logging.info(
        "Done. Total time: %.1f s  →  %r",
        time.time() - t_start,
        args.output,
    )


if __name__ == "__main__":
    main()
