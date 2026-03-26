"""
Read a CSV produced by run_all_experiments.py and generate comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _parse_float(s: str) -> Optional[float]:
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(csv_path: str) -> Tuple[List[Dict], Dict[Tuple[str, str], Dict]]:
    """Load the experiments CSV."""
    rows: List[Dict] = []
    averages: Dict[Tuple[str, str], Dict] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            task = row["task"]
            method = row["method"]
            psnr = _parse_float(row.get("psnr", ""))
            psnr_y = _parse_float(row.get("psnr_y", ""))
            lpips = _parse_float(row.get("lpips", ""))

            if row["image_name"] == "AVERAGE":
                averages[(task, method)] = {
                    "psnr": psnr,
                    "psnr_y": psnr_y,
                    "lpips": lpips,
                }
            else:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "image_name": row["image_name"],
                        "psnr": psnr,
                        "psnr_y": psnr_y,
                        "lpips": lpips,
                    }
                )

    return rows, averages


_METHOD_COLORS = {
    "diffpir": "#2196F3",  # blue
    "dps_y0": "#FF9800",  # orange
    "dps_yt": "#FF5722",  # deep orange
    "pnp_gaussian": "#4CAF50",  # green
    "pnp_drunet": "#9C27B0",  # purple
    "pnp_diffbir": "#E91E63",  # pink
}

_METHOD_LABELS = {
    "diffpir": "DiffPIR",
    "dps_y0": "DPS-y₀",
    "dps_yt": "DPS-yₜ",
    "pnp_gaussian": "PnP-Gaussian",
    "pnp_drunet": "PnP-DRUNet",
    "pnp_diffbir": "PnP-DiffBIR",
}

_TASK_LABELS = {
    "inpaint": "Inpainting",
    "deblur": "Deblurring",
    "sr": "Super-Resolution",
}

_METRIC_LABELS = {
    "psnr": "PSNR (dB)  ↑",
    "psnr_y": "PSNR-Y (dB)  ↑",
    "lpips": "LPIPS  ↓",
}


def _method_color(method: str) -> str:
    return _METHOD_COLORS.get(method, "#607D8B")


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def _task_label(task: str) -> str:
    return _TASK_LABELS.get(task, task)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.4,
        "figure.dpi": 150,
    }
)


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_bar(
    averages: Dict[Tuple[str, str], Dict],
    task: str,
    metric: str,
    methods: List[str],
    output_dir: str,
) -> None:
    """Grouped bar chart of metric across methods for a single task."""
    values = []
    labels = []
    colors = []
    for method in methods:
        val = averages.get((task, method), {}).get(metric)
        if val is not None:
            values.append(val)
            labels.append(_method_label(method))
            colors.append(_method_color(method))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(values) * 1.2), 4.5))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.6, zorder=3)

    # value labels on bars
    is_lpips = metric == "lpips"
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.002 if is_lpips else 0.05),
            f"{val:.3f}" if is_lpips else f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(f"{_task_label(task)} — {_METRIC_LABELS[metric]}", fontweight="bold")

    # highlight best bar
    best_idx = int(np.argmin(values) if is_lpips else np.argmax(values))
    bars[best_idx].set_edgecolor("#222")
    bars[best_idx].set_linewidth(2)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"bar_{task}_{metric}.png"))


def plot_scatter_psnr_lpips(
    averages: Dict[Tuple[str, str], Dict],
    tasks: List[str],
    methods: List[str],
    output_dir: str,
) -> None:
    """Scatter plot of avg PSNR vs avg LPIPS per (task, method), with markers
    shaped by task and coloured by method."""
    task_markers = {"inpaint": "o", "deblur": "s", "sr": "^"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for task in tasks:
        marker = task_markers.get(task, "D")
        for method in methods:
            data = averages.get((task, method), {})
            psnr = data.get("psnr")
            lpips = data.get("lpips")
            if psnr is None or lpips is None:
                continue
            ax.scatter(
                psnr,
                lpips,
                marker=marker,
                s=100,
                color=_method_color(method),
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                label=f"{_task_label(task)} / {_method_label(method)}",
            )
            ax.annotate(
                _method_label(method),
                (psnr, lpips),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=_method_color(method),
            )

    ax.set_xlabel("PSNR (dB)  ↑")
    ax.set_ylabel("LPIPS  ↓")
    ax.set_title("PSNR vs LPIPS across all tasks & methods", fontweight="bold")

    # Deduplicate legend entries by method colour
    seen_methods: set = set()
    handles, legend_labels = [], []
    for task in tasks:
        for method in methods:
            if (task, method) in averages and method not in seen_methods:
                seen_methods.add(method)
                patch = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=_method_color(method),
                    markersize=9,
                    label=_method_label(method),
                )
                handles.append(patch)
                legend_labels.append(_method_label(method))

    # Add task-shape legend
    for task in tasks:
        mk = task_markers.get(task, "D")
        patch = plt.Line2D(
            [0],
            [0],
            marker=mk,
            color="w",
            markerfacecolor="#888",
            markersize=9,
            label=_task_label(task),
        )
        handles.append(patch)
        legend_labels.append(_task_label(task))

    ax.legend(handles, legend_labels, fontsize=8, loc="best", framealpha=0.7)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "scatter_psnr_lpips.png"))


def plot_heatmap(
    averages: Dict[Tuple[str, str], Dict],
    tasks: List[str],
    methods: List[str],
    metric: str,
    output_dir: str,
) -> None:
    """Heatmap with tasks as rows and methods as columns."""
    # filter to methods that actually appear in averages
    active_methods = [m for m in methods if any((t, m) in averages for t in tasks)]
    if not active_methods or not tasks:
        return

    grid = np.full((len(tasks), len(active_methods)), np.nan)
    for i, task in enumerate(tasks):
        for j, method in enumerate(active_methods):
            val = averages.get((task, method), {}).get(metric)
            if val is not None:
                grid[i, j] = val

    is_lpips = metric == "lpips"
    cmap = "RdYlGn_r" if is_lpips else "RdYlGn"

    fig, ax = plt.subplots(
        figsize=(max(6, len(active_methods) * 1.4), max(3, len(tasks) * 1.1))
    )
    im = ax.imshow(grid, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(active_methods)))
    ax.set_xticklabels(
        [_method_label(m) for m in active_methods], rotation=30, ha="right"
    )
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([_task_label(t) for t in tasks])
    ax.set_title(
        f"Average {_METRIC_LABELS[metric]} — all tasks × methods", fontweight="bold"
    )

    # annotate cells
    for i in range(len(tasks)):
        for j in range(len(active_methods)):
            val = grid[i, j]
            if not np.isnan(val):
                text = f"{val:.3f}" if is_lpips else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                )

    plt.colorbar(im, ax=ax, label=_METRIC_LABELS[metric], shrink=0.8)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"heatmap_{metric}.png"))


def plot_per_image_lines(
    rows: List[Dict],
    tasks: List[str],
    methods: List[str],
    metric: str,
    output_dir: str,
) -> None:
    """Line plots showing per-image metric values for each task."""
    data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["task"] in tasks and row["method"] in methods:
            val = row.get(metric)
            if val is not None:
                data[row["task"]][row["method"]][row["image_name"]] = val

    active_tasks = [t for t in tasks if data[t]]
    if not active_tasks:
        return

    fig, axes = plt.subplots(
        1,
        len(active_tasks),
        figsize=(5 * len(active_tasks), 4),
        sharey=False,
    )
    if len(active_tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, active_tasks):
        task_data = data[task]
        # Gather a consistent x-axis (sorted image names)
        all_imgs = sorted({img for m_data in task_data.values() for img in m_data})
        x = np.arange(len(all_imgs))

        for method in methods:
            if method not in task_data:
                continue
            y = [task_data[method].get(img) for img in all_imgs]
            valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5,
                label=_method_label(method),
                color=_method_color(method),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [os.path.splitext(n)[0] for n in all_imgs],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(_METRIC_LABELS[metric])
        ax.set_title(_task_label(task), fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-image {_METRIC_LABELS[metric]}", fontweight="bold")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, f"per_image_{metric}.png"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot experiment results from a CSV produced by run_all_experiments.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default="results/experiments.csv",
        help="Path to the experiments CSV.",
    )
    p.add_argument(
        "--output_dir",
        default="results/plots",
        help="Directory to write plot PNG files.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["inpaint", "deblur", "sr"],
        help="Tasks to include in plots.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "diffpir",
            "dps_y0",
            "dps_yt",
            "pnp_gaussian",
            "pnp_drunet",
            "pnp_diffbir",
        ],
        help="Methods to include in plots.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "lpips", "psnr_y"],
        choices=["psnr", "lpips", "psnr_y"],
        help="Metrics to plot.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.csv} …")
    rows, averages = load_csv(args.csv)
    print(f"  {len(rows)} per-image rows, {len(averages)} (task, method) averages")

    os.makedirs(args.output_dir, exist_ok=True)

    # Bar charts (one per task × metric)
    for task in args.tasks:
        for metric in args.metrics:
            if metric == "psnr_y" and task != "sr":
                continue  # psnr_y is SR-specific
            plot_bar(averages, task, metric, args.methods, args.output_dir)

    # PSNR vs LPIPS scatter
    if "psnr" in args.metrics and "lpips" in args.metrics:
        plot_scatter_psnr_lpips(averages, args.tasks, args.methods, args.output_dir)

    # Heatmaps
    for metric in args.metrics:
        if metric == "psnr_y":
            continue  # better shown per-task only
        plot_heatmap(averages, args.tasks, args.methods, metric, args.output_dir)

    # Per-image line plots
    for metric in ["psnr", "lpips"]:
        if metric in args.metrics:
            plot_per_image_lines(
                rows, args.tasks, args.methods, metric, args.output_dir
            )

    print(f"\nAll plots written to {args.output_dir!r}")


if __name__ == "__main__":
    main()
