#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect residual-point files from many fills, reject problematic fill-groups,
make diagnostic combined plots, and build a final SBIL-binned year summary.

The final binned plot is fill-weighted:
  1. Points are binned in SBIL independently inside each fill.
  2. A trimmed mean residual is computed for each (fill, kind, label, SBIL bin).
  3. Those per-fill bin values are averaged with equal fill weights.

This avoids giving a long fill more weight simply because it contains more
residual points.

Example
-------
python3 plot_residual_year_summary.py \
  --input-dir /eos/home-a/alshevel/hfpipe/plots/24/ \
  --output-dir residual_year_summary/24/ \
  --year 2024 \
  --max-fill 10232 \
  --trim-frac 0.10 \
  --bin-width 0.5
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd


DATASET_PREFIXES = {
    "type1_": "type1",
    "type2_": "type2",
}

PETROFF_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return value.strip("_")


def trimmed_mean(values: np.ndarray, trim_frac: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])

    if values.size == 0:
        return np.nan

    if values.size < 5 or trim_frac <= 0.0:
        return float(np.mean(values))

    n_trim = int(math.floor(trim_frac * values.size))
    if n_trim == 0 or 2 * n_trim >= values.size:
        return float(np.mean(values))

    return float(np.mean(values[n_trim:-n_trim]))


def median_abs_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    median = np.median(values)
    return float(np.median(np.abs(values - median)))


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------

def parse_fill_from_name(path: Path) -> int | None:
    match = re.search(r"residual_points_fill_(\d+)\.h5$", path.name)
    return int(match.group(1)) if match else None


def find_residual_files(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("residual_points_fill_*.h5"))


def dataset_kind_and_label(dataset_name: str) -> tuple[str, str] | None:
    for prefix, kind in DATASET_PREFIXES.items():
        if dataset_name.startswith(prefix):
            return kind, dataset_name[len(prefix):]
    return None


def read_residual_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fill_from_name = parse_fill_from_name(path)

    with h5py.File(path, "r") as h5:
        for dataset_name, dataset in h5.items():
            parsed = dataset_kind_and_label(dataset_name)
            if parsed is None:
                continue

            kind, label = parsed
            array = np.asarray(dataset, dtype=np.float64)

            if array.ndim != 2 or array.shape[1] != 2:
                continue

            fill = (
                int(dataset.attrs["fill"])
                if "fill" in dataset.attrs
                else fill_from_name
            )
            if fill is None:
                continue

            x = array[:, 0]
            y = array[:, 1]
            finite = np.isfinite(x) & np.isfinite(y)

            for x_value, y_value in zip(x[finite], y[finite]):
                rows.append(
                    {
                        "fill": fill,
                        "kind": kind,
                        "label": label,
                        "x": float(x_value),
                        "y": float(y_value),
                        "source_file": str(path),
                    }
                )

    return rows


def collect_residual_points(base_dir: Path, max_fill: int) -> pd.DataFrame:
    files = find_residual_files(base_dir)
    if not files:
        raise FileNotFoundError(
            f"No residual_points_fill_*.h5 files found under {base_dir}"
        )

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(read_residual_file(path))

    if not rows:
        raise RuntimeError("Matched HDF5 files contain no residual point datasets")

    df = pd.DataFrame(rows)
    df = df[df["fill"] <= max_fill].copy()

    if df.empty:
        raise RuntimeError(f"No residual points remain after max-fill={max_fill}")

    return df.sort_values(["kind", "label", "fill", "x"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Fill-group quality selection
# ----------------------------------------------------------------------

def build_fill_group_summary(
    df: pd.DataFrame,
    *,
    residual_threshold: float,
    sbil_threshold: float,
    trim_frac: float,
    min_points: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (fill, kind, label), sub in df.groupby(
        ["fill", "kind", "label"],
        sort=True,
    ):
        x = sub["x"].to_numpy(dtype=np.float64)
        y = sub["y"].to_numpy(dtype=np.float64)

        central_shift = trimmed_mean(y, trim_frac)
        mad = median_abs_deviation(y)

        n_points = len(sub)
        max_x = float(np.max(x))

        bad_residual = (
            n_points >= min_points
            and np.isfinite(central_shift)
            and abs(central_shift) > residual_threshold
        )
        bad_sbil = max_x > sbil_threshold

        rows.append(
            {
                "fill": int(fill),
                "kind": str(kind),
                "label": str(label),
                "n_points": int(n_points),
                "min_x": float(np.min(x)),
                "max_x": max_x,
                "min_pct": float(np.min(y)),
                "max_pct": float(np.max(y)),
                "max_abs_pct": float(np.max(np.abs(y))),
                "trimmed_mean_pct": central_shift,
                "mad_pct": mad,
                "bad_residual": bool(bad_residual),
                "bad_sbil": bool(bad_sbil),
                "exclude_from_plot": bool(bad_residual or bad_sbil),
            }
        )

    if not rows:
        raise RuntimeError("Could not build fill-group summary")

    return (
        pd.DataFrame(rows)
        .sort_values(["kind", "label", "fill"])
        .reset_index(drop=True)
    )


def select_accepted_points(
    df: pd.DataFrame,
    fill_summary: pd.DataFrame,
) -> pd.DataFrame:
    accepted_groups = fill_summary.loc[
        ~fill_summary["exclude_from_plot"],
        ["fill", "kind", "label"],
    ].copy()

    if accepted_groups.empty:
        return df.iloc[0:0].copy()

    accepted_groups["accepted"] = True

    return (
        df.merge(
            accepted_groups,
            on=["fill", "kind", "label"],
            how="inner",
        )
        .drop(columns="accepted")
        .sort_values(["kind", "label", "fill", "x"])
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# Final SBIL binning
# ----------------------------------------------------------------------

def make_bin_edges(x: np.ndarray, bin_width: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("Cannot build SBIL bins from an empty array")
    if bin_width <= 0.0:
        raise ValueError("--bin-width must be > 0")

    low = math.floor(float(np.min(x)) / bin_width) * bin_width
    high = math.ceil(float(np.max(x)) / bin_width) * bin_width

    if np.isclose(low, high):
        high = low + bin_width

    n_bins = int(round((high - low) / bin_width))
    return low + np.arange(n_bins + 1, dtype=np.float64) * bin_width


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Return integer bin indices in [0, n_bins-1].

    The upper edge of the final bin is inclusive so a point exactly on the
    global maximum is not lost.
    """
    x = np.asarray(x, dtype=np.float64)
    indices = np.searchsorted(edges, x, side="right") - 1

    last_edge = np.isclose(x, edges[-1], rtol=0.0, atol=1e-12)
    indices[last_edge] = len(edges) - 2

    return indices


def build_binned_summary(
    df: pd.DataFrame,
    *,
    bin_width: float,
    trim_frac: float,
    min_fills_per_bin: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a two-stage, fill-weighted SBIL summary.

    Returns
    -------
    fill_bins
        One row per (fill, kind, label, SBIL bin).
    summary
        One row per (kind, label, SBIL bin), combining fill-bin values with
        equal fill weights.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    fill_bin_rows: list[dict[str, Any]] = []

    for (kind, label), group in df.groupby(["kind", "label"], sort=True):
        edges = make_bin_edges(group["x"].to_numpy(), bin_width)
        n_bins = len(edges) - 1

        work = group[["fill", "x", "y"]].copy()
        work["bin_index"] = assign_bins(work["x"].to_numpy(), edges)
        work = work[
            (work["bin_index"] >= 0)
            & (work["bin_index"] < n_bins)
        ]

        for (fill, bin_index), sub in work.groupby(
            ["fill", "bin_index"],
            sort=True,
        ):
            bin_index = int(bin_index)
            low = float(edges[bin_index])
            high = float(edges[bin_index + 1])

            fill_bin_rows.append(
                {
                    "fill": int(fill),
                    "kind": str(kind),
                    "label": str(label),
                    "bin_index": bin_index,
                    "bin_low": low,
                    "bin_high": high,
                    "bin_center": 0.5 * (low + high),
                    "n_points": int(len(sub)),
                    "mean_x": float(sub["x"].mean()),
                    "residual_pct": trimmed_mean(
                        sub["y"].to_numpy(dtype=np.float64),
                        trim_frac,
                    ),
                }
            )

    fill_bins = pd.DataFrame(fill_bin_rows)
    if fill_bins.empty:
        return fill_bins, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []

    for (kind, label, bin_index), sub in fill_bins.groupby(
        ["kind", "label", "bin_index"],
        sort=True,
    ):
        values = sub["residual_pct"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        n_fills = int(values.size)
        std = float(np.std(values, ddof=1)) if n_fills > 1 else np.nan
        sem = std / math.sqrt(n_fills) if n_fills > 1 else np.nan
        p16, p84 = np.percentile(values, [16.0, 84.0])

        first = sub.iloc[0]
        summary_rows.append(
            {
                "kind": str(kind),
                "label": str(label),
                "bin_index": int(bin_index),
                "bin_low": float(first["bin_low"]),
                "bin_high": float(first["bin_high"]),
                "bin_center": float(first["bin_center"]),
                "n_fills": n_fills,
                "n_points": int(sub["n_points"].sum()),
                "mean_pct": float(np.mean(values)),
                "std_pct": std,
                "sem_pct": sem,
                "median_pct": float(np.median(values)),
                "p16_pct": float(p16),
                "p84_pct": float(p84),
                "use_in_plot": n_fills >= min_fills_per_bin,
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["kind", "label", "bin_index"])
        .reset_index(drop=True)
    )

    return fill_bins, summary


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_figure(
    *,
    year: int,
    ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_prop_cycle(color=PETROFF_10)
    ax.set_xlabel("Mean SBIL [Hz/µb]")
    ax.set_ylabel(ylabel)

    hep.cms.label(
        "Preliminary",
        ax=ax,
        loc=0,
        data=True,
        year=year,
        rlabel=f"{year}, 13.6 TeV",
    )

    return fig, ax


def compute_symmetric_ylim(
    values: np.ndarray,
    *,
    percentile: float = 99.0,
    min_limit: float = 0.5,
    hard_cap: float = 5.0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return -min_limit, min_limit

    limit = max(
        min_limit,
        1.15 * float(np.percentile(np.abs(values), percentile)),
    )
    limit = min(limit, hard_cap)

    return -limit, limit


def style_residual_axis(
    ax: plt.Axes,
    y_values: np.ndarray,
) -> None:
    ymin, ymax = compute_symmetric_ylim(y_values)
    ax.set_ylim(ymin, ymax)

    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(+0.2, linestyle="--", linewidth=1.0)
    ax.axhline(-0.2, linestyle="--", linewidth=1.0)
    ax.axhspan(-0.2, 0.2, alpha=0.08)
    ax.grid(True, alpha=0.3)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_points(
    sub: pd.DataFrame,
    *,
    output_path: Path,
    year: int,
    color_by_fill: bool,
) -> None:
    if sub.empty:
        return

    kind = str(sub["kind"].iloc[0])
    ylabel = f"{kind.capitalize()} Residual [% of mean SBIL]"

    fig, ax = make_figure(year=year, ylabel=ylabel)

    if color_by_fill:
        fills = sorted(sub["fill"].unique())
        for fill in fills:
            fill_data = sub[sub["fill"] == fill]
            ax.plot(
                fill_data["x"],
                fill_data["y"],
                ".",
                markersize=2.5,
                alpha=0.45,
            )
    else:
        ax.plot(
            sub["x"],
            sub["y"],
            ".",
            markersize=3.0,
            alpha=0.65,
        )

    style_residual_axis(ax, sub["y"].to_numpy(dtype=np.float64))
    save_figure(fig, output_path)


def plot_final_binned_summary(
    summary: pd.DataFrame,
    *,
    output_path: Path,
    year: int,
) -> None:
    if summary.empty:
        return

    plot_data = summary[summary["use_in_plot"]].copy()
    if plot_data.empty:
        return

    kind = str(plot_data["kind"].iloc[0])
    ylabel = f"{kind.capitalize()} Residual [% of mean SBIL]"

    x = plot_data["bin_center"].to_numpy(dtype=np.float64)
    mean = plot_data["mean_pct"].to_numpy(dtype=np.float64)
    sem = plot_data["sem_pct"].to_numpy(dtype=np.float64)
    p16 = plot_data["p16_pct"].to_numpy(dtype=np.float64)
    p84 = plot_data["p84_pct"].to_numpy(dtype=np.float64)

    fig, ax = make_figure(year=year, ylabel=ylabel)

    # Fill-to-fill central 68% interval.
    ax.fill_between(
        x,
        p16,
        p84,
        alpha=0.18,
        linewidth=0.0,
        label="16–84% across fills",
    )

    # Equal-fill-weight mean; SEM describes uncertainty on the mean.
    finite_sem = np.where(np.isfinite(sem), sem, 0.0)
    ax.errorbar(
        x,
        mean,
        yerr=finite_sem,
        fmt="o-",
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        label="Fill-weighted mean ± SEM",
    )

    y_for_limits = np.concatenate([p16, p84, mean - finite_sem, mean + finite_sem])
    style_residual_axis(ax, y_for_limits)
    ax.legend(frameon=False, fontsize=10)

    save_figure(fig, output_path)


def make_all_plots(
    df: pd.DataFrame,
    binned_summary: pd.DataFrame,
    *,
    output_dir: Path,
    year: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for (kind, label), sub in df.groupby(["kind", "label"], sort=True):
        stem = f"{kind}_{safe_name(label)}"

        plot_combined_points(
            sub,
            output_path=output_dir / f"{stem}_all_fills.png",
            year=year,
            color_by_fill=False,
        )

        plot_combined_points(
            sub,
            output_path=output_dir / f"{stem}_all_fills_colored.png",
            year=year,
            color_by_fill=True,
        )

        binned = binned_summary[
            (binned_summary["kind"] == kind)
            & (binned_summary["label"] == label)
        ]
        plot_final_binned_summary(
            binned,
            output_path=output_dir / f"{stem}_FINAL_binned.png",
            year=year,
        )


# ----------------------------------------------------------------------
# Output tables
# ----------------------------------------------------------------------

def save_reports(
    *,
    fill_summary: pd.DataFrame,
    fill_bins: pd.DataFrame,
    binned_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fill_summary.to_csv(
        output_dir / "fill_group_summary.csv",
        index=False,
    )

    flagged = fill_summary[fill_summary["exclude_from_plot"]].copy()
    flagged.to_csv(
        output_dir / "flagged_fill_groups.csv",
        index=False,
    )

    fill_bins.to_csv(
        output_dir / "binned_values_per_fill.csv",
        index=False,
    )

    binned_summary.to_csv(
        output_dir / "binned_year_summary.csv",
        index=False,
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect residual points over many fills, reject problematic "
            "fill-groups, and make raw and SBIL-binned year-summary plots."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory scanned recursively for residual_points_fill_*.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for plots and CSV summaries",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year shown in the CMS label",
    )
    parser.add_argument(
        "--max-fill",
        type=int,
        default=10232,
        help="Maximum fill number to keep",
    )

    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=1.0,
        help=(
            "Reject a fill-group if |trimmed mean residual| is above this "
            "threshold in percent"
        ),
    )
    parser.add_argument(
        "--sbil-threshold",
        type=float,
        default=15.0,
        help=(
            "Reject a fill-group if any of its mean-SBIL points exceeds "
            "this value"
        ),
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.10,
        help="Fraction trimmed from each side when computing trimmed means",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help=(
            "Minimum number of points before the residual-threshold "
            "rejection is applied"
        ),
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.5,
        help="SBIL bin width for the final year-summary plot [Hz/µb]",
    )
    parser.add_argument(
        "--min-fills-per-bin",
        type=int,
        default=3,
        help="Minimum number of fills required to draw a final binned point",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.trim_frac < 0.5:
        raise ValueError("--trim-frac must satisfy 0 <= trim-frac < 0.5")
    if args.bin_width <= 0.0:
        raise ValueError("--bin-width must be > 0")
    if args.min_points < 1:
        raise ValueError("--min-points must be >= 1")
    if args.min_fills_per_bin < 1:
        raise ValueError("--min-fills-per-bin must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    df = collect_residual_points(
        args.input_dir,
        max_fill=args.max_fill,
    )

    fill_summary = build_fill_group_summary(
        df,
        residual_threshold=args.residual_threshold,
        sbil_threshold=args.sbil_threshold,
        trim_frac=args.trim_frac,
        min_points=args.min_points,
    )

    accepted = select_accepted_points(df, fill_summary)
    if accepted.empty:
        raise RuntimeError("No fill-groups survived the quality selection")

    fill_bins, binned_summary = build_binned_summary(
        accepted,
        bin_width=args.bin_width,
        trim_frac=args.trim_frac,
        min_fills_per_bin=args.min_fills_per_bin,
    )

    save_reports(
        fill_summary=fill_summary,
        fill_bins=fill_bins,
        binned_summary=binned_summary,
        output_dir=args.output_dir,
    )

    make_all_plots(
        accepted,
        binned_summary,
        output_dir=args.output_dir,
        year=args.year,
    )

    n_flagged = int(fill_summary["exclude_from_plot"].sum())
    n_accepted = int((~fill_summary["exclude_from_plot"]).sum())

    print(f"Collected residual points: {len(df)}")
    print(f"Accepted fill-groups: {n_accepted}")
    print(f"Flagged fill-groups: {n_flagged}")
    print(f"Accepted residual points: {len(accepted)}")
    print(f"Per-fill SBIL-bin values: {len(fill_bins)}")
    print(
        "Final plotted SBIL bins: "
        f"{int(binned_summary['use_in_plot'].sum()) if not binned_summary.empty else 0}"
    )
    print(f"Output written to: {args.output_dir}")

    flagged = fill_summary[fill_summary["exclude_from_plot"]]
    if not flagged.empty:
        columns = [
            "fill",
            "kind",
            "label",
            "n_points",
            "min_x",
            "max_x",
            "trimmed_mean_pct",
            "mad_pct",
            "bad_residual",
            "bad_sbil",
        ]
        print("\nFlagged fill-groups:")
        print(flagged[columns].to_string(index=False))


if __name__ == "__main__":
    main()
