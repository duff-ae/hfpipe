#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python3 plot_residual_year_summary.py \
  --input-dir /eos/home-a/alshevel/hf_plots_2024 \
  --output-dir residual_year_summary \
  --year 2024 \
  --max-fill 10232 \
  --trim-frac 0.10
"""

from __future__ import annotations

import os
import re
import glob
import math
import argparse
from typing import List, Dict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep


# ----------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------

def create_figure(x_axis, y_axis, year=2024, plot_type="Preliminary", title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})

    rlabel = f"{year}, 13.6 TeV"
    cms_status = plot_type
    petroff_10 = [
        "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
        "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"
    ]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", petroff_10)

    hep.cms.label(cms_status, loc=0, data=True, year=year, rlabel=rlabel)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    return fig


def _compute_ylim(y, q=99.0, min_lim=0.25, hard_cap=None):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return (-min_lim, min_lim)

    y_abs_q = np.percentile(np.abs(y), q)
    y_lim = max(min_lim, 1.15 * y_abs_q)

    if hard_cap is not None:
        y_lim = min(y_lim, hard_cap)

    return (-y_lim, y_lim)


def _style_residual_axis(ax, ylabel, yvals):
    ax.set_xlabel("Mean SBIL [Hz/µb]")
    ax.set_ylabel(ylabel)

    ymin, ymax = _compute_ylim(yvals, q=99.0, min_lim=0.5, hard_cap=5.0)
    ax.set_ylim(ymin, ymax)

    ax.axhline(0.0, linestyle="-", linewidth=1.0)
    ax.axhline(+0.2, linestyle="--", linewidth=1.0)
    ax.axhline(-0.2, linestyle="--", linewidth=1.0)
    ax.axhspan(-0.2, 0.2, alpha=0.08)

    ax.grid(True, alpha=0.3)


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_")


# ----------------------------------------------------------------------
# Reading helpers
# ----------------------------------------------------------------------

def find_residual_files(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "**", "residual_points_fill_*.h5")
    return sorted(glob.glob(pattern, recursive=True))


def _parse_fill_from_name(path: str) -> int | None:
    m = re.search(r"residual_points_fill_(\d+)\.h5$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def read_one_residual_file(path: str) -> List[Dict]:
    rows: List[Dict] = []

    fill_from_name = _parse_fill_from_name(path)

    with h5py.File(path, "r") as h5:
        dataset_names = list(h5.keys())

        for dsname in dataset_names:
            if not (dsname.startswith("type1_") or dsname.startswith("type2_")):
                continue

            arr = np.asarray(h5[dsname], dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue

            kind = "type1" if dsname.startswith("type1_") else "type2"
            label = dsname[len("type1_"):] if kind == "type1" else dsname[len("type2_"):]

            dset = h5[dsname]
            fill = int(dset.attrs["fill"]) if "fill" in dset.attrs else fill_from_name
            sbil_min = float(dset.attrs.get("sbil_min_for_plot", np.nan))

            x = arr[:, 0]
            y = arr[:, 1]

            finite_mask = np.isfinite(x) & np.isfinite(y)
            x = x[finite_mask]
            y = y[finite_mask]

            for xi, yi in zip(x, y):
                rows.append(
                    {
                        "fill": fill,
                        "label": label,
                        "kind": kind,
                        "x": float(xi),
                        "y": float(yi),
                        "sbil_min": sbil_min,
                        "source_file": path,
                    }
                )

    return rows


def collect_dataframe(base_dir: str, max_fill: int = 10232) -> pd.DataFrame:
    files = find_residual_files(base_dir)
    if not files:
        raise FileNotFoundError(f"No residual_points_fill_*.h5 files found under: {base_dir}")

    rows: List[Dict] = []
    for path in files:
        rows.extend(read_one_residual_file(path))

    if not rows:
        raise RuntimeError("No residual points found in the matched HDF5 files")

    df = pd.DataFrame(rows)
    df = df[df["fill"].notna()].copy()
    df["fill"] = df["fill"].astype(int)

    # keep only pp fills
    df = df[df["fill"] <= max_fill].copy()

    df = df.sort_values(["kind", "label", "fill"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Robust filtering
# ----------------------------------------------------------------------

def trimmed_mean(values: np.ndarray, trim_frac: float = 0.10) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    arr = np.sort(arr)
    n = arr.size

    if n < 5:
        return float(np.mean(arr))

    k = int(math.floor(trim_frac * n))

    # avoid trimming away everything
    if 2 * k >= n:
        return float(np.mean(arr))

    if k == 0:
        return float(np.mean(arr))

    return float(np.mean(arr[k:n - k]))


def median_abs_deviation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))

def build_fill_group_summary(
    df: pd.DataFrame,
    *,
    residual_threshold: float = 1.0,
    sbil_threshold: float = 50.0,
    trim_frac: float = 0.10,
    min_points: int = 20,
) -> pd.DataFrame:
    summary_rows: List[Dict] = []

    grouped = df.groupby(["fill", "kind", "label"], sort=True)

    for (fill, kind, label), sub in grouped:
        x = sub["x"].to_numpy(dtype=np.float64)
        y = sub["y"].to_numpy(dtype=np.float64)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        n_points = int(y.size)
        if n_points == 0:
            continue

        min_x = float(np.min(x))
        max_x = float(np.max(x))
        min_y = float(np.min(y))
        max_y = float(np.max(y))
        max_abs_y = float(np.max(np.abs(y)))

        central_shift = trimmed_mean(y, trim_frac=trim_frac)
        mad_y = median_abs_deviation(y)

        # one-point spikes should not reject a fill-group
        bad_residual = bool(
            n_points >= min_points and np.isfinite(central_shift) and abs(central_shift) > residual_threshold
        )
        bad_sbil = bool(max_x > sbil_threshold)

        exclude = bool(bad_residual or bad_sbil)

        summary_rows.append(
            {
                "fill": int(fill),
                "kind": str(kind),
                "label": str(label),
                "n_points": n_points,
                "min_x": min_x,
                "max_x": max_x,
                "min_pct": min_y,
                "max_pct": max_y,
                "max_abs_pct": max_abs_y,
                "trimmed_mean_pct": float(central_shift),
                "mad_pct": float(mad_y) if np.isfinite(mad_y) else np.nan,
                "residual_threshold_pct": float(residual_threshold),
                "sbil_threshold": float(sbil_threshold),
                "bad_residual": bad_residual,
                "bad_sbil": bad_sbil,
                "exclude_from_plot": exclude,
            }
        )

    if not summary_rows:
        raise RuntimeError("Could not build fill-group summary")

    out = pd.DataFrame(summary_rows)
    out = out.sort_values(
        ["exclude_from_plot", "kind", "label", "fill"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return out

def apply_fill_group_filter(df: pd.DataFrame, fill_summary: pd.DataFrame) -> pd.DataFrame:
    keep = fill_summary[~fill_summary["exclude_from_plot"]][["fill", "kind", "label"]].copy()
    if keep.empty:
        return df.iloc[0:0].copy()

    keep["__keep__"] = True

    merged = df.merge(keep, on=["fill", "kind", "label"], how="left")
    filtered = merged[merged["__keep__"] == True].drop(columns="__keep__").copy()
    filtered = filtered.sort_values(["kind", "label", "fill"]).reset_index(drop=True)
    return filtered


def save_small_reports(fill_summary: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    flagged_full_corr = fill_summary[
        (fill_summary["exclude_from_plot"]) & (fill_summary["label"] == "full_corr")
    ].copy()

    flagged_full_corr = flagged_full_corr[
        [
            "fill",
            "kind",
            "label",
            "n_points",
            "min_x",
            "max_x",
            "trimmed_mean_pct",
            "mad_pct",
            "min_pct",
            "max_pct",
            "max_abs_pct",
            "bad_residual",
            "bad_sbil",
        ]
    ].sort_values(["kind", "fill"]).reset_index(drop=True)

    flagged_csv = os.path.join(output_dir, "flagged_full_corr.csv")
    flagged_full_corr.to_csv(flagged_csv, index=False)

# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def plot_combined_residual(
    df: pd.DataFrame,
    *,
    kind: str,
    label: str,
    output_dir: str,
    year: int = 2024,
) -> None:
    sub = df[(df["kind"] == kind) & (df["label"] == label)].copy()
    if sub.empty:
        return

    x = sub["x"].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.float64)

    fig = create_figure(
        "Mean SBIL [Hz/µb]",
        f"{kind.capitalize()} Residual [% of mean SBIL]",
        year=year,
        plot_type="Preliminary",
        title_suffix=f"All accepted fills combined, label={label}",
    )
    ax = plt.gca()

    ax.plot(x, y, ".", markersize=3)
    _style_residual_axis(ax, f"{kind.capitalize()} Residual [% of mean SBIL]", y)

    png_path = os.path.join(output_dir, f"{kind}_residuals_allfills_{_safe_name(label)}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_residual_colored_by_fill(
    df: pd.DataFrame,
    *,
    kind: str,
    label: str,
    output_dir: str,
    year: int = 2024,
    max_fills_for_legend: int = 12,
) -> None:
    sub = df[(df["kind"] == kind) & (df["label"] == label)].copy()
    if sub.empty:
        return

    fills = sorted(sub["fill"].dropna().unique())

    fig = create_figure(
        "Mean SBIL [Hz/µb]",
        f"{kind.capitalize()} Residual [% of mean SBIL]",
        year=year,
        plot_type="Preliminary",
        title_suffix=f"Accepted fills combined (colored), label={label}",
    )
    ax = plt.gca()

    for i, fill in enumerate(fills):
        ss = sub[sub["fill"] == fill]
        ax.plot(
            ss["x"].to_numpy(dtype=np.float64),
            ss["y"].to_numpy(dtype=np.float64),
            ".",
            markersize=2.5,
            alpha=0.5,
            label=f"{fill}" if i < max_fills_for_legend else None,
        )

    _style_residual_axis(ax, f"{kind.capitalize()} Residual [% of mean SBIL]", sub["y"].to_numpy(dtype=np.float64))

    if len(fills) <= max_fills_for_legend:
        ax.legend(loc="best", frameon=False, fontsize=10, ncol=2)

    png_path = os.path.join(output_dir, f"{kind}_residuals_allfills_colored_{_safe_name(label)}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(df: pd.DataFrame, output_dir: str, year: int = 2024) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted(df["label"].dropna().unique())
    kinds = ["type1", "type2"]

    for label in labels:
        for kind in kinds:
            plot_combined_residual(
                df,
                kind=kind,
                label=label,
                output_dir=output_dir,
                year=year,
            )
            plot_combined_residual_colored_by_fill(
                df,
                kind=kind,
                label=label,
                output_dir=output_dir,
                year=year,
            )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect residual point clouds over all fills, reject problematic fill-groups by robust mean residual, and make combined year-style plots."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Base directory to scan recursively for residual_points_fill_*.h5",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plots and small reports will be written",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year label for CMS plot styling",
    )
    parser.add_argument(
        "--max-fill",
        type=int,
        default=10232,
        help="Maximum fill number to keep (pp-only selection)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=1.0,
        help="Exclude fill-group from combined plots if |trimmed mean residual| exceeds this threshold in percent",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.10,
        help="Fraction to trim on each side when computing trimmed mean",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Minimum number of points in a fill-group before exclusion logic is applied",
    )
    parser.add_argument(
        "--sbil-threshold",
        type=float,
        default=15.0,
        help="Exclude fill-group if any x point (mean SBIL) exceeds this threshold",
    )
    args = parser.parse_args()

    if not (0.0 <= args.trim_frac < 0.5):
        raise ValueError("--trim-frac must satisfy 0 <= trim-frac < 0.5")

    df = collect_dataframe(args.input_dir, max_fill=args.max_fill)

    fill_summary = build_fill_group_summary(
        df,
        residual_threshold=args.residual_threshold,
        sbil_threshold=args.sbil_threshold,
        trim_frac=args.trim_frac,
        min_points=args.min_points,
    )

    df_filtered = apply_fill_group_filter(df, fill_summary)

    os.makedirs(args.output_dir, exist_ok=True)
    save_small_reports(fill_summary, args.output_dir)
    make_all_plots(df_filtered, args.output_dir, year=args.year)

    flagged = fill_summary[fill_summary["exclude_from_plot"]].copy()
    used = fill_summary[~fill_summary["exclude_from_plot"]].copy()

    print(f"Collected residual points in memory: {len(df)}")
    print(f"Accepted fill-groups for plotting: {len(used)}")
    print(f"Flagged fill-groups for manual check: {len(flagged)}")
    print(f"Plots are made from accepted fill-groups only")
    print(f"Output written to: {args.output_dir}")

    if not flagged.empty:
        print("\nFlagged fill-groups:")
        print(
            flagged[
                ["fill", "kind", "label", "n_points", "min_x", "max_x", "min_pct", "max_pct", "max_abs_pct", "bad_residual", "bad_sbil"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()