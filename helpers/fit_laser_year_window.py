#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_laser_year_window.py  (with emittance-scan comparison)

Unchanged laser pipeline + optional sigma_vis comparison from emittance-scan CSV.

Usage:
  python3 fit_laser_year_window.py \
      --input-dir /eos/home-a/alshevel/hfpipe/plots/24_test/ \
      --output-dir laser_year_window_2024 \
      --year 2024 \
      --bril-pattern "input/luminosity/*.csv" \
      --reference-lumi delivered \
      --stable-beams-only \
      --sbil-min 6  --sbil-max 8 \
      --response-min 2  --response-max 3 \
      --trim-frac 0.2  --min-points 20 \
      --fit-nsigma 3   --relative-ylim 5 \
      --emittance-csv input/scans/scan24.csv \
      --det HFET \
      --make-diagnostics
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep

LASER_BCIDS = [3489, 3490, 3491, 3492]
#LASER_BCIDS = [3490]


# ==============================================================================
# CMS style helpers
# ==============================================================================

def _apply_cms_rcparams() -> None:
    plt.style.use(hep.style.CMS)
    mpl.rcParams.update({
        "font.size":             14,
        "axes.labelsize":        16,
        "axes.titlesize":        14,
        "xtick.labelsize":       13,
        "ytick.labelsize":       13,
        "legend.fontsize":       12,
        "figure.dpi":            150,
        "lines.linewidth":       1.8,
        "axes.linewidth":        1.2,
        "xtick.major.width":     1.2,
        "ytick.major.width":     1.2,
        "xtick.minor.visible":   True,
        "ytick.minor.visible":   True,
    })


def _cms_label(ax, year: Optional[int]) -> None:
    rlabel = f"{year}, 13.6 TeV" if year else "13.6 TeV"
    hep.cms.label("Preliminary", data=True, loc=0, rlabel=rlabel, ax=ax)


def cms_figure(
    xlabel: str,
    ylabel: str,
    year:   Optional[int] = None,
    title:  str = "",
    figsize: tuple = (9.0, 6.5),
) -> tuple:
    _apply_cms_rcparams()
    fig, ax = plt.subplots(figsize=figsize)
    _cms_label(ax, year)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10, fontsize=13)
    ax.grid(True, which="major", alpha=0.15, linewidth=0.8, color="0.5")
    return fig, ax


# ==============================================================================
# Parsing / IO helpers  (ORIGINAL, unchanged)
# ==============================================================================

def parse_fill(path: str) -> int:
    m = re.search(r"laser_summary_fill_(\d+)\.h5$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse fill from {path}")
    return int(m.group(1))


def find_laser_files(input_dir: str) -> list:
    pat = os.path.join(input_dir, "**", "laser_summary_fill_*.h5")
    return sorted(glob.glob(pat, recursive=True))


# ==============================================================================
# Robust statistics  (ORIGINAL, unchanged)
# ==============================================================================

def trimmed_mean(arr: np.ndarray, trim_frac: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.sort(arr)
    k = int(np.floor(trim_frac * arr.size))
    if 2 * k >= arr.size:
        return float(np.mean(arr))
    return float(np.mean(arr[k:arr.size - k]))


def trimmed_std(arr: np.ndarray, trim_frac: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    arr = np.sort(arr)
    k = int(np.floor(trim_frac * arr.size))
    if 2 * k >= arr.size:
        return float(np.std(arr, ddof=1))
    arr2 = arr[k:arr.size - k]
    if arr2.size < 2:
        return np.nan
    return float(np.std(arr2, ddof=1))


def mad_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad)


# ==============================================================================
# Laser data pipeline  (ORIGINAL, unchanged)
# ==============================================================================

def read_fill_sum(path: str, bcids: list = LASER_BCIDS) -> tuple:
    fill = parse_fill(path)
    arrays = []
    with h5py.File(path, "r") as h5:
        for bcid in bcids:
            arr = np.asarray(h5[f"bcid_{bcid}"]["corr_sbil_points"], dtype=np.float64)
            arrays.append(arr)

    n = arrays[0].shape[0]
    x = arrays[0][:, 0].copy()
    for arr in arrays:
        if arr.shape != (n, 2):
            raise ValueError(f"{path}: inconsistent corr_sbil_points shapes")
        if not np.allclose(arr[:, 0], x, rtol=0.0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{path}: SBIL x-values differ between BCIDs")

    y_sum = np.zeros_like(x, dtype=np.float64)
    for arr in arrays:
        y_sum += arr[:, 1]
    return fill, x, y_sum


def build_fill_table(
    input_dir: str,
    *,
    sbil_min: float,
    sbil_max: float,
    y_min: float,
    y_max: float,
    trim_frac: float,
    min_points: int,
) -> pd.DataFrame:
    rows = []
    for path in find_laser_files(input_dir):
        fill, x, y = read_fill_sum(path)
        mask = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= sbil_min) & (x <= sbil_max)
            & (y >= y_min) & (y <= y_max)
        )
        xx, yy = x[mask], y[mask]
        row = {
            "fill": fill, "n_points": int(yy.size),
            "sbil_mean": float(np.mean(xx)) if xx.size else np.nan,
            "response": np.nan, "response_std": np.nan, "response_err": np.nan,
            "ok": False, "source_file": path,
        }
        if yy.size >= min_points:
            mu  = trimmed_mean(yy, trim_frac)
            sig = trimmed_std(yy, trim_frac)
            err = sig / np.sqrt(yy.size) if np.isfinite(sig) and yy.size > 0 else np.nan
            if np.isfinite(mu) and mu > 0:
                row.update(response=mu, response_std=sig, response_err=err, ok=True)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No laser files found")
    return df.sort_values("fill").reset_index(drop=True)


def read_brilcalc_tables(
    pattern: str,
    prefer: str = "delivered",
    stable_beams_only: bool = True,
) -> pd.DataFrame:
    files = sorted(glob.iglob(pattern))
    if not files:
        raise RuntimeError(f"No BRIL CSV found at {pattern}")

    def pick_col(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    parts = []
    for fp in files:
        df = pd.read_csv(fp, sep=",", skiprows=1, engine="python")
        df.columns = df.columns.astype(str).str.strip()
        if "#run:fill" not in df.columns:
            continue
        col = pick_col(
            df.columns,
            ["recorded(/fb)", "recorded(/ub)"] if prefer == "recorded"
            else ["delivered(/fb)", "delivered(/ub)"],
        )
        if col is None:
            continue
        val    = pd.to_numeric(df[col], errors="coerce")
        val_fb = val * 1e-9 if col.endswith("(/ub)") else val
        rf = df["#run:fill"].astype(str).str.split(":", n=1, expand=True)
        if rf.shape[1] < 2:
            continue
        out = pd.DataFrame({
            "fill": pd.to_numeric(rf[1], errors="coerce"),
            "value_fb": pd.to_numeric(val_fb, errors="coerce"),
        }).dropna()
        out["fill"] = out["fill"].astype(int)
        if stable_beams_only and "beamstatus" in df.columns:
            bs  = df["beamstatus"].astype(str).str.strip().str.upper()
            out = out.loc[bs == "STABLE BEAMS"].copy()
        parts.append(out[["fill", "value_fb"]])

    if not parts:
        raise RuntimeError(f"No usable BRIL rows found in {pattern}")
    return pd.concat(parts, ignore_index=True)


def make_fill_to_ilumi(
    pattern: str,
    prefer: str = "delivered",
    stable_beams_only: bool = True,
    zero_at_first: bool = True,
) -> dict:
    byls = read_brilcalc_tables(pattern, prefer=prefer, stable_beams_only=stable_beams_only)
    per_fill = (
        byls.groupby("fill", as_index=False)["value_fb"]
        .sum()
        .rename(columns={"value_fb": "lumi_fb"})
        .sort_values("fill")
        .reset_index(drop=True)
    )
    if per_fill.empty:
        raise RuntimeError("No fills in BRIL table")
    per_fill["ilumi_fb"] = per_fill["lumi_fb"].cumsum()
    if zero_at_first:
        per_fill["ilumi_fb"] -= float(per_fill["ilumi_fb"].iloc[0])
    return dict(zip(per_fill["fill"].astype(int), per_fill["ilumi_fb"].astype(float)))


def attach_ilumi(df: pd.DataFrame, fill_to_ilumi: dict) -> pd.DataFrame:
    out = df.copy()
    out["ilumi"] = out["fill"].map(fill_to_ilumi)
    out = out.dropna(subset=["ilumi"]).copy()
    out["ilumi"] = out["ilumi"].astype(float)
    return out.sort_values("ilumi").reset_index(drop=True)


def robust_line_fit(df: pd.DataFrame, nsigma: float = 3.0) -> tuple:
    """Original laser fit: works on DataFrame with 'ok', 'ilumi', 'response'."""
    fit_df = df[df["ok"]].copy()
    fit_df = fit_df[np.isfinite(fit_df["ilumi"]) & np.isfinite(fit_df["response"])].copy()
    if len(fit_df) < 2:
        raise RuntimeError("Not enough points for line fit")
    for _ in range(3):
        x = fit_df["ilumi"].to_numpy(dtype=np.float64)
        y = fit_df["response"].to_numpy(dtype=np.float64)
        m, b  = np.polyfit(x, y, 1)
        resid = y - (m * x + b)
        sigma = mad_std(resid)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        keep = np.abs(resid - np.median(resid)) < nsigma * sigma
        if keep.all():
            break
        fit_df = fit_df.loc[keep].copy()
        if len(fit_df) < 2:
            raise RuntimeError("Too few points left after outlier rejection")
    x = fit_df["ilumi"].to_numpy(dtype=np.float64)
    y = fit_df["response"].to_numpy(dtype=np.float64)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b), fit_df.reset_index(drop=True)


def robust_line_fit_xy(x: np.ndarray, y: np.ndarray, nsigma: float = 3.0) -> tuple:
    """Generic fit on raw arrays. Returns (m, b, used_mask over input arrays)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        raise RuntimeError("Not enough finite points for line fit")
    for _ in range(5):
        xi, yi = x[mask], y[mask]
        m, b   = np.polyfit(xi, yi, 1)
        resid  = yi - (m * xi + b)
        sigma  = mad_std(resid)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_mask = mask.copy()
        new_mask[mask] = np.abs(resid - np.median(resid)) < nsigma * sigma
        if new_mask.sum() < 2 or new_mask.sum() == mask.sum():
            break
        mask = new_mask
    xi, yi = x[mask], y[mask]
    m, b   = np.polyfit(xi, yi, 1)
    return float(m), float(b), mask


def normalize_to_fit(df: pd.DataFrame, fit_df: pd.DataFrame, m: float, b: float) -> pd.DataFrame:
    out = df.copy()
    x  = out["ilumi"].to_numpy(dtype=np.float64)
    y  = out["response"].to_numpy(dtype=np.float64)
    ye = out["response_err"].to_numpy(dtype=np.float64)
    x0 = float(fit_df["ilumi"].min())
    y0 = m * x0 + b
    if not np.isfinite(y0) or y0 == 0.0:
        raise RuntimeError(f"Bad normalization value at fit start: y0={y0}")
    fit_y = m * x + b
    out["fit_response"]          = fit_y
    out["response_norm_pct"]     = 100.0 * y   / y0
    out["response_err_norm_pct"] = 100.0 * ye  / y0
    out["fit_norm_pct"]          = 100.0 * fit_y / y0
    out["fit_start_ilumi"]       = x0
    out["fit_start_value"]       = y0
    return out


def fit_summary_numbers(fit_df: pd.DataFrame, m: float, b: float) -> dict:
    x0 = float(fit_df["ilumi"].min())
    x1 = float(fit_df["ilumi"].max())
    y0 = m * x0 + b
    y1 = m * x1 + b
    if not np.isfinite(y0) or y0 == 0.0:
        raise RuntimeError(f"Bad fit normalization value y0={y0}")
    return dict(
        x0=x0, x1=x1, y0=y0, y1=y1,
        slope_pct_per_fb = 100.0 * m / y0,
        total_change_pct = 100.0 * (y1 / y0 - 1.0),
    )


# ==============================================================================
# Emittance-scan data  (NEW)
# ==============================================================================

# Scan quality cuts applied before per-fill averaging
_VALID_SCAN_NAMES  = {"emit9", "emit15"}
_VALID_SCAN_TIMING = "early"
_VALID_SCAN_STEPS  = 9
_VALID_SCAN_BETA   = 120
_MIN_NBCID         = 400
_SVIS_OUTLIER_FRAC = 0.10   # drop fills where svis deviates > 10% from median


def load_emittance_csv(
    csv_path: str,
    det: str = "HFET",
    fill_to_ilumi: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Read emittance-scan CSV, apply quality cuts, average svis per fill
    (weighted by 1/rms^2), then drop per-fill outliers (>10% from median).

    Quality cuts (applied row-by-row before averaging):
      scanName  in {emit9, emit15}
      scanTiming == early
      scanSteps == 9
      scanBeta  == 120
      nbcid     >  400

    Per-fill outlier cut:
      |svis_mean - median(svis_mean)| / median(svis_mean) > 10%  -> dropped

    iLumi: matched via fill_to_ilumi (same brilcalc table as laser),
           fallback to 'iLumi' column in CSV if present.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ── detector filter ───────────────────────────────────────────────────────
    df = df[df["det"].astype(str).str.upper() == det.upper()].copy()
    if df.empty:
        raise RuntimeError(f"No rows for detector '{det}' in {csv_path}")

    # ── numeric coercion ──────────────────────────────────────────────────────
    for col in ("svis", "svisrms", "fill", "iLumi", "scanSteps", "scanBeta", "nbcid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["fill", "svis"]).copy()
    df["fill"] = df["fill"].astype(int)

    # ── quality cuts ──────────────────────────────────────────────────────────
    n_before = len(df)

    if "scanName" in df.columns:
        df = df[df["scanName"].astype(str).str.strip().isin(_VALID_SCAN_NAMES)]
    if "scanTiming" in df.columns:
        df = df[df["scanTiming"].astype(str).str.strip().str.lower() == _VALID_SCAN_TIMING]
    if "scanSteps" in df.columns:
        df = df[df["scanSteps"] == _VALID_SCAN_STEPS]
    if "scanBeta" in df.columns:
        df = df[df["scanBeta"] == _VALID_SCAN_BETA]
    if "nbcid" in df.columns:
        df = df[df["nbcid"] > _MIN_NBCID]

    n_after = len(df)
    print(f"[emittance] quality cuts: {n_before} -> {n_after} rows kept")

    if df.empty:
        raise RuntimeError("No rows survive quality cuts")

    # ── per-fill weighted average ─────────────────────────────────────────────
    rows = []
    for fill, grp in df.groupby("fill"):
        svis  = grp["svis"].to_numpy(dtype=np.float64)
        rms   = grp["svisrms"].to_numpy(dtype=np.float64) if "svisrms" in grp.columns else None
        valid = np.isfinite(svis)
        if not valid.any():
            continue
        svis = svis[valid]
        if rms is not None:
            rms = rms[valid]
            w   = np.where((rms > 0) & np.isfinite(rms), 1.0 / rms**2, 1.0)
        else:
            w = np.ones_like(svis)
        mu     = float(np.average(svis, weights=w))
        mu_err = float(np.std(svis, ddof=1) / np.sqrt(len(svis))) if len(svis) > 1 else np.nan
        rows.append({"fill": int(fill), "svis_mean": mu, "svis_err": mu_err, "n_scans": len(svis)})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid emittance-scan points after averaging")

    # ── per-fill outlier cut: >10% from median ────────────────────────────────
    med     = float(np.median(out["svis_mean"].to_numpy()))
    rel_dev = np.abs(out["svis_mean"].to_numpy() - med) / med
    mask_ok = rel_dev <= _SVIS_OUTLIER_FRAC
    n_drop  = int((~mask_ok).sum())
    if n_drop:
        dropped = out.loc[~mask_ok, "fill"].tolist()
        print(f"[emittance] outlier cut (>10% from median={med:.1f}): "
              f"dropping {n_drop} fill(s): {dropped}")
    out = out.loc[mask_ok].copy()

    if out.empty:
        raise RuntimeError("No fills survive the svis outlier cut")

    # ── attach iLumi ──────────────────────────────────────────────────────────
    if fill_to_ilumi is not None:
        out["ilumi"] = out["fill"].map(fill_to_ilumi)
    elif "iLumi" in df.columns:
        ilumi_map = df.groupby("fill")["iLumi"].mean().to_dict()
        out["ilumi"] = out["fill"].map(ilumi_map)
    else:
        raise RuntimeError("No iLumi source: provide --bril-pattern or a CSV with iLumi column")

    out = out.dropna(subset=["ilumi"]).sort_values("ilumi").reset_index(drop=True)
    return out


# ==============================================================================
# Standard laser plots  (improved CMS style, same logic as original)
# ==============================================================================

# ==============================================================================
# Standard laser plots  — с residual-панелью
# ==============================================================================

def plot_response_vs_ilumi(
    df: pd.DataFrame, fit_df: pd.DataFrame, m: float, b: float,
    output_dir: str, year: Optional[int],
) -> None:
    good = df[df["ok"]].copy()
    if good.empty:
        return
    info = fit_summary_numbers(fit_df, m, b)

    _apply_cms_rcparams()
    fig = plt.figure(figsize=(9.0, 8.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax  = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    _cms_label(ax, year)
    for a in (ax, ax_r):
        a.grid(True, which="major", alpha=0.15, linewidth=0.8, color="0.5")

    x, y, ye = (good[c].to_numpy(dtype=np.float64) for c in ("ilumi", "response", "response_err"))
    fit_y = m * x + b
    resid = y - fit_y

    m_err = np.isfinite(ye)
    if np.any(m_err):
        ax.errorbar(x[m_err], y[m_err], yerr=ye[m_err],
                    fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.75, label="Per-fill average")
    if np.any(~m_err):
        ax.plot(x[~m_err], y[~m_err], "o", ms=4, alpha=0.75)

    xx = np.linspace(info["x0"], info["x1"], 300)
    ax.plot(xx, m * xx + b, "-", lw=2.0, label="Linear fit")
    ax.set_ylabel("Summed laser response")
    ax.legend(framealpha=0.85)
    txt = (
        f"slope = {m:.4g} / fb$^{{-1}}$\n"
        f"total change = {info['total_change_pct']:+.2f}%"
    )
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))

    # residual panel
    if np.any(m_err):
        ax_r.errorbar(x[m_err], resid[m_err], yerr=ye[m_err],
                      fmt="o", ms=3.5, lw=0.7, capsize=2, alpha=0.7, color="C0")
    if np.any(~m_err):
        ax_r.plot(x[~m_err], resid[~m_err], "o", ms=3.5, alpha=0.7, color="C0")
    ax_r.axhline(0, lw=1.4, ls="--", color="C1")
    ax_r.set_ylabel("Residual")
    ax_r.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")

    fig.savefig(os.path.join(output_dir, "response_vs_ilumi.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative_vs_ilumi(
    df: pd.DataFrame, fit_df: pd.DataFrame, m: float, b: float,
    output_dir: str, year: Optional[int], rel_ylim: Optional[float],
) -> None:
    good = df[df["ok"]].copy()
    if good.empty:
        return
    info = fit_summary_numbers(fit_df, m, b)

    _apply_cms_rcparams()
    fig = plt.figure(figsize=(9.0, 8.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax  = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    _cms_label(ax, year)
    for a in (ax, ax_r):
        a.grid(True, which="major", alpha=0.15, linewidth=0.8, color="0.5")

    x  = good["ilumi"].to_numpy(dtype=np.float64)
    y  = good["response_norm_pct"].to_numpy(dtype=np.float64)
    ye = good["response_err_norm_pct"].to_numpy(dtype=np.float64)

    fit_y_pct = 100.0 * (m * x + b) / info["y0"]
    resid     = y - fit_y_pct

    m_err = np.isfinite(ye)
    if np.any(m_err):
        ax.errorbar(x[m_err], y[m_err], yerr=ye[m_err],
                    fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.75, label="Per-fill average")
    if np.any(~m_err):
        ax.plot(x[~m_err], y[~m_err], "o", ms=4, alpha=0.75)

    xx = np.linspace(info["x0"], info["x1"], 300)
    ax.plot(xx, 100.0 * (m * xx + b) / info["y0"], "-", lw=2.0, label="Linear fit")
    ax.axhline(100.0, ls="--", lw=1.0, color="0.55", alpha=0.8, label="100%")
    ax.set_ylabel("Relative response [%]")
    ax.legend(framealpha=0.85)
    txt = (
        f"slope = {info['slope_pct_per_fb']:+.4f} % / fb$^{{-1}}$\n"
        f"total change = {info['total_change_pct']:+.2f}%"
    )
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))
    if rel_ylim is not None:
        ax.set_ylim(100.0 - rel_ylim, 100.0 + rel_ylim)

    # residual panel
    if np.any(m_err):
        ax_r.errorbar(x[m_err], resid[m_err], yerr=ye[m_err],
                      fmt="o", ms=3.5, lw=0.7, capsize=2, alpha=0.7, color="C0")
    if np.any(~m_err):
        ax_r.plot(x[~m_err], resid[~m_err], "o", ms=3.5, alpha=0.7, color="C0")
    ax_r.axhline(0, lw=1.4, ls="--", color="C1")
    ax_r.set_ylabel("Residual [%]")
    ax_r.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")

    fig.savefig(os.path.join(output_dir, "response_relative_vs_ilumi.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Degradation comparison — автоматический одинаковый скейл для обеих осей
# ==============================================================================

def plot_degradation_comparison(
    laser_df:   pd.DataFrame,
    m_las:      float, b_las: float, fit_las_df: pd.DataFrame,
    emit_df:    pd.DataFrame,
    m_emit:     float, b_emit: float, mask_emit: np.ndarray,
    output_dir: str,
    year:       Optional[int],
    rel_ylim:   Optional[float],
    det:        str,
) -> None:
    xl  = laser_df["ilumi"].to_numpy(dtype=np.float64)
    yl  = laser_df["response"].to_numpy(dtype=np.float64)
    yel = (laser_df["response_err"].to_numpy(dtype=np.float64)
           if "response_err" in laser_df.columns
           else np.full(len(laser_df), np.nan))

    xe  = emit_df["ilumi"].to_numpy(dtype=np.float64)
    ye  = emit_df["svis_mean"].to_numpy(dtype=np.float64)
    yee = emit_df["svis_err"].to_numpy(dtype=np.float64)

    x0_vdm  = float(xe.min())
    y0_las  = m_las  * x0_vdm + b_las
    y0_emit = m_emit * x0_vdm + b_emit

    if not np.isfinite(y0_las)  or y0_las  == 0:
        raise RuntimeError(f"Bad laser normalisation y0={y0_las}")
    if not np.isfinite(y0_emit) or y0_emit == 0:
        raise RuntimeError(f"Bad sigma_vis normalisation y0={y0_emit}")

    yl_rel  = 100.0 * yl  / y0_las
    yel_rel = 100.0 * yel / y0_las
    ye_rel  = 100.0 * ye  / y0_emit
    yee_rel = 100.0 * yee / y0_emit

    def flas(x):  return 100.0 * (m_las  * x + b_las)  / y0_las
    def femit(x): return 100.0 * (m_emit * x + b_emit) / y0_emit

    x_lo = min(x0_vdm, float(xl.min()))
    x_hi = max(float(fit_las_df["ilumi"].max()), float(xe[mask_emit].max()))
    xx   = np.linspace(x_lo, x_hi, 500)

    x_ov_lo = max(x0_vdm, float(xl.min()))
    x_ov_hi = min(float(fit_las_df["ilumi"].max()), float(xe[mask_emit].max()))

    slope_las_pct  = 100.0 * m_las  / y0_las
    slope_emit_pct = 100.0 * m_emit / y0_emit
    avg_slope_pct  = 0.5 * (slope_las_pct + slope_emit_pct)
    chg_las_pct    = flas(x_ov_hi)  - flas(x_ov_lo)
    chg_emit_pct   = femit(x_ov_hi) - femit(x_ov_lo)

    if x_ov_hi > x_ov_lo:
        xx_ov        = np.linspace(x_ov_lo, x_ov_hi, 1000)
        avg_diff_pct = float(np.trapz(flas(xx_ov) - femit(xx_ov), xx_ov) / (x_ov_hi - x_ov_lo))
    else:
        avg_diff_pct = float(flas(x_ov_lo) - femit(x_ov_lo))

    # ── автоматический общий Y-диапазон ──────────────────────────────────────
    # Собираем ВСЕ относительные значения (точки + линии фита) для обеих осей,
    # берём общий min/max и добавляем 20% отступа.
    good_las = laser_df["ok"].to_numpy(dtype=bool) if "ok" in laser_df.columns else np.ones(len(laser_df), dtype=bool)

    all_y_vals = np.concatenate([
        yl_rel[good_las & np.isfinite(yl_rel)],
        ye_rel[np.isfinite(ye_rel)],
        flas(xx),
        femit(xx),
    ])
    finite_vals = all_y_vals[np.isfinite(all_y_vals)]

    if rel_ylim is not None:
        # если передан --relative-ylim, используем его (оба пола одинаковые)
        shared_lo = 100.0 - rel_ylim
        shared_hi = 100.0 + rel_ylim
    else:
        pad = 0.20 * (finite_vals.ptp() if finite_vals.ptp() > 0 else 1.0)
        shared_lo = float(finite_vals.min()) - pad
        shared_hi = float(finite_vals.max()) + pad

    _apply_cms_rcparams()
    fig = plt.figure(figsize=(10.5, 9.5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1.4], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    _cms_label(ax_top, year)
    for ax in (ax_top, ax_bot):
        ax.grid(True, which="major", alpha=0.15, lw=0.8, color="0.5")

    C_LAS  = "#2166ac"
    C_EMIT = "#d6604d"

    # левая ось — laser
    ax_top.set_ylabel("Relative laser response [%]", color=C_LAS, labelpad=8)
    ax_top.tick_params(axis="y", labelcolor=C_LAS)
    ax_top.set_ylim(shared_lo, shared_hi)          # ← автоматически

    m_err_l   = good_las & np.isfinite(yel_rel)
    m_noerr_l = good_las & ~np.isfinite(yel_rel)
    if np.any(m_err_l):
        ax_top.errorbar(xl[m_err_l], yl_rel[m_err_l], yerr=yel_rel[m_err_l],
                        fmt="o", ms=5, lw=0.9, capsize=2, alpha=0.75,
                        color=C_LAS, label="Laser (4 BCIDs)")
    if np.any(m_noerr_l):
        ax_top.plot(xl[m_noerr_l], yl_rel[m_noerr_l], "o", ms=5, alpha=0.75, color=C_LAS)
    ax_top.plot(xx, flas(xx), "-", lw=2.2, color=C_LAS,
                label=f"Laser fit:  {slope_las_pct:+.4f}% / fb$^{{-1}}$")
    ax_top.axhline(100.0, ls=":", lw=0.8, color=C_LAS, alpha=0.35)

    # правая ось — sigma_vis (те же пределы!)
    ax_r = ax_top.twinx()
    ax_r.set_ylabel(fr"Relative $\sigma_{{\rm vis}}$ ({det}) [%]", color=C_EMIT, labelpad=8)
    ax_r.tick_params(axis="y", labelcolor=C_EMIT)
    ax_r.set_ylim(shared_lo, shared_hi)             # ← такой же диапазон

    m_err_e = np.isfinite(yee_rel)
    if np.any(m_err_e):
        ax_r.errorbar(xe[m_err_e], ye_rel[m_err_e], yerr=yee_rel[m_err_e],
                      fmt="s", ms=6, lw=0.9, capsize=2, alpha=0.85,
                      color=C_EMIT, label=fr"$\sigma_{{\rm vis}}$ per fill")
    if np.any(~m_err_e):
        ax_r.plot(xe[~m_err_e], ye_rel[~m_err_e], "s", ms=6, alpha=0.85, color=C_EMIT)
    ax_r.plot(xx, femit(xx), "--", lw=2.2, color=C_EMIT,
              label=fr"$\sigma_{{\rm vis}}$ fit:  {slope_emit_pct:+.4f}% / fb$^{{-1}}$")
    ax_r.axhline(100.0, ls=":", lw=0.8, color=C_EMIT, alpha=0.35)

    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.90, fontsize=11)

    # нижняя панель
    flas_xx  = flas(xx)
    femit_xx = femit(xx)
    ax_bot.plot(xx, flas_xx,  "-",  lw=2.0, color=C_LAS,  label="Laser fit")
    ax_bot.plot(xx, femit_xx, "--", lw=2.0, color=C_EMIT, label=fr"$\sigma_{{\rm vis}}$ fit")
    ax_bot.fill_between(xx, flas_xx, femit_xx,
                        where=(flas_xx >= femit_xx), alpha=0.25, color=C_LAS,
                        label="Laser > $\\sigma_{\\rm vis}$")
    ax_bot.fill_between(xx, flas_xx, femit_xx,
                        where=(flas_xx <  femit_xx), alpha=0.25, color=C_EMIT,
                        label="$\\sigma_{\\rm vis}$ > Laser")
    ax_bot.axhline(100.0, ls=":", lw=0.8, color="0.55", alpha=0.6)
    ax_bot.set_ylabel("Relative [%]", fontsize=12)
    ax_bot.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")
    ax_bot.legend(loc="lower left", fontsize=10, framealpha=0.88, ncol=2)

    txt = "\n".join([
        r"$\bf{Degradation\ slopes}$",
        f"  Laser:    {slope_las_pct:+.4f} % / fb$^{{-1}}$",
        f"  σ_vis:    {slope_emit_pct:+.4f} % / fb$^{{-1}}$",
        f"  Average:  {avg_slope_pct:+.4f} % / fb$^{{-1}}$",
        "",
        r"$\bf{Total\ change\ (overlap\ range)}$",
        f"  Laser:  {chg_las_pct:+.2f}%",
        f"  σ_vis:  {chg_emit_pct:+.2f}%",
        "",
        r"$\bf{Lumi\ impact\ estimate}$",
        fr"  $\langle\Delta L/L\rangle \approx$ {avg_diff_pct:+.3f} %",
    ])
    ax_bot.text(
        0.985, 0.97, txt, transform=ax_bot.transAxes,
        ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec="0.75"),
    )

    fig.savefig(os.path.join(output_dir, "degradation_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] degradation_comparison.png")
    print(f"     Shared Y-range:  [{shared_lo:.3f}, {shared_hi:.3f}]%")
    print(f"     Laser slope:     {slope_las_pct:+.5f} % / fb-1")
    print(f"     sigma_vis slope: {slope_emit_pct:+.5f} % / fb-1")
    print(f"     Average slope:   {avg_slope_pct:+.5f} % / fb-1")
    print(f"     Laser total d:   {chg_las_pct:+.3f} %  (overlap range)")
    print(f"     sigma_vis total: {chg_emit_pct:+.3f} %  (overlap range)")
    print(f"     <dL/L> est:      {avg_diff_pct:+.4f} %")


# ==============================================================================
# sigma_vis standalone relative plot  (NEW, mirrors laser relative plot)
# ==============================================================================

def plot_svis_relative(
    emit_df: pd.DataFrame,
    m: float, b: float, fit_mask: np.ndarray,
    output_dir: str, year: Optional[int],
    det: str, rel_ylim: Optional[float],
) -> None:
    x   = emit_df["ilumi"].to_numpy(dtype=np.float64)
    y   = emit_df["svis_mean"].to_numpy(dtype=np.float64)
    ye  = emit_df["svis_err"].to_numpy(dtype=np.float64)
    x0  = float(x[fit_mask].min())
    x1  = float(x[fit_mask].max())
    y0  = m * x0 + b
    if not np.isfinite(y0) or y0 == 0:
        raise RuntimeError(f"Bad sigma_vis normalisation y0={y0}")

    y_rel  = 100.0 * y  / y0
    ye_rel = 100.0 * ye / y0
    slope_pct     = 100.0 * m / y0
    total_chg_pct = 100.0 * (m * x1 + b) / y0 - 100.0

    fit_y_pct = 100.0 * (m * x + b) / y0
    resid     = y_rel - fit_y_pct

    _apply_cms_rcparams()
    fig = plt.figure(figsize=(9.0, 8.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax  = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    _cms_label(ax, year)
    for a in (ax, ax_r):
        a.grid(True, which="major", alpha=0.15, linewidth=0.8, color="0.5")

    m_err = np.isfinite(ye_rel)
    if np.any(m_err):
        ax.errorbar(x[m_err], y_rel[m_err], yerr=ye_rel[m_err],
                    fmt="s", ms=5, lw=0.9, capsize=2, alpha=0.8,
                    label=fr"$\sigma_{{\rm vis}}$ per fill")
    if np.any(~m_err):
        ax.plot(x[~m_err], y_rel[~m_err], "s", ms=5, alpha=0.8)

    xx = np.linspace(x0, x1, 300)
    ax.plot(xx, 100.0 * (m * xx + b) / y0, "-", lw=2.0, label="Linear fit")
    ax.axhline(100.0, ls="--", lw=1.0, color="0.55", alpha=0.8, label="100%")
    ax.set_ylabel(fr"Relative $\sigma_{{\rm vis}}$ ({det}) [%]")
    ax.legend(framealpha=0.88)
    txt = (
        f"slope = {slope_pct:+.4f} % / fb$^{{-1}}$\n"
        f"total change = {total_chg_pct:+.2f}%"
    )
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))
    if rel_ylim is not None:
        ax.set_ylim(100.0 - rel_ylim, 100.0 + rel_ylim)

    # residual panel
    if np.any(m_err):
        ax_r.errorbar(x[m_err], resid[m_err], yerr=ye_rel[m_err],
                      fmt="s", ms=3.5, lw=0.7, capsize=2, alpha=0.7, color="C1")
    if np.any(~m_err):
        ax_r.plot(x[~m_err], resid[~m_err], "s", ms=3.5, alpha=0.7, color="C1")
    ax_r.axhline(0, lw=1.4, ls="--", color="0.45")
    ax_r.set_ylabel("Residual [%]")
    ax_r.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")

    fig.savefig(os.path.join(output_dir, "svis_relative_vs_ilumi.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] svis_relative_vs_ilumi.png")

# ==============================================================================
# Diagnostics  (ORIGINAL logic, improved style)
# ==============================================================================

def make_diagnostics(
    df: pd.DataFrame, output_dir: str, year: Optional[int],
    *, sbil_min, sbil_max, y_min, y_max, trim_frac, min_points_plot, max_plots,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    good = df[df["ok"] & (df["n_points"] >= min_points_plot)].sort_values("n_points", ascending=False)
    made = 0
    for _, row in good.iterrows():
        if made >= max_plots:
            break
        fill, x, y = read_fill_sum(row["source_file"])
        mask_all = np.isfinite(x) & np.isfinite(y) & (y > 0)
        mask_sel = mask_all & (x >= sbil_min) & (x <= sbil_max) & (y >= y_min) & (y <= y_max)
        xx, yy = x[mask_sel], y[mask_sel]
        mu = trimmed_mean(yy, trim_frac) if yy.size else np.nan

        fig, ax = cms_figure(
            "SBIL [Hz/µb]", "Summed corrected laser response",
            year=year, title=f"Fill {int(fill)}",
        )
        ax.axvspan(sbil_min, sbil_max, color="0.80", alpha=0.30, zorder=0, label="SBIL window")
        ax.axhspan(y_min, y_max, color="0.88", alpha=0.25, zorder=0, label="Response window")
        ax.plot(x[mask_all], y[mask_all], ".", ms=2.5, alpha=0.12, color="0.5")
        ax.plot(xx, yy, ".", ms=3.5, alpha=0.80, label="Selected points")
        if np.isfinite(mu):
            ax.axhline(mu, lw=1.6, label=f"Trimmed mean = {mu:.4f}")
        ax.legend(fontsize=10, framealpha=0.85)
        txt = f"N = {int(row['n_points'])}" + (f"\ntrimmed mean = {mu:.4f}" if np.isfinite(mu) else "")
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))
        fig.savefig(os.path.join(output_dir, f"diag_fill_{int(fill)}.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)
        made += 1


# ==============================================================================
# CLI  (ORIGINAL args + two new emittance args)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Yearly laser trend + optional emittance-scan comparison."
    )
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--year",       type=int, default=None)

    parser.add_argument("--bril-pattern",      default=None)
    parser.add_argument("--reference-lumi",    choices=["delivered", "recorded"], default="delivered")
    parser.add_argument("--stable-beams-only", action="store_true")

    parser.add_argument("--sbil-min",     type=float, default=6.0)
    parser.add_argument("--sbil-max",     type=float, default=8.0)
    parser.add_argument("--response-min", type=float, default=2.0)
    parser.add_argument("--response-max", type=float, default=3.0)
    parser.add_argument("--trim-frac",    type=float, default=0.2)
    parser.add_argument("--min-points",   type=int,   default=20)
    parser.add_argument("--fit-nsigma",   type=float, default=3.0)
    parser.add_argument("--relative-ylim",type=float, default=5.0)

    # new: emittance scan comparison
    parser.add_argument("--emittance-csv", default=None,
                        help="Emittance-scan CSV (optional); enables comparison plots")
    parser.add_argument("--det",           default="HFET",
                        help="Detector to use from emittance CSV (default: HFET)")

    parser.add_argument("--make-diagnostics",  action="store_true")
    parser.add_argument("--min-points-plot",   type=int, default=50)
    parser.add_argument("--max-diagnostics",   type=int, default=40)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- laser pipeline (ORIGINAL, unchanged) ---------------------------------
    df = build_fill_table(
        args.input_dir,
        sbil_min=args.sbil_min, sbil_max=args.sbil_max,
        y_min=args.response_min, y_max=args.response_max,
        trim_frac=args.trim_frac, min_points=args.min_points,
    )

    fill_to_ilumi = None
    if args.bril_pattern:
        fill_to_ilumi = make_fill_to_ilumi(
            args.bril_pattern,
            prefer=args.reference_lumi,
            stable_beams_only=args.stable_beams_only,
            zero_at_first=True,
        )
        df = attach_ilumi(df, fill_to_ilumi)
    else:
        df = df.copy()
        df["ilumi"] = df["fill"].astype(float)

    m_las, b_las, fit_las_df = robust_line_fit(df, nsigma=args.fit_nsigma)
    df = normalize_to_fit(df, fit_las_df, m_las, b_las)

    csv_path = os.path.join(args.output_dir, "laser_year_window.csv")
    df.to_csv(csv_path, index=False)

    plot_response_vs_ilumi(df, fit_las_df, m_las, b_las, args.output_dir, args.year)
    plot_relative_vs_ilumi(df, fit_las_df, m_las, b_las, args.output_dir, args.year, args.relative_ylim)

    # -- emittance comparison (optional, NEW) ---------------------------------
    if args.emittance_csv:
        emit_df = load_emittance_csv(
            args.emittance_csv,
            det=args.det,
            fill_to_ilumi=fill_to_ilumi,
        )
        m_emit, b_emit, mask_emit = robust_line_fit_xy(
            emit_df["ilumi"].to_numpy(dtype=np.float64),
            emit_df["svis_mean"].to_numpy(dtype=np.float64),
            nsigma=args.fit_nsigma,
        )
        plot_svis_relative(
            emit_df, m_emit, b_emit, mask_emit,
            args.output_dir, args.year, args.det, args.relative_ylim,
        )
        plot_degradation_comparison(
            df, m_las, b_las, fit_las_df,
            emit_df, m_emit, b_emit, mask_emit,
            args.output_dir, args.year, args.relative_ylim, args.det,
        )

    # -- diagnostics (ORIGINAL) -----------------------------------------------
    if args.make_diagnostics:
        make_diagnostics(
            df, os.path.join(args.output_dir, "diagnostics"), args.year,
            sbil_min=args.sbil_min, sbil_max=args.sbil_max,
            y_min=args.response_min, y_max=args.response_max,
            trim_frac=args.trim_frac,
            min_points_plot=args.min_points_plot, max_plots=args.max_diagnostics,
        )

    print(f"\nTotal fills:      {len(df)}")
    print(f"Good fills:       {int(df['ok'].sum())}")
    print(f"Fit points used:  {len(fit_las_df)}")
    print(f"Trend fit:        y = {m_las:.8g} * x + {b_las:.8g}")
    print(f"CSV:              {csv_path}")
    print(f"Plots:            {args.output_dir}")


if __name__ == "__main__":
    main()