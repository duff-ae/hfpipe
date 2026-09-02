#!/usr/bin/env python3
"""
plot_svis_type1_corrected.py

Apply Type-1 afterglow correction (p0, BX+1) to emittance-scan sigma_vis values
and compare raw vs corrected side-by-side.

Correction formula:
    svis_corr = svis_raw / (1 + p0)

Usage:
  python3 plot_svis_type1_corrected.py \
      --emittance-csv  input/scans/scan24_new.csv \
      --type1-dir      /eos/home-a/alshevel/hf_plots_2024/ \
      --output-dir     plots_svis_corrected/ \
      --year           2024 \
      --det            HFET \
      --bril-pattern   "input/luminosity/*.csv" \
      --stable-beams-only \
      --fit-nsigma     3.0 \
      --relative-ylim  5.0
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

try:
    import mplhep as hep
    _HEP = True
except ImportError:
    _HEP = False
    print("[warn] mplhep not found — plain matplotlib style")


# ── constants ─────────────────────────────────────────────────────────────────

TYPE1_OFFSET = 1   # BX+1 only

# emittance-scan quality cuts (same as fit_laser_year_window.py)
_VALID_SCAN_NAMES  = {"emit9", "emit15"}
_VALID_SCAN_TIMING = "early"
_VALID_SCAN_STEPS  = 9
_VALID_SCAN_BETA   = 120
_MIN_NBCID         = 400
_SVIS_OUTLIER_FRAC = 0.10


# ── CMS style ─────────────────────────────────────────────────────────────────

def _apply_cms_style() -> None:
    if _HEP:
        plt.style.use(hep.style.CMS)
    mpl.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
        "figure.dpi": 150, "lines.linewidth": 1.6, "axes.linewidth": 1.1,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })


def _cms_label(ax, year: Optional[int]) -> None:
    if _HEP:
        rlabel = f"{year}, 13.6 TeV" if year else "13.6 TeV"
        hep.cms.label("Preliminary", data=True, loc=0, rlabel=rlabel, ax=ax)


def _grid(ax) -> None:
    ax.grid(True, which="major", alpha=0.15, lw=0.8, color="0.5")


# ── brilcalc ──────────────────────────────────────────────────────────────────

def make_fill_to_ilumi(pattern: str, prefer: str, stable_only: bool) -> dict:
    files = sorted(glob.iglob(pattern))
    if not files:
        raise RuntimeError(f"No BRIL CSV at {pattern}")
    parts = []
    for fp in files:
        df = pd.read_csv(fp, sep=",", skiprows=1, engine="python")
        df.columns = df.columns.astype(str).str.strip()
        if "#run:fill" not in df.columns:
            continue
        cands = (["recorded(/fb)", "recorded(/ub)"] if prefer == "recorded"
                 else ["delivered(/fb)", "delivered(/ub)"])
        col = next((c for c in cands if c in df.columns), None)
        if col is None:
            continue
        val    = pd.to_numeric(df[col], errors="coerce")
        val_fb = val * 1e-9 if col.endswith("(/ub)") else val
        rf     = df["#run:fill"].astype(str).str.split(":", n=1, expand=True)
        if rf.shape[1] < 2:
            continue
        out = pd.DataFrame({"fill":     pd.to_numeric(rf[1],   errors="coerce"),
                            "value_fb": pd.to_numeric(val_fb,  errors="coerce")}).dropna()
        out["fill"] = out["fill"].astype(int)
        if stable_only and "beamstatus" in df.columns:
            bs  = df["beamstatus"].astype(str).str.strip().str.upper()
            out = out.loc[bs == "STABLE BEAMS"].copy()
        parts.append(out[["fill", "value_fb"]])
    if not parts:
        raise RuntimeError("No usable BRIL rows")
    byls = pd.concat(parts, ignore_index=True)
    per  = (byls.groupby("fill", as_index=False)["value_fb"].sum()
            .sort_values("fill").reset_index(drop=True))
    per["ilumi"] = per["value_fb"].cumsum() - per["value_fb"].cumsum().iloc[0]
    return dict(zip(per["fill"].astype(int), per["ilumi"].astype(float)))


# ── type1 p0 reader — same logic as read_p0() in plot_type1_drift.py ─────────

def find_type1_files(d: str) -> list:
    return sorted(glob.glob(os.path.join(d, "**", "type1_coeffs_fill*.h5"), recursive=True))


def parse_fill_from_type1(path: str) -> int:
    m = re.search(r"type1_coeffs_fill(\d+)\.h5$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse fill from {path}")
    return int(m.group(1))


def build_fill_to_p0(type1_dir: str, offset: int = TYPE1_OFFSET) -> dict:
    result = {}
    for path in find_type1_files(type1_dir):
        try:
            with h5py.File(path, "r") as h5:
                fill    = int(h5.attrs.get("fill", parse_fill_from_type1(path)))
                p0      = np.asarray(h5["p0"],      dtype=np.float64)
                offsets = np.asarray(h5["offsets"], dtype=np.int64)
            fitted = set(int(v) for v in offsets)
            if offset not in fitted or offset >= len(p0):
                continue
            val = float(p0[offset])
            if np.isfinite(val):
                result[fill] = val
        except Exception as e:
            print(f"[warn] {path}: {e}")
    print(f"[type1] loaded p0 (BX+{offset}) for {len(result)} fills")
    return result


# ── emittance-scan loader ─────────────────────────────────────────────────────

def load_emittance_csv(
    csv_path: str,
    det: str,
    fill_to_ilumi: Optional[dict],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df["det"].astype(str).str.upper() == det.upper()].copy()
    if df.empty:
        raise RuntimeError(f"No rows for detector '{det}' in {csv_path}")

    for col in ("svis", "svisrms", "fill", "iLumi", "scanSteps", "scanBeta", "nbcid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["fill", "svis"]).copy()
    df["fill"] = df["fill"].astype(int)

    n_before = len(df)
    if "scanName"   in df.columns:
        df = df[df["scanName"].astype(str).str.strip().isin(_VALID_SCAN_NAMES)]
    if "scanTiming" in df.columns:
        df = df[df["scanTiming"].astype(str).str.strip().str.lower() == _VALID_SCAN_TIMING]
    if "scanSteps"  in df.columns:
        df = df[df["scanSteps"] == _VALID_SCAN_STEPS]
    if "scanBeta"   in df.columns:
        df = df[df["scanBeta"]  == _VALID_SCAN_BETA]
    if "nbcid"      in df.columns:
        df = df[df["nbcid"] > _MIN_NBCID]
    print(f"[emittance] quality cuts: {n_before} -> {len(df)} rows kept")
    if df.empty:
        raise RuntimeError("No rows survive quality cuts")

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
        rows.append({"fill": int(fill), "svis_raw": mu, "svis_err": mu_err, "n_scans": len(svis)})

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid points after per-fill averaging")

    med     = float(np.median(out["svis_raw"].to_numpy()))
    rel_dev = np.abs(out["svis_raw"].to_numpy() - med) / med
    mask_ok = rel_dev <= _SVIS_OUTLIER_FRAC
    n_drop  = int((~mask_ok).sum())
    if n_drop:
        print(f"[emittance] outlier cut (>10% from median={med:.1f}): "
              f"dropping {n_drop} fill(s): {out.loc[~mask_ok,'fill'].tolist()}")
    out = out.loc[mask_ok].copy()
    if out.empty:
        raise RuntimeError("No fills survive outlier cut")

    if fill_to_ilumi is not None:
        out["ilumi"] = out["fill"].map(fill_to_ilumi)
    elif "iLumi" in df.columns:
        ilumi_map = df.groupby("fill")["iLumi"].mean().to_dict()
        out["ilumi"] = out["fill"].map(ilumi_map)
    else:
        raise RuntimeError("No iLumi source available")

    return out.dropna(subset=["ilumi"]).sort_values("ilumi").reset_index(drop=True)


# ── type1 correction ──────────────────────────────────────────────────────────

def apply_type1_correction(emit_df: pd.DataFrame, fill_to_p0: dict) -> pd.DataFrame:
    out = emit_df.copy()
    out["p0"] = out["fill"].map(fill_to_p0)

    n_before = len(out)
    out = out.dropna(subset=["p0"]).copy()
    n_after  = len(out)
    if n_before != n_after:
        print(f"[type1] dropped {n_before - n_after} fills with no p0 match "
              f"({n_after} remain)")

    denom = 1.0 + out["p0"].to_numpy(dtype=np.float64)
    out["svis_corr"]     = out["svis_raw"].to_numpy(dtype=np.float64) / denom
    out["svis_corr_err"] = out["svis_err"].to_numpy(dtype=np.float64) / denom
    return out.reset_index(drop=True)


# ── robust fit ────────────────────────────────────────────────────────────────

def mad_std(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(arr - np.median(arr))))


def robust_line_fit_xy(x: np.ndarray, y: np.ndarray, nsigma: float = 3.0) -> tuple:
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


# ── plot helper ───────────────────────────────────────────────────────────────

def _draw_relative_panel(
    fig, gs_top, gs_bot,
    x: np.ndarray, y_rel: np.ndarray, ye_rel: np.ndarray,
    m: float, b: float, y0: float,
    x0: float, x1: float,
    marker: str, color: str, label: str,
    title: str, ylabel: str,
    year: Optional[int],
    rel_ylim: Optional[float],
) -> None:
    ax     = fig.add_subplot(gs_top)
    ax_res = fig.add_subplot(gs_bot, sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    _cms_label(ax, year)
    _grid(ax)
    _grid(ax_res)

    xx        = np.linspace(x0, x1, 300)
    fit_pct   = 100.0 * (m * xx + b) / y0
    fit_at_x  = 100.0 * (m * x  + b) / y0
    resid     = y_rel - fit_at_x

    slope_pct     = 100.0 * m / y0
    total_chg_pct = 100.0 * (m * x1 + b) / y0 - 100.0

    m_err = np.isfinite(ye_rel)
    if np.any(m_err):
        ax.errorbar(x[m_err], y_rel[m_err], yerr=ye_rel[m_err],
                    fmt=marker, ms=5, lw=0.9, capsize=2, alpha=0.80,
                    color=color, label=label)
    if np.any(~m_err):
        ax.plot(x[~m_err], y_rel[~m_err], marker, ms=5, alpha=0.80, color=color)

    ax.plot(xx, fit_pct, "-", lw=2.0, color=color, alpha=0.7, label="Linear fit")
    ax.axhline(100.0, ls="--", lw=1.0, color="0.55", alpha=0.8, label="100%")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.88)
    if rel_ylim is not None:
        ax.set_ylim(100.0 - rel_ylim, 100.0 + rel_ylim)
    ax.set_title(title, pad=6, fontsize=12)

    txt = (
        f"slope = {slope_pct:+.4f} % / fb$^{{-1}}$\n"
        f"total change = {total_chg_pct:+.2f}%"
    )
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.8"))

    if np.any(m_err):
        ax_res.errorbar(x[m_err], resid[m_err], yerr=ye_rel[m_err],
                        fmt=marker, ms=3.5, lw=0.7, capsize=2, alpha=0.70, color=color)
    if np.any(~m_err):
        ax_res.plot(x[~m_err], resid[~m_err], marker, ms=3.5, alpha=0.70, color=color)
    ax_res.axhline(0, lw=1.4, ls="--", color="0.45")
    ax_res.set_ylabel("Residual [%]")
    ax_res.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_svis_raw_vs_corrected(
    df: pd.DataFrame,
    output_dir: str,
    year: Optional[int],
    det: str,
    fit_nsigma: float,
    rel_ylim: Optional[float],
) -> None:
    x       = df["ilumi"].to_numpy(dtype=np.float64)
    raw     = df["svis_raw"].to_numpy(dtype=np.float64)
    cor     = df["svis_corr"].to_numpy(dtype=np.float64)
    raw_err = df["svis_err"].to_numpy(dtype=np.float64)
    cor_err = df["svis_corr_err"].to_numpy(dtype=np.float64)

    m_raw, b_raw, _ = robust_line_fit_xy(x, raw, nsigma=fit_nsigma)
    m_cor, b_cor, _ = robust_line_fit_xy(x, cor, nsigma=fit_nsigma)

    x0 = float(x.min())
    x1 = float(x.max())

    y0_raw = m_raw * x0 + b_raw
    y0_cor = m_cor * x0 + b_cor

    raw_rel     = 100.0 * raw     / y0_raw
    raw_err_rel = 100.0 * raw_err / y0_raw
    cor_rel     = 100.0 * cor     / y0_cor
    cor_err_rel = 100.0 * cor_err / y0_cor

    C_RAW = "#2166ac"
    C_COR = "#1a9641"

    _apply_cms_style()
    fig = plt.figure(figsize=(17.0, 8.0))

    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28)
    gs_l  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                             height_ratios=[3, 1], hspace=0.08)
    gs_r  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                             height_ratios=[3, 1], hspace=0.08)

    _draw_relative_panel(
        fig, gs_l[0], gs_l[1],
        x=x, y_rel=raw_rel, ye_rel=raw_err_rel,
        m=m_raw, b=b_raw, y0=y0_raw, x0=x0, x1=x1,
        marker="s", color=C_RAW,
        label=fr"$\sigma_{{\rm vis}}^{{\rm raw}}$ per fill",
        title=fr"Raw $\sigma_{{\rm vis}}$ ({det})",
        ylabel=fr"Relative $\sigma_{{\rm vis}}^{{\rm raw}}$ ({det}) [%]",
        year=year, rel_ylim=rel_ylim,
    )

    _draw_relative_panel(
        fig, gs_r[0], gs_r[1],
        x=x, y_rel=cor_rel, ye_rel=cor_err_rel,
        m=m_cor, b=b_cor, y0=y0_cor, x0=x0, x1=x1,
        marker="^", color=C_COR,
        label=fr"$\sigma_{{\rm vis}}^{{\rm corr}}$ per fill",
        title=fr"Type-1 corrected $\sigma_{{\rm vis}}$ ({det})",
        ylabel=fr"Relative $\sigma_{{\rm vis}}^{{\rm corr}}$ ({det}) [%]",
        year=year, rel_ylim=rel_ylim,
    )

    fig.suptitle(
        fr"$\sigma_{{\rm vis}}$ ({det}): raw vs Type-1 corrected (BX+{TYPE1_OFFSET})",
        y=1.01, fontsize=14,
    )

    out_path = os.path.join(output_dir, "svis_type1_corrected.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")

    slope_raw_pct = 100.0 * m_raw / y0_raw
    slope_cor_pct = 100.0 * m_cor / y0_cor
    print(f"     Raw slope:  {slope_raw_pct:+.5f} % / fb-1")
    print(f"     Corr slope: {slope_cor_pct:+.5f} % / fb-1")
    print(f"     Delta:      {slope_cor_pct - slope_raw_pct:+.5f} % / fb-1")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emittance-csv",     required=True)
    parser.add_argument("--type1-dir",         required=True)
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--year",              type=int, default=None)
    parser.add_argument("--det",               default="HFET")
    parser.add_argument("--bril-pattern",      default=None)
    parser.add_argument("--reference-lumi",    choices=["delivered", "recorded"], default="delivered")
    parser.add_argument("--stable-beams-only", action="store_true")
    parser.add_argument("--fit-nsigma",        type=float, default=3.0)
    parser.add_argument("--relative-ylim",     type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fill_to_ilumi = None
    if args.bril_pattern:
        fill_to_ilumi = make_fill_to_ilumi(
            args.bril_pattern, args.reference_lumi, args.stable_beams_only)

    emit_df    = load_emittance_csv(args.emittance_csv, det=args.det,
                                    fill_to_ilumi=fill_to_ilumi)
    fill_to_p0 = build_fill_to_p0(args.type1_dir, offset=TYPE1_OFFSET)
    df         = apply_type1_correction(emit_df, fill_to_p0)

    if df.empty:
        raise RuntimeError("No fills survived after matching svis with p0")

    csv_path = os.path.join(args.output_dir, "svis_type1_corrected.csv")
    df[["fill", "ilumi", "p0", "svis_raw", "svis_err", "svis_corr", "svis_corr_err"]].to_csv(
        csv_path, index=False)
    print(f"[CSV] {csv_path}")

    plot_svis_raw_vs_corrected(
        df, output_dir=args.output_dir, year=args.year, det=args.det,
        fit_nsigma=args.fit_nsigma, rel_ylim=args.relative_ylim,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()