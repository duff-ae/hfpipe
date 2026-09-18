#!/usr/bin/env python3
'''
plot_type1_drift.py

Plot Type-1 afterglow coefficients and the resulting correction functions
over the year to visualise detector drift and fill-to-fill spread.

Reads type1_coeffs_fill{fill}.h5 files produced by save_type1_coeffs(),
cross-matches fills with brilcalc luminosity, and produces one PNG per
offset showing coefficients + relative change [%].

Usage:
  python3 plot_type1_drift.py \
    --input-dir /eos/home-a/alshevel/hfpipe/plots/24/ \
    --output-dir type1/plots_type1_drift_24/ \
    --year 2024 \
    --offsets 1 \
    --bril-pattern "input/luminosity/*.csv" \
    --stable-beams-only \
    --mad-nsigma 10
'''

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

try:
    import mplhep as hep
    _HEP = True
except ImportError:
    _HEP = False
    print("[warn] mplhep not found — plain matplotlib style")


# ── style ────────────────────────────────────────────────────────────────────

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

def _cms_label(ax, year):
    if _HEP:
        rlabel = f"{year}, 13.6 TeV" if year else "13.6 TeV"
        hep.cms.label("Preliminary", data=True, loc=0, rlabel=rlabel, ax=ax)

def _grid(ax):
    ax.grid(True, which="major", alpha=0.15, lw=0.8, color="0.5")


# ── IO ───────────────────────────────────────────────────────────────────────

def find_type1_files(d: str) -> list[str]:
    return sorted(glob.glob(os.path.join(d, "**", "type1_coeffs_fill*.h5"), recursive=True))

def parse_fill(path: str) -> int:
    m = re.search(r"type1_coeffs_fill(\d+)\.h5$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse fill from {path}")
    return int(m.group(1))

def read_coeffs(path: str, requested_offsets: list[int]) -> list[dict]:
    """Read p0, p1, p2 for all requested Type-1 offsets."""
    rows = []
    with h5py.File(path, "r") as h5:
        fill    = int(h5.attrs.get("fill", parse_fill(path)))
        p0      = np.asarray(h5["p0"],      dtype=np.float64)
        p1      = np.asarray(h5["p1"],      dtype=np.float64)
        p2      = np.asarray(h5["p2"],      dtype=np.float64)
        offsets = np.asarray(h5["offsets"], dtype=np.int64)
        orders  = np.asarray(h5["orders"],  dtype=np.int32)

    fitted = set(int(v) for v in offsets)
    for off in requested_offsets:
        if off not in fitted or off >= len(p0):
            continue
        rows.append(dict(
            fill=fill,
            offset=off,
            order=int(orders[off]),
            p0=float(p0[off]),
            p1=float(p1[off]),
            p2=float(p2[off]),
        ))
    return rows


# ── brilcalc ─────────────────────────────────────────────────────────────────

def make_fill_to_ilumi(pattern: str, prefer: str, stable_only: bool) -> dict[int, float]:
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


# ── MAD rejection ─────────────────────────────────────────────────────────────

def mad_mask(vals: np.ndarray, nsigma: float) -> np.ndarray:
    """True = good point."""
    fin = np.isfinite(vals)
    if fin.sum() == 0:
        return fin
    med = np.median(vals[fin])
    mad = np.median(np.abs(vals[fin] - med))
    sig = 1.4826 * mad
    if sig == 0.0:
        return fin
    return fin & (np.abs(vals - med) <= nsigma * sig)


# ── per-offset plot ───────────────────────────────────────────────────────────

C_GOOD    = "#2166ac"
C_OUTLIER = "#d6604d"

def plot_p0_offset(
    sub: pd.DataFrame,
    offset: int,
    output_dir: str,
    year: Optional[int],
    x_label: str,
    mad_nsigma: float,
    highlight_fill : int,
) -> None:
    order         = int(sub["order"].iloc[0]) if "order" in sub.columns else 1
    fit_order_str = f"poly{order}" if order > 1 else "linear"

    x    = sub["ilumi"].to_numpy(dtype=np.float64)
    y    = sub["p0"].to_numpy(dtype=np.float64)
    good = mad_mask(y, nsigma=mad_nsigma)

    n_total   = int(np.isfinite(y).sum())
    n_dropped = int((~good & np.isfinite(y)).sum())

    ref = float(np.median(y[good])) if good.sum() > 0 else np.nan

    _apply_cms_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    _cms_label(ax, year)
    _grid(ax)

    # outliers first (behind)
    out_mask = ~good & np.isfinite(y)

    # good points
    ax.plot(x[good], y[good],
            "o", ms=5, color=C_GOOD, alpha=0.80, zorder=3)

    # median line
    if np.isfinite(ref):
        ax.axhline(ref, ls="--", lw=1.4, color="0.35", zorder=1,
                   label=f"Median = {ref:.5g}")

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"$p_0$  [BX+{offset}, {fit_order_str} fit]")
    ax.legend(framealpha=0.88, loc="best")

    box_txt = "\n".join([
        rf"$\bf{{BX+{offset}}}$  ({fit_order_str})",
        f"Median $p_0$ = {ref:.5g}" if np.isfinite(ref) else "",
    ]).strip()
    ax.text(0.985, 0.97, box_txt,
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.92, ec="0.75"))

    if highlight_fill is not None and highlight_fill in sub["fill"].values:
        idx = sub["fill"].values == highlight_fill
        ax.plot(x[idx], y[idx],
                "*", ms=14, color="gold", markeredgecolor="k", markeredgewidth=0.8,
                zorder=5, label=f"Fill {highlight_fill}")

    fig.suptitle(f"$p_0$ drift — BX+{offset}", y=1.01, fontsize=13)
    out_path = os.path.join(output_dir, f"p0_drift_offset{offset}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {out_path}  (dropped {n_dropped}/{n_total}, median={ref:.5g})")


# ── summary ───────────────────────────────────────────────────────────────────

def plot_p0_summary(
    df: pd.DataFrame,
    offsets: list[int],
    output_dir: str,
    year: Optional[int],
    x_label: str,
    mad_nsigma: float,
) -> None:
    _apply_cms_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    _cms_label(ax, year)
    _grid(ax)

    colors  = plt.cm.tab10(np.linspace(0, 0.75, len(offsets)))
    markers = ["o", "s", "^", "D"]

    for i, off in enumerate(offsets):
        sub = df[df["offset"] == off].sort_values("ilumi")
        if sub.empty:
            continue
        x    = sub["ilumi"].to_numpy(dtype=np.float64)
        y    = sub["p0"].to_numpy(dtype=np.float64)
        good = mad_mask(y, nsigma=mad_nsigma)
        if good.sum() == 0:
            continue
        ref = float(np.median(y[good]))
        if ref == 0.0:
            continue
        y_rel = 100.0 * (y - ref) / np.abs(ref)

        c  = colors[i]
        mk = markers[i % len(markers)]
        ax.plot(x[good], y_rel[good],
                ls="none", marker=mk, ms=5, color=c, alpha=0.75,
                label=f"BX+{off}  (med={ref:.4g})")

    ax.axhline(0.0, ls=":", lw=1.0, color="0.5", alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$(p_0 - \mathrm{med})\,/\,|\mathrm{med}|$  [%]")
    ax.legend(framealpha=0.88, title=f"MAD {mad_nsigma:.0f}σ cut")
    ax.set_title(r"$p_0$ — all offsets (normalised to median)", pad=10)

    out_path = os.path.join(output_dir, "p0_drift_summary.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  {out_path}")



# ── correction functions over the year ────────────────────────────────────────

def plot_type1_function_family(
    df: pd.DataFrame,
    offset: int,
    output_dir: str,
    year: Optional[int],
    sbil_max: float,
    highlight_fill: Optional[int],
    color_label: str,
) -> None:
    """
    Overlay the fitted Type-1 residual-fraction function for every fill:

        f_t(SBIL) = p0[t] + p1[t] * SBIL + p2[t] * SBIL^2

    No MAD rejection is applied here on purpose: the point of this plot is
    to expose the full fill-to-fill spread, including anomalous fits.
    """
    sub = df[df["offset"] == offset].sort_values("ilumi").copy()
    if sub.empty:
        return

    finite = (
        np.isfinite(sub["p0"].to_numpy(dtype=np.float64))
        & np.isfinite(sub["p1"].to_numpy(dtype=np.float64))
        & np.isfinite(sub["p2"].to_numpy(dtype=np.float64))
    )
    sub = sub.loc[finite].copy()
    if sub.empty:
        return

    x = np.linspace(0.0, float(sbil_max), 301, dtype=np.float64)

    p0 = sub["p0"].to_numpy(dtype=np.float64)[:, None]
    p1 = sub["p1"].to_numpy(dtype=np.float64)[:, None]
    p2 = sub["p2"].to_numpy(dtype=np.float64)[:, None]
    curves = p0 + p1 * x[None, :] + p2 * x[None, :] ** 2

    # Use integrated luminosity for the colour progression when available.
    cvals = sub["ilumi"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(cvals)):
        cvals = sub["fill"].to_numpy(dtype=np.float64)

    cmin = float(np.nanmin(cvals))
    cmax = float(np.nanmax(cvals))
    if cmax == cmin:
        cmax = cmin + 1.0

    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = plt.get_cmap("viridis")

    _apply_cms_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    _cms_label(ax, year)
    _grid(ax)

    # Every fill is drawn: no outlier rejection here.
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.plot(
            x,
            curves[i],
            color=cmap(norm(cvals[i])),
            alpha=0.22,
            lw=0.9,
            zorder=1,
        )

    # Distribution envelope makes the amount of fill-to-fill spread readable
    # even when there are many fills.
    q05, q16, q50, q84, q95 = np.nanpercentile(
        curves, [5, 16, 50, 84, 95], axis=0
    )
    ax.fill_between(x, q05, q95, color="0.5", alpha=0.10, lw=0,
                    label="5–95% of fills", zorder=2)
    ax.fill_between(x, q16, q84, color="0.35", alpha=0.16, lw=0,
                    label="16–84% of fills", zorder=2)
    ax.plot(x, q50, color="black", lw=2.0, label="Median", zorder=4)

    # Optional highlighted fill.
    if highlight_fill is not None:
        hit = sub[sub["fill"] == highlight_fill]
        if not hit.empty:
            row = hit.iloc[0]
            y_hi = (
                float(row["p0"])
                + float(row["p1"]) * x
                + float(row["p2"]) * x**2
            )
            ax.plot(
                x, y_hi,
                color="gold", lw=2.6,
                markeredgecolor="black",
                label=f"Fill {highlight_fill}",
                zorder=5,
            )

    ax.axhline(0.0, ls=":", lw=1.0, color="0.4", alpha=0.8)
    ax.set_xlim(0.0, float(sbil_max))
    ax.set_xlabel(r"SBIL / colliding-BX $\mu$")
    ax.set_ylabel(r"Type-1 residual fraction  $p_0 + p_1 x + p_2 x^2$")
    ax.set_title(f"Type-1 correction functions — BX+{offset}", pad=10)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label(color_label)

    ax.legend(framealpha=0.88, loc="best")

    out_path = os.path.join(output_dir, f"type1_function_family_offset{offset}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[OK]  {out_path}  "
        f"({len(sub)} fills, SBIL 0..{sbil_max:g})"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",         required=True)
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--year",              type=int, default=None)
    parser.add_argument("--offsets",           type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--bril-pattern",      default=None)
    parser.add_argument("--reference-lumi",    choices=["delivered", "recorded"], default="delivered")
    parser.add_argument("--stable-beams-only", action="store_true")
    parser.add_argument("--mad-nsigma",        type=float, default=3.0)
    parser.add_argument("--highlight-fill", type=int, default=8999)
    parser.add_argument("--sbil-max", type=float, default=3.0,
                        help="Maximum x value for Type-1 function-family plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fill_to_ilumi = None
    if args.bril_pattern:
        fill_to_ilumi = make_fill_to_ilumi(
            args.bril_pattern, args.reference_lumi, args.stable_beams_only)
        x_label = r"Integrated luminosity [fb$^{-1}$]"
        color_label = r"Integrated luminosity [fb$^{-1}$]"
        print(f"[info] {len(fill_to_ilumi)} fills in brilcalc table")
    else:
        x_label = "Fill number"
        color_label = "Fill number"
        print("[info] no bril-pattern — using fill number as x-axis")

    files = find_type1_files(args.input_dir)
    if not files:
        raise RuntimeError(f"No type1_coeffs_fill*.h5 under {args.input_dir}")
    print(f"[info] {len(files)} coefficient file(s)")

    rows = []
    for path in files:
        try:
            for r in read_coeffs(path, args.offsets):
                fill   = r["fill"]
                r["ilumi"] = fill_to_ilumi.get(fill, np.nan) if fill_to_ilumi else float(fill)
                rows.append(r)
        except Exception as e:
            print(f"[warn] {path}: {e}")

    df = pd.DataFrame(rows).sort_values(["offset", "ilumi"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No data loaded")
    print(f"[info] {len(df)} rows, offsets: {sorted(df['offset'].unique())}")

    df.to_csv(os.path.join(args.output_dir, "type1_coeffs_all.csv"), index=False)
    # Keep the old filename as well so existing downstream use does not break.
    df.to_csv(os.path.join(args.output_dir, "p0_all.csv"), index=False)

    for off in args.offsets:
        sub = df[df["offset"] == off].copy()
        if sub.empty:
            print(f"[warn] offset {off}: no data")
            continue
        plot_p0_offset(
            sub, off, args.output_dir, args.year,
            x_label, args.mad_nsigma, args.highlight_fill,
        )
        plot_type1_function_family(
            df=df,
            offset=off,
            output_dir=args.output_dir,
            year=args.year,
            sbil_max=args.sbil_max,
            highlight_fill=args.highlight_fill,
            color_label=color_label,
        )

    plot_p0_summary(df, args.offsets, args.output_dir, args.year, x_label, args.mad_nsigma)
    print("\nDone.")


if __name__ == "__main__":
    main()