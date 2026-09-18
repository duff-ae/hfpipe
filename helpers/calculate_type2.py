#!/usr/bin/env python3
"""Standalone single-bunch Type-2 / HFSBR calibration."""
from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from hfcore.config import load_config
from hfcore.hd5schema import BX_LEN
from hfcore.io import load_active_mask, load_hd5_to_arrays
from hfcore.pedestal import subtract_fixed_pedestal_mod4
from hfcore.plotter import create_double_figure, create_figure

log = logging.getLogger("hfpipe.calculate_type2")

# Analysis defaults.  These are intentionally here rather than hidden in the
# production config: this is a calibration helper, not a production step.
PEDESTAL_ROWS = 10
PEDESTAL_BX_START = 500
PEDESTAL_BX_STOP = 3000
MIN_SBIL = 2.0
MIN_PHYSICS_ROWS = 100
SBIL_BIN_WIDTH = 2
TAIL_FIT_START = 1500
FIT_STOP = 3000
FULL_FIT_START = 4


def exp_fit(x, a, b, c):
    return a * np.exp(-b * np.asarray(x)) + c


def moyal_pdf(x, loc, scale):
    z = (np.asarray(x) - loc) / scale
    return np.exp(-0.5 * (z + np.exp(np.clip(-z, -700, 700)))) / (
        np.sqrt(2 * np.pi) * scale
    )


def full_fit(x, a1, b1, a2, b2, am, mu, sigma, c):
    x = np.asarray(x, dtype=float)
    return (
        a1 * np.exp(-b1 * x)
        + a2 * np.exp(-b2 * x)
        + am * moyal_pdf(x, mu, sigma)
        + c
    )


def select_fill(data, fill):
    if "fillnum" not in data:
        return data
    sel = np.asarray(data["fillnum"]) == fill
    if not np.any(sel):
        raise RuntimeError(f"No data for fill {fill}")
    return {k: np.asarray(v)[sel] for k, v in data.items()}


def pedestal4_from_window(rows, start=PEDESTAL_BX_START, stop=PEDESTAL_BX_STOP):
    """Estimate one pedestal per BX%4 from all samples in a BX window.

    Every value from every supplied row in [start, stop) contributes directly
    to the corresponding mod-4 pedestal.
    """
    rows = np.asarray(rows, dtype=float)
    bx = np.arange(BX_LEN)
    ped4 = np.empty(4, dtype=float)
    for mod in range(4):
        sel = (bx >= start) & (bx < stop) & (bx % 4 == mod)
        ped4[mod] = np.nanmedian(rows[:, sel])
    return ped4


def pedestal4_per_row(rows, start=PEDESTAL_BX_START, stop=PEDESTAL_BX_STOP):
    """Per-row diagnostic pedestal values using the same BX window."""
    rows = np.asarray(rows, dtype=float)
    bx = np.arange(BX_LEN)
    out = np.empty((len(rows), 4), dtype=float)
    for mod in range(4):
        sel = (bx >= start) & (bx < stop) & (bx % 4 == mod)
        out[:, mod] = np.nanmedian(rows[:, sel], axis=1)
    return out


def derive_tail_pedestal(data, n_rows=PEDESTAL_ROWS):
    """Estimate the clean mod-4 pedestal from the last raw rows.

    The last N rows are assumed to be the post-dump sample.  The pedestal is
    measured from *all* BX samples in [500, 3000), separately for BX % 4.
    """
    bxraw = np.asarray(data["bxraw"], dtype=float)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(f"bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})")
    if len(bxraw) < n_rows:
        raise RuntimeError(
            f"Only {len(bxraw)} rows in input, cannot use last {n_rows} for pedestal"
        )

    tail_idx = np.arange(len(bxraw) - n_rows, len(bxraw))
    before = bxraw[tail_idx]

    # Use every post-dump sample in the pedestal BX window directly.
    ped4 = pedestal4_from_window(before)
    ped4_rows_before = pedestal4_per_row(before)

    corrected = subtract_fixed_pedestal_mod4(bxraw, ped4)
    after = corrected[tail_idx]
    ped4_rows_after = pedestal4_per_row(after)

    return {
        "ped4": ped4,
        "tail_idx": tail_idx,
        "before": before,
        "after": after,
        "ped4_before": ped4_rows_before,
        "ped4_after": ped4_rows_after,
    }

def clean_masks(colliding_bx, bx_to_clean):
    clean_abs = np.ones(BX_LEN, dtype=bool)
    clean_rel = np.ones(BX_LEN, dtype=bool)
    for bx in bx_to_clean or ():
        bx = int(bx) % BX_LEN
        clean_abs[bx] = False
        clean_rel[(bx - colliding_bx) % BX_LEN] = False
    return clean_abs, clean_rel


def make_profile(aligned, row_sel, sbil):
    rows = np.asarray(aligned[row_sel], dtype=float)
    good = np.isfinite(rows[:, 0]) & (rows[:, 0] > 0)
    rows = rows[good]
    selected_sbil = np.asarray(sbil[row_sel])[good]

    # Normalize every orbit separately.  This is useful for the SBIL overlay:
    # its RMS/SEM really reflects the statistical quality at that SBIL.
    norm = rows / rows[:, 0, None]
    return {
        "n": len(norm),
        "sbil": float(np.nanmean(selected_sbil)),
        "mean": np.nanmean(norm, axis=0),
        "std": np.nanstd(norm, axis=0, ddof=1),
        "sem": np.nanstd(norm, axis=0, ddof=1) / np.sqrt(len(norm)),
    }


def interpolate_bad(profile, clean_mask, stop):
    out = np.asarray(profile).copy()
    x = np.arange(BX_LEN)
    good = clean_mask & (x < stop) & np.isfinite(out)
    bad = (~clean_mask) & (x < stop)
    if np.count_nonzero(good) >= 2:
        out[bad] = np.interp(x[bad], x[good], out[good])
    return out


def fit_tail(profile, clean_mask):
    x = np.arange(BX_LEN, dtype=float)
    sel = (
        (x >= TAIL_FIT_START)
        & (x < FIT_STOP)
        & clean_mask
        & np.isfinite(profile)
    )
    y = profile[sel]
    c0 = np.nanmedian(y[-100:])
    a0 = max(np.nanpercentile(y, 90) - c0, 1e-6)
    pars, _ = curve_fit(
        exp_fit,
        x[sel],
        y,
        p0=(a0, 1e-3, c0),
        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=100000,
    )
    return pars


def fit_full_profile(profile, clean_mask, tail_pars):
    """Unweighted double-exponential + Moyal full fit."""
    x = np.arange(BX_LEN, dtype=float)
    sel = (
        (x >= FULL_FIT_START)
        & (x < FIT_STOP)
        & clean_mask
        & np.isfinite(profile)
    )

    at, bt, ct = tail_pars
    early = sel & (x < 500)
    excess = np.nanmax(profile[early] - exp_fit(x[early], at, bt, ct))
    excess = max(float(excess), 1e-5)

    p0 = [
        max(at, 1e-6), max(bt, 1e-6),
        excess, 2e-2,
        max(excess * 50, 1e-4), 80.0, 20.0,
        ct,
    ]
    low = [0, 1e-7, 0, 1e-5, 0, 5, 1, -1e-2]
    high = [1e-1, 1e-1, 1e-1, 1, 1, 600, 300, 1e-2]

    pars, _ = curve_fit(
        full_fit,
        x[sel],
        profile[sel],
        p0=p0,
        bounds=(low, high),
        maxfev=500000,
    )

    # Keep the first exponential as the slower one for readable output.
    if pars[1] > pars[3]:
        pars[[0, 1, 2, 3]] = pars[[2, 3, 0, 1]]

    return pars


def build_candidates(profile, clean_mask):
    x = np.arange(BX_LEN, dtype=float)

    raw = profile.copy()
    raw[0] = 1.0

    tail_pars = fit_tail(profile, clean_mask)
    tail = interpolate_bad(profile, clean_mask, TAIL_FIT_START)
    tail[TAIL_FIT_START:] = exp_fit(x[TAIL_FIT_START:], *tail_pars)
    tail[0] = 1.0

    try:
        full_pars = fit_full_profile(profile, clean_mask, tail_pars)
        full = profile.copy()
        full[FULL_FIT_START:] = full_fit(x[FULL_FIT_START:], *full_pars)
        full[:FULL_FIT_START] = profile[:FULL_FIT_START]
        full[0] = 1.0
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        log.warning("Full fit failed, using tail candidate as fallback: %s", exc)
        full_pars = None
        full = tail.copy()

    return raw, tail, full, tail_pars, full_pars


def plot_pedestals(ped, output_dir, fill, sigvis, clean_abs, year):
    scale = 11245.6 / sigvis
    before4 = ped["ped4_before"] * scale
    after4 = ped["ped4_after"] * scale
    bx = np.arange(BX_LEN)

    # 1) Actual four pedestal values before/after.
    fig = create_figure("BCID % 4", "Pedestal [Hz/µb]", fill, year=year)
    mod = np.arange(4)
    plt.errorbar(
        mod - 0.06,
        np.nanmedian(before4, axis=0),
        yerr=np.nanstd(before4, axis=0),
        fmt="o", capsize=3, label="Before",
    )
    plt.errorbar(
        mod + 0.06,
        np.nanmedian(after4, axis=0),
        yerr=np.nanstd(after4, axis=0),
        fmt="o", capsize=3, label="After",
    )
    plt.xticks(mod)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pedestal_mod4_before_after.png"), dpi=300)
    plt.close(fig)

    # 2) Histograms of the actual pedestal samples in the same BX window
    #    used for the pedestal estimate.
    fig, ax = create_double_figure(
        "Signal [Hz/µb]", "Before", "After", fill, ratio=1, year=year
    )
    ped_window = (bx >= PEDESTAL_BX_START) & (bx < PEDESTAL_BX_STOP)
    for m in range(4):
        sel = ped_window & clean_abs & (bx % 4 == m)
        ax[0].hist((ped["before"][:, sel] * scale).ravel(), bins=70,
                   histtype="step", label=f"BX % 4 = {m}")
        ax[1].hist((ped["after"][:, sel] * scale).ravel(), bins=70,
                   histtype="step", label=f"BX % 4 = {m}")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pedestal_hist_before_after.png"), dpi=300)
    plt.close(fig)

    # 3) Full post-dump mean orbit before/after as a function of BX.
    mean_before = np.nanmean(ped["before"], axis=0) * scale
    mean_after = np.nanmean(ped["after"], axis=0) * scale
    fig, ax = create_double_figure(
        "BCID", "Before [Hz/µb]", "After [Hz/µb]", fill, ratio=1, year=year
    )
    for m in range(4):
        sel = clean_abs & (bx % 4 == m)
        ax[0].plot(bx[sel], mean_before[sel], ".", ms=2, label=f"BX % 4 = {m}")
        ax[1].plot(bx[sel], mean_after[sel], ".", ms=2, label=f"BX % 4 = {m}")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pedestal_vs_bx_before_after.png"), dpi=300)
    plt.close(fig)


def plot_luminosity_vs_entry(sbil, output_dir, fill, year):
    """Plot the full-fill single-bunch luminosity versus input entry."""
    sbil = np.asarray(sbil, dtype=float)
    entries = np.arange(sbil.size)
    good = np.isfinite(sbil)

    fig = create_figure(
        "Entry",
        "Instantaneous luminosity [Hz/µb]",
        fill,
        year=year,
    )
    plt.plot(entries[good], sbil[good], ".", ms=2, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "luminosity_vs_entry.png"), dpi=300)
    plt.close(fig)


def plot_sbil_profiles(profiles, clean_rel, output_dir, fill, year):
    if not profiles:
        return
    x = np.arange(BX_LEN)
    shown = clean_rel & (x > 3)

    fig, ax = create_double_figure(
        "Relative BX after colliding bunch",
        "Signal / colliding signal",
        "Stat. uncertainty on mean",
        fill, ratio=2, year=year,
    )
    for p in profiles:
        label = f"<SBIL>={p['sbil']:.2f}, N={p['n']}"
        ax[0].plot(x[shown], p["mean"][shown], ".", ms=2, alpha=0.55, label=label)
        good = shown & np.isfinite(p["sem"]) & (p["sem"] > 0)
        ax[1].plot(x[good], p["sem"][good], ".", ms=2, alpha=0.55)

    ax[0].set_ylim(-2e-4, 1.2e-3)
    ax[1].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sbr_by_sbil.png"), dpi=300)
    plt.close(fig)


def plot_measured_sbr(profile, clean_rel, output_dir, fill, year):
    """Measured Type-2 profile with no model overlaid."""
    x = np.arange(BX_LEN)
    shown = clean_rel & (x > 0)

    fig = create_figure(
        "Relative BX after colliding bunch",
        "Signal / colliding signal",
        fill,
        year=year,
    )
    plt.plot(x[shown], profile[shown], ".", ms=2, alpha=0.45)
    plt.ylim(-2e-4, 1.2e-3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sbr_measured.png"), dpi=300)
    plt.close(fig)


def plot_tail_fit(profile, tail, clean_rel, output_dir, fill, year):
    """Measured SBR and the long-tail exponential model only."""
    x = np.arange(BX_LEN)
    shown = clean_rel & (x > 0)
    model_sel = clean_rel & (x >= TAIL_FIT_START)

    fig, ax = create_double_figure(
        "Relative BX after colliding bunch",
        "Signal / colliding signal",
        "Data - tail model",
        fill, ratio=2, year=year,
    )
    ax[0].plot(x[shown], profile[shown], ".", ms=2, alpha=0.35, label="Measured SBR")
    ax[0].plot(x[model_sel], tail[model_sel], "-", lw=1.2, label="Exponential tail model")

    # Show the residual wherever the model is used.  The part beyond FIT_STOP
    # is an extrapolation and is separated visually from the fitted region.
    ax[1].plot(
        x[model_sel],
        (profile - tail)[model_sel],
        ".",
        ms=2,
    )

    for a in ax:
        a.axvline(FIT_STOP, ls="--", lw=1.0, alpha=0.7)

    ax[0].set_ylim(-2e-4, 1.2e-3)
    ax[1].set_ylim(-1e-4, 1e-4)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sbr_tail_fit.png"), dpi=300)
    plt.close(fig)


def plot_full_fit(profile, full, full_pars, clean_rel, output_dir, fill, year):
    """Measured SBR and the double-exponential + Moyal model."""
    if full_pars is None:
        return

    x = np.arange(BX_LEN)
    shown = clean_rel & (x > 0)
    model_sel = clean_rel & (x >= FULL_FIT_START)

    fig, ax = create_double_figure(
        "Relative BX after colliding bunch",
        "Signal / colliding signal",
        "Data - full model",
        fill, ratio=2, year=year,
    )
    ax[0].plot(x[shown], profile[shown], ".", ms=2, alpha=0.35, label="Measured SBR")
    ax[0].plot(
        x[model_sel],
        full[model_sel],
        "-",
        lw=1.2,
        label="Double exponential + Moyal",
    )
    ax[1].plot(
        x[model_sel],
        (profile - full)[model_sel],
        ".",
        ms=2,
    )

    for a in ax:
        a.axvline(FIT_STOP, ls="--", lw=1.0, alpha=0.7)

    ax[0].set_ylim(-2e-4, 1.2e-3)
    ax[1].set_ylim(-1e-4, 1e-4)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sbr_full_fit.png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--fill", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--pedestal-rows", type=int, default=PEDESTAL_ROWS)
    parser.add_argument("--sbil-bin-width", type=float, default=SBIL_BIN_WIDTH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    fill = args.fill
    output_dir = args.output_dir or os.path.join(
        getattr(cfg.io, "type1_dir", None) or cfg.io.output_dir,
        "type2", str(fill),
    )
    os.makedirs(output_dir, exist_ok=True)

    data = load_hd5_to_arrays(
        cfg.io.input_dir,
        cfg.io.input_pattern.format(fill=fill),
        node=cfg.io.node,
    )
    data = select_fill(data, fill)
    active = load_active_mask(cfg.io.active_mask_pattern.format(fill=fill), expected_len=BX_LEN)
    active_bx = np.flatnonzero(active)
    if len(active_bx) != 1:
        raise RuntimeError(f"Need single-bunch fill, found {len(active_bx)} active BX")
    colliding_bx = int(active_bx[0])

    # Input is already raw HF data.  Do not undo any online correction here,
    # and do not apply the existing offline LSQ Type-2 correction: this script
    # is measuring that response directly.

    clean_abs, clean_rel = clean_masks(colliding_bx, cfg.afterglow.bx_to_clean)

    # The input contains the post-dump tail at the end.  Use exactly the last
    # N raw rows to measure the clean pedestal; no beam data are needed.
    ped = derive_tail_pedestal(data, n_rows=args.pedestal_rows)
    ped4 = ped["ped4"]
    log.info(
        "Pedestal from last %d rows: %s",
        args.pedestal_rows,
        np.array2string(ped4, precision=8),
    )
    plot_pedestals(
        ped, output_dir, fill, cfg.afterglow.sigvis, clean_abs, args.year
    )

    bxraw = subtract_fixed_pedestal_mod4(np.asarray(data["bxraw"], dtype=float), ped4)
    aligned = np.roll(bxraw, -colliding_bx, axis=1)

    scale = 11245.6 / cfg.afterglow.sigvis
    sbil = aligned[:, 0] * scale

    # Full-fill diagnostic before any SBIL selection.  This is simply the
    # pedestal-subtracted colliding-BX signal converted to luminosity units,
    # shown versus input entry (our proxy for time here).
    plot_luminosity_vs_entry(sbil, output_dir, fill, args.year)

    # Discard the 0--1 SBIL region completely: it is dominated by junk and
    # does not contribute useful Type-2 information.
    sbil_min = max(float(getattr(cfg.type1, "sbil_min", 0.0)), MIN_SBIL)
    physics = np.isfinite(sbil) & (sbil >= sbil_min)
    if np.count_nonzero(physics) < MIN_PHYSICS_ROWS:
        raise RuntimeError("Too few physics rows after SBIL selection")

    global_profile = make_profile(aligned, physics, sbil)

    # One single diagnostic plot with all useful SBIL slices.
    sbil_profiles = []
    width = args.sbil_bin_width
    selected_sbil = sbil[physics]
    lo = np.floor(selected_sbil.min() / width) * width
    hi = np.ceil(selected_sbil.max() / width) * width
    for left in np.arange(lo, hi, width):
        sel = physics & (sbil >= left) & (sbil < left + width)
        if np.count_nonzero(sel) >= MIN_PHYSICS_ROWS:
            sbil_profiles.append(make_profile(aligned, sel, sbil))
    plot_sbil_profiles(sbil_profiles, clean_rel, output_dir, fill, args.year)

    profile = global_profile["mean"]
    raw, tail, full, tail_pars, full_pars = build_candidates(profile, clean_rel)
    plot_measured_sbr(profile, clean_rel, output_dir, fill, args.year)
    plot_tail_fit(profile, tail, clean_rel, output_dir, fill, args.year)
    plot_full_fit(profile, full, full_pars, clean_rel, output_dir, fill, args.year)

    np.savetxt(os.path.join(output_dir, f"type2_raw_fill{fill}.txt"),
               raw[None, :], fmt="%.8e", delimiter=",")
    np.savetxt(os.path.join(output_dir, f"type2_tail_fill{fill}.txt"),
               tail[None, :], fmt="%.8e", delimiter=",")
    np.savetxt(os.path.join(output_dir, f"type2_full_fill{fill}.txt"),
               full[None, :], fmt="%.8e", delimiter=",")

    log.info("Tail fit: %s", np.array2string(tail_pars, precision=6))
    if full_pars is not None:
        log.info("Full fit: %s", np.array2string(full_pars, precision=6))
    log.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
