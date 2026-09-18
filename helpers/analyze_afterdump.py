#!/usr/bin/env python3
"""
Standalone after-dump / pedestal diagnostics.

Beam data is loaded only here, never by the production HF pipeline.  The helper
aligns beam intensity to HF rows by timestamp, finds the low-intensity tail
_after_ the last above-threshold point, and writes the old after-dump/pedestal
plots plus a compact HDF5 summary useful for later pedestal prediction.

Example
-------
python3 helpers/analyze_afterdump.py \
    --config configs/analysis_25_physics.yaml \
    --fill 10826 \
    --beam-path /eos/cms/store/group/dpg_bril/comm_bril/2025/physics
"""
from __future__ import annotations

import argparse
import logging
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from hfcore.beam import align_beam_columns_to_lumi, load_beam_fill, timestamp_seconds
from hfcore.calibration import recover_online_full_fill
from hfcore.config import PipelineConfig, load_config
from hfcore.hd5schema import BX_LEN
from hfcore.io import load_active_mask, load_hd5_to_arrays
from hfcore.pedestal import calculate_dynamic_pedestal, subtract_fixed_pedestal_mod4
from hfcore.plotter import create_double_figure, create_figure

log = logging.getLogger("hfpipe.afterdump")


def double_exp_decay(
    t: np.ndarray,
    a_fast: float,
    tau_fast: float,
    a_slow: float,
    tau_slow: float,
    c: float,
) -> np.ndarray:
    return c + a_fast * np.exp(-t / tau_fast) + a_slow * np.exp(-t / tau_slow)


def _fit_double_exp(t: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    good = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t[good], dtype=np.float64)
    y = np.asarray(y[good], dtype=np.float64)
    if t.size < 20 or np.ptp(t) <= 0:
        return None

    tail_n = min(max(5, t.size // 10), 50)
    c0 = float(np.median(y[-tail_n:]))
    amp = max(float(y[0] - c0), np.finfo(float).eps)
    span = max(float(np.ptp(t)), 1.0)
    p0 = (0.65 * amp, max(span / 20.0, 1.0), 0.35 * amp, max(span / 3.0, 2.0), c0)

    try:
        pars, _ = curve_fit(
            double_exp_decay,
            t,
            y,
            p0=p0,
            bounds=(
                [0.0, 1.0e-6, 0.0, 1.0e-6, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
            maxfev=200000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    pars = np.asarray(pars, dtype=np.float64)
    # Canonical ordering makes downstream comparisons between fills easier.
    if pars[1] > pars[3]:
        pars = pars[[2, 3, 0, 1, 4]]
    return pars


def _select_fill(data: dict[str, np.ndarray], fill: int) -> dict[str, np.ndarray]:
    if "fillnum" not in data:
        return data
    sel = np.asarray(data["fillnum"], dtype=np.int64) == int(fill)
    if not np.any(sel):
        raise ValueError(f"No rows with fillnum={fill} in selected HF input")
    if np.all(sel):
        return data
    return {k: np.asarray(v)[sel] for k, v in data.items()}


def _sort_by_time(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t = timestamp_seconds(data)
    order = np.argsort(t, kind="stable")
    if np.all(order == np.arange(order.size)):
        return data
    return {k: np.asarray(v)[order] for k, v in data.items()}


def _find_afterdump_rows(
    intensity1: np.ndarray,
    threshold: float,
    skip_rows: int,
) -> np.ndarray:
    """Return row indices in the post-dump low-intensity tail only."""
    intensity1 = np.asarray(intensity1, dtype=np.float64)
    finite = np.isfinite(intensity1)
    above = finite & (intensity1 >= threshold)
    if not np.any(above):
        raise RuntimeError(
            "Cannot identify beam dump: no beam-1 intensity point is above the threshold"
        )

    last_above = int(np.flatnonzero(above)[-1])
    idx = np.arange(intensity1.size)
    after = idx[(idx > last_above) & finite & (intensity1 < threshold)]
    if skip_rows > 0:
        after = after[skip_rows:]
    return after


def _pedestal4_rows(bxraw: np.ndarray) -> np.ndarray:
    return np.stack([calculate_dynamic_pedestal(row) for row in bxraw], axis=0).astype(np.float64)


def analyze_afterdump(
    lumi: dict[str, np.ndarray],
    beam: dict[str, np.ndarray],
    active_mask: np.ndarray,
    fill: int,
    output_dir: str,
    *,
    sigvis: float,
    fixed_pedestal_4=None,
    intensity1_column: str = "intensity1",
    intensity2_column: str = "intensity2",
    dump_threshold: float = 1.0e11,
    skip_rows: int = 1,
    min_rows: int = 50,
    max_beam_dt: float | None = None,
    pedestal_bx_start: int = 1000,
    pedestal_bx_stop: int = 3000,
    year: int | None = None,
) -> dict[str, np.ndarray]:
    """Run after-dump diagnostics and save plots + HDF5 summary."""
    lumi = _sort_by_time(lumi)
    active_mask = np.asarray(active_mask, dtype=np.int32).ravel()
    if active_mask.shape != (BX_LEN,):
        raise ValueError(f"active_mask has shape {active_mask.shape}, expected ({BX_LEN},)")
    n_active = int(active_mask.sum())
    if n_active == 0:
        raise ValueError("active_mask contains no colliding BX")

    aligned_beam = align_beam_columns_to_lumi(
        lumi,
        beam,
        columns=(intensity1_column, intensity2_column),
        max_dt=max_beam_dt,
    )
    intensity1 = aligned_beam[intensity1_column]
    intensity2 = aligned_beam[intensity2_column]
    dt_beam = aligned_beam["dt"]

    after_idx = _find_afterdump_rows(intensity1, dump_threshold, skip_rows)
    if after_idx.size < min_rows:
        raise RuntimeError(
            f"After-dump tail contains only {after_idx.size} HF rows after filtering; need >= {min_rows}"
        )

    bxraw_raw = np.asarray(lumi["bxraw"], dtype=np.float64)
    if bxraw_raw.ndim != 2 or bxraw_raw.shape[1] != BX_LEN:
        raise ValueError(f"bxraw has shape {bxraw_raw.shape}, expected (T, {BX_LEN})")
    bxraw_pedsub = subtract_fixed_pedestal_mod4(bxraw_raw, fixed_pedestal_4)

    scale = 11245.6 / float(sigvis)
    active = active_mask.astype(bool)
    rate_raw = bxraw_raw[:, active].sum(axis=1) * scale
    rate_pedsub = bxraw_pedsub[:, active].sum(axis=1) * scale

    t_abs = timestamp_seconds(lumi)
    fill_elapsed = t_abs - t_abs[0]
    t_after = t_abs[after_idx] - t_abs[after_idx[0]]

    tail_raw = bxraw_raw[after_idx]
    tail_pedsub = bxraw_pedsub[after_idx]
    pedestal4_raw = _pedestal4_rows(tail_raw) * scale
    pedestal4_pedsub = _pedestal4_rows(tail_pedsub) * scale

    rate_fit = _fit_double_exp(t_after, rate_raw[after_idx])
    pedestal_fits = np.full((4, 5), np.nan, dtype=np.float64)
    for mod in range(4):
        pars = _fit_double_exp(t_after, pedestal4_raw[:, mod])
        if pars is not None:
            pedestal_fits[mod] = pars

    os.makedirs(output_dir, exist_ok=True)

    # 1) Full-fill beam intensity and HF response.  Beam I/O exists only here.
    fig, ax = create_double_figure(
        "Fill elapsed time [s]",
        "Beam intensity [protons]",
        "HF active-BX sum [Hz/µb]",
        fill,
        year=year or 2025,
    )
    ax[0].plot(fill_elapsed, intensity1, ".", ms=3, label="B1 intensity")
    ax[0].plot(fill_elapsed, intensity2, ".", ms=3, label="B2 intensity")
    ax[1].plot(fill_elapsed, rate_raw, ".", ms=3, label="Raw")
    ax[1].plot(fill_elapsed, rate_pedsub, ".", ms=3, label="Fixed-pedestal subtracted")
    ax[0].legend(loc="upper right", frameon=False, fontsize=10)
    ax[1].legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "beam_intensity.png"), dpi=300)
    plt.close(fig)

    # 2) Decay after dump.
    fig = create_figure(
        "Elapsed time after beam dump [s]",
        "HF active-BX sum [Hz/µb]",
        fill,
        year=year or 2025,
    )
    plt.plot(t_after, rate_raw[after_idx], ".", ms=4, label="Raw")
    plt.plot(t_after, rate_pedsub[after_idx], ".", ms=4, label="Fixed-pedestal subtracted")
    if rate_fit is not None:
        tt = np.linspace(float(t_after.min()), float(t_after.max()), 500)
        plt.plot(tt, double_exp_decay(tt, *rate_fit), "-", label="Raw: double-exp fit")
    plt.legend(loc="upper right", frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "after_dump_rate.png"), dpi=300)
    plt.close(fig)

    # 3) Pedestal evolution.  This is the directly useful quantity for a later
    #    prediction of the next fill's initial pedestal.
    fig, ax = create_double_figure(
        "Elapsed time after beam dump [s]",
        "Raw pedestal [Hz/µb]",
        "After fixed pedestal subtraction [Hz/µb]",
        fill,
        ratio=1,
        year=year or 2025,
    )
    tt = np.linspace(float(t_after.min()), float(t_after.max()), 500)
    for mod in range(4):
        ax[0].plot(t_after, pedestal4_raw[:, mod], ".", ms=3, label=f"BCID % 4 = {mod}")
        ax[1].plot(t_after, pedestal4_pedsub[:, mod], ".", ms=3, label=f"BCID % 4 = {mod}")
        if np.all(np.isfinite(pedestal_fits[mod])):
            ax[0].plot(tt, double_exp_decay(tt, *pedestal_fits[mod]), "-", lw=1)
    ax[0].legend(loc="upper right", frameon=False, fontsize=9)
    ax[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pedestal_evolution.png"), dpi=300)
    plt.close(fig)

    # 4) Old pedestal-distribution diagnostic, fixed to use all after-dump rows.
    # The old code renormalized each raw orbit to the final after-dump level;
    # preserve that intent, but keep the 2D (time, BX) shape until BCID selection.
    tail_rate_raw = rate_raw[after_idx]
    tail_raw_renorm = np.array(tail_raw, copy=True)
    final_rate = float(tail_rate_raw[-1])
    if np.isfinite(final_rate) and abs(final_rate) > np.finfo(float).eps:
        factors = tail_rate_raw / final_rate
        valid_factor = np.isfinite(factors) & (np.abs(factors) > np.finfo(float).eps)
        tail_raw_renorm[valid_factor] /= factors[valid_factor, None]

    lo = max(0, int(pedestal_bx_start))
    hi = min(BX_LEN, int(pedestal_bx_stop))
    if lo >= hi:
        raise ValueError("pedestal_bx_start must be smaller than pedestal_bx_stop")
    fig, ax = create_double_figure(
        "Signal [Hz/µb]",
        "Raw entries",
        "Pedestal-subtracted entries",
        fill,
        ratio=1,
        year=year or 2025,
    )
    for mod in range(4):
        bcids = np.arange(lo, hi, dtype=np.int64)
        bcids = bcids[bcids % 4 == mod]
        vals_raw = (tail_raw_renorm[:, bcids] * scale).ravel()
        vals_sub = (tail_pedsub[:, bcids] * scale).ravel()
        ax[0].hist(vals_raw[np.isfinite(vals_raw)], bins=50, histtype="step", label=f"BCID % 4 = {mod}")
        ax[1].hist(vals_sub[np.isfinite(vals_sub)], bins=50, histtype="step", label=f"BCID % 4 = {mod}")
    ax[0].legend(loc="upper right", frameon=False, fontsize=9)
    ax[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pedestals_dump.png"), dpi=300)
    plt.close(fig)

    # 5) Last measured after-dump orbit, grouped by mod4.
    fig = create_figure(
        "BCID",
        "Signal after fixed pedestal subtraction [Hz/µb]",
        fill,
        year=year or 2025,
    )
    last = tail_pedsub[-1] * scale
    for mod in range(4):
        bcids = np.arange(BX_LEN, dtype=np.int64)
        bcids = bcids[bcids % 4 == mod]
        plt.plot(bcids, last[bcids], ".", ms=3, label=f"BCID % 4 = {mod}")
    plt.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bx_pedestals.png"), dpi=300)
    plt.close(fig)

    out = {
        "afterdump_row_index": after_idx.astype(np.int64),
        "timestamp_s": t_abs[after_idx].astype(np.float64),
        "time_after_dump_s": t_after.astype(np.float64),
        "intensity1": intensity1[after_idx].astype(np.float64),
        "intensity2": intensity2[after_idx].astype(np.float64),
        "beam_match_dt_s": dt_beam[after_idx].astype(np.float64),
        "active_rate_raw": rate_raw[after_idx].astype(np.float64),
        "active_rate_pedsub": rate_pedsub[after_idx].astype(np.float64),
        "pedestal4_raw": pedestal4_raw.astype(np.float64),
        "pedestal4_pedsub": pedestal4_pedsub.astype(np.float64),
        "pedestal4_fit_params": pedestal_fits,
    }
    if rate_fit is not None:
        out["active_rate_fit_params"] = rate_fit

    h5_path = os.path.join(output_dir, f"afterdump_fill{fill}.h5")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["fill"] = int(fill)
        h5.attrs["sigvis"] = float(sigvis)
        h5.attrs["dump_threshold"] = float(dump_threshold)
        h5.attrs["n_active_bx"] = n_active
        h5.attrs["afterdump_reference_timestamp_s"] = float(t_abs[after_idx[0]])
        h5.attrs["fit_model"] = "c + a_fast*exp(-t/tau_fast) + a_slow*exp(-t/tau_slow)"
        h5.attrs["fit_param_order"] = np.array(
            ["a_fast", "tau_fast_s", "a_slow", "tau_slow_s", "c"], dtype="S16"
        )
        h5.attrs["fixed_pedestal_4_mu"] = np.asarray(
            fixed_pedestal_4 if fixed_pedestal_4 is not None else [0, 0, 0, 0],
            dtype=np.float64,
        )
        for name, arr in out.items():
            h5.create_dataset(name, data=arr)

    return out


def _default_output_dir(cfg: PipelineConfig, fill: int) -> str:
    base = getattr(cfg.io, "type1_dir", None) or cfg.io.output_dir
    return os.path.join(base, "afterdump", str(fill))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HF after-dump tail and pedestals")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--fill", type=int, required=True)
    parser.add_argument("--beam-path", required=True, help="Base directory containing <fill>/*.hd5 beam files")
    parser.add_argument("--beam-node", default="beam")
    parser.add_argument("--intensity1-column", default="intensity1")
    parser.add_argument("--intensity2-column", default="intensity2")
    parser.add_argument("--dump-threshold", type=float, default=1.0e11)
    parser.add_argument("--max-beam-dt", type=float)
    parser.add_argument("--skip-rows", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--pedestal-bx-start", type=int, default=1000)
    parser.add_argument("--pedestal-bx-stop", type=int, default=3000)
    parser.add_argument("--output-dir")
    parser.add_argument("--year", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    fill = args.fill

    if cfg.afterglow.sigvis is None:
        raise ValueError("afterglow.sigvis must be set in the config")

    lumi = load_hd5_to_arrays(
        cfg.io.input_dir,
        cfg.io.input_pattern.format(fill=fill),
        node=cfg.io.node,
    )
    lumi = _select_fill(lumi, fill)
    mask = load_active_mask(cfg.io.active_mask_pattern.format(fill=fill), expected_len=BX_LEN)
    lumi = recover_online_full_fill(lumi, cfg, mask, fill)

    beam_columns = [
        "fillnum", "timestampsec", "timestampmsec",
        args.intensity1_column, args.intensity2_column,
    ]
    beam = load_beam_fill(
        args.beam_path,
        fill,
        node=args.beam_node,
        columns=beam_columns,
    )

    output_dir = args.output_dir or _default_output_dir(cfg, fill)
    result = analyze_afterdump(
        lumi=lumi,
        beam=beam,
        active_mask=mask,
        fill=fill,
        output_dir=output_dir,
        sigvis=cfg.afterglow.sigvis,
        fixed_pedestal_4=cfg.afterglow.fixed_pedestal_4,
        intensity1_column=args.intensity1_column,
        intensity2_column=args.intensity2_column,
        dump_threshold=args.dump_threshold,
        skip_rows=args.skip_rows,
        min_rows=args.min_rows,
        max_beam_dt=args.max_beam_dt,
        pedestal_bx_start=args.pedestal_bx_start,
        pedestal_bx_stop=args.pedestal_bx_stop,
        year=args.year,
    )

    max_dt = float(np.nanmax(result["beam_match_dt_s"]))
    log.info(
        "fill %d: %d after-dump rows, max beam/HF timestamp mismatch %.3f s -> %s",
        fill, result["time_after_dump_s"].size, max_dt, output_dir,
    )


if __name__ == "__main__":
    main()
