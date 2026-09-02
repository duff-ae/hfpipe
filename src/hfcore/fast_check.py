#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hfcore.config import load_config
from hfcore.hd5schema import BX_LEN
from hfcore.io import load_hd5_to_arrays
from hfcore.pipeline import _load_hfsbr_for_online
from hfcore.online_recovery import (
    reconstruct_from_online_batch,
    _call_revert_afterglow_inplace,
    _validate_active_mask,
    _validate_hfsbr,
)

import matplotlib.pyplot as plt

from hfcore.plotter import create_double_figure

# python3 src/hfcore/fast_check.py --config configs/analysis_25_physics.yaml --fill 10639 --max-rows 500 --save-npz online_recovery.npz

KEYS = ("fillnum", "runnum", "lsnum", "nbnum")

def _per_bx_residual_stats(
    test: np.ndarray,
    ref: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-BX statistics of (test - ref).

    Returns:
        count : number of valid rows for each BX
        bias  : mean(test - ref) for each BX
        rms   : sqrt(mean((test - ref)^2)) for each BX

    BXs without table coverage are returned as NaN.
    """

    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if test.shape != ref.shape:
        raise ValueError(
            f"test/ref shape mismatch: {test.shape} vs {ref.shape}"
        )

    if valid.shape != test.shape:
        raise ValueError(
            f"valid shape mismatch: {valid.shape} vs {test.shape}"
        )

    mask = (
        valid
        & np.isfinite(test)
        & np.isfinite(ref)
    )

    diff = np.where(mask, test - ref, 0.0)

    count = mask.sum(axis=0)

    bias = np.full(BX_LEN, np.nan, dtype=np.float64)
    rms = np.full(BX_LEN, np.nan, dtype=np.float64)

    good = count > 0

    bias[good] = (
        diff[:, good].sum(axis=0)
        / count[good]
    )

    rms[good] = np.sqrt(
        (diff[:, good] ** 2).sum(axis=0)
        / count[good]
    )

    return count, bias, rms


def plot_online_recovery_per_bx(
    raw_ref: np.ndarray,
    afterglow_only: np.ndarray,
    recovered: np.ndarray,
    valid: np.ndarray,
    active_mask: np.ndarray,
    cfg,
    fill: int,
    year: int = 2025,
    zero_bx: tuple[int, ...] = (),
) -> str:
    """
    Per-BX closure diagnostic.

    Top:
        mean signed recovery residual per BX.

    Bottom:
        RMS recovery residual per BX.

    Both are expressed as % of the mean table-reference response
    over colliding BX, so non-colliding BX remain well-defined.

    Compares:
        - HFSBR inversion with the true table pedestal
        - full recovery with pedestal reconstructed from zero BX
    """

    raw_ref = np.asarray(raw_ref, dtype=np.float64)
    afterglow_only = np.asarray(afterglow_only, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    active = np.asarray(active_mask, dtype=bool).ravel()

    if active.shape != (BX_LEN,):
        raise ValueError(
            f"active_mask shape={active.shape}, expected {(BX_LEN,)}"
        )

    # ------------------------------------------------------------
    # Common normalization:
    #
    # mean reference signal over all valid colliding BX and rows.
    #
    # Do NOT divide residual by each individual BX:
    # non-colliding BX can be zero / tiny and the ratio becomes
    # numerically meaningless.
    # ------------------------------------------------------------

    norm_mask = (
        valid
        & active[None, :]
        & np.isfinite(raw_ref)
    )

    n_norm = np.count_nonzero(norm_mask)

    if n_norm == 0:
        raise RuntimeError(
            "Cannot determine mean colliding-BX normalization"
        )

    mean_colliding = (
        np.where(norm_mask, raw_ref, 0.0).sum()
        / n_norm
    )

    if not np.isfinite(mean_colliding) or mean_colliding == 0.0:
        raise RuntimeError(
            f"Invalid mean colliding normalization: {mean_colliding}"
        )

    # ------------------------------------------------------------
    # Per-BX residual statistics
    # ------------------------------------------------------------

    count_hfsbr, bias_hfsbr, rms_hfsbr = _per_bx_residual_stats(
        afterglow_only,
        raw_ref,
        valid,
    )

    count_full, bias_full, rms_full = _per_bx_residual_stats(
        recovered,
        raw_ref,
        valid,
    )

    # Same validity should normally give identical coverage.
    if not np.array_equal(count_hfsbr, count_full):
        print(
            "[WARN] HFSBR-only and full-recovery per-BX "
            "coverage are not identical"
        )

    scale_pct = 100.0 / mean_colliding

    bias_hfsbr_pct = bias_hfsbr * scale_pct
    bias_full_pct = bias_full * scale_pct

    rms_hfsbr_pct = rms_hfsbr * scale_pct
    rms_full_pct = rms_full * scale_pct

    bx = np.arange(BX_LEN)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = create_double_figure(
        "BCID",
        "Mean residual\n[% of mean colliding BX]",
        "Residual RMS\n[% of mean colliding BX]",
        fill,
        ratio=2,
        year=year,
        plot_type="Preliminary",
    )

    # Signed mean residual -> direct test for systematic BX structure.
    ax[0].plot(
        bx,
        bias_hfsbr_pct,
        linewidth=1.0,
        label="HFSBR only (table pedestal)",
    )

    ax[0].plot(
        bx,
        bias_full_pct,
        linewidth=1.0,
        label="Full recovery (zero-BX pedestal)",
    )

    ax[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    # RMS -> local loss of precision / problematic BX regions.
    ax[1].plot(
        bx,
        rms_hfsbr_pct,
        linewidth=1.0,
        label="HFSBR only (table pedestal)",
    )

    ax[1].plot(
        bx,
        rms_full_pct,
        linewidth=1.0,
        label="Full recovery (zero-BX pedestal)",
    )

    # ------------------------------------------------------------
    # Mark regions important for pedestal reconstruction
    # ------------------------------------------------------------

    # CMS dynamic pedestal sampling region:
    # 3500 ... 3551 for 13 samples x 4 channels.
    for a in ax:
        a.axvline(
            3500,
            linestyle=":",
            linewidth=1.0,
        )

        a.axvline(
            3552,
            linestyle=":",
            linewidth=1.0,
        )

    # Mark the zero-BX constraint range, if supplied.
    if zero_bx:
        zmin = min(zero_bx)
        zmax = max(zero_bx)

        for a in ax:
            a.axvspan(
                zmin - 0.5,
                zmax + 0.5,
                alpha=0.08,
            )

    ax[0].legend(
        loc="upper right",
        frameon=False,
        fontsize=11,
    )

    ax[1].legend(
        loc="upper right",
        frameon=False,
        fontsize=11,
    )

    ax[0].grid(True, alpha=0.25)
    ax[1].grid(True, alpha=0.25)

    ax[0].set_xlim(0, BX_LEN - 1)
    ax[1].set_xlim(0, BX_LEN - 1)

    fig.tight_layout()

    # ------------------------------------------------------------
    # Output directory -- same convention as the existing plotter
    # ------------------------------------------------------------

    plot_dir = getattr(cfg.io, "type1_dir", None)

    if plot_dir is None:
        plot_dir = os.path.join(
            cfg.io.output_dir,
            "type1",
        )

    output_dir = os.path.join(
        plot_dir,
        str(fill),
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"online_recovery_per_bx_fill_{fill}.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Small textual summary, useful together with the PNG
    # ------------------------------------------------------------

    finite_full = np.isfinite(bias_full_pct)
    finite_hfsbr = np.isfinite(bias_hfsbr_pct)

    print()
    print("PER-BX CLOSURE SUMMARY")
    print("----------------------")

    if np.any(finite_hfsbr):
        print(
            "HFSBR only:"
            f" max |mean residual| = "
            f"{np.nanmax(np.abs(bias_hfsbr_pct)):.6e}%"
            f", max RMS = "
            f"{np.nanmax(rms_hfsbr_pct):.6e}%"
        )

    if np.any(finite_full):
        print(
            "Full recovery:"
            f" max |mean residual| = "
            f"{np.nanmax(np.abs(bias_full_pct)):.6e}%"
            f", max RMS = "
            f"{np.nanmax(rms_full_pct):.6e}%"
        )

    no_coverage = np.where(count_full == 0)[0]

    print(
        f"BX with no table reference coverage: "
        f"{len(no_coverage)} / {BX_LEN}"
    )

    if len(no_coverage) > 0:
        print(
            "First BX without coverage:",
            no_coverage[:30].tolist(),
        )

    print(f"Saved: {output_path}")

    return output_path

def select_fill(data: dict, fill: int) -> dict:
    fillnum = np.asarray(data["fillnum"])
    sel = fillnum == fill

    out = {}
    n = len(fillnum)

    for name, value in data.items():
        arr = np.asarray(value)

        if arr.ndim > 0 and arr.shape[0] == n:
            out[name] = arr[sel]
        else:
            out[name] = arr

    return out

def build_pedestal_response(
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    response[k] = R(pattern_k),

    where pattern_k[bx] = 1 for bx % 4 == k.
    """
    response = np.empty((4, BX_LEN), dtype=np.float32)

    bx_mod4 = np.arange(BX_LEN) % 4

    for k in range(4):
        row = (bx_mod4 == k).astype(np.float32)

        _call_revert_afterglow_inplace(
            active_mask_i32=active_mask,
            mu_hist_f32=row,
            hfsbr_f32=hfsbr,
        )

        response[k] = row

    return response


def recover_online_with_zero_bx(
    bxraw_final: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
    zero_bx: tuple[int, ...],
):
    """
    Recover original pre-online-correction bxraw and the unknown
    4-component online pedestal.

    Constraint:
        recovered_raw[zero_bx] ~= 0
    """

    bxraw_final = np.asarray(bxraw_final, dtype=np.float32)

    active_i32 = _validate_active_mask(active_mask)
    hfsbr_f32 = _validate_hfsbr(hfsbr)

    zero_bx_arr = np.asarray(zero_bx, dtype=np.int64)

    # ------------------------------------------------------------
    # Response to the four possible pedestal components.
    # Shape: (4, BX_LEN)
    # ------------------------------------------------------------

    response = build_pedestal_response(
        hfsbr=hfsbr_f32,
        active_mask=active_i32,
    )

    # Matrix connecting pedestal[4] to recovered values
    # in the zero-BX constraint region.
    #
    # Shape: (Nzero, 4)
    A = response[:, zero_bx_arr].T.astype(np.float64)

    print(
        "online pedestal system:",
        f"shape={A.shape}",
        f"rank={np.linalg.matrix_rank(A)}",
        f"condition={np.linalg.cond(A):.6e}",
    )

    if np.linalg.matrix_rank(A) < 4:
        raise RuntimeError(
            "zero_bx pedestal system has rank < 4"
        )

    # Pseudoinverse is identical for every row in the fill.
    A_pinv = np.linalg.pinv(A)

    T = bxraw_final.shape[0]

    base = np.empty_like(bxraw_final, dtype=np.float32)

    # ------------------------------------------------------------
    # R(final)
    # ------------------------------------------------------------

    for i in range(T):
        row = np.ascontiguousarray(
            bxraw_final[i],
            dtype=np.float32,
        ).copy()

        _call_revert_afterglow_inplace(
            active_mask_i32=active_i32,
            mu_hist_f32=row,
            hfsbr_f32=hfsbr_f32,
        )

        base[i] = row

    # ------------------------------------------------------------
    # Solve:
    #
    # A p = -R(final)[zero_bx]
    #
    # pedestal shape: (T, 4)
    # ------------------------------------------------------------

    rhs = -base[:, zero_bx_arr].astype(np.float64)

    pedestal = rhs @ A_pinv.T

    # ------------------------------------------------------------
    # By linearity:
    #
    # R(final + pedestal_pattern)
    # =
    # R(final) + sum_k p_k R(pattern_k)
    #
    # No second C call required.
    # ------------------------------------------------------------

    recovered = (
        base.astype(np.float64)
        + pedestal @ response.astype(np.float64)
    )

    return (
        recovered.astype(np.float32),
        pedestal.astype(np.float32),
    )

def strict_align_aux(
    main: dict,
    aux: dict,
    colname: str,
) -> np.ndarray:
    """
    Strict equivalent of _align_aux_by_keys.

    Unlike the production helper:
      - missing keys are fatal;
      - duplicate aux keys are fatal.
    """

    for key in KEYS:
        if key not in main:
            raise KeyError(f"main is missing key column {key!r}")
        if key not in aux:
            raise KeyError(f"aux is missing key column {key!r}")

    if colname not in aux:
        raise KeyError(f"aux is missing requested column {colname!r}")

    main_keys = np.stack(
        [np.asarray(main[k], dtype=np.int64) for k in KEYS],
        axis=1,
    )

    aux_keys = np.stack(
        [np.asarray(aux[k], dtype=np.int64) for k in KEYS],
        axis=1,
    )

    index = {}

    for i, key_arr in enumerate(aux_keys):
        key = tuple(key_arr.tolist())

        if key in index:
            raise RuntimeError(
                f"Duplicate auxiliary key {key}: "
                f"rows {index[key]} and {i}"
            )

        index[key] = i

    source = np.asarray(aux[colname])

    out_shape = (len(main_keys),) + source.shape[1:]
    out = np.empty(out_shape, dtype=source.dtype)

    missing = []

    for i, key_arr in enumerate(main_keys):
        key = tuple(key_arr.tolist())
        j = index.get(key)

        if j is None:
            missing.append(key)
            continue

        out[i] = source[j]

    if missing:
        preview = missing[:10]
        raise RuntimeError(
            f"{len(missing)} main rows have no matching auxiliary row. "
            f"First missing keys: {preview}"
        )

    return out


def pedestal_pattern(pedestal_4: np.ndarray) -> np.ndarray:
    idx = np.arange(BX_LEN) % 4
    return pedestal_4[:, idx]


def calculate_dynamic_pedestal_batch(
    hist: np.ndarray,
    n_sample: int,
) -> np.ndarray:
    """
    Calculate pedestal[4] from BX 3500 onward using n_sample values
    for each modulo-4 channel.
    """

    hist = np.asarray(hist, dtype=np.float64)

    if hist.ndim != 2 or hist.shape[1] != BX_LEN:
        raise ValueError(
            f"hist has shape {hist.shape}, expected (T, {BX_LEN})"
        )

    result = np.empty((hist.shape[0], 4), dtype=np.float64)

    for imod in range(4):
        idx = 3500 + imod + 4 * np.arange(n_sample)

        if idx[-1] >= BX_LEN:
            raise ValueError(
                f"Pedestal indices exceed orbit: last={idx[-1]}"
            )

        result[:, imod] = np.mean(hist[:, idx], axis=1)

    return result


def revert_afterglow_legacy_only(
    hist: np.ndarray,
    active_mask: np.ndarray,
    hfsbr: np.ndarray,
) -> np.ndarray:
    """
    Run ONLY the current C revert_afterglow(), without pedestal subtraction.
    """

    hist = np.asarray(hist, dtype=np.float32)
    active = _validate_active_mask(active_mask)
    hfsbr = _validate_hfsbr(hfsbr)

    out = np.empty_like(hist, dtype=np.float32)

    for i in range(hist.shape[0]):
        row = np.ascontiguousarray(hist[i], dtype=np.float32).copy()

        _call_revert_afterglow_inplace(
            active_mask_i32=active,
            mu_hist_f32=row,
            hfsbr_f32=hfsbr,
        )

        out[i] = row

    return out


def revert_afterglow_snapshot_fft(
    hist: np.ndarray,
    active_mask: np.ndarray,
    hfsbr: np.ndarray,
) -> np.ndarray:
    """
    Alternative interpretation of HFSBR inversion:

        raw[j] = corrected[j]
               + sum_i active[i] * corrected[i] * HFSBR[(j-i) mod BX_LEN]

    Source amplitudes are frozen BEFORE any destination BX is modified.

    This is useful specifically to diagnose whether the in-place descending
    C loop changes active source amplitudes through orbit wrap-around.
    """

    hist = np.asarray(hist, dtype=np.float64)
    active = np.asarray(active_mask, dtype=np.float64)

    kernel = np.asarray(hfsbr[:BX_LEN], dtype=np.float64).copy()

    # Current C loop starts at offset 1.
    kernel[0] = 0.0

    sources = hist * active[None, :]

    kernel_fft = np.fft.fft(kernel)
    source_fft = np.fft.fft(sources, axis=1)

    tail = np.fft.ifft(
        source_fft * kernel_fft[None, :],
        axis=1,
    ).real

    return hist + tail


def print_metric(
    name: str,
    test: np.ndarray,
    ref: np.ndarray,
    valid: np.ndarray | None = None,
) -> None:

    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    if test.shape != ref.shape:
        raise ValueError(
            f"{name}: shape mismatch {test.shape} vs {ref.shape}"
        )

    mask = np.isfinite(test) & np.isfinite(ref)

    if valid is not None:
        mask &= np.broadcast_to(valid, test.shape)

    if not np.any(mask):
        print(f"{name:45s}: NO VALID VALUES")
        return

    d = test[mask] - ref[mask]
    r = ref[mask]

    rms = np.sqrt(np.mean(d * d))
    ref_rms = np.sqrt(np.mean(r * r))

    rel_rms = rms / ref_rms if ref_rms > 0 else np.nan

    print(
        f"{name:45s} "
        f"N={len(d):10d}  "
        f"bias={np.mean(d): .6e}  "
        f"RMS={rms: .6e}  "
        f"MAE={np.mean(np.abs(d)): .6e}  "
        f"P95={np.percentile(np.abs(d), 95): .6e}  "
        f"max={np.max(np.abs(d)): .6e}  "
        f"RMS/refRMS={rel_rms: .6e}"
    )


def bx_mask_2d(mask_1d: np.ndarray, nrow: int) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(mask_1d, dtype=bool)[None, :],
        (nrow, BX_LEN),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--fill", required=True, type=int)

    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Use at most this many rows for expensive recovery comparisons.",
    )

    parser.add_argument(
        "--ped-column",
        default="bxraw",
    )

    parser.add_argument(
        "--afterglow-column",
        default="bxraw",
    )

    parser.add_argument(
        "--save-npz",
        default=None,
        help="Optional output NPZ with sampled validation arrays.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    fill = args.fill

    input_name = cfg.io.input_pattern.format(fill=fill)

    print("============================================================")
    print(f"ONLINE RECOVERY VALIDATION -- fill {fill}")
    print("============================================================")
    print(f"Input: {os.path.join(cfg.io.input_dir, input_name)}")
    print()

    # ------------------------------------------------------------------
    # Main HFET data
    # ------------------------------------------------------------------

    main_data = load_hd5_to_arrays(
        cfg.io.input_dir,
        input_name,
        node=cfg.io.node,
    )
    main_data = select_fill(main_data, fill)

    final_online = np.asarray(main_data["bxraw"], dtype=np.float64)

    if final_online.ndim != 2 or final_online.shape[1] != BX_LEN:
        raise RuntimeError(
            f"Main bxraw shape is {final_online.shape}, "
            f"expected (T, {BX_LEN})"
        )

    T = final_online.shape[0]

    print(f"Main rows: {T}")

    # ------------------------------------------------------------------
    # Reference tables
    # ------------------------------------------------------------------

    ped_data = load_hd5_to_arrays(
        cfg.io.input_dir,
        input_name,
        node=cfg.online_recovery.pedestal_node,
    )
    ped_data = select_fill(ped_data, fill)

    aft_data = load_hd5_to_arrays(
        cfg.io.input_dir,
        input_name,
        node=cfg.online_recovery.afterglow_node,
    )
    aft_data = select_fill(aft_data, fill)

    pedestal_4 = strict_align_aux(
        main_data,
        ped_data,
        args.ped_column,
    ).astype(np.float64)

    afterglow_frac = strict_align_aux(
        main_data,
        aft_data,
        args.afterglow_column,
    ).astype(np.float64)

    if pedestal_4.shape != (T, 4):
        raise RuntimeError(
            f"pedestal shape={pedestal_4.shape}, expected {(T, 4)}"
        )

    if afterglow_frac.shape != (T, BX_LEN):
        raise RuntimeError(
            f"afterglow_frac shape={afterglow_frac.shape}, "
            f"expected {(T, BX_LEN)}"
        )

    # ------------------------------------------------------------------
    # Exact tables reference
    # ------------------------------------------------------------------

    ped_bx = pedestal_pattern(pedestal_4)

    pre_pedestal_ref = final_online + ped_bx

    valid_frac = (
        np.isfinite(afterglow_frac)
        & (afterglow_frac > 0.0)
    )

    pre_afterglow_ref = np.full_like(
        pre_pedestal_ref,
        np.nan,
        dtype=np.float64,
    )

    pre_afterglow_ref[valid_frac] = (
        pre_pedestal_ref[valid_frac]
        / afterglow_frac[valid_frac]
    )

    print()
    print("TABLE COVERAGE")
    print("--------------")
    print(
        f"afterglow_frac > 0: "
        f"{np.count_nonzero(valid_frac)}/{valid_frac.size} "
        f"({100*np.mean(valid_frac):.3f}%)"
    )

    # ------------------------------------------------------------------
    # Table algebra closure
    # ------------------------------------------------------------------

    roundtrip = (
        pre_afterglow_ref * afterglow_frac
        - ped_bx
    )

    print()
    print("1. TABLE SELF-CLOSURE")
    print("---------------------")

    print_metric(
        "raw_table * frac - pedestal -> final",
        roundtrip,
        final_online,
        valid=valid_frac,
    )

    # ------------------------------------------------------------------
    # Pedestal semantics
    # ------------------------------------------------------------------

    print()
    print("2. PEDESTAL VALIDATION")
    print("----------------------")

    for n_sample in (10, 13):

        ped_from_pre_ped = calculate_dynamic_pedestal_batch(
            pre_pedestal_ref,
            n_sample=n_sample,
        )

        print_metric(
            f"ped table vs D(pre_ped), n={n_sample}",
            ped_from_pre_ped,
            pedestal_4,
        )

        # Only meaningful if table raw is defined in pedestal region.
        ped_from_raw = calculate_dynamic_pedestal_batch(
            pre_afterglow_ref,
            n_sample=n_sample,
        )

        print_metric(
            f"ped table vs D(pre_afterglow), n={n_sample}",
            ped_from_raw,
            pedestal_4,
        )

    # ------------------------------------------------------------------
    # Sample rows for expensive HFSBR tests
    # ------------------------------------------------------------------

    if T <= args.max_rows:
        sample_idx = np.arange(T)
    else:
        sample_idx = np.linspace(
            0,
            T - 1,
            args.max_rows,
            dtype=int,
        )

    print()
    print(f"HFSBR validation rows: {len(sample_idx)} / {T}")

    final_s = final_online[sample_idx]
    pre_ped_s = pre_pedestal_ref[sample_idx]
    raw_ref_s = pre_afterglow_ref[sample_idx]
    frac_s = afterglow_frac[sample_idx]
    valid_s = valid_frac[sample_idx]

    # ------------------------------------------------------------------
    # Active mask
    # ------------------------------------------------------------------

    mask_path = cfg.io.active_mask_pattern.format(fill=fill)

    with open(mask_path, "r") as f:
        active_mask = np.asarray(json.load(f), dtype=np.int32)

    if active_mask.shape != (BX_LEN,):
        raise RuntimeError(
            f"active mask shape={active_mask.shape}, "
            f"expected {(BX_LEN,)}"
        )

    active_cols = bx_mask_2d(
        active_mask == 1,
        len(sample_idx),
    )

    nonactive_cols = bx_mask_2d(
        active_mask == 0,
        len(sample_idx),
    )

    pedestal_cols_13 = np.zeros(BX_LEN, dtype=bool)
    pedestal_cols_13[3500:3552] = True
    pedestal_cols_13 = bx_mask_2d(
        pedestal_cols_13,
        len(sample_idx),
    )

    # ------------------------------------------------------------------
    # HFSBR
    # ------------------------------------------------------------------

    hfsbr = _load_hfsbr_for_online(cfg, fill)

    print()
    print("3. AFTERGLOW-ONLY CLOSURE")
    print("-------------------------")
    print(
        "Input to HFSBR inversion here is TABLE pre-pedestal state.\n"
        "Therefore pedestal recovery is completely removed from this test."
    )

    legacy_afterglow_only = revert_afterglow_legacy_only(
        pre_ped_s,
        active_mask,
        hfsbr,
    )

    snapshot_afterglow_only = revert_afterglow_snapshot_fft(
        pre_ped_s,
        active_mask,
        hfsbr,
    )

    print_metric(
        "legacy C revert vs table raw [all]",
        legacy_afterglow_only,
        raw_ref_s,
        valid=valid_s,
    )

    print_metric(
        "legacy C revert vs table raw [active]",
        legacy_afterglow_only,
        raw_ref_s,
        valid=valid_s & active_cols,
    )

    print_metric(
        "legacy C revert vs table raw [non-active]",
        legacy_afterglow_only,
        raw_ref_s,
        valid=valid_s & nonactive_cols,
    )

    print_metric(
        "legacy C revert vs table raw [ped region]",
        legacy_afterglow_only,
        raw_ref_s,
        valid=valid_s & pedestal_cols_13,
    )

    print()
    print("Alternative frozen-source / circular-convolution interpretation:")

    print_metric(
        "snapshot FFT vs table raw [all]",
        snapshot_afterglow_only,
        raw_ref_s,
        valid=valid_s,
    )

    print_metric(
        "snapshot FFT vs table raw [active]",
        snapshot_afterglow_only,
        raw_ref_s,
        valid=valid_s & active_cols,
    )

    print_metric(
        "snapshot FFT vs table raw [non-active]",
        snapshot_afterglow_only,
        raw_ref_s,
        valid=valid_s & nonactive_cols,
    )

    print()
    print("Legacy C vs snapshot interpretation:")

    print_metric(
        "legacy C vs snapshot FFT",
        legacy_afterglow_only,
        snapshot_afterglow_only,
    )

    # ------------------------------------------------------------------
    # Implied afterglow fractions
    # ------------------------------------------------------------------

    print()
    print("4. IMPLIED AFTERGLOW FRACTION")
    print("-----------------------------")

    eps = 1e-12

    good_legacy = np.abs(legacy_afterglow_only) > eps
    good_snapshot = np.abs(snapshot_afterglow_only) > eps

    frac_from_legacy = np.full_like(
        legacy_afterglow_only,
        np.nan,
        dtype=np.float64,
    )

    frac_from_snapshot = np.full_like(
        snapshot_afterglow_only,
        np.nan,
        dtype=np.float64,
    )

    frac_from_legacy[good_legacy] = (
        pre_ped_s[good_legacy]
        / legacy_afterglow_only[good_legacy]
    )

    frac_from_snapshot[good_snapshot] = (
        pre_ped_s[good_snapshot]
        / snapshot_afterglow_only[good_snapshot]
    )

    print_metric(
        "frac implied by legacy C vs table frac",
        frac_from_legacy,
        frac_s,
        valid=valid_s & good_legacy,
    )

    print_metric(
        "frac implied by snapshot vs table frac",
        frac_from_snapshot,
        frac_s,
        valid=valid_s & good_snapshot,
    )

    # ------------------------------------------------------------------
    # Current combined online implementation
    # ------------------------------------------------------------------

    print()
    print("5. CURRENT FULL LEGACY ONLINE METHOD")
    print("------------------------------------")

    legacy_full = reconstruct_from_online_batch(
        bxraw_final=final_s.astype(np.float32),
        hfsbr=hfsbr,
        active_mask=active_mask,
        show_progress=False,
    )

    print_metric(
        "CURRENT online mu_before vs table raw",
        legacy_full.mu_before,
        raw_ref_s,
        valid=valid_s,
    )

    print_metric(
        "CURRENT online mu_after vs table raw",
        legacy_full.mu_after,
        raw_ref_s,
        valid=valid_s,
    )

    print_metric(
        "CURRENT online pedestal vs table pedestal",
        legacy_full.pedestal,
        pedestal_4[sample_idx],
    )

    print()
    print("6. ZERO-BX PEDESTAL RECOVERY")
    print("----------------------------")

    zero_bx = (3553, 3554, 3555, 3556, 3557)

    # First: are these actually valid zero constraints?
    print_metric(
        "table raw at zero BX",
        raw_ref_s[:, zero_bx],
        np.zeros((len(sample_idx), len(zero_bx))),
    )

    recovered_zero, pedestal_zero = recover_online_with_zero_bx(
        bxraw_final=final_s,
        hfsbr=hfsbr,
        active_mask=active_mask,
        zero_bx=zero_bx,
    )

    print_metric(
        "zeroBX pedestal vs table pedestal",
        pedestal_zero,
        pedestal_4[sample_idx],
    )

    print_metric(
        "zeroBX recovered raw vs table raw [all]",
        recovered_zero,
        raw_ref_s,
        valid=valid_s,
    )

    print_metric(
        "zeroBX recovered raw vs table raw [active]",
        recovered_zero,
        raw_ref_s,
        valid=valid_s & active_cols,
    )

    print_metric(
        "zeroBX recovered raw vs table raw [non-active]",
        recovered_zero,
        raw_ref_s,
        valid=valid_s & nonactive_cols,
    )

    plot_online_recovery_per_bx(
        raw_ref=raw_ref_s,
        afterglow_only=legacy_afterglow_only,
        recovered=recovered_zero,
        valid=valid_s,
        active_mask=active_mask,
        cfg=cfg,
        fill=fill,
        year=2025,
        zero_bx=zero_bx,
    )

    print()
    print("7. EXACT ZERO-BX MODEL CHECK")
    print("----------------------------")

    zero_idx = np.asarray(zero_bx, dtype=np.int64)

    # ------------------------------------------------------------
    # A. TRUE table pedestal + our revert_afterglow
    #
    # legacy_afterglow_only was constructed as:
    #
    #   R(final + table_pedestal)
    #
    # Since the original software histogram has zero_bx == 0,
    # these values MUST be zero if our inverse model is exact.
    # ------------------------------------------------------------

    z_with_true_ped = legacy_afterglow_only[:, zero_idx]

    print("A. R(final + TABLE pedestal) at artificial zero BX")

    for j, bcid in enumerate(zero_idx):
        vals = z_with_true_ped[:, j]

        print(
            f"  BX {bcid}: "
            f"mean={np.mean(vals): .8e}  "
            f"RMS={np.sqrt(np.mean(vals**2)): .8e}  "
            f"max|x|={np.max(np.abs(vals)): .8e}"
        )

    print(
        "  ALL:     "
        f"mean={np.mean(z_with_true_ped): .8e}  "
        f"RMS={np.sqrt(np.mean(z_with_true_ped**2)): .8e}  "
        f"max|x|={np.max(np.abs(z_with_true_ped)): .8e}"
    )


    # ------------------------------------------------------------
    # B. Difference between pedestal obtained from zero BX
    #    and the true table pedestal.
    # ------------------------------------------------------------

    ped_ref_s = pedestal_4[sample_idx].astype(np.float64)
    ped_fit_s = pedestal_zero.astype(np.float64)

    delta_ped = ped_fit_s - ped_ref_s

    print()
    print("B. zero-BX fitted pedestal - TABLE pedestal")

    for k in range(4):
        vals = delta_ped[:, k]

        print(
            f"  mod4 {k}: "
            f"mean={np.mean(vals): .8e}  "
            f"RMS={np.sqrt(np.mean(vals**2)): .8e}  "
            f"max|x|={np.max(np.abs(vals)): .8e}"
        )


    # ------------------------------------------------------------
    # C. Direct C evaluation using the fitted pedestal.
    #
    # This is important because recovered_zero was obtained using
    # linear decomposition:
    #
    #   R(final) + pedestal @ R(pattern)
    #
    # Mathematically this equals R(final + pedestal_pattern),
    # but the C implementation operates in float32.
    #
    # Check that the linear solver itself is not introducing the
    # ~1e-4 discrepancy.
    # ------------------------------------------------------------

    idx_mod4 = np.arange(BX_LEN) % 4

    pre_ped_fitted = (
        final_s.astype(np.float64)
        + ped_fit_s[:, idx_mod4]
    ).astype(np.float32)

    recovered_zero_direct = revert_afterglow_legacy_only(
        pre_ped_fitted,
        active_mask,
        hfsbr,
    )

    print()
    print("C. Direct C evaluation vs linear-response reconstruction")

    print_metric(
        "direct R(final + fitted ped) vs linear solution",
        recovered_zero_direct,
        recovered_zero,
    )

    z_fitted_direct = recovered_zero_direct[:, zero_idx]

    print()
    print("D. Direct R(final + FITTED pedestal) at zero BX")

    for j, bcid in enumerate(zero_idx):
        vals = z_fitted_direct[:, j]

        print(
            f"  BX {bcid}: "
            f"mean={np.mean(vals): .8e}  "
            f"RMS={np.sqrt(np.mean(vals**2)): .8e}  "
            f"max|x|={np.max(np.abs(vals)): .8e}"
        )


    # ------------------------------------------------------------
    # E. Does the pedestal difference exactly explain the residual
    #    obtained with the true pedestal?
    #
    # From linearity:
    #
    #   R(final + p_fit)
    # = R(final + p_table) + A (p_fit - p_table)
    #
    # and the LHS should be zero at zero BX.
    # ------------------------------------------------------------

    response = build_pedestal_response(
        hfsbr=hfsbr,
        active_mask=active_mask,
    )

    A = response[:, zero_idx].T.astype(np.float64)

    predicted_zero_shift = delta_ped @ A.T

    closure_prediction = (
        z_with_true_ped.astype(np.float64)
        + predicted_zero_shift
    )

    print()
    print("E. z(table pedestal) + A * (fitted-table pedestal)")

    print(
        f"  mean={np.mean(closure_prediction): .8e}  "
        f"RMS={np.sqrt(np.mean(closure_prediction**2)): .8e}  "
        f"max|x|={np.max(np.abs(closure_prediction)): .8e}"
    )

    # ------------------------------------------------------------------
    # Save detailed sample
    # ------------------------------------------------------------------

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            sample_idx=sample_idx,
            final_online=final_s,
            pedestal_table=pedestal_4[sample_idx],
            afterglow_frac_table=frac_s,
            pre_pedestal_table=pre_ped_s,
            pre_afterglow_table=raw_ref_s,
            legacy_afterglow_only=legacy_afterglow_only,
            snapshot_afterglow_only=snapshot_afterglow_only,
            legacy_full_mu_before=legacy_full.mu_before,
            legacy_full_mu_after=legacy_full.mu_after,
            legacy_full_pedestal=legacy_full.pedestal,
            active_mask=active_mask,
            hfsbr=hfsbr,
        )

        print()
        print(f"Saved detailed validation arrays to {args.save_npz}")


if __name__ == "__main__":
    main()