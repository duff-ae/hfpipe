# src/hfcore/type1_fit.py

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

from .hd5schema import BX_LEN
from .decorators import log_step, timeit

log = logging.getLogger("hfpipe.type1_fit")


# ---------------------------------------------------------------------------
# Helpers for Type-1 coefficient extraction
# ---------------------------------------------------------------------------

def _collect_type1_points(
    bxraw: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    offset: int,
    sbil_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (y, frac) points for a given offset.

      y    = mu_colliding  = bxraw[ibx]
      frac = mu_afterglow / mu_colliding
           = bxraw[ibx + offset] / bxraw[ibx]

    We only use:
      - rows with avg > sbil_min (avg is treated as SBIL or its proxy),
      - BX indices such that:
          active_mask[bx] == 1      (colliding BX)
        and active_mask[bx + dt] == 0 for all dt = 1..offset
        (following 'offset' BX are non-colliding).
    """
    active_mask = np.asarray(active_mask, dtype=np.int32)
    bxraw = np.asarray(bxraw, dtype=np.float64)
    avg = np.asarray(avg, dtype=np.float64)

    assert bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    # SBIL / avg filter
    mask_sbil = avg > sbil_min
    hists = bxraw[mask_sbil]
    if hists.shape[0] == 0:
        return np.array([]), np.array([])

    N = BX_LEN
    upper = N - offset

    colliding_indices: list[int] = []
    for bx in range(upper):
        if active_mask[bx] != 1:
            continue
        # require that the next 'offset' BX are non-active
        ok = True
        for dt in range(1, offset + 1):
            if active_mask[bx + dt] != 0:
                ok = False
                break
        if ok:
            colliding_indices.append(bx)

    if not colliding_indices:
        return np.array([]), np.array([])

    colliding_indices = np.asarray(colliding_indices, dtype=np.int64)
    afterglow_indices = colliding_indices + offset

    # collect values across all selected histograms
    y_vals = []
    frac_vals = []
    for hist in hists:
        y = hist[colliding_indices]       # mu_colliding
        y_after = hist[afterglow_indices] # mu_afterglow

        # avoid division by zero
        mask_nonzero = y > 0.0
        if not np.any(mask_nonzero):
            continue

        y = y[mask_nonzero]
        y_after = y_after[mask_nonzero]
        frac = y_after / y

        y_vals.append(y)
        frac_vals.append(frac)

    if not y_vals:
        return np.array([]), np.array([])

    y_all = np.concatenate(y_vals)
    frac_all = np.concatenate(frac_vals)
    return y_all, frac_all


def _fit_type1_for_offset(
    bxraw: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    offset: int,
    sbil_min: float,
    order: int,
) -> Tuple[float, float, float]:
    """
    Fit Type-1 fraction as a polynomial in y = mu_colliding:

      frac(y) ≈ poly_order(y)

    order:
      1 -> linear      : frac ≈ k * y      + b
      2 -> quadratic   : frac ≈ k2 * y^2 + k1 * y + b

    Returns (p0, p1, p2) where:
      - for order=1: p0 = b,  p1 = k,  p2 = 0
      - for order=2: p0 = b,  p1 = k1, p2 = k2

    This layout is convenient for direct use in Type-1 subtraction.
    """
    y_all, frac_all = _collect_type1_points(bxraw, avg, active_mask, offset, sbil_min)

    if y_all.size == 0:
        # no points -> all coefficients are zero
        return 0.0, 0.0, 0.0

    # np.polyfit returns [c_k, ..., c_0] for poly(x) = c_k x^k + ... + c_0
    coeffs = np.polyfit(y_all, frac_all, order)
    coeffs = coeffs[::-1]  # now coeffs[0] = c_0, coeffs[1] = c_1, ...

    # linear case: frac ≈ k * y + b
    if order == 1:
        b = float(coeffs[0])
        k = float(coeffs[1]) if coeffs.size > 1 else 0.0
        return b, k, 0.0

    # quadratic case: frac ≈ k2 * y^2 + k1 * y + b
    if order == 2:
        b = float(coeffs[0])
        k1 = float(coeffs[1]) if coeffs.size > 1 else 0.0
        k2 = float(coeffs[2]) if coeffs.size > 2 else 0.0
        return b, k1, k2

    # fallback (should not be used in practice)
    p0 = float(coeffs[0]) if coeffs.size > 0 else 0.0
    p1 = float(coeffs[1]) if coeffs.size > 1 else 0.0
    p2 = float(coeffs[2]) if coeffs.size > 2 else 0.0
    return p0, p1, p2


@log_step("compute_type1_coeffs")
@timeit("compute_type1_coeffs")
def compute_type1_coeffs(
    bxraw: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    offsets: Sequence[int],
    sbil_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arrays p0, p1, p2 and orders for all requested offsets.

    Fit order convention:
      - offset == 1 -> quadratic (order = 2)
      - offset >  1 -> linear    (order = 1)

    Returned arrays:
      p0, p1, p2, orders of length max_offset + 1;
      index t corresponds to BX + t, i.e. t=1 -> BX+1, etc.
      t=0 is kept zero (no subtraction for the colliding BX itself).
    """
    bxraw = np.asarray(bxraw, dtype=np.float64)
    avg = np.asarray(avg, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=np.int32)

    assert bxraw.ndim == 2 and bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    if not offsets:
        max_offset = 0
    else:
        max_offset = max(offsets)

    p0 = np.zeros(max_offset + 1, dtype=np.float64)
    p1 = np.zeros(max_offset + 1, dtype=np.float64)
    p2 = np.zeros(max_offset + 1, dtype=np.float64)
    orders = np.zeros(max_offset + 1, dtype=np.int32)

    for off in offsets:
        if off <= 0:
            continue  # offset = 0 is not used

        # convention: BX+1 -> quadratic, all others -> linear
        if off == 1:
            order = 2
        else:
            order = 1

        c0, c1, c2 = _fit_type1_for_offset(
            bxraw=bxraw,
            avg=avg,
            active_mask=active_mask,
            offset=off,
            sbil_min=sbil_min,
            order=order,
        )
        p0[off] = c0
        p1[off] = c1
        p2[off] = c2
        orders[off] = order

    return p0, p1, p2, orders


def save_type1_coeffs(
    fill: int,
    output_dir: str,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    offsets: Sequence[int],
    orders: np.ndarray,
) -> str:
    """
    Save Type-1 coefficients into a small HDF5 file:

      type1_coeffs_fill{fill}.h5

    Layout:
      datasets:
        - "p0", "p1", "p2"     (length = max_offset+1)
        - "offsets"            (list of offsets actually fitted)
        - "orders"             (fit order for each t)
      attrs:
        - "fill"
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"type1_coeffs_fill{fill}.h5")

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    orders = np.asarray(orders, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("p0", data=p0)
        h5.create_dataset("p1", data=p1)
        h5.create_dataset("p2", data=p2)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("orders", data=orders)
        h5.attrs["fill"] = int(fill)

    return path


# ---------------------------------------------------------------------------
# Helpers for diagnostic analysis (scatter / binned / fit)
# ---------------------------------------------------------------------------

def _select_type1_pairs(active_mask: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select (colliding BX, afterglow BX) index pairs for a given offset.

    Logic mirrors the old code:

      colliding_value = [
          bx
          for bx in range(upper)
          if active_mask[bx] == 1
          and all(active_mask[bx + dt] == 0 for dt in range(1, offset + 1))
      ]
      afterglow_value = [bx + offset for bx in colliding_value]

    where 'upper' is min(3480, N - offset).
    """
    active_mask = np.asarray(active_mask, dtype=np.int8)
    N = active_mask.shape[0]

    upper = min(3480, N - offset)

    colliding: list[int] = []
    for bx in range(upper):
        if active_mask[bx] != 1:
            continue
        # all following 'offset' BX must be non-active
        if all(active_mask[bx + dt] == 0 for dt in range(1, offset + 1)):
            colliding.append(bx)

    colliding = np.asarray(colliding, dtype=np.int64)
    afterglow = colliding + offset
    return colliding, afterglow


def _do_binning(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    """
    Perform approximate equal-population binning in x, return binned (x, y) and errors.

    Returns:
      x_bins : mean x in each bin
      y_bins : mean y in each bin
      s_bins : standard error of y in each bin
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    # sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # split into 'nbins' bins with (approximately) equal number of points
    n = x_sorted.size
    nbins = min(n, nbins)
    bin_edges = np.linspace(0, n, nbins + 1, dtype=int)

    x_bins = []
    y_bins = []
    s_bins = []

    for i in range(nbins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if end <= start:
            continue
        xx = x_sorted[start:end]
        yy = y_sorted[start:end]
        if xx.size == 0:
            continue
        x_bins.append(xx.mean())
        y_bins.append(yy.mean())
        # standard error of the mean
        if yy.size > 1:
            s_bins.append(yy.std(ddof=1) / np.sqrt(yy.size))
        else:
            s_bins.append(0.0)

    return np.asarray(x_bins), np.asarray(y_bins), np.asarray(s_bins)


def _h5_get_or_create(h5: h5py.File, path: str, dtype) -> h5py.Dataset:
    """
    Get or create a 1D dataset with the given dtype.
    """
    if path in h5:
        ds = h5[path]
        # minimal dtype check
        if ds.dtype != np.dtype(dtype):
            raise TypeError(f"HDF5 dataset {path} has dtype={ds.dtype}, expected {dtype}")
        return ds
    else:
        return h5.create_dataset(path, shape=(0,), maxshape=(None,), dtype=dtype)


def _h5_append_1d(ds: h5py.Dataset, arr: np.ndarray):
    """
    Append a 1D array at the end of the dataset.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("_h5_append_1d: only 1D arrays supported")
    old_n = ds.shape[0]
    new_n = old_n + arr.shape[0]
    ds.resize((new_n,))
    ds[old_n:new_n] = arr


# ---------------------------------------------------------------------------
# Main diagnostic step
# ---------------------------------------------------------------------------

@log_step("analyze_type1_step")
def analyze_type1_step(data, cfg, active_mask, fill: int, tag: str = "before"):
    """
    Diagnostic step: reproduces the old calculate_type1 logic
    using the current data/cfg structure.

    For each offset in cfg.type1.offsets:
      - builds a scatter of Type-1 fractions (afterglow/colliding),
      - performs x-binning,
      - fits a polynomial (order=2 for offset=1, else order=1),
      - optionally stores debug HDF5 (if cfg.type1.save_hd5 is True),
      - optionally produces PNG plots (if cfg.type1.make_plots is True).

    This function does NOT modify 'data'.

    The 'tag' argument ("before" / "after") is used only to distinguish
    the output debug files for pre- and post-Type-1 application.
    """
    if "bxraw" not in data:
        raise KeyError("analyze_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"analyze_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # SBIL / avg – same rules as in compute_type1_step
    if "sbil" in data:
        avg = np.asarray(data["sbil"], dtype=np.float64)
    elif "avg" in data:
        avg = np.asarray(data["avg"], dtype=np.float64)
    else:
        mask = np.asarray(active_mask, dtype=np.int32)
        n_active = int(mask.sum())
        if n_active == 0:
            raise ValueError("analyze_type1_step: active_mask has zero active BX")
        avg = (bxraw * mask[None, :]).sum(axis=1) / float(n_active)

    offsets: Sequence[int] = list(getattr(cfg.type1, "offsets", [1, 2, 3, 4]))
    sbil_min: float = float(getattr(cfg.type1, "sbil_min", 0.1))
    make_plots: bool = bool(getattr(cfg.type1, "make_plots", False))
    save_hd5: bool = bool(getattr(cfg.type1, "save_hd5", False))

    # base directory for Type-1 debug output
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

    tag_suffix = f"_{tag}" if tag else ""

    for offset in offsets:
        # fit order convention
        order = 2 if offset == 1 else 1

        # --- SBIL selection ---
        time_mask = avg > sbil_min
        if not np.any(time_mask):
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no points with SBIL > %g",
                fill,
                offset,
                sbil_min,
            )
            continue

        hists = bxraw[time_mask, :]   # shape (T_selected, BX_LEN)

        # --- select BX pairs ---
        colliding_idx, afterglow_idx = _select_type1_pairs(active_mask, offset)
        if colliding_idx.size == 0:
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no (colliding, afterglow) pairs found",
                fill,
                offset,
            )
            continue

        # --- extract values ---
        coll = hists[:, colliding_idx]      # (T_sel, Npairs)
        aft = hists[:, afterglow_idx]       # (T_sel, Npairs)

        # protect against division by zero
        valid = coll > 0.0
        coll = coll[valid]
        aft = aft[valid]

        if coll.size == 0:
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no positive colliding BX values",
                fill,
                offset,
            )
            continue

        bx_value = coll.astype(np.float64)              # base intensity
        bx_type1 = (aft / coll).astype(np.float64)      # Type-1 fraction

        # --- binning ---
        x_bins, y_bins, s_bins = _do_binning(bx_value, bx_type1, nbins=20)

        # --- fit ---
        coeffs = np.polyfit(bx_value, bx_type1, order)
        new_x = np.linspace(
            float(np.min(bx_value)),
            float(np.max(bx_value)),
            num=bx_value.size,
        )
        new_line = np.polyval(coeffs, new_x)

        # --- HDF5 output (optional, controlled by cfg.type1.save_hd5) ---
        # Layout:
        #   <type1_dir>/<fill>/hd5/type1_{offset}{tag_suffix}.h5
        output_dir = os.path.join(type1_dir, str(fill))
        os.makedirs(output_dir, exist_ok=True)

        if save_hd5:
            hd5_dir = os.path.join(output_dir, "hd5")
            os.makedirs(hd5_dir, exist_ok=True)
            hd5_path = os.path.join(hd5_dir, f"type1_{offset}{tag_suffix}.h5")

            fill_arr_scatter = np.full(bx_value.size, int(fill), dtype=np.int64)
            fill_arr_binned  = np.full(x_bins.size,   int(fill), dtype=np.int64)
            fill_arr_fit     = np.full(new_x.size,    int(fill), dtype=np.int64)

            with h5py.File(hd5_path, "a") as h5:
                h5.attrs["fill"] = int(fill)
                h5.attrs["type"] = int(offset)
                h5.attrs["order"] = int(order)
                h5.attrs["tag"] = str(tag)

                # --- scatter ---
                ds_fill = _h5_get_or_create(h5, "scatter/fill", dtype=np.int64)
                ds_x    = _h5_get_or_create(h5, "scatter/x",    dtype=np.float64)
                ds_y    = _h5_get_or_create(h5, "scatter/y",    dtype=np.float64)
                _h5_append_1d(ds_fill, fill_arr_scatter)
                _h5_append_1d(ds_x, bx_value)
                _h5_append_1d(ds_y, bx_type1)

                # --- binned ---
                db_fill = _h5_get_or_create(h5, "binned/fill", dtype=np.int64)
                db_x    = _h5_get_or_create(h5, "binned/x",    dtype=np.float64)
                db_y    = _h5_get_or_create(h5, "binned/y",    dtype=np.float64)
                db_ey   = _h5_get_or_create(h5, "binned/yerr", dtype=np.float64)
                _h5_append_1d(db_fill, fill_arr_binned)
                _h5_append_1d(db_x, x_bins)
                _h5_append_1d(db_y, y_bins)
                _h5_append_1d(db_ey, s_bins)

                # --- fit curve ---
                df_fill = _h5_get_or_create(h5, "fit/fill", dtype=np.int64)
                df_x    = _h5_get_or_create(h5, "fit/x",    dtype=np.float64)
                df_y    = _h5_get_or_create(h5, "fit/y",    dtype=np.float64)
                _h5_append_1d(df_fill, fill_arr_fit)
                _h5_append_1d(df_x, new_x.astype(np.float64))
                _h5_append_1d(df_y, new_line.astype(np.float64))

                # --- poly coefficients ---
                pc_fill = _h5_get_or_create(h5, "poly/fill",   dtype=np.int64)
                pc_deg  = _h5_get_or_create(h5, "poly/order",  dtype=np.int64)
                pc_coef = _h5_get_or_create(h5, "poly/coeffs", dtype=np.float64)
                _h5_append_1d(pc_fill, np.array([int(fill)], dtype=np.int64))
                _h5_append_1d(pc_deg,  np.array([int(order)], dtype=np.int64))
                _h5_append_1d(pc_coef, np.asarray(coeffs, dtype=np.float64))

            log.info(
                "[analyze_type1_step] fill %d offset %d tag=%s: saved debug HDF5 to %s",
                fill,
                offset,
                tag,
                hd5_path,
            )

        # --- PNG (optional) ---
        if make_plots:
            fig = plt.figure(figsize=(7, 5))
            # scatter
            plt.plot(
                bx_value,
                bx_type1,
                ".",
                alpha=0.2,
                label=f"Type-1 fraction for BX[i+{offset}]",
            )
            # binned
            plt.errorbar(
                x_bins,
                y_bins,
                yerr=s_bins,
                fmt="o",
                linestyle="",
                markersize=4,
                lw=1,
                zorder=10,
                capsize=3,
                capthick=1,
                label="Binned",
            )

            # fit curve
            if order == 1:
                plt.plot(
                    new_x,
                    new_line,
                    label=f"Linear fit: {coeffs[0]:.5f} x + {coeffs[1]:.5f}",
                )
            else:
                # coeffs is [c_k, ..., c_0] as returned by np.polyfit
                poly_str = " + ".join(
                    f"{c:.5f} x^{i}"
                    for i, c in zip(range(order, -1, -1), coeffs)
                )
                plt.plot(new_x, new_line, label=f"Poly{order} fit: {poly_str}")

            plt.xlabel("Instantaneous luminosity [Hz/μb]")
            plt.ylabel("Type-1 fraction")
            plt.title(f"Fill {fill}, offset={offset}, tag={tag}")
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()

            png_path = os.path.join(output_dir, f"type1_{offset}{tag_suffix}.png")
            plt.savefig(png_path, dpi=300)
            plt.close(fig)

            log.info(
                "[analyze_type1_step] fill %d offset %d tag=%s: saved PNG to %s",
                fill,
                offset,
                tag,
                png_path,
            )