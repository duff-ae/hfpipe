# src/hfcore/bunch_train.py

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

from .hd5schema import BX_LEN
from .decorators import log_step, timeit
from .type1_fit import _do_binning, _h5_get_or_create, _h5_append_1d

log = logging.getLogger("hfpipe.bunch_train")


# ---------------------------------------------------------------------------
# Helpers for bunch train coefficient extraction
# ---------------------------------------------------------------------------

def _find_head(heads, idx):
    """
    Find the head bx associated with the given tail bx
    """
    return heads[np.searchsorted(heads, idx, side="left") - 1]

def _downsample(arr, max_len=15000):
    step = max(1, len(arr) // max_len)
    return arr[::step]

def _collect_bunch_train_points(
    bxraw: np.ndarray,
    bxraw_ref: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    sbil_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (y, ratio) points
    """
    #active_mask = np.asarray(active_mask, dtype=np.int32)
    bxraw = np.asarray(bxraw, dtype=np.float64)
    bxraw_ref = np.asarray(bxraw_ref, dtype=np.float64)
    #avg = np.asarray(avg, dtype=np.float64)

    assert bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    # SBIL / avg filter
    mask_sbil = avg > sbil_min
    hists = bxraw[mask_sbil]
    hists_ref = bxraw_ref[mask_sbil]
    if hists.shape[0] == 0:
        return np.array([]), np.array([])

    # using too much memory so remove a bunch of the points
    hists = _downsample(hists)
    hists_ref = _downsample(hists_ref)

    shift = np.roll(active_mask, 1)
    heads = np.where(active_mask & ~shift)[0]
    tails = np.where(active_mask & shift)[0]

    if tails.size == 0:
        return np.array([]), np.array([])

    # collect values across all selected histograms
    y_vals = []
    frac_vals = []
    for hist, hist_ref in zip(hists, hists_ref):
        head = _find_head(heads, tails)

        # avoid division by zero
        mask_nonzero = hist[head] > 0.0
        if not np.any(mask_nonzero):
            continue

        y = hist[tails][mask_nonzero]
        frac     = (hist[tails] / hist[head])[mask_nonzero]
        frac_ref = (hist_ref[tails] / hist_ref[head])[mask_nonzero]
        
        y_vals.append(y)
        frac_vals.append(frac_ref / frac - 1)

    if not y_vals:
        return np.array([]), np.array([])

    y_all = np.concatenate(y_vals)
    frac_all = np.concatenate(frac_vals)

    return y_all, frac_all

@log_step("compute_bunch_train_coeffs")
@timeit("compute_bunch_train_coeffs")
def compute_bunch_train_coeffs(
    bxraw: np.ndarray,
    bxraw_ref: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    sbil_min: float,
    order: int,
) -> Tuple[float, float, float]:
    """
    Fit bunch train residuals (relative to reference luminometer)
    """
    y_all, frac_all = _collect_bunch_train_points(bxraw, bxraw_ref, avg, active_mask, sbil_min)

    if y_all.size == 0:
        # no points -> all coefficients are zero
        return np.array([0.0, 0.0, 0.0])

    # np.polyfit returns [c_k, ..., c_0] for poly(x) = c_k x^k + ... + c_0
    coeffs = np.polyfit(y_all, frac_all, order)
    coeffs = coeffs[::-1]  # now coeffs[0] = c_0, coeffs[1] = c_1, ...

    return coeffs

def save_bunch_train_coeffs(
    fill: int,
    output_dir: str,
    coeffs: np.ndarray,
) -> str:
    """
    Save bunch train coefficients into a small HDF5 file:
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bunch_train_coeffs_fill{fill}.h5")

    coeffs = np.asarray(coeffs, dtype=np.float64)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("coeffs", data=coeffs)
        h5.attrs["fill"] = int(fill)

    return path


# ---------------------------------------------------------------------------
# Helpers for diagnostic analysis (scatter / binned / fit)
# ---------------------------------------------------------------------------

def _select_bunch_train_pairs(active_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select (tail BX, head BX) index pairs
    """
    active_mask = np.asarray(active_mask, dtype=np.int8)
    shift = np.roll(active_mask, 1)
    heads = np.where(active_mask & ~shift)[0]
    tails = np.where(active_mask & shift)[0]
    tails = tails[tails < 3480]

    if tails.size == 0:
        return np.array([]), np.array([])

    return tails, _find_head(heads, tails)


# ---------------------------------------------------------------------------
# Main diagnostic step
# ---------------------------------------------------------------------------

@log_step("analyze_bunch_train_step")
def analyze_bunch_train_step(data, cfg, active_mask, fill: int, tag: str = "before"):
    """
    Make diagnostic plots
    """
    if "bxraw_ref" not in data:
        raise KeyError("analyze_bunch_train_step: 'bxraw_ref' not found in data")

    #bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    #bxraw_ref = np.asarray(data["bxraw_ref"], dtype=np.float64)
    
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
        avg = (data["bxraw"] * mask[None, :]).sum(axis=1) / float(n_active)

    sbil_min: float = float(getattr(cfg.bunch_train, "sbil_min", 0.1))
    make_plots: bool = bool(getattr(cfg.bunch_train, "make_plots", False))
    save_hd5: bool = bool(getattr(cfg.bunch_train, "save_hd5", False))
    order: int = int(getattr(cfg.bunch_train, "order", 1))

    # base directory for debug output
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

    tag_suffix = f"_{tag}" if tag else ""

    # --- SBIL selection ---
    time_mask = avg > sbil_min
    if not np.any(time_mask):
        log.warning(
            "[analyze_bunch_train_step] fill %d: no points with SBIL > %g",
            fill,
            sbil_min,
        )
    else:
        hists = data["bxraw"][time_mask, :] * 11245.6/cfg.afterglow.sigvis   # shape (T_selected, BX_LEN)
        hists_ref = data["bxraw_ref"][time_mask, :] * 11245.6/cfg.afterglow.sigvis

        # remove some of the points, because the full data is enough to cause the plotter to crash
        hists = _downsample(hists, max_len=500)
        hists_ref = _downsample(hists_ref, max_len=500)

        # --- select BX pairs ---
        tail_idx, head_idx = _select_bunch_train_pairs(active_mask)

        if tail_idx.size == 0:
            log.warning(
                "[analyze_bunch_train_step] fill %d: no (head, tail) pairs found",
                fill,
            )
        else:
            # --- extract values ---
            head     = hists[:, head_idx]      # (T_sel, Npairs)
            tail     = hists[:, tail_idx]       # (T_sel, Npairs)
            head_ref = hists_ref[:, head_idx]      # (T_sel, Npairs)
            tail_ref = hists_ref[:, tail_idx]       # (T_sel, Npairs)

            # protect against division by zero
            valid = head > 0.0
            head     = head[valid]
            tail     = tail[valid]
            head_ref = head_ref[valid]
            tail_ref = tail_ref[valid]

            if head.size == 0:
                log.warning(
                    "[analyze_bunch_train_step] fill %d: no positive colliding BX values",
                    fill,
                )
            else:
                frac     = (tail / head)
                frac_ref = (tail_ref / head_ref)
                bx_ratio = (frac_ref / frac - 1)

                # --- binning ---
                x = tail
                x_bins, y_bins, s_bins = _do_binning(x, bx_ratio, nbins=20)

                # --- fit ---
                coeffs = np.polyfit(x, bx_ratio, order)
                new_x = np.linspace(
                    float(np.min(x)),
                    float(np.max(x)),
                    num=x.size,
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
                    hd5_path = os.path.join(hd5_dir, f"bunch_train{tag_suffix}.h5")

                    fill_arr_scatter = np.full(x.size, int(fill), dtype=np.int64)
                    fill_arr_binned  = np.full(x_bins.size,   int(fill), dtype=np.int64)
                    fill_arr_fit     = np.full(new_x.size,    int(fill), dtype=np.int64)

                    with h5py.File(hd5_path, "a") as h5:
                        h5.attrs["fill"] = int(fill)
                        h5.attrs["order"] = int(order)
                        h5.attrs["tag"] = str(tag)

                        # --- scatter ---
                        ds_fill = _h5_get_or_create(h5, "scatter/fill", dtype=np.int64)
                        ds_x    = _h5_get_or_create(h5, "scatter/x",    dtype=np.float64)
                        ds_y    = _h5_get_or_create(h5, "scatter/y",    dtype=np.float64)
                        _h5_append_1d(ds_fill, fill_arr_scatter)
                        _h5_append_1d(ds_x, x)
                        _h5_append_1d(ds_y, bx_ratio)

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
                        "[analyze_bunch_train_step] fill %d tag=%s: saved debug HDF5 to %s",
                        fill,
                        tag,
                        hd5_path,
                    )

                # --- PNG (optional) ---
                if make_plots:
                    fig = plt.figure(figsize=(7, 5))
                    # scatter
                    plt.plot(x, bx_ratio, ".", alpha=0.002, label=f"Bunch Train fraction to reference")
                    # binned
                    plt.errorbar(x_bins, y_bins, yerr=s_bins, fmt="o", linestyle="",
                                 markersize=4, lw=1, zorder=10, capsize=3, capthick=1, label="Binned")

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
                    plt.ylabel("Bunch Train Ratio to Reference")
                    plt.title(f"Fill {fill}, tag={tag}")
                    plt.legend(loc="upper right", frameon=False)
                    plt.tight_layout()

                    png_path = os.path.join(output_dir, f"bunch_train{tag_suffix}.png")
                    plt.savefig(png_path, dpi=300)
                    plt.close(fig)

                    log.info(
                        "[analyze_bunch_train_step] fill %d tag=%s: saved PNG to %s",
                        fill,
                        tag,
                        png_path,
                    )


def apply_bunch_train_batch(
    bxraw: np.ndarray,
    active_mask: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Apply the bunch train correction
    """
    hists = np.asarray(bxraw, dtype=np.float64)
    T, N = hists.shape

    # Work on a copy to avoid in-place modification of the input array
    out = hists.copy()

    active_mask = np.asarray(active_mask, dtype=np.int8)

    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    for ibx in range(1, N):
        if active_mask[ibx] != 1 or active_mask[ibx - 1] != 1:
            continue

        y = out[:, ibx]
        out[:, ibx] += y * np.polyval(coeffs[::-1], y)

    return out


def load_bunch_train_coeffs(fill: int, output_dir: str) -> np.ndarray:
    """
    Load coeffs from bunch_train_coeffs_fill{fill}.h5 in the given directory.
    """
    path = os.path.join(output_dir, f"bunch_train_coeffs_fill{fill}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bunch Train coeffs file not found: {path}")

    with h5py.File(path, "r") as h5:
        p = np.asarray(h5["coeffs"], dtype=np.float64)

    return p
