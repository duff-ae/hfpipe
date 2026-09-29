# src/hfcore/bunch_train.py

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

from .hd5schema import BX_LEN
from .decorators import log_step, timeit
from .type1_fit import _do_binning, _h5_get_or_create, _h5_append_1d, get_sbil_like_column

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
        mask_nonzero = hist[head] > sbil_min
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
# Streaming / chunked equivalent of the fit logic above
# ---------------------------------------------------------------------------
#
# Mirrors type1_fit.Type1OffsetAccumulator / accumulate_type1_offset_chunk:
# lets a caller reproduce compute_bunch_train_coeffs' (y, frac) point
# collection and final np.polyfit call without ever holding the whole
# fill's bxraw / bxraw_ref arrays in memory -- process the fill in row
# chunks, accumulate (y, frac) points via BunchTrainAccumulator, then fit
# once all chunks have been scanned.
#
# Unlike Type-1, bunch-train fits a single order with no sequential
# per-offset chain (no in-place subtraction of one offset's estimated
# contribution before fitting the next), so a single accumulate-then-
# finalize pass over the fill is enough here -- there's no ping-pong
# scratch-file dance needed, unlike compute_type1_fill.

class BunchTrainAccumulator:
    """
    Incrementally collects the (y, frac) points that
    `_collect_bunch_train_points` would otherwise need the whole-fill
    `bxraw` / `bxraw_ref` arrays in memory to compute.

    Call `.add()` once per chunk (via `accumulate_bunch_train_chunk`),
    then `.finalize(order)` once all chunks have been seen to get back
    the same coefficients `compute_bunch_train_coeffs` would produce
    from the full arrays, since `np.polyfit` is called on the
    concatenation of all chunk-level points either way.
    """

    def __init__(self) -> None:
        self._y_parts: List[np.ndarray] = []
        self._frac_parts: List[np.ndarray] = []

    def add(self, y: np.ndarray, frac: np.ndarray) -> None:
        if y.size:
            self._y_parts.append(y)
            self._frac_parts.append(frac)

    @property
    def n_points(self) -> int:
        return sum(p.size for p in self._y_parts)

    def finalize(self, order: int) -> np.ndarray:
        """
        Same convention as `compute_bunch_train_coeffs`: returns
        `coeffs` such that `coeffs[0]` is the constant term, `coeffs[1]`
        the linear term, etc. (already reversed from what `np.polyfit`
        returns), ready to hand straight to `apply_bunch_train_batch`.

        Returns an all-zero array of length `order + 1` if no points
        were ever added. (The original in-memory
        `compute_bunch_train_coeffs` always returns a fixed length-3
        zero array in this "no points" case, regardless of `order`;
        since every entry is zero either way this makes no numerical
        difference through `apply_bunch_train_batch` -- `order + 1`
        zeros is just the more internally-consistent shape to return.)
        """
        if not self._y_parts:
            return np.zeros(order + 1, dtype=np.float64)

        y_all = np.concatenate(self._y_parts)
        frac_all = np.concatenate(self._frac_parts)

        coeffs = np.polyfit(y_all, frac_all, order)
        coeffs = coeffs[::-1]  # coeffs[0] = c_0, coeffs[1] = c_1, ...

        return coeffs


def accumulate_bunch_train_chunk(
    acc: BunchTrainAccumulator,
    bxraw_chunk: np.ndarray,
    bxraw_ref_chunk: np.ndarray,
    avg_chunk: np.ndarray,
    active_mask: np.ndarray,
    sbil_min: float,
) -> None:
    """
    Feed one chunk's worth of points into `acc`. Thin wrapper around
    `_collect_bunch_train_points` so call sites in the chunked pipeline
    stay symmetric with the accumulator API (and with
    `accumulate_type1_offset_chunk` in type1_fit.py).
    """
    y, frac = _collect_bunch_train_points(
        bxraw_chunk, bxraw_ref_chunk, avg_chunk, active_mask, sbil_min,
    )
    acc.add(y, frac)


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
# Streaming / chunked equivalent of the diagnostic scatter collection
# ---------------------------------------------------------------------------
#
# Mirrors type1_fit.Type1DiagnosticAccumulator / analyze_type1_offset_finalize
# / analyze_type1_fill_chunked. Like the Type-1 diagnostic (and unlike the
# fit in BunchTrainAccumulator above, which re-derives a head for every
# tail per row via `_find_head`), the (tail_idx, head_idx) pairing here
# only depends on `active_mask`, so it's computed once per fill and reused
# across all chunks -- everything can be accumulated in a SINGLE streaming
# pass, no sequential/ping-pong semantics involved.
#
# Because only the (head, tail) BX columns are ever extracted per chunk --
# never the full (T, BX_LEN) histogram -- there's no need for
# `analyze_bunch_train_step`'s `_downsample(..., max_len=500)` row
# subsampling here; the chunked path never holds more of the fill in
# memory than the (head, tail) scatter itself, same as the chunked Type-1
# diagnostic accumulator.

class BunchTrainDiagnosticAccumulator:
    """
    Accumulates (tail_value, bx_ratio) scatter points across chunks,
    matching `analyze_bunch_train_step`'s inner selection exactly:
      1) time_mask = avg > sbil_min (select rows/histograms),
      2) extract head = hists[:, head_idx], tail = hists[:, tail_idx]
         (and the same from the reference luminometer's histograms)
         for the fixed (tail, head) BX pairs,
      3) keep only head > sbil_min entries,
      4) bx_ratio = (tail_ref / head_ref) / (tail / head) - 1.

    `tail_idx` / `head_idx` come from `_select_bunch_train_pairs` and
    only depend on `active_mask`, so they're computed once per fill and
    reused across all chunks.
    """

    def __init__(self, tail_idx: np.ndarray, head_idx: np.ndarray, sbil_min: float):
        self.tail_idx = tail_idx
        self.head_idx = head_idx
        self.sbil_min = float(sbil_min)
        self._x_parts: List[np.ndarray] = []
        self._y_parts: List[np.ndarray] = []

    def add_chunk(
        self,
        bxraw_chunk: np.ndarray,
        bxraw_ref_chunk: np.ndarray,
        avg_chunk: np.ndarray,
        scale: float,
    ) -> None:
        bxraw_chunk = np.asarray(bxraw_chunk, dtype=np.float64)
        bxraw_ref_chunk = np.asarray(bxraw_ref_chunk, dtype=np.float64)
        avg_chunk = np.asarray(avg_chunk, dtype=np.float64)

        time_mask = avg_chunk > self.sbil_min
        if not np.any(time_mask):
            return
        if self.tail_idx.size == 0:
            return

        hists = bxraw_chunk[time_mask, :] * scale
        hists_ref = bxraw_ref_chunk[time_mask, :] * scale

        head = hists[:, self.head_idx]
        tail = hists[:, self.tail_idx]
        head_ref = hists_ref[:, self.head_idx]
        tail_ref = hists_ref[:, self.tail_idx]

        valid = head > self.sbil_min
        head = head[valid]
        tail = tail[valid]
        head_ref = head_ref[valid]
        tail_ref = tail_ref[valid]
        if head.size == 0:
            return

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = tail / head
            frac_ref = tail_ref / head_ref
            bx_ratio = frac_ref / frac - 1

        self._x_parts.append(tail.astype(np.float64))
        self._y_parts.append(bx_ratio.astype(np.float64))

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._x_parts:
            return np.array([]), np.array([])
        return np.concatenate(self._x_parts), np.concatenate(self._y_parts)


def analyze_bunch_train_finalize(
    x: np.ndarray,
    bx_ratio: np.ndarray,
    cfg,
    fill: int,
    order: int,
    tag: str = "before",
) -> None:
    """
    Shared plotting/saving core for the bunch-train diagnostic, extracted
    from the tail of `analyze_bunch_train_step` -- binning, poly fit,
    optional HDF5 point-cloud save, optional PNG. Operates on
    already-collected (x, bx_ratio) arrays instead of a full
    `data['bxraw']` / `data['bxraw_ref']`, so it works identically
    whether those arrays came from
    `BunchTrainDiagnosticAccumulator.finalize()` (chunked) or directly
    from the full in-memory arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    bx_ratio = np.asarray(bx_ratio, dtype=np.float64)

    make_plots: bool = bool(getattr(cfg.bunch_train, "make_plots", False))
    save_hd5: bool = bool(getattr(cfg.bunch_train, "save_hd5", False))

    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

    tag_suffix = f"_{tag}" if tag else ""

    if x.size == 0:
        log.warning(
            "[analyze_bunch_train_finalize] fill %d: no points collected",
            fill,
        )
        return

    # --- binning ---
    x_bins, y_bins, s_bins = _do_binning(x, bx_ratio, nbins=20)

    # --- fit ---
    coeffs = np.polyfit(x, bx_ratio, order)
    new_x = np.linspace(
        float(np.min(x)),
        float(np.max(x)),
        num=x.size,
    )
    new_line = np.polyval(coeffs, new_x)

    output_dir = os.path.join(type1_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    if save_hd5:
        hd5_dir = os.path.join(output_dir, "hd5")
        os.makedirs(hd5_dir, exist_ok=True)
        hd5_path = os.path.join(hd5_dir, f"bunch_train{tag_suffix}.h5")

        fill_arr_scatter = np.full(x.size, int(fill), dtype=np.int64)
        fill_arr_binned = np.full(x_bins.size, int(fill), dtype=np.int64)
        fill_arr_fit = np.full(new_x.size, int(fill), dtype=np.int64)

        with h5py.File(hd5_path, "a") as h5:
            h5.attrs["fill"] = int(fill)
            h5.attrs["order"] = int(order)
            h5.attrs["tag"] = str(tag)

            # --- scatter ---
            ds_fill = _h5_get_or_create(h5, "scatter/fill", dtype=np.int64)
            ds_x = _h5_get_or_create(h5, "scatter/x", dtype=np.float64)
            ds_y = _h5_get_or_create(h5, "scatter/y", dtype=np.float64)
            _h5_append_1d(ds_fill, fill_arr_scatter)
            _h5_append_1d(ds_x, x)
            _h5_append_1d(ds_y, bx_ratio)

            # --- binned ---
            db_fill = _h5_get_or_create(h5, "binned/fill", dtype=np.int64)
            db_x = _h5_get_or_create(h5, "binned/x", dtype=np.float64)
            db_y = _h5_get_or_create(h5, "binned/y", dtype=np.float64)
            db_ey = _h5_get_or_create(h5, "binned/yerr", dtype=np.float64)
            _h5_append_1d(db_fill, fill_arr_binned)
            _h5_append_1d(db_x, x_bins)
            _h5_append_1d(db_y, y_bins)
            _h5_append_1d(db_ey, s_bins)

            # --- fit curve ---
            df_fill = _h5_get_or_create(h5, "fit/fill", dtype=np.int64)
            df_x = _h5_get_or_create(h5, "fit/x", dtype=np.float64)
            df_y = _h5_get_or_create(h5, "fit/y", dtype=np.float64)
            _h5_append_1d(df_fill, fill_arr_fit)
            _h5_append_1d(df_x, new_x.astype(np.float64))
            _h5_append_1d(df_y, new_line.astype(np.float64))

            # --- poly coefficients ---
            pc_fill = _h5_get_or_create(h5, "poly/fill", dtype=np.int64)
            pc_deg = _h5_get_or_create(h5, "poly/order", dtype=np.int64)
            pc_coef = _h5_get_or_create(h5, "poly/coeffs", dtype=np.float64)
            _h5_append_1d(pc_fill, np.array([int(fill)], dtype=np.int64))
            _h5_append_1d(pc_deg, np.array([int(order)], dtype=np.int64))
            _h5_append_1d(pc_coef, np.asarray(coeffs, dtype=np.float64))

        log.info(
            "[analyze_bunch_train_finalize] fill %d tag=%s: saved debug HDF5 to %s",
            fill,
            tag,
            hd5_path,
        )

    # --- PNG (optional) ---
    if make_plots:
        fig = plt.figure(figsize=(7, 5))
        # scatter
        plt.plot(x, bx_ratio, ".", alpha=0.2, label=f"Bunch Train fraction to reference")
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
            "[analyze_bunch_train_finalize] fill %d tag=%s: saved PNG to %s",
            fill,
            tag,
            png_path,
        )


def analyze_bunch_train_fill_chunked(
    chunks_iter_factory,
    cfg,
    active_mask: np.ndarray,
    fill: int,
    tag: str = "before",
) -> None:
    """
    Chunked equivalent of `analyze_bunch_train_step`: streams over
    chunks ONCE, accumulating the (tail, bx_ratio) scatter via
    `BunchTrainDiagnosticAccumulator`, then finalizes via
    `analyze_bunch_train_finalize`. Mirrors
    `type1_fit.analyze_type1_fill_chunked`.

    `chunks_iter_factory` is a zero-arg callable that returns a fresh
    iterator of chunk dicts each time it's called (a plain generator
    would be exhausted after one use; the caller typically passes
    something like `lambda: iter_hd5_row_chunks(dirname, basename, ...)`).
    Each yielded chunk must already carry "bxraw_ref" (merged in during
    `_pass0_prepare` when `cfg.steps.bunch_train` is enabled).
    """
    sbil_min: float = float(getattr(cfg.bunch_train, "sbil_min", 0.1))
    order: int = int(getattr(cfg.bunch_train, "order", 1))
    scale = 11245.6 / float(cfg.afterglow.sigvis)

    tail_idx, head_idx = _select_bunch_train_pairs(active_mask)
    if tail_idx.size == 0:
        log.warning(
            "[analyze_bunch_train_fill_chunked] fill %d: no (head, tail) pairs found",
            fill,
        )
        return

    acc = BunchTrainDiagnosticAccumulator(tail_idx, head_idx, sbil_min)

    for chunk in chunks_iter_factory():
        bxraw_chunk = np.asarray(chunk["bxraw"], dtype=np.float64)
        bxraw_ref_chunk = np.asarray(chunk["bxraw_ref"], dtype=np.float64)
        avg_chunk = get_sbil_like_column(chunk, active_mask)
        acc.add_chunk(bxraw_chunk, bxraw_ref_chunk, avg_chunk, scale)

    x, bx_ratio = acc.finalize()
    analyze_bunch_train_finalize(x, bx_ratio, cfg, fill, order, tag=tag)


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
            valid = head > sbil_min #0.0
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
