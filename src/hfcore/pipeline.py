from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import logging
import h5py
import json
import re

from .decorators import log_step, timeit
from .io import (
    load_hd5_to_arrays,
    iter_hd5_row_chunks,
    Hd5ChunkWriter,
)
from .hd5schema import BX_LEN
from .afterglow_lsq import build_afterglow_solver_from_file, AfterglowSolver
from .type1_fit import (
    save_type1_coeffs,
    Type1OffsetAccumulator,
    accumulate_type1_offset_chunk,
    subtract_type1_offset_inplace,
    get_sbil_like_column,
    analyze_type1_fill_chunked,
)
from .type1_apply import apply_type1_batch
#from .online_recovery import reconstruct_from_tables_batch, reconstruct_from_online_batch
from .plotter import (
    BxProfileAccumulator,
    plot_hist_bx_from_profile,
    compute_scaled_active_sum_chunk,
    plot_lumi_comparison_from_series,
    build_residual_masks,
    compute_residual_row_averages,
    plot_residuals_finalize,
    compute_laser_columns_chunk,
    plot_lasers_from_series,
    LASER_BCID,
)

from .config import PipelineConfig


log = logging.getLogger("hfpipe")

DEFAULT_CHUNK_SIZE = 500


def _align_aux_by_keys(
    main: Dict[str, np.ndarray],
    aux: Dict[str, np.ndarray],
    colname: str,
) -> np.ndarray:

    keys = ("fillnum", "runnum", "lsnum", "nbnum")

    T = main[keys[0]].shape[0]
    main_key = np.stack([main[k].astype(np.int64) for k in keys], axis=1)  # (T, 4)

    aux_T = aux[keys[0]].shape[0]
    aux_key = np.stack([aux[k].astype(np.int64) for k in keys], axis=1)    # (aux_T, 4)

    index: Dict[tuple, int] = {}
    for j in range(aux_T):
        index[tuple(aux_key[j])] = j

    aux_col = aux[colname]
    tail_shape = aux_col.shape[1:]

    out = np.zeros((T,) + tail_shape, dtype=aux_col.dtype)

    missing = 0
    for i in range(T):
        key = tuple(main_key[i])
        j = index.get(key, None)
        if j is None:
            missing += 1
            continue
        out[i] = aux_col[j]

    if missing > 0:
        print(f"[WARN] _align_aux_by_keys: {missing} rows had no match in aux node")

    return out


def _subtract_fixed_pedestal_mod4_inplace(data: dict, ped4) -> None:
    """
    Subtract constant mod4 pedestal from bxraw only:
      bxraw[:, bx] -= ped4[bx % 4]

    Purely elementwise -- works on a full-fill dict or a single chunk.
    """
    if ped4 is None:
        return

    if "bxraw" not in data:
        raise KeyError("fixed_pedestal_4 is set but data has no 'bxraw'")

    ped4 = np.asarray(ped4, dtype=np.float32).ravel()
    if ped4.shape[0] != 4:
        raise ValueError(f"fixed_pedestal_4 must have length 4, got shape {ped4.shape}")

    bxraw = np.asarray(data["bxraw"])
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(f"bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})")

    ped_vec = ped4[np.arange(BX_LEN) % 4][None, :]  # (1, BX_LEN)
    data["bxraw"] = (bxraw - ped_vec).astype(bxraw.dtype, copy=False)


def _recompute_derived_from_bxraw_inplace(
    data: dict,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
) -> None:
    """
    Recompute ONLY derived columns from bxraw:
      - bx      = bxraw * scale
      - avgraw  = sum(bxraw over active BX)   (NOT mean)
      - avg     = avgraw * scale

    This must be the single source of truth for bx/avgraw/avg. Works
    equally on a full-fill dict or a single chunk.
    """
    if "bxraw" not in data:
        raise KeyError("_recompute_derived_from_bxraw_inplace: missing 'bxraw' in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float32)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"_recompute_derived_from_bxraw_inplace: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    mask = np.asarray(active_mask, dtype=np.int32).ravel()
    if mask.shape[0] != BX_LEN:
        raise ValueError(
            f"_recompute_derived_from_bxraw_inplace: active_mask len={mask.shape[0]} != BX_LEN={BX_LEN}"
        )

    sigvis = getattr(cfg.afterglow, "sigvis", None)
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    data["bx"] = (bxraw * scale).astype(np.float32, copy=False)

    avgraw = (bxraw * mask[None, :]).sum(axis=1)
    data["avgraw"] = avgraw.astype(np.float32, copy=False)

    data["avg"] = (avgraw * scale).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Helpers for Type-1 paths
# ---------------------------------------------------------------------------

def _get_type1_dir(cfg: PipelineConfig) -> str:
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")
    os.makedirs(type1_dir, exist_ok=True)
    return type1_dir


def _get_type1_coeff_path(cfg: PipelineConfig, fill: int) -> str:
    type1_dir = _get_type1_dir(cfg)
    return os.path.join(type1_dir, f"type1_coeffs_fill{fill}.h5")


# ---------------------------------------------------------------------------
# Step 0: recover origin rates (unchanged; not wired into the pipeline,
# same as it was commented out before)
# ---------------------------------------------------------------------------
def _load_hfsbr_for_online(cfg: PipelineConfig, fill: int) -> np.ndarray:
    pattern = cfg.online_recovery.hfsbr_pattern or cfg.afterglow.hfsbr_pattern
    if not pattern:
        raise ValueError(
            "No HFSBR pattern for online recovery "
            "(both online_recovery.hfsbr_pattern and afterglow.hfsbr_pattern are empty)."
        )

    path = pattern.format(fill=fill)
    if not os.path.exists(path):
        raise FileNotFoundError(f"HFSBR file for online recovery not found: {path}")

    if path.endswith(".npy"):
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float64).ravel()

    if path.endswith(".txt") or path.endswith(".dat"):
        with open(path, "r") as f:
            text = f.read()
        text = text.replace("[", " ").replace("]", " ")
        tokens = re.split(r"[,\s]+", text)
        values = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            try:
                values.append(float(tok))
            except ValueError:
                continue
        if not values:
            raise RuntimeError(f"HFSBR .txt file {path} did not contain any numeric tokens")
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.shape[0] < BX_LEN:
            raise RuntimeError(
                f"HFSBR from {path} has length {arr.shape[0]} < BX_LEN={BX_LEN}"
            )
        return arr

    if path.endswith(".h5") or path.endswith(".hd5"):
        with h5py.File(path, "r") as h5:
            if "hfsbr" in h5:
                return np.asarray(h5["hfsbr"][:], dtype=np.float64).ravel()
            for name, obj in h5.items():
                if hasattr(obj, "shape"):
                    return np.asarray(obj[...], dtype=np.float64).ravel()
        raise RuntimeError(f"Could not find HFSBR dataset in {path}")

    raise RuntimeError(
        f"Unknown HFSBR file format for path {path}. "
        f"Please adapt _load_hfsbr_for_online."
    )


def recover_bxraw_step(
    data: dict,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    fill: int,
    input_pattern: str,
) -> dict:
    bxraw_final = np.asarray(data["bxraw"], dtype=np.float32)
    rec_cfg = cfg.online_recovery

    use_tables = (rec_cfg.method == "tables")
    use_online = (rec_cfg.method == "online")

    states_tables = None
    states_online = None

    '''
    if use_tables:
        ped_node = rec_cfg.pedestal_node
        aft_node = rec_cfg.afterglow_node

        ped_data = load_hd5_to_arrays(cfg.io.input_dir, input_pattern, node=ped_node)
        aft_data = load_hd5_to_arrays(cfg.io.input_dir, input_pattern, node=aft_node)

        ped_4 = _align_aux_by_keys(main=data, aux=ped_data, colname="bxraw").astype(np.float32)
        aft_frac = _align_aux_by_keys(main=data, aux=aft_data, colname="bxraw").astype(np.float32)

        states_tables = reconstruct_from_tables_batch(
            bxraw_final=bxraw_final, pedestal_4=ped_4, afterglow_frac=aft_frac,
        )

    
    if use_online:
        hfsbr = _load_hfsbr_for_online(cfg, fill)
        states_online = reconstruct_from_online_batch(
            bxraw_final=bxraw_final, hfsbr=hfsbr, active_mask=active_mask,
            zero_bx=(3553, 3554, 3555, 3556, 3557), show_progress=True,
        )
    '''
    if use_tables:
        data["bxraw"] = states_tables.mu_before
    else:
        if states_online is None:
            raise RuntimeError("online_recovery: states_online is None, check config")
        data["bxraw"] = states_online.mu_before

    return data


def calculate_dynamic_pedestal(mu_hist: np.ndarray) -> np.ndarray:
    """
    Exact copy of CMS dynamic pedestal logic.

    We take the last 13*4 BXs (3500..3500+4*13-1 = 3500..3551),
    group them by HF subdetector (0..3), and return pedestal[4].
    """
    n_sample = 13
    pedestal = np.zeros(4, dtype=np.float32)
    base = 3500
    for ibx in range(4):
        s = 0.0
        for j in range(ibx, 4 * n_sample, 4):
            s += mu_hist[base + j]
        pedestal[ibx] = s / n_sample
    return pedestal


# ---------------------------------------------------------------------------
# Tiny generic row-series accumulator (O(T) memory, independent of BX_LEN)
# ---------------------------------------------------------------------------
class SeriesAccumulator:
    """
    Collects 1D per-row arrays across chunks and concatenates them once,
    at the end. Used for every plotting quantity that turned out to be a
    per-row scalar (mean SBIL, residual averages, laser BX values, ...)
    rather than needing the full (T, BX_LEN) bxraw.
    """

    def __init__(self) -> None:
        self._parts: List[np.ndarray] = []

    def add(self, arr: np.ndarray) -> None:
        self._parts.append(np.asarray(arr, dtype=np.float64))

    def finalize(self) -> np.ndarray:
        if not self._parts:
            return np.array([], dtype=np.float64)
        return np.concatenate(self._parts)


# ---------------------------------------------------------------------------
# Chunk-level pipeline steps
# ---------------------------------------------------------------------------

def restore_rates_chunk(
    chunk: dict,
    active_mask: np.ndarray,
    solver: AfterglowSolver,
    warm_state: dict,
    n_jobs: int = -1,
) -> dict:
    """
    Chunk-level equivalent of the LSQ-afterglow + dynamic-pedestal
    restoration.

    - use_warm_start=True: rows solved one at a time, in order, threading
      `warm_state['prev_tail_params']` across chunk boundaries as well
      as row boundaries -- reproduces AfterglowSolver.apply_batch's
      sequential warm-start chain at O(chunk) memory.
    - use_warm_start=False: rows are independent, solved in parallel via
      joblib, chunk by chunk, exactly like apply_batch's non-warm-start
      branch.

    Does not recompute bx/avg/avgraw. Returns a new dict (shallow copy).
    """
    bxraw_obs = np.asarray(chunk["bxraw"], dtype=np.float64)
    T = bxraw_obs.shape[0]
    mu_corr = np.empty((T, BX_LEN), dtype=np.float32)

    if solver.use_warm_start:
        prev_tail_params = warm_state.get("prev_tail_params", None)

        for i in range(T):
            x0 = prev_tail_params if prev_tail_params is not None else None
            mu_true, _ped = solver._solve_one(bxraw_obs[i], x0=x0)

            dbg = solver.last_tail_debug
            if dbg is not None and dbg.get("fit_ok", False):
                prev_tail_params = np.asarray(dbg["fit_params"], dtype=np.float64)
            else:
                prev_tail_params = None

            ped = calculate_dynamic_pedestal(mu_true)
            corr = mu_true - ped[np.arange(BX_LEN) % 4]
            mu_corr[i] = corr.astype(np.float32)

        warm_state["prev_tail_params"] = prev_tail_params
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=8)(
            delayed(solver._solve_one)(bxraw_obs[i], x0=None) for i in range(T)
        )
        for i, (mu_true, _ped) in enumerate(results):
            ped = calculate_dynamic_pedestal(mu_true)
            corr = mu_true - ped[np.arange(BX_LEN) % 4]
            mu_corr[i] = corr.astype(np.float32)

    out = dict(chunk)
    out["bxraw"] = mu_corr
    return out


def apply_type1_chunk(
    chunk: dict,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> dict:
    """
    Chunk-level equivalent of apply_type1_step: combined (all-offsets-
    at-once) Type-1 subtraction via `apply_type1_batch`, followed by
    derived-column recompute. Returns a new dict.
    """
    bxraw = np.asarray(chunk["bxraw"], dtype=np.float64)
    corrected = apply_type1_batch(
        bxraw=bxraw, active_mask=active_mask, p0=p0, p1=p1, p2=p2,
    ).astype(np.float32)

    out = dict(chunk)
    out["bxraw"] = corrected
    _recompute_derived_from_bxraw_inplace(out, cfg, active_mask)
    return out


# ---------------------------------------------------------------------------
# Pass 0: input -> (optional fixed pedestal + LSQ afterglow + dynamic
# pedestal + derived recompute) -> scratch_stage0, feeding all the exact
# streaming plot/diagnostic accumulators along the way.
# ---------------------------------------------------------------------------
def _pass0_prepare(
    fill: int,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    input_name: str,
    node: str,
    chunk_size: int,
    scratch_stage0: str,
    scale: float,
    active_bool: np.ndarray,
    type1_mask: np.ndarray,
    type2_mask: np.ndarray,
    raw_profile_acc: Optional[BxProfileAccumulator],
    t2_profile_acc: Optional[BxProfileAccumulator],
    t2_residual_accs: Optional[tuple],
    raw_lumi_sum_acc: Optional[SeriesAccumulator],
    raw_laser_accs: Optional[dict],
) -> int:
    solver: Optional[AfterglowSolver] = None
    ped4 = None
    warm_state: dict = {}

    if cfg.steps.restore_rates:
        hfsbr_path = cfg.afterglow.hfsbr_pattern.format(fill=fill)
        if not os.path.exists(hfsbr_path):
            raise FileNotFoundError(f"HFSBR file not found: {hfsbr_path}")

        solver = build_afterglow_solver_from_file(
            hfsbr_path=hfsbr_path,
            active_mask=active_mask,
            bx_to_clean=cfg.afterglow.bx_to_clean or [],
            p0_guess=None,
            lambda_reg=cfg.afterglow.lambda_reg,
            lambda_nonactive=cfg.afterglow.lambda_nonactive,
        )
        ped4 = getattr(cfg.afterglow, "fixed_pedestal_4", None)

    n_rows_total = 0

    with Hd5ChunkWriter(scratch_stage0, node=node) as writer:
        for chunk in iter_hd5_row_chunks(
            cfg.io.input_dir, input_name, node=node,
            chunk_size=chunk_size, fill_filter=fill,
        ):
            raw_bxraw = np.asarray(chunk["bxraw"], dtype=np.float64)

            if raw_profile_acc is not None:
                raw_profile_acc.add_chunk(raw_bxraw)
            if raw_lumi_sum_acc is not None:
                raw_lumi_sum_acc.add(compute_scaled_active_sum_chunk(raw_bxraw, active_bool, scale))
            if raw_laser_accs is not None:
                laser_vals = compute_laser_columns_chunk(raw_bxraw, scale)
                for bcid, vals in laser_vals.items():
                    raw_laser_accs[bcid].add(vals)

            if cfg.steps.restore_rates:
                if ped4 is not None:
                    _subtract_fixed_pedestal_mod4_inplace(chunk, ped4)

                chunk = restore_rates_chunk(
                    chunk, active_mask, solver, warm_state,
                    n_jobs=cfg.afterglow.n_jobs,
                )
                _recompute_derived_from_bxraw_inplace(chunk, cfg, active_mask)

            t2_bxraw = np.asarray(chunk["bxraw"], dtype=np.float64)
            if t2_profile_acc is not None:
                t2_profile_acc.add_chunk(t2_bxraw)
            if t2_residual_accs is not None:
                avg_col_acc, avg_type1_acc, avg_type2_acc = t2_residual_accs
                avg_col, avg_type1, avg_type2 = compute_residual_row_averages(
                    t2_bxraw, active_bool, type1_mask, type2_mask, scale,
                )
                avg_col_acc.add(avg_col)
                avg_type1_acc.add(avg_type1)
                avg_type2_acc.add(avg_type2)

            writer.write_chunk(chunk)
            n_rows_total += next(iter(chunk.values())).shape[0]

    return n_rows_total


# ---------------------------------------------------------------------------
# Type-1 coefficient fit: sequential per-offset, via ping-pong scratch
# files, reproducing compute_type1_coeffs' exact semantics without ever
# holding the whole fill's bxraw in memory.
# ---------------------------------------------------------------------------
def compute_type1_fill(
    fill: int,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    node: str,
    chunk_size: int,
    scratch_dir: str,
    scratch_stage0: str,
) -> str:
    """
    Fits Type-1 coefficients for the whole fill by streaming over
    `scratch_stage0` in multiple passes -- one pair of passes per
    offset, largest offset first, exactly matching the sequential
    "fit, then subtract estimated contribution before fitting the next
    (smaller) offset" semantics of the in-memory `compute_type1_coeffs`.

    The scratch files created here (`fit_a.h5` / `fit_b.h5`) only ever
    hold *fit-time* mutated bxraw, used purely to accumulate fit
    statistics -- they are discarded at the end and never affect
    `scratch_stage0`.

    Returns the path to the saved coefficient file.
    """
    offsets: List[int] = list(getattr(cfg.type1, "offsets", [1, 2, 3, 4]))
    sbil_min = float(getattr(cfg.type1, "sbil_min", 0.1))
    max_offset = max(offsets) if offsets else 0

    p0 = np.zeros(max_offset + 1, dtype=np.float64)
    p1 = np.zeros(max_offset + 1, dtype=np.float64)
    p2 = np.zeros(max_offset + 1, dtype=np.float64)
    orders = np.zeros(max_offset + 1, dtype=np.int32)

    fit_scratch_a = os.path.join(scratch_dir, "fit_a.h5")
    fit_scratch_b = os.path.join(scratch_dir, "fit_b.h5")

    cur_path = scratch_stage0
    toggle_paths = [fit_scratch_a, fit_scratch_b]
    toggle_idx = 0

    for off in reversed(offsets):
        if off <= 0:
            continue
        order = 2 if off == 1 else 1

        acc = Type1OffsetAccumulator()
        for chunk in iter_hd5_row_chunks(
            os.path.dirname(cur_path), os.path.basename(cur_path),
            node=node, chunk_size=chunk_size,
        ):
            bxraw_chunk = np.asarray(chunk["bxraw"], dtype=np.float64)
            avg_chunk = get_sbil_like_column(chunk, active_mask)
            accumulate_type1_offset_chunk(acc, bxraw_chunk, avg_chunk, active_mask, off, sbil_min)

        c0, c1, c2 = acc.finalize(order)
        p0[off], p1[off], p2[off], orders[off] = c0, c1, c2, order

        next_path = toggle_paths[toggle_idx % 2]
        toggle_idx += 1

        with Hd5ChunkWriter(next_path, node=node) as writer:
            for chunk in iter_hd5_row_chunks(
                os.path.dirname(cur_path), os.path.basename(cur_path),
                node=node, chunk_size=chunk_size,
            ):
                bxraw_chunk = np.asarray(chunk["bxraw"], dtype=np.float64).copy()
                subtract_type1_offset_inplace(bxraw_chunk, active_mask, off, c0, c1, c2)
                chunk = dict(chunk)
                chunk["bxraw"] = bxraw_chunk.astype(np.float32)
                writer.write_chunk(chunk)

        cur_path = next_path

    type1_dir = _get_type1_dir(cfg)
    coeff_path = save_type1_coeffs(
        fill=fill, output_dir=type1_dir,
        p0=p0, p1=p1, p2=p2, offsets=offsets, orders=orders,
    )
    log.info(
        "[compute_type1_fill] fill %d: Type-1 coeffs saved to %s (offsets=%s)",
        fill, coeff_path, offsets,
    )

    for p in (fit_scratch_a, fit_scratch_b):
        if os.path.exists(p):
            os.remove(p)

    return coeff_path


# ---------------------------------------------------------------------------
# Final pass: apply Type-1 (if enabled) on scratch_stage0, recompute
# derived, feed final-stage accumulators, write final output.
# ---------------------------------------------------------------------------
def _pass_final_apply_and_write(
    fill: int,
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    node: str,
    chunk_size: int,
    scratch_stage0: str,
    output_name: str,
    scale: float,
    active_bool: np.ndarray,
    type1_mask: np.ndarray,
    type2_mask: np.ndarray,
    final_profile_acc: Optional[BxProfileAccumulator],
    final_residual_accs: Optional[tuple],
    final_lumi_sum_acc: Optional[SeriesAccumulator],
    final_laser_accs: Optional[dict],
) -> None:
    output_full_path = os.path.join(cfg.io.output_dir, output_name)

    p0 = p1 = p2 = None
    if cfg.steps.apply_type1:
        coeff_path = _get_type1_coeff_path(cfg, fill)
        if not os.path.exists(coeff_path):
            raise FileNotFoundError(f"apply_type1: Type-1 coeff file not found: {coeff_path}")
        with h5py.File(coeff_path, "r") as h5:
            p0 = h5["p0"][:]
            p1 = h5["p1"][:]
            p2 = h5["p2"][:]

    with Hd5ChunkWriter(output_full_path, node=node) as writer:
        for chunk in iter_hd5_row_chunks(
            os.path.dirname(scratch_stage0), os.path.basename(scratch_stage0),
            node=node, chunk_size=chunk_size,
        ):
            if cfg.steps.apply_type1:
                chunk = apply_type1_chunk(chunk, cfg, active_mask, p0, p1, p2)
            else:
                # "Recompute rates for safety" -- run_fill always did this
                # unconditionally after the enabled-steps block, regardless
                # of which steps actually ran. Preserve that here.
                chunk = dict(chunk)
                _recompute_derived_from_bxraw_inplace(chunk, cfg, active_mask)

            final_bxraw = np.asarray(chunk["bxraw"], dtype=np.float64)

            if final_profile_acc is not None:
                final_profile_acc.add_chunk(final_bxraw)
            if final_residual_accs is not None:
                avg_col_acc, avg_type1_acc, avg_type2_acc = final_residual_accs
                avg_col, avg_type1, avg_type2 = compute_residual_row_averages(
                    final_bxraw, active_bool, type1_mask, type2_mask, scale,
                )
                avg_col_acc.add(avg_col)
                avg_type1_acc.add(avg_type1)
                avg_type2_acc.add(avg_type2)
            if final_lumi_sum_acc is not None:
                final_lumi_sum_acc.add(compute_scaled_active_sum_chunk(final_bxraw, active_bool, scale))
            if final_laser_accs is not None:
                laser_vals = compute_laser_columns_chunk(final_bxraw, scale)
                for bcid, vals in laser_vals.items():
                    final_laser_accs[bcid].add(vals)

            writer.write_chunk(chunk)


# ---------------------------------------------------------------------------
# Main entry point for a single fill
# ---------------------------------------------------------------------------

@log_step("run_fill")
@timeit("run_fill")
def run_fill(
    fill: int,
    cfg: PipelineConfig,
    chunk_size: Optional[int] = None,
    keep_scratch_on_error: bool = False,
) -> None:
    """
    Full pipeline for a single fill, processed in bounded-memory chunks:

      - stream input HDF5 (rows filtered by fillnum == fill as read),
      - load active BX mask,
      - pass 0: optional fixed-mod4-pedestal + LSQ afterglow + dynamic
        pedestal + derived recompute, streamed to a scratch file,
      - Type-1 fit (if enabled): sequential per-offset fit via ping-pong
        scratch passes, exactly matching the original offset-by-offset
        semantics,
      - final pass: Type-1 apply (if enabled) + derived recompute,
        streamed to the output file.

    Plotting and diagnostics (plot_hist_bx, plot_residuals,
    analyze_type1_step, and optionally plot_lumi_comparison /
    plot_lasers) are all fed from EXACT streaming accumulators computed
    along the way (per-BX mean profile, per-row SBIL/residual/laser
    series) -- these are the only quantities those plots ever actually
    needed, so nothing here is a sampled approximation; memory cost for
    all of them is O(T) or O(BX_LEN), never O(T * BX_LEN).

    Config flags (all default to the previous behaviour):
      - cfg.type1.make_plots           -> plot_hist_bx + plot_residuals
      - cfg.type1.debug                -> analyze_type1_step, tag="before"
      - cfg.type1.debug_after_apply    -> analyze_type1_step, tag="after"
      - cfg.type1.make_lumi_comparison_plot -> plot_lumi_comparison (off by
        default, same as it was commented out in the original pipeline)
      - cfg.type1.make_laser_plots     -> plot_lasers (off by default,
        same reason)

    `chunk_size` defaults to `cfg.io.chunk_size` if set, else
    `DEFAULT_CHUNK_SIZE`.
    """
    if chunk_size is None:
        chunk_size = getattr(cfg.io, "chunk_size", DEFAULT_CHUNK_SIZE)

    input_name = cfg.io.input_pattern.format(fill=fill)
    output_name = cfg.io.output_pattern.format(fill=fill)
    node = cfg.io.node

    # --- active BX mask from JSON file ---
    if not cfg.io.active_mask_pattern:
        raise ValueError("io.active_mask_pattern is not set in config")

    mask_path = cfg.io.active_mask_pattern.format(fill=fill)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Active BX mask not found: {mask_path}")

    with open(mask_path, "r") as f:
        active_mask = json.load(f)
    active_mask = np.asarray(active_mask, dtype=np.int32)

    if active_mask.ndim != 1:
        raise ValueError(
            f"Active BX mask must be a one-dimensional JSON list, got shape {active_mask.shape}"
        )
    if active_mask.shape[0] != BX_LEN:
        raise ValueError(f"active_mask len={active_mask.shape[0]} != BX_LEN={BX_LEN}")
    if not np.all((active_mask == 0) | (active_mask == 1)):
        bad_values = np.unique(active_mask[(active_mask != 0) & (active_mask != 1)])
        raise ValueError(f"Active BX mask must contain only 0 and 1, found: {bad_values.tolist()}")

    # --- flags ---
    make_plots = bool(getattr(cfg.type1, "make_plots", False))
    debug_before = bool(getattr(cfg.type1, "debug", False))
    debug_after = bool(getattr(cfg.type1, "debug_after_apply", False))
    make_lumi_comparison_plot = bool(getattr(cfg.type1, "make_lumi_comparison_plot", False))
    make_laser_plots = bool(getattr(cfg.type1, "make_laser_plots", False))

    need_t2_residuals = make_plots or debug_before
    need_final_residuals = make_plots or debug_after
    need_lumi_extras = make_lumi_comparison_plot or make_laser_plots

    scale = 11245.6 / float(cfg.afterglow.sigvis)
    active_bool, type1_mask, type2_mask = build_residual_masks(active_mask, cfg.afterglow.bx_to_clean)

    # --- accumulators (created lazily, only if actually needed) ---
    raw_profile_acc = BxProfileAccumulator() if make_plots else None
    t2_profile_acc = BxProfileAccumulator() if make_plots else None
    final_profile_acc = BxProfileAccumulator() if make_plots else None

    t2_residual_accs = (SeriesAccumulator(), SeriesAccumulator(), SeriesAccumulator()) if need_t2_residuals else None
    final_residual_accs = (SeriesAccumulator(), SeriesAccumulator(), SeriesAccumulator()) if need_final_residuals else None

    raw_lumi_sum_acc = SeriesAccumulator() if need_lumi_extras else None
    final_lumi_sum_acc = SeriesAccumulator() if need_lumi_extras else None

    raw_laser_accs = {bcid: SeriesAccumulator() for bcid in LASER_BCID} if make_laser_plots else None
    final_laser_accs = {bcid: SeriesAccumulator() for bcid in LASER_BCID} if make_laser_plots else None

    scratch_dir = os.path.join(cfg.io.output_dir, ".scratch", str(fill))
    os.makedirs(scratch_dir, exist_ok=True)
    scratch_stage0 = os.path.join(scratch_dir, "stage0.h5")

    try:
        n_rows = _pass0_prepare(
            fill=fill, cfg=cfg, active_mask=active_mask,
            input_name=input_name, node=node, chunk_size=chunk_size,
            scratch_stage0=scratch_stage0,
            scale=scale, active_bool=active_bool, type1_mask=type1_mask, type2_mask=type2_mask,
            raw_profile_acc=raw_profile_acc,
            t2_profile_acc=t2_profile_acc,
            t2_residual_accs=t2_residual_accs,
            raw_lumi_sum_acc=raw_lumi_sum_acc,
            raw_laser_accs=raw_laser_accs,
        )

        if n_rows == 0:
            log.warning(
                "[run_fill] fill %d: no rows with fillnum=%d in input, skipping", fill, fill,
            )
            return

        if make_plots:
            plot_hist_bx_from_profile(raw_profile_acc.mean_profile(), cfg, fill, 'Uncorr. Luminosity')
            plot_hist_bx_from_profile(t2_profile_acc.mean_profile(), cfg, fill, 'T2 Corr. Luminosity')

        if need_t2_residuals:
            avg_col, avg_type1, avg_type2 = (a.finalize() for a in t2_residual_accs)
            if make_plots:
                plot_residuals_finalize(
                    avg_col, avg_type1, avg_type2, cfg, fill, 't2_corr',
                    n_col=int(active_bool.sum()), n_type1=int(type1_mask.sum()), n_type2=int(type2_mask.sum()),
                )
            if debug_before:
                analyze_type1_fill_chunked(
                    lambda: iter_hd5_row_chunks(
                        os.path.dirname(scratch_stage0), os.path.basename(scratch_stage0),
                        node=node, chunk_size=chunk_size,
                    ),
                    cfg, active_mask, fill, tag="before",
                )

        if cfg.steps.compute_type1:
            compute_type1_fill(
                fill=fill, cfg=cfg, active_mask=active_mask,
                node=node, chunk_size=chunk_size,
                scratch_dir=scratch_dir, scratch_stage0=scratch_stage0,
            )

        _pass_final_apply_and_write(
            fill=fill, cfg=cfg, active_mask=active_mask,
            node=node, chunk_size=chunk_size,
            scratch_stage0=scratch_stage0, output_name=output_name,
            scale=scale, active_bool=active_bool, type1_mask=type1_mask, type2_mask=type2_mask,
            final_profile_acc=final_profile_acc,
            final_residual_accs=final_residual_accs,
            final_lumi_sum_acc=final_lumi_sum_acc,
            final_laser_accs=final_laser_accs,
        )

        if need_final_residuals:
            avg_col_f, avg_type1_f, avg_type2_f = (a.finalize() for a in final_residual_accs)
            if make_plots:
                plot_residuals_finalize(
                    avg_col_f, avg_type1_f, avg_type2_f, cfg, fill, 'full_corr',
                    n_col=int(active_bool.sum()), n_type1=int(type1_mask.sum()), n_type2=int(type2_mask.sum()),
                )
            if debug_after:
                output_full_path = os.path.join(cfg.io.output_dir, output_name)
                analyze_type1_fill_chunked(
                    lambda: iter_hd5_row_chunks(
                        os.path.dirname(output_full_path), os.path.basename(output_full_path),
                        node=node, chunk_size=chunk_size,
                    ),
                    cfg, active_mask, fill, tag="after",
                )

        if make_plots:
            plot_hist_bx_from_profile(final_profile_acc.mean_profile(), cfg, fill, 'T2 and T1 Corr. Luminosity')

        if make_lumi_comparison_plot:
            plot_lumi_comparison_from_series(
                final_lumi_sum_acc.finalize(), raw_lumi_sum_acc.finalize(), cfg, fill,
            )

        if make_laser_plots:
            n_active = int(active_bool.sum())
            avg_final = final_lumi_sum_acc.finalize() / n_active
            avg_raw = raw_lumi_sum_acc.finalize() / n_active
            laser_corr = {bcid: acc.finalize() for bcid, acc in final_laser_accs.items()}
            laser_uncorr = {bcid: acc.finalize() for bcid, acc in raw_laser_accs.items()}
            plot_lasers_from_series(avg_final, avg_raw, laser_corr, laser_uncorr, cfg, fill)

    finally:
        if not keep_scratch_on_error:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    log.info("[run_fill] fill %d: done -> %s", fill, os.path.join(cfg.io.output_dir, output_name))


def run_many_fills(cfg: PipelineConfig, fills: list[int], chunk_size: Optional[int] = None):
    """
    Run run_fill() for each fill.

    Any exception inside run_fill is treated as non-fatal:
      - the fill is added to the failed_fills list,
      - processing continues for the remaining fills.

    At the end, a summary of failed / skipped fills is printed.
    """
    failed: List[int] = []

    for fill in fills:
        try:
            run_fill(fill, cfg, chunk_size=chunk_size)

        except FileNotFoundError as e:
            print(f"[WARN] Fill {fill} skipped: {e}")
            failed.append(fill)

        except Exception as e:
            print(f"[ERROR] Fill {fill} failed with exception:")
            print(e)
            failed.append(fill)

    if failed:
        print("\n====================================")
        print("   ⚠ Some fills FAILED or SKIPPED")
        print("====================================")
        print("Failed fills:")
        for f in failed:
            print(f" - {f}")
        print("====================================\n")
    else:
        print("\nAll fills processed successfully.\n")