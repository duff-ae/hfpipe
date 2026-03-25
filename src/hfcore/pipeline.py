from __future__ import annotations

import os
from typing import List

import numpy as np
import logging
import h5py
import copy
import math
import traceback
from tqdm import tqdm
from joblib import Parallel, delayed

from .decorators import log_step, timeit
from .io import load_hd5_to_arrays, arrays_to_rows, save_to_hd5
from .hd5schema import BX_LEN
from .afterglow_lsq import build_afterglow_solver_from_file
from .type1_fit import compute_type1_coeffs, save_type1_coeffs, analyze_type1_step
from .type1_apply import apply_type1_batch
from .bunch_train import compute_bunch_train_coeffs, save_bunch_train_coeffs, analyze_bunch_train_step, apply_bunch_train_batch
from .plotter import plot_hist_bx, plot_lumi_comparison, plot_residuals, plot_lasers

from .config import PipelineConfig


log = logging.getLogger("hfpipe")


# ---------------------------------------------------------------------------
# Helpers for Type-1 paths
# ---------------------------------------------------------------------------

def _get_type1_dir(cfg: PipelineConfig) -> str:
    """
    Return the directory where Type-1 coefficient files are stored.

    Priority:
      1) cfg.io.type1_dir if set;
      2) <output_dir>/type1 as a default.

    Ensures that the directory exists.
    """
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")
    os.makedirs(type1_dir, exist_ok=True)
    return type1_dir


def _get_type1_coeff_path(cfg: PipelineConfig, fill: int) -> str:
    """
    Return the full path to the Type-1 coefficients file for a given fill.
    """
    type1_dir = _get_type1_dir(cfg)
    return os.path.join(type1_dir, f"type1_coeffs_fill{fill}.h5")


# ---------------------------------------------------------------------------
# Helpers for merging tables
# ---------------------------------------------------------------------------

# for double checking if a column is all unique values
# sanity check used before merging two tables on a column
def _is_unique_columns(cols):
    stacked = np.column_stack(cols)
    return np.unique(stacked, axis=0).shape[0] == stacked.shape[0]

# for merging one column from a dictionary into another
# useful for getting the pedestal information
def _inner_merge_one_column(left, right, on, right_col, new_col_name):
   # Build structured arrays for keys
    left_keys = np.core.records.fromarrays([left[c] for c in on], names=",".join(on))
    right_keys = np.core.records.fromarrays([right[c] for c in on], names=",".join(on))

    # Sort right keys for fast lookup
    order = np.argsort(right_keys)
    right_keys_sorted = right_keys[order]
    right_vals_sorted = right[right_col][order]

    # Find matches
    idx = np.searchsorted(right_keys_sorted, left_keys)
    valid = idx < right_keys_sorted.size
    mask = np.zeros_like(valid, dtype=bool)
    mask[valid] = right_keys_sorted[idx[valid]] == left_keys[valid]

    # Build merged dict
    merged = {}

    for k, v in left.items():
        merged[k] = v[mask]

    merged[new_col_name] = right_vals_sorted[idx[mask]]

    return merged


# ---------------------------------------------------------------------------
# Step 0: revert the online corrections
# ---------------------------------------------------------------------------
def revert_pedestal(mu_batch, pedestal):
    mu = mu_batch.copy()
    for i in range(4):
        mu[:, i::4] += pedestal[:,i][:, None]
    return mu

def revert_afterglow(mu_batch, HFSBR, linear, quad, active_mask):
    mu = mu_batch.copy()
    B, N = mu.shape
    
    for ibx in range(N - 1, -1, -1):
        if not active_mask[ibx]:
            continue

        base = mu[:, ibx]
        base2 = base * base

        type_ = 0
        for d in range(1, N):
            idx = (ibx + d) % N

            if type_ < 3:
                SBR = base2 * quad[type_] + base * linear[type_] + HFSBR[d]
            else:
                SBR = HFSBR[d]

            mu[:, idx] += base * SBR

            type_ += 1

    return mu

@log_step("revert_online")
@timeit("revert_online")
def revert_online_corrections(data: dict, cfg: PipelineConfig, active_mask: np.ndarray) -> dict:
    fills = np.unique(data["fillnum"])
    if fills.size != 1:
        raise ValueError(f"Expected exactly one fill in file, got {fills}")
    fill = int(fills[0])
    
    # --- load HFSBR matrix ---
    hfsbr_path = cfg.online.hfsbr.format(fill=fill)
    if not os.path.exists(hfsbr_path):
        raise FileNotFoundError(f"HFSBR file not found: {hfsbr_path}")

    HFSBR = np.loadtxt(hfsbr_path, dtype=np.float64, delimiter=",")
    HFSBR = HFSBR.astype(np.float64, copy=False)
    N     = HFSBR.shape[0]

    # --- online type1 corrections ---
    lin_type1 = cfg.online.linear_type1
    quad_type1 = cfg.online.quad_type1

    if N != BX_LEN:
        raise ValueError(f"HFSBR len={N} does not match BX_LEN={BX_LEN}")

    active_mask = np.asarray(active_mask, dtype=np.int32)
    if active_mask.shape[0] != N:
        raise ValueError("active_mask must match HFSBR length")
    
    bxraw_obs = data["bxraw"]
    assert bxraw_obs.shape[1] == BX_LEN

    # first revert pedestal
    if "pedestal" in data.keys():
        bxraw_obs = revert_pedestal(bxraw_obs, data["pedestal"])

    # Split T into batches
    n_jobs = cfg.afterglow.n_jobs
    T = bxraw_obs.shape[0]
    batch_size = min(T, max(10, math.ceil(T / (4 * n_jobs))))
    batches = [bxraw_obs[i : i + batch_size] for i in range(0, bxraw_obs.shape[0], batch_size)]

    # Parallel processing
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(revert_afterglow)(batch, HFSBR, lin_type1, quad_type1, active_mask)
        for batch in tqdm(batches, desc="Reverting Online Afterglow", unit="batch")
    )

    # Stack results back
    data["bxraw"] = np.vstack(results)

    # ------------------------------------------------------------------
    # Recompute bx and avg
    # ------------------------------------------------------------------
    sigvis = cfg.afterglow.sigvis
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    bx_lumi = (data["bxraw"] * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    return data

# ---------------------------------------------------------------------------
# Step 1: afterglow / recovery of mu_true
# ---------------------------------------------------------------------------

def calculate_dynamic_pedestal(mu_hist: np.ndarray) -> np.ndarray:
    """
    Exact copy of CMS dynamic pedestal logic.

    We take the last 13*4 BXs (3500..3500+4*13-1 = 3500..3551),
    group them by HF subdetector (0..3),
    and return pedestal[4].
    """
    n_sample = 13
    pedestal = np.zeros(4, dtype=np.float32)

    # last 52 BX (3500..3551)
    base = 3500
    for ibx in range(4):
        s = 0.0
        for j in range(ibx, 4 * n_sample, 4):
            s += mu_hist[base + j]
        pedestal[ibx] = s / n_sample
    return pedestal


@log_step("restore_rates")
@timeit("restore_rates")
def restore_rates_step(data: dict, cfg: PipelineConfig, active_mask: np.ndarray) -> dict:
    """
    Step 1: restore true rates (mu_true) via LSQ,
    then subtract the dynamic pedestal (CMS logic),
    then recompute bx and avg.

    Result:
      - data["bxraw"] contains mu_true after dynamic pedestal subtraction
      - data["bx"] / data["avg"] contain lumi after scaling by sigvis
    """
    fills = np.unique(data["fillnum"])
    if fills.size != 1:
        raise ValueError(f"Expected exactly one fill in file, got {fills}")
    fill = int(fills[0])

    # --- load HFSBR matrix ---
    hfsbr_path = cfg.afterglow.hfsbr_pattern.format(fill=fill)
    if not os.path.exists(hfsbr_path):
        raise FileNotFoundError(f"HFSBR file not found: {hfsbr_path}")

    bx_to_clean = cfg.afterglow.bx_to_clean or []
    lambda_reg = cfg.afterglow.lambda_reg
    lambda_nonactive = cfg.afterglow.lambda_nonactive
    n_jobs = cfg.afterglow.n_jobs

    # --- build LSQ solver ---
    solver = build_afterglow_solver_from_file(
        hfsbr_path=hfsbr_path,
        active_mask=active_mask,
        bx_to_clean=bx_to_clean,
        p0_guess=None,
        lambda_reg=lambda_reg,
        lambda_nonactive=lambda_nonactive,
    )

    bxraw_obs = data["bxraw"]
    assert bxraw_obs.shape[1] == BX_LEN

    # --- main LSQ pass ---
    mu_true, ped_lsq = solver.apply_batch(
        bxraw_obs,
        n_jobs=n_jobs,
        desc=f"LSQ afterglow (fill {fill})",
    )

    # ------------------------------------------------------------------
    # Dynamic pedestal subtraction (CMS logic)
    # ------------------------------------------------------------------
    mu_corr = np.empty_like(mu_true, dtype=np.float32)

    for i in range(mu_true.shape[0]):
        hist = mu_true[i]
        ped = calculate_dynamic_pedestal(hist)

        # subtract pedestal for BX 0..3551 using HF scheme (subdet = bx % 4)
        corr = hist - ped[np.arange(BX_LEN) % 4]
        mu_corr[i] = corr.astype(np.float32)

    # --- update bxraw with mu_true after pedestal subtraction ---
    data["bxraw"] = mu_corr

    # ------------------------------------------------------------------
    # Recompute bx and avg (these should be AFTER pedestal subtraction)
    # ------------------------------------------------------------------
    sigvis = cfg.afterglow.sigvis
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    bx_lumi = (mu_corr * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    return data


# ---------------------------------------------------------------------------
# Step 2: fit Type-1
# ---------------------------------------------------------------------------

@log_step("compute_type1_step")
@timeit("compute_type1_step")
def compute_type1_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Pipeline step: estimate Type-1 coefficients for a given fill.

    - Uses bxraw (already restored from afterglow & pedestal).
    - As "avg" (SBIL) it takes:
        * data["sbil"] if present;
        * else data["avg"] if present;
        * else quickly computes sbil = sum(bxraw * mask) / Nactive.
    - For each offset in cfg.type1.offsets:
        * offset == 1 -> quadratic fit (order=2);
        * offset >  1 -> linear fit (order=1).
    - Saves p0, p1, p2, offsets, orders into an HDF5 file.
    """
    if "bxraw" not in data:
        raise KeyError("compute_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"compute_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # --- SBIL / avg for Type-1 fit ---
    if "sbil" in data:
        avg = np.asarray(data["sbil"], dtype=np.float64)
    elif "avg" in data:
        avg = np.asarray(data["avg"], dtype=np.float64)
    else:
        # fallback: compute SBIL from bxraw and active mask
        mask = np.asarray(active_mask, dtype=np.int32)
        n_active = int(mask.sum())
        if n_active == 0:
            raise ValueError("compute_type1_step: active_mask has zero active BX")
        avg = (bxraw * mask[None, :]).sum(axis=1) / float(n_active)

    offsets = list(getattr(cfg.type1, "offsets", [1, 2, 3, 4]))
    sbil_min = float(getattr(cfg.type1, "sbil_min", 0.1))

    # --- compute Type-1 coefficients ---
    p0, p1, p2, orders = compute_type1_coeffs(
        bxraw=bxraw,
        avg=avg,
        active_mask=active_mask,
        offsets=offsets,
        sbil_min=sbil_min,
    )

    # --- where to save ---
    type1_dir = _get_type1_dir(cfg)

    path = save_type1_coeffs(
        fill=fill,
        output_dir=type1_dir,
        p0=p0,
        p1=p1,
        p2=p2,
        offsets=offsets,
        orders=orders,
    )

    log.info(
        "[compute_type1_step] fill %d: Type-1 coeffs saved to %s (offsets=%s)",
        fill,
        path,
        offsets,
    )

    # --- optional debug / analysis block ---
    # This is the "before Type-1 subtraction" diagnostic.
    if getattr(cfg.type1, "debug", False):
        analyze_type1_step(
            data=data,
            cfg=cfg,
            active_mask=active_mask,
            fill=fill,
            tag="before"
        )

    return data


# ---------------------------------------------------------------------------
# Step 3: apply Type-1
# ---------------------------------------------------------------------------

@log_step("apply_type1_step")
@timeit("apply_type1_step")
def apply_type1_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Pipeline step: apply Type-1 subtraction to bxraw.

    - Reads coefficients from type1_coeffs_fill{fill}.h5.
    - Calls apply_type1_batch(bxraw, active_mask, p0, p1, p2).
    - Updates:
        * data["bxraw"] (mu_true after Type-1 subtraction),
        * data["bx"] / data["avg"] (lumi after Type-1 subtraction).
    - Optionally runs analyze_type1_step again if cfg.type1.debug_after_apply is True.
    """
    if "bxraw" not in data:
        raise KeyError("apply_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"apply_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # --- where to find Type-1 coefficients ---
    coeff_path = _get_type1_coeff_path(cfg, fill)
    if not os.path.exists(coeff_path):
        raise FileNotFoundError(
            f"apply_type1_step: Type-1 coeff file not found: {coeff_path}"
        )

    # --- read coefficients ---
    with h5py.File(coeff_path, "r") as h5:
        p0 = h5["p0"][:]
        p1 = h5["p1"][:]
        p2 = h5["p2"][:]
        # offsets and orders are kept for logging / sanity checks
        offsets = h5["offsets"][:]
        orders = h5["orders"][:]

    log.info(
        "[apply_type1_step] fill %d: loaded Type-1 coeffs from %s (offsets=%s, orders=%s)",
        fill,
        coeff_path,
        list(offsets),
        list(orders),
    )

    # --- apply Type-1 subtraction in mu-space ---
    corrected_mu = apply_type1_batch(
        bxraw=bxraw,
        active_mask=active_mask,
        p0=p0,
        p1=p1,
        p2=p2,
    ).astype(np.float32)

    data["bxraw"] = corrected_mu

    # --- recompute bx and avg AFTER Type-1 subtraction ---
    sigvis = getattr(cfg.afterglow, "sigvis", None)
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    bx_lumi = (corrected_mu * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    # --- optional "after Type-1" diagnostics ---
    if getattr(cfg.type1, "debug_after_apply", False):
        analyze_type1_step(
            data=data,
            cfg=cfg,
            active_mask=active_mask,
            fill=fill,
            tag="after"
        )

    return data

# ---------------------------------------------------------------------------
# Step 4: bunch train corrections
# ---------------------------------------------------------------------------

@log_step("compute_bunch_train_step")
@timeit("compute_bunch_train_step")
def compute_bunch_train_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Pipeline step: estimate bunch train coefficients for a given fill.
    """
    if "bxraw" not in data:
        raise KeyError("compute_bunch_train_step: 'bxraw' not found in data")
    if "bxraw_ref" not in data:
        raise KeyError("compute_bunch_train_step: 'bxraw_ref' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    bxraw_ref = np.asarray(data["bxraw_ref"], dtype=np.float64)

    # --- SBIL / avg for Type-1 fit ---
    if "sbil" in data:
        avg = np.asarray(data["sbil"], dtype=np.float64)
    elif "avg" in data:
        avg = np.asarray(data["avg"], dtype=np.float64)
    else:
        # fallback: compute SBIL from bxraw and active mask
        mask = np.asarray(active_mask, dtype=np.int32)
        n_active = int(mask.sum())
        if n_active == 0:
            raise ValueError("compute_bunch_train_step: active_mask has zero active BX")
        avg = (bxraw * mask[None, :]).sum(axis=1) / float(n_active)

    order = int(getattr(cfg.bunch_train, "order", 1))
    sbil_min = float(getattr(cfg.bunch_train, "sbil_min", 0.1))

    # --- compute Type-1 coefficients ---
    p = compute_bunch_train_coeffs(
        bxraw=bxraw,
        bxraw_ref=bxraw_ref,
        avg=avg,
        active_mask=active_mask,
        order=order,
        sbil_min=sbil_min,
    )

    # --- where to save ---
    type1_dir = _get_type1_dir(cfg)

    path = save_bunch_train_coeffs(
        fill=fill,
        output_dir=type1_dir,
        coeffs=p,
    )

    log.info(
        "[compute_bunch_train_step] fill %d: Type-1 coeffs saved to %s",
        fill,
        path,
    )

    # --- optional debug / analysis block ---
    # This is the "before Type-1 subtraction" diagnostic.
    if getattr(cfg.bunch_train, "make_plots", False):
        analyze_bunch_train_step(
            data=data,
            cfg=cfg,
            active_mask=active_mask,
            fill=fill,
            tag="before"
        )

    return data

@log_step("apply_bunch_train_step")
@timeit("apply_bunch_train_step")
def apply_bunch_train_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Pipeline step: apply bunch train subtraction to bxraw.
    """
    if "bxraw" not in data:
        raise KeyError("apply_bunch_train_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)

    # --- where to find bunch train coefficients ---
    type1_dir = _get_type1_dir(cfg)
    coeff_path = os.path.join(type1_dir, f"bunch_train_coeffs_fill{fill}.h5")
    if not os.path.exists(coeff_path):
        raise FileNotFoundError(
            f"apply_bunch_train_step: Bunch Train coeff file not found: {coeff_path}"
        )

    # --- read coefficients ---
    with h5py.File(coeff_path, "r") as h5:
        p = h5["coeffs"][:]

    log.info(
        "[apply_bunch_train_step] fill %d: loaded Bunch Train coeffs from %s",
        fill,
        coeff_path,
    )

    # --- apply bunch_train subtraction in mu-space ---
    corrected_mu = apply_bunch_train_batch(
        bxraw=bxraw,
        active_mask=active_mask,
        coeffs=p,
    ).astype(np.float32)

    data["bxraw"] = corrected_mu

    # --- recompute bx and avg AFTER bunch train subtraction ---
    sigvis = getattr(cfg.afterglow, "sigvis", None)
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    bx_lumi = (corrected_mu * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    # --- optional "after bunch train" diagnostics ---
    if getattr(cfg.bunch_train, "make_plots", False):
        analyze_bunch_train_step(
            data=data,
            cfg=cfg,
            active_mask=active_mask,
            fill=fill,
            tag="after"
        )

    return data


# ---------------------------------------------------------------------------
# Main entry point for a single fill
# ---------------------------------------------------------------------------

@log_step("run_fill")
@timeit("run_fill")
def run_fill(fill: int, cfg: PipelineConfig) -> None:
    """
    Full pipeline for a single fill:

      - load input HDF5 (possibly multiple files and multiple fills),
      - filter rows with fillnum == fill,
      - load active BX mask,
      - sequentially apply enabled steps,
      - save the result to a new HDF5 file.
    """
    input_name = cfg.io.input_pattern.format(fill=fill)
    output_name = cfg.io.output_pattern.format(fill=fill)

    # --- active BX mask from npy file ---
    if not cfg.io.active_mask_pattern:
        raise ValueError("io.active_mask_pattern is not set in config")

    mask_path = cfg.io.active_mask_pattern.format(fill=fill)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"ActiveBX mask not found: {mask_path}")

    active_mask = np.load(mask_path)
    active_mask = np.asarray(active_mask, dtype=np.int32)
    if active_mask.shape[0] != BX_LEN:
        raise ValueError(f"active_mask len={active_mask.shape[0]} != BX_LEN={BX_LEN}")

    # --- 1) load input data (may contain multiple files and multiple fills) ---
    data = load_hd5_to_arrays(cfg.io.input_dir, input_name, node=cfg.io.node)
    
    # columns used for merging two tables together
    merge_cols = ['fillnum', 'runnum', 'lsnum', 'nbnum']

    # add the pedestal data (if needed)
    if cfg.online.pedestal_table is not None and cfg.steps.revert_online:
        ped = load_hd5_to_arrays(cfg.io.input_dir, input_name, node=cfg.online.pedestal_table)

        if data["timestampsec"].size != ped["timestampsec"].size:
            log.warning(
                "[run_fill] fill %d: pedestal data not present for %d entries. Dropping rows...",
                fill,
                data["timestampsec"].size - ped["timestampsec"].size,
            )

        if not _is_unique_columns([ped[c] for c in merge_cols]):
            raise ValueError(f"{merge_cols} values are not unique. Choose another column to merge on.")

        data = _inner_merge_one_column(data, ped, merge_cols, "bxraw", "pedestal")
    
    # add the reference data for the bunch train correction (if needed)
    if cfg.steps.bunch_train:
        ref = load_hd5_to_arrays(cfg.bunch_train.linear_reference, cfg.bunch_train.input_pattern.format(fill=fill), node=cfg.bunch_train.node)

        if data["timestampsec"].size != ref["timestampsec"].size:
            log.warning(
                "[run_fill] fill %d: reference luminometer data not present for %d entries. Dropping rows...",
                fill,
                data["timestampsec"].size - ref["timestampsec"].size,
            )

        if not _is_unique_columns([ref[c] for c in merge_cols]):
            raise ValueError(f"{merge_cols} values are not unique. Choose another column to merge on.")

        data = _inner_merge_one_column(data, ref, merge_cols, "bxraw", "bxraw_ref")
    

    # --- 1a) keep only rows corresponding to the current fill ---
    fill_arr = data.get("fillnum", None)
    if fill_arr is not None:
        fill_arr = np.asarray(fill_arr)
        unique_fills = np.unique(fill_arr)

        if fill not in unique_fills:
            log.warning(
                "[run_fill] fill %d: no rows with fillnum=%d in input (found fills: %s), skipping",
                fill,
                fill,
                unique_fills,
            )
            return

        mask = (fill_arr == fill)
        n_before = fill_arr.size
        n_after = int(mask.sum())

        if n_after == 0:
            log.warning(
                "[run_fill] fill %d: selection on fillnum left zero rows (fills in file: %s), skipping",
                fill,
                unique_fills,
            )
            return

        if unique_fills.size > 1:
            log.info(
                "[run_fill] fill %d: filtering by fillnum -> kept %d of %d rows (fills in file: %s)",
                fill,
                n_after,
                n_before,
                unique_fills,
            )

        for key, arr in data.items():
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_before:
                data[key] = arr[mask]
    else:
        log.warning(
            "[run_fill] fill %d: 'fillnum' column not found in data; proceeding without fill filtering",
            fill,
        )

    # filter out any nans TODO is this physically correct?
    if not np.all(np.isfinite(data['bxraw'])):
        log.warning(
            "[run_fill] fill %d: nan values found in data. Replacing with 0...",
            fill,
        )
        np.nan_to_num(data['bxraw'], copy=False, posinf=0.0, neginf=0.0)



    # before doing any processing, revert the online corrections
    if cfg.steps.revert_online:
        if cfg.type1.make_plots:
            plot_hist_bx(data, cfg, fill, 'Online Luminosity')
        data = revert_online_corrections(data, cfg, active_mask)

    # plot the uncorrected rates
    # TODO also need to tell the plot which year this is
    if cfg.type1.make_plots:
        plot_hist_bx(data, cfg, fill, 'Uncorr. Luminosity')

        # save the uncorrected rates for later comparison
        data_origin = copy.deepcopy(data)

    # --- 2) pipeline steps ---

    if cfg.steps.restore_rates:
        data = restore_rates_step(data, cfg, active_mask)

        # plot after t2 but before t1
        if cfg.type1.make_plots:
            plot_hist_bx(data, cfg, fill, 'T2 Corr. Luminosity')
            plot_residuals(data, cfg, active_mask, fill, 't2_corr')
    
    if cfg.steps.compute_type1:
        data = compute_type1_step(data, cfg, active_mask, fill)

    if cfg.steps.apply_type1:
        data = apply_type1_step(data, cfg, active_mask, fill)

        # plot the corrected rates
        # TODO also need to tell the plot which year this is
        if cfg.type1.make_plots:
            plot_hist_bx(data, cfg, fill, 'T2 and T1 Corr. Luminosity')
            plot_residuals(data, cfg, active_mask, fill, 't1_t2_corr')
    
    if cfg.steps.bunch_train:
        data = compute_bunch_train_step(data, cfg, active_mask, fill)
        data = apply_bunch_train_step(data, cfg, active_mask, fill)

    if cfg.type1.make_plots:
        plot_hist_bx(data, cfg, fill, 'Full Corr. Luminosity')
        plot_residuals(data, cfg, active_mask, fill, 'full_corr')
        plot_lumi_comparison(data, data_origin, cfg, active_mask, fill)
        plot_lasers(data, data_origin, cfg, active_mask, fill)

    # --- 3) save result ---
    rows = arrays_to_rows(data)
    save_to_hd5(rows, node=cfg.io.node, path=cfg.io.output_dir, name=output_name)


def run_many_fills(cfg: PipelineConfig, fills: list[int]):
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
            run_fill(fill, cfg)

        except FileNotFoundError as e:
            # Common case: missing mask, missing hd5, missing beam file, etc.
            print(f"[WARN] Fill {fill} skipped: {e}")
            failed.append(fill)

        except Exception as e:
            # Any other exception — also mark as failed and continue.
            print(f"[ERROR] Fill {fill} failed with exception:")
            print(e)
            print(traceback.format_exc())
            failed.append(fill)

    # Final summary
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
