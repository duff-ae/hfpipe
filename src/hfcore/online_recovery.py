# src/hfcore/online_recovery.py

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .hd5schema import BX_LEN

log = logging.getLogger("hfpipe")

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from cffi import FFI
except Exception as e:
    raise RuntimeError(
        "online_recovery.py requires cffi. Install it with `pip install cffi`."
    ) from e


# ----------------------------------------------------------------------
# C backend
# ----------------------------------------------------------------------

_ffi = FFI()
_ffi.cdef(
    """
    void revert_afterglow(const int * activeBXMask, float * muHistPerBX, const float * HFSBR);
    void subtract_pedestal(float * muHistPerBX, float * pedestal);
    """
)

_C = _ffi.verify(
    r"""
    void revert_afterglow(const int * activeBXMask, float * muHistPerBX, const float * HFSBR) {
        int ibx, jbx;
        int bx_len = 3564;
        int sbr_len = 3564;

        for (ibx = bx_len; ibx-- > 0; ) {
            if (activeBXMask[ibx] == 1) {
                for (jbx = ibx + 1; jbx < ibx + sbr_len; jbx++) {
                    if (jbx < bx_len) {
                        muHistPerBX[jbx] += muHistPerBX[ibx] * HFSBR[jbx - ibx];
                    } else {
                        muHistPerBX[jbx - bx_len] += muHistPerBX[ibx] * HFSBR[jbx - ibx];
                    }
                }
            }
        }
    }

    void subtract_pedestal(float * muHistPerBX, float * pedestal) {
        int nSample = 10;
        int bx_len = 3564;
        int ibx, jbx;

        for (ibx = 0; ibx < 4; ibx++) {
            pedestal[ibx] = 0.0f;
            for (jbx = ibx; jbx < 4 * nSample; jbx += 4) {
                pedestal[ibx] += muHistPerBX[3500 + jbx] / nSample;
            }
        }

        for (ibx = 0; ibx < bx_len; ibx++) {
            muHistPerBX[ibx] -= pedestal[ibx % 4];
        }
    }
    """,
    extra_compile_args=["-O3"],
)


@dataclass
class OnlineStates:
    """
    Reconstructed online-like states per row:

      - mu_before : BX rates after revert_afterglow and pedestal subtraction
      - mu_after  : BX rates after revert_afterglow, before pedestal subtraction
      - pedestal  : 4-element pedestal values (per row)
    """
    mu_before: np.ndarray   # shape (T, BX_LEN)
    mu_after: np.ndarray    # shape (T, BX_LEN)
    pedestal: np.ndarray    # shape (T, 4)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _validate_bx_hist_2d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != BX_LEN:
        raise ValueError(f"{name} has shape {arr.shape}, expected (T, {BX_LEN})")
    return arr


def _validate_bx_hist_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.shape[0] != BX_LEN:
        raise ValueError(f"{name} has shape {arr.shape}, expected ({BX_LEN},)")
    return arr


def _validate_active_mask(active_mask: np.ndarray) -> np.ndarray:
    active = np.asarray(active_mask, dtype=np.int32).ravel()
    if active.shape[0] != BX_LEN:
        raise ValueError(
            f"active_mask has length {active.shape[0]}, expected {BX_LEN}"
        )
    return active


def _validate_hfsbr(hfsbr: np.ndarray) -> np.ndarray:
    hfsbr = np.asarray(hfsbr, dtype=np.float32).ravel()
    if hfsbr.shape[0] < BX_LEN:
        raise ValueError(
            f"hfsbr length {hfsbr.shape[0]} is smaller than BX_LEN={BX_LEN}"
        )
    return hfsbr


def _make_progress(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total)
    return iterable


def _as_c_float32_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr = _validate_bx_hist_1d(name, arr)
    return arr


def _call_revert_afterglow_inplace(
    active_mask_i32: np.ndarray,
    mu_hist_f32: np.ndarray,
    hfsbr_f32: np.ndarray,
) -> None:
    _C.revert_afterglow(
        _ffi.cast("const int *", _ffi.from_buffer(active_mask_i32)),
        _ffi.cast("float *", _ffi.from_buffer(mu_hist_f32)),
        _ffi.cast("const float *", _ffi.from_buffer(hfsbr_f32)),
    )


def _call_subtract_pedestal_inplace(
    mu_hist_f32: np.ndarray,
    pedestal_f32: np.ndarray,
) -> None:
    _C.subtract_pedestal(
        _ffi.cast("float *", _ffi.from_buffer(mu_hist_f32)),
        _ffi.cast("float *", _ffi.from_buffer(pedestal_f32)),
    )


# ----------------------------------------------------------------------
# Method A: reconstruction using extra tables
# ----------------------------------------------------------------------

def reconstruct_from_tables_batch(
    bxraw_final: np.ndarray,
    pedestal_4: np.ndarray,
    afterglow_frac: np.ndarray,
) -> OnlineStates:
    """
    Reconstruct using hfEtPedestal and hfafterglowfrac.

    Interpretation kept consistent with previous code:
      - mu_after  = bxraw_final + pedestal
      - mu_before = mu_after / afterglow_frac   where frac > 0
    """
    bxraw_final = np.asarray(bxraw_final, dtype=np.float64)
    pedestal_4 = np.asarray(pedestal_4, dtype=np.float64)
    afterglow_frac = np.asarray(afterglow_frac, dtype=np.float64)

    bxraw_final = _validate_bx_hist_2d("bxraw_final", bxraw_final)

    T, _ = bxraw_final.shape
    if pedestal_4.shape != (T, 4):
        raise ValueError(f"pedestal_4 shape {pedestal_4.shape}, expected ({T}, 4)")
    if afterglow_frac.shape != (T, BX_LEN):
        raise ValueError(
            f"afterglow_frac shape {afterglow_frac.shape}, expected ({T}, {BX_LEN})"
        )

    idx_mod4 = np.arange(BX_LEN) % 4

    mu_after = np.empty_like(bxraw_final, dtype=np.float64)
    mu_before = np.zeros_like(bxraw_final, dtype=np.float64)

    for i in range(T):
        ped_pattern = pedestal_4[i][idx_mod4]
        mu_after[i] = bxraw_final[i] + ped_pattern

        mask = afterglow_frac[i] > 0.0
        mu_before[i, mask] = mu_after[i, mask] / afterglow_frac[i, mask]

    return OnlineStates(
        mu_before=mu_before.astype(np.float32),
        mu_after=mu_after.astype(np.float32),
        pedestal=pedestal_4.astype(np.float32),
    )


# ----------------------------------------------------------------------
# Method B: exact C-style online reconstruction
# ----------------------------------------------------------------------

def reconstruct_single_hist_online(
    bxraw_final: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact C-style sequence for one histogram:

      input histogram = bxraw_final
      1) revert_afterglow(mu)
      2) save mu_after = reverted histogram before pedestal subtraction
      3) subtract_pedestal(mu, pedestal)
      4) save mu_before = final corrected histogram

    No fit, no zero_bx constraints, no iteration.
    """
    mu = _as_c_float32_1d("bxraw_final", bxraw_final).copy()
    hfsbr_f32 = _validate_hfsbr(hfsbr)
    active_i32 = _validate_active_mask(active_mask)

    _call_revert_afterglow_inplace(
        active_mask_i32=active_i32,
        mu_hist_f32=mu,
        hfsbr_f32=hfsbr_f32,
    )

    mu_after = mu.copy()
    pedestal = np.zeros(4, dtype=np.float32)

    _call_subtract_pedestal_inplace(
        mu_hist_f32=mu,
        pedestal_f32=pedestal,
    )

    mu_before = mu

    return mu_before, mu_after, pedestal


def reconstruct_from_online_batch(
    bxraw_final: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
    zero_bx: tuple[int, ...] = (3553, 3554, 3555, 3556, 3557),  # kept for API compatibility
    n_iter: int = 0,                                             # kept for API compatibility
    step: float = 0.0,                                           # kept for API compatibility
    show_progress: bool = False,
) -> OnlineStates:
    """
    Batch wrapper around exact per-histogram C-style reconstruction.

    Note:
      zero_bx / n_iter / step are ignored intentionally.
      They are kept only so the existing pipeline call site does not break.
    """
    bxraw_final = np.asarray(bxraw_final, dtype=np.float32)
    bxraw_final = _validate_bx_hist_2d("bxraw_final", bxraw_final)

    hfsbr = _validate_hfsbr(hfsbr)
    active_mask = _validate_active_mask(active_mask)

    T, _ = bxraw_final.shape

    mu_before_all = np.zeros_like(bxraw_final, dtype=np.float32)
    mu_after_all = np.zeros_like(bxraw_final, dtype=np.float32)
    pedestal_all = np.zeros((T, 4), dtype=np.float32)

    row_iter = _make_progress(
        range(T),
        enabled=show_progress,
        desc="Online recovery (C-style)",
        total=T,
    )

    for i in row_iter:
        mu_before_i, mu_after_i, pedestal_i = reconstruct_single_hist_online(
            bxraw_final=bxraw_final[i],
            hfsbr=hfsbr,
            active_mask=active_mask,
        )

        mu_before_all[i] = mu_before_i
        mu_after_all[i] = mu_after_i
        pedestal_all[i] = pedestal_i

    return OnlineStates(
        mu_before=mu_before_all,
        mu_after=mu_after_all,
        pedestal=pedestal_all,
    )