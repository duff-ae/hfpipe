from __future__ import annotations

import numpy as np

from .hd5schema import BX_LEN


def fixed_pedestal_vector(pedestal_4) -> np.ndarray:
    """Expand four mod-4 pedestal values to one full-orbit BX vector."""
    ped4 = np.asarray(pedestal_4, dtype=np.float32).ravel()
    if ped4.shape != (4,):
        raise ValueError(f"fixed pedestal must have length 4, got shape {ped4.shape}")
    return ped4[np.arange(BX_LEN) % 4]


def subtract_fixed_pedestal_mod4_inplace(data: dict, pedestal_4) -> None:
    """
    Subtract a fixed mod-4 pedestal from ``data['bxraw']`` in place.

    This is intentionally the same operation used by the production pipeline
    before LSQ afterglow recovery.  Keeping it here lets standalone calibration
    helpers (notably single-bunch Type-2 extraction) use exactly the same
    pedestal convention without importing private pipeline internals.
    """
    if pedestal_4 is None:
        return
    if "bxraw" not in data:
        raise KeyError("fixed pedestal is set but data has no 'bxraw'")

    bxraw = np.asarray(data["bxraw"])
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(f"bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})")

    ped_vec = fixed_pedestal_vector(pedestal_4)[None, :]
    data["bxraw"] = (bxraw - ped_vec).astype(bxraw.dtype, copy=False)


def subtract_fixed_pedestal_mod4(bxraw: np.ndarray, pedestal_4) -> np.ndarray:
    """Return a pedestal-subtracted copy of a ``(T, BX_LEN)`` array."""
    out = {"bxraw": np.array(bxraw, copy=True)}
    subtract_fixed_pedestal_mod4_inplace(out, pedestal_4)
    return out["bxraw"]


def calculate_dynamic_pedestal(mu_hist: np.ndarray) -> np.ndarray:
    """
    Exact CMS dynamic-pedestal logic used by the production pipeline.

    Take BX 3500..3551 (13 samples for each BX % 4) and return the four
    pedestal components.
    """
    arr = np.asarray(mu_hist)
    if arr.ndim != 1 or arr.size != BX_LEN:
        raise ValueError(f"mu_hist has shape {arr.shape}, expected ({BX_LEN},)")

    n_sample = 13
    pedestal = np.zeros(4, dtype=np.float32)
    base = 3500
    for ibx in range(4):
        s = 0.0
        for j in range(ibx, 4 * n_sample, 4):
            s += arr[base + j]
        pedestal[ibx] = s / n_sample
    return pedestal