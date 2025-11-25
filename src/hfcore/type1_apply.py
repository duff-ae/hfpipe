# src/hfcore/type1_apply.py

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import h5py

from .hd5schema import BX_LEN


def apply_type1_batch(
    bxraw: np.ndarray,
    active_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized analogue of the original C function subtract_type1,
    applied to all histograms at once.

    Parameters
    ----------
    bxraw : np.ndarray
        Array of shape (T, BX_LEN) with per-BX mu (after afterglow + pedestal).
    active_mask : np.ndarray
        1D array of shape (BX_LEN,) with active / colliding BX pattern.
        Here it is only used to decide which BX can serve as "colliding" seeds.
        (We require active_mask[ibx] == 1 for a colliding BX.)
    p0, p1, p2 : np.ndarray
        1D arrays of length type_len + 1, where type_len is the maximum
        offset. Index t corresponds to BX+ t:
            t = 0 -> colliding BX itself
            t = 1 -> first afterglow BX
            ...
        Coefficients define the Type-1 fraction as:
            frac_t(y) = p0[t] + p1[t] * y + p2[t] * y^2
        where y is the colliding mu (mu_colliding).

    Returns
    -------
    out : np.ndarray
        Array of shape (T, BX_LEN) with Type-1 contribution subtracted.

    Notes
    -----
    The subtraction formula is:

        mu_corr[j] = mu_true[j] - y * frac_t(y),

    where
        y        = mu_true[ibx] (colliding BX),
        j        = ibx + t,
        frac_t   = p0[t] + p1[t] * y + p2[t] * y^2.

    For t = 0, p0[0], p1[0], p2[0] are expected to be zero
    (i.e. no subtraction on the colliding BX itself).
    """
    hists = np.asarray(bxraw, dtype=np.float64)
    if hists.ndim != 2:
        raise ValueError(f"apply_type1_batch: bxraw must be 2D, got shape {hists.shape}")

    T, N = hists.shape
    if N != BX_LEN:
        raise ValueError(f"apply_type1_batch: bxraw.shape[1]={N} != BX_LEN={BX_LEN}")

    # Work on a copy to avoid in-place modification of the input array
    out = hists.copy()

    active_mask = np.asarray(active_mask, dtype=np.int8)
    if active_mask.shape[0] != N:
        raise ValueError(
            f"apply_type1_batch: active_mask length={active_mask.shape[0]} "
            f"!= BX_LEN={BX_LEN}"
        )

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    if not (p0.shape == p1.shape == p2.shape):
        raise ValueError(
            "apply_type1_batch: p0, p1, p2 must have the same shape, got "
            f"{p0.shape}, {p1.shape}, {p2.shape}"
        )

    type_len = len(p0) - 1  # if p0 has length 5 -> offsets t=0..4 => type_len=4
    if type_len < 0:
        # empty coefficient arrays: nothing to do
        return out

    # Full analogue of the C loop:
    #
    # for (ibx = 0; ibx <= N - (type_len + 1); ++ibx) {
    #   if (active_mask[ibx] != 1) continue;
    #   y = out[:, ibx];
    #   for (t = 0; t <= type_len; ++t) {
    #       j = ibx + t;
    #       poly = p0[t] + p1[t] * y + p2[t] * y^2;
    #       out[:, j] -= y * poly;
    #   }
    # }
    #
    # Here we do the same in Python/NumPy.
    for ibx in range(0, N - (type_len + 1) + 1):
        if active_mask[ibx] != 1:
            continue

        # y for ALL events in this colliding BX: shape (T,)
        y = out[:, ibx]
        y2 = y * y

        # Loop over all offsets t = 0..type_len
        for t in range(type_len + 1):
            j = ibx + t
            # poly(y) = p0[t] + p1[t] * y + p2[t] * y^2
            poly = p0[t] + p1[t] * y + p2[t] * y2
            # subtract the Type-1 contribution for all events at BX j
            out[:, j] -= y * poly

    return out


def load_type1_coeffs(fill: int, output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load p0, p1, p2 from type1_coeffs_fill{fill}.h5 in the given directory.

    Parameters
    ----------
    fill : int
        Fill number.
    output_dir : str
        Directory where type1_coeffs_fill{fill}.h5 is stored.

    Returns
    -------
    p0, p1, p2 : np.ndarray
        Coefficient arrays as stored in the file.
    """
    path = os.path.join(output_dir, f"type1_coeffs_fill{fill}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Type-1 coeffs file not found: {path}")

    with h5py.File(path, "r") as h5:
        p0 = np.asarray(h5["p0"], dtype=np.float64)
        p1 = np.asarray(h5["p1"], dtype=np.float64)
        p2 = np.asarray(h5["p2"], dtype=np.float64)

    return p0, p1, p2
