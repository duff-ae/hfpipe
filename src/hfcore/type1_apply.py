# src/hfcore/type1_apply.py

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import h5py

from .hd5schema import BX_LEN
from .decorators import log_step, timeit


def subtract_type1_numpy(
    mu_hist_per_bx: np.ndarray,
    active_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Чистый перенос логики вычитания Type1:
      для каждого коллайдерного BX (где active_mask == 1)
      и каждого сдвига t (offset):
        y = mu[ibx]
        frac(y) = p0[t] + p1[t]*y + p2[t]*y^2
        mu[ibx + t] -= y * frac(y)

    Предполагаем, что len(p0) == len(p1) == len(p2) == max_offset+1.
    """
    mu = np.array(mu_hist_per_bx, dtype=np.float64, copy=True)
    active = np.asarray(active_mask, dtype=np.int32)

    bx_len = mu.shape[0]
    type_len = len(p0) - 1  # максимальный offset

    for ibx in range(0, bx_len - (type_len + 1) + 1):
        if active[ibx] != 1:
            continue
        y = mu[ibx]
        if y <= 0.0:
            continue
        y2 = y * y
        # t=0 – не трогаем (коллайдерный BX), но даже если p*=0 – всё равно безопасно
        for t in range(1, type_len + 1):
            j = ibx + t
            if j >= bx_len:
                break
            frac = p0[t] + p1[t] * y + p2[t] * y2
            mu[j] -= y * frac

    return mu


def apply_type1_batch(
    bxraw: np.ndarray,
    active_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Применяет вычитание Type1 ко всем гистограммам:
      bxraw: (T, BX_LEN)
    Возвращает новый массив той же формы.
    """
    hists = np.asarray(bxraw, dtype=np.float64)
    T, N = hists.shape
    assert N == BX_LEN, f"bxraw.shape[1]={N} != BX_LEN={BX_LEN}"

    corrected = np.empty_like(hists, dtype=np.float64)
    for i in range(T):
        corrected[i] = subtract_type1_numpy(hists[i], active_mask, p0, p1, p2)

    return corrected


def load_type1_coeffs(fill: int, output_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает p0, p1, p2 из файла type1_coeffs_fill{fill}.h5.
    """
    path = os.path.join(output_dir, f"type1_coeffs_fill{fill}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Type1 coeffs file not found: {path}")

    with h5py.File(path, "r") as h5:
        p0 = np.asarray(h5["p0"], dtype=np.float64)
        p1 = np.asarray(h5["p1"], dtype=np.float64)
        p2 = np.asarray(h5["p2"], dtype=np.float64)
    return p0, p1, p2