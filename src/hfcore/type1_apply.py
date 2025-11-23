# src/hfcore/type1_apply.py

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import h5py

from .hd5schema import BX_LEN
from .decorators import log_step, timeit

def apply_type1_batch(
    bxraw: np.ndarray,
    active_mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Векторизованный аналог C-функции subtract_type1
    для всех гистограмм сразу.

    bxraw: (T, BX_LEN)
    active_mask: (BX_LEN,)
    p0, p1, p2: массивы длины type_len+1 (как в C: t=0..type_len)
    """
    hists = np.asarray(bxraw, dtype=np.float64)
    T, N = hists.shape
    assert N == BX_LEN, f"bxraw.shape[1]={N} != BX_LEN={BX_LEN}"

    # Работаем копией, чтобы не портить входной массив по месту
    out = hists.copy()

    active_mask = np.asarray(active_mask, dtype=np.int8)
    coll_idx = np.nonzero(active_mask == 1)[0]

    type_len = len(p0) - 1  # если p0/p1/p2 длины 5 → type_len = 4

    # Полный аналог C-цикла
    for ibx in range(0, N - (type_len + 1) + 1):
        if active_mask[ibx] != 1:
            continue

        # y для ВСЕХ событий в этом bx: shape (T,)
        y = out[:, ibx]
        y2 = y * y

        for t in range(type_len + 1):
            j = ibx + t
            # poly(y) = p0[t] + y*p1[t] + y^2 * p2[t]
            poly = p0[t] + p1[t] * y + p2[t] * y2
            # вычитаем вклад для всех событий
            out[:, j] -= y * poly

    return out


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