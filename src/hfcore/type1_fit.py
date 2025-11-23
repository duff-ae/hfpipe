# src/hfcore/type1_fit.py

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
import h5py

from .hd5schema import BX_LEN
from .decorators import log_step, timeit


def _collect_type1_points(
    bxraw: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    offset: int,
    sbil_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Собирает точки (y, frac) для заданного offset:
      y     = mu_colliding (bxraw[ibx])
      frac  = mu_afterglow / mu_colliding = bxraw[ibx+offset] / bxraw[ibx]
    Использует только строки с avg > sbil_min и только BX,
    где bx активен, а следующие 'offset' BX неактивны.
    """
    active_mask = np.asarray(active_mask, dtype=np.int32)
    bxraw = np.asarray(bxraw, dtype=np.float64)
    avg = np.asarray(avg, dtype=np.float64)

    assert bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    # фильтр по светимости (SBIL)
    mask_sbil = avg > sbil_min
    hists = bxraw[mask_sbil]
    if hists.shape[0] == 0:
        return np.array([]), np.array([])

    N = BX_LEN
    upper = N - offset

    colliding_indices = []
    for bx in range(upper):
        if active_mask[bx] != 1:
            continue
        # требуем, чтобы следующие offset BX были неактивны
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

    # собираем значения
    y_vals = []
    frac_vals = []
    for hist in hists:
        y = hist[colliding_indices]              # mu_colliding
        y_after = hist[afterglow_indices]        # mu_afterglow
        # избегаем деления на ноль
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
    Строит fit Type1Fraction(y) ~ poly_order(y), где y=mu_colliding.
    Возвращает (p0, p1, p2) для данного offset.
    Если order < 2, старшие коэффициенты будут 0.
    """
    y_all, frac_all = _collect_type1_points(bxraw, avg, active_mask, offset, sbil_min)

    if y_all.size == 0:
        # нет точек — все коэффициенты нули
        return 0.0, 0.0, 0.0

    # polyfit возвращает [c_k, ..., c_0] для poly(x) = c_k x^k + ... + c_0
    coeffs = np.polyfit(y_all, frac_all, order)
    # переводим в [c_0, c_1, c_2] (снизу вверх)
    coeffs = coeffs[::-1]

    # заполняем p0,p1,p2 (поддерживаем максимум квадратичный полином)
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
    order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Считает массивы p0,p1,p2 для всех offsets.
    Длина массивов = max_offset + 1, индекс t соответствует сдвигу t (BX+i).
    t=0 оставляем нулевым (для самого коллайдерного BX вычитания нет).
    """
    if not offsets:
        max_offset = 0
    else:
        max_offset = max(offsets)

    p0 = np.zeros(max_offset + 1, dtype=np.float64)
    p1 = np.zeros(max_offset + 1, dtype=np.float64)
    p2 = np.zeros(max_offset + 1, dtype=np.float64)

    for off in offsets:
        if off <= 0:
            continue  # offset=0 не трогаем
        c0, c1, c2 = _fit_type1_for_offset(bxraw, avg, active_mask, off, sbil_min, order)
        p0[off] = c0
        p1[off] = c1
        p2[off] = c2

    return p0, p1, p2


def save_type1_coeffs(
    fill: int,
    output_dir: str,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    offsets: Sequence[int],
    order: int,
) -> str:
    """
    Сохраняет коэффициенты Type1 в небольшой HDF5-файл.
    Формат простой: p0, p1, p2, offsets, order.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"type1_coeffs_fill{fill}.h5")

    with h5py.File(path, "w") as h5:
        h5.create_dataset("p0", data=p0)
        h5.create_dataset("p1", data=p1)
        h5.create_dataset("p2", data=p2)
        h5.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        h5.attrs["order"] = int(order)
        h5.attrs["fill"] = int(fill)

    return path