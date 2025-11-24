# src/hfcore/type1_fit.py

from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging

from .hd5schema import BX_LEN
from .decorators import log_step, timeit

log = logging.getLogger("hfpipe.type1_fit")

def _collect_type1_points(
    bxraw: np.ndarray,
    avg: np.ndarray,
    active_mask: np.ndarray,
    offset: int,
    sbil_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Собирает точки (y, frac) для заданного offset:

      y     = mu_colliding  = bxraw[ibx]
      frac  = mu_afterglow / mu_colliding = bxraw[ibx+offset] / bxraw[ibx]

    Используем только:
      - строки, где avg > sbil_min (avg здесь трактуем как SBIL или аналог);
      - BX, где:
          active_mask[bx] == 1  (коллайдерный BX)
        и active_mask[bx+dt] == 0 для dt=1..offset (последующие offset BX неактивны).
    """
    active_mask = np.asarray(active_mask, dtype=np.int32)
    bxraw = np.asarray(bxraw, dtype=np.float64)
    avg = np.asarray(avg, dtype=np.float64)

    assert bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    # фильтр по светимости (SBIL / avg)
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
        y = hist[colliding_indices]       # mu_colliding
        y_after = hist[afterglow_indices] # mu_afterglow

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
    Строит фит Type1Fraction(y) ~ poly_order(y), где y = mu_colliding.

    order:
      1 -> линейный фит  frac ≈ k*y + b
      2 -> квадратичный  frac ≈ k2*y^2 + k1*y + b

    Возвращает (p0, p1, p2), где:
      - для order=1: p0 = b,  p1 = k,  p2 = 0
      - для order=2: p0 = b,  p1 = k1, p2 = k2

    (в таком виде удобно напрямую подставлять в формулу вычитания).
    """
    y_all, frac_all = _collect_type1_points(bxraw, avg, active_mask, offset, sbil_min)

    if y_all.size == 0:
        # нет точек — все коэффициенты нули
        return 0.0, 0.0, 0.0

    # polyfit возвращает [c_k, ..., c_0] для poly(x) = c_k x^k + ... + c_0
    coeffs = np.polyfit(y_all, frac_all, order)
    coeffs = coeffs[::-1]  # теперь coeffs[0] = c_0, coeffs[1] = c_1, ...

    # линейный случай: frac ≈ k*y + b
    if order == 1:
        b = float(coeffs[0])
        k = float(coeffs[1]) if coeffs.size > 1 else 0.0
        return b, k, 0.0

    # квадратичный случай: frac ≈ k2*y^2 + k1*y + b
    if order == 2:
        b = float(coeffs[0])
        k1 = float(coeffs[1]) if coeffs.size > 1 else 0.0
        k2 = float(coeffs[2]) if coeffs.size > 2 else 0.0
        return b, k1, k2

    # на всякий случай (не должны сюда попадать)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Считает массивы p0,p1,p2 и orders для всех offsets.

    Логика порядка фита:
      - offset == 1 -> quadratric (order = 2)
      - offset >  1 -> linear     (order = 1)

    Возвращаемые массивы:
      p0, p1, p2, orders  длиной max_offset+1;
      индекс t соответствует смещению BX+i, т.е. t=1 -> BX+1 и т.д.
      t=0 оставляем нулевым (для самого коллайдерного BX вычитания нет).
    """
    bxraw = np.asarray(bxraw, dtype=np.float64)
    avg = np.asarray(avg, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=np.int32)

    assert bxraw.ndim == 2 and bxraw.shape[1] == BX_LEN, "bxraw must be (T, BX_LEN)"

    if not offsets:
        max_offset = 0
    else:
        max_offset = max(offsets)

    p0 = np.zeros(max_offset + 1, dtype=np.float64)
    p1 = np.zeros(max_offset + 1, dtype=np.float64)
    p2 = np.zeros(max_offset + 1, dtype=np.float64)
    orders = np.zeros(max_offset + 1, dtype=np.int32)

    for off in offsets:
        if off <= 0:
            continue  # offset=0 не трогаем

        # правило: BX+1 -> квадратичный, остальные -> линейный
        if off == 1:
            order = 2
        else:
            order = 1

        c0, c1, c2 = _fit_type1_for_offset(
            bxraw=bxraw,
            avg=avg,
            active_mask=active_mask,
            offset=off,
            sbil_min=sbil_min,
            order=order,
        )
        p0[off] = c0
        p1[off] = c1
        p2[off] = c2
        orders[off] = order

    return p0, p1, p2, orders


def save_type1_coeffs(
    fill: int,
    output_dir: str,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    offsets: Sequence[int],
    orders: np.ndarray,
) -> str:
    """
    Сохраняет коэффициенты Type1 в небольшой HDF5-файл:

      type1_coeffs_fill{fill}.h5

    Формат:
      datasets:
        - "p0", "p1", "p2"     (длины max_offset+1)
        - "offsets"            (список используемых смещений)
        - "orders"             (порядок фита для каждого t)
      attrs:
        - "fill"
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"type1_coeffs_fill{fill}.h5")

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    orders = np.asarray(orders, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("p0", data=p0)
        h5.create_dataset("p1", data=p1)
        h5.create_dataset("p2", data=p2)
        h5.create_dataset("offsets", data=offsets)
        h5.create_dataset("orders", data=orders)
        h5.attrs["fill"] = int(fill)

    return path

def _select_type1_pairs(active_mask: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выбор пар (colliding BX, afterglow BX) для данного offset.

    Логика повторяет старый код:
      colliding_value = [bx for bx in range(upper)
                         if activeBXMask[bx] == 1
                         and all(activeBXMask[bx + dt] == 0 for dt in range(1, offset + 1))]
      afterglow_value = [bx + offset for bx in colliding_value]
    """
    active_mask = np.asarray(active_mask, dtype=np.int8)
    N = active_mask.shape[0]
    # верхняя граница, как в старом коде
    upper = min(3480, N - offset)

    colliding = []
    for bx in range(upper):
        if active_mask[bx] != 1:
            continue
        # все последующие offset BX должны быть неактивными
        if all(active_mask[bx + dt] == 0 for dt in range(1, offset + 1)):
            colliding.append(bx)

    colliding = np.asarray(colliding, dtype=np.int64)
    afterglow = colliding + offset
    return colliding, afterglow


def _do_binning(x: np.ndarray, y: np.ndarray, nbins: int = 20):
    """
    Аналог do_binning(x, y, 20).

    Делает биннинг по x с равной численностью точек (примерно),
    возвращает:
      x_bins — центр по x,
      y_bins — среднее y,
      s_bins — стандартная ошибка среднего по y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    # сортируем по x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # делим на nbins примерно равных по числу точек
    n = x_sorted.size
    nbins = min(n, nbins)
    bin_edges = np.linspace(0, n, nbins + 1, dtype=int)

    x_bins = []
    y_bins = []
    s_bins = []

    for i in range(nbins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if end <= start:
            continue
        xx = x_sorted[start:end]
        yy = y_sorted[start:end]
        if xx.size == 0:
            continue
        x_bins.append(xx.mean())
        y_bins.append(yy.mean())
        # ошибка среднего
        if yy.size > 1:
            s_bins.append(yy.std(ddof=1) / np.sqrt(yy.size))
        else:
            s_bins.append(0.0)

    return np.asarray(x_bins), np.asarray(y_bins), np.asarray(s_bins)


def _h5_get_or_create(h5: h5py.File, path: str, dtype) -> h5py.Dataset:
    """
    Полный аналог твоей вспомогательной функции:
    создаёт или возвращает 1D dataset с указанным dtype.
    """
    if path in h5:
        ds = h5[path]
        # минимальная проверка dtype
        if ds.dtype != np.dtype(dtype):
            raise TypeError(f"HDF5 dataset {path} has dtype={ds.dtype}, expected {dtype}")
        return ds
    else:
        return h5.create_dataset(path, shape=(0,), maxshape=(None,), dtype=dtype)


def _h5_append_1d(ds: h5py.Dataset, arr: np.ndarray):
    """
    Аппендим 1D массив в конец dataset’а.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("_h5_append_1d: only 1D arrays supported")
    old_n = ds.shape[0]
    new_n = old_n + arr.shape[0]
    ds.resize((new_n,))
    ds[old_n:new_n] = arr


@log_step("analyze_type1_step")
def analyze_type1_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Аналитический шаг: полностью имитирует старый calculate_type1,
    но работает с текущей структурой data/cfg.

    Для каждого offset из cfg.type1.offsets:
      - строит скаттер Type1-фракции (afterglow/colliding),
      - делает биннинг,
      - фит (order=2 для offset=1, иначе order=1),
      - сохраняет всё в HDF5: scatter/*, binned/*, fit/*, poly/*,
      - опционально рисует PNG (если cfg.type1.make_plots == True).

    Ничего не меняет в data.
    """
    if "bxraw" not in data:
        raise KeyError("analyze_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"analyze_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # SBIL / avg — те же правила, что и в compute_type1_step
    if "sbil" in data:
        avg = np.asarray(data["sbil"], dtype=np.float64)
    elif "avg" in data:
        avg = np.asarray(data["avg"], dtype=np.float64)
    else:
        mask = np.asarray(active_mask, dtype=np.int32)
        n_active = int(mask.sum())
        if n_active == 0:
            raise ValueError("analyze_type1_step: active_mask has zero active BX")
        avg = (bxraw * mask[None, :]).sum(axis=1) / float(n_active)

    offsets: Sequence[int] = list(getattr(cfg.type1, "offsets", [1, 2, 3, 4]))
    sbil_min: float = float(getattr(cfg.type1, "sbil_min", 0.1))
    make_plots: bool = bool(getattr(cfg.type1, "make_plots", False))

    # базовый путь, куда класть /{fill}/hd5/type1_{offset}.h5
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

    for offset in offsets:
        # порядок фита -> как в старом коде
        order = 2 if offset == 1 else 1

        # --- отбор по SBIL ---
        time_mask = avg > sbil_min
        if not np.any(time_mask):
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no points with SBIL > %g",
                fill,
                offset,
                sbil_min,
            )
            continue

        hists = bxraw[time_mask, :]   # (T_sel, BX_LEN)

        # --- выбираем пары BX ---
        colliding_idx, afterglow_idx = _select_type1_pairs(active_mask, offset)
        if colliding_idx.size == 0:
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no (colliding, afterglow) pairs found",
                fill,
                offset,
            )
            continue

        # --- вытаскиваем значения ---
        coll = hists[:, colliding_idx]      # shape (T_sel, Npairs)
        aft = hists[:, afterglow_idx]       # shape (T_sel, Npairs)

        # защищаемся от деления на ноль
        valid = coll > 0.0
        coll = coll[valid]
        aft = aft[valid]

        if coll.size == 0:
            log.warning(
                "[analyze_type1_step] fill %d offset %d: no positive colliding BX values",
                fill,
                offset,
            )
            continue

        bx_value = coll.astype(np.float64)              # базовая интенсивность
        bx_type1 = (aft / coll).astype(np.float64)      # Type1 фракция

        # --- биннинг ---
        x_bins, y_bins, s_bins = _do_binning(bx_value, bx_type1, nbins=20)

        # --- фит ---
        # polyfit ожидает x, y 1D
        coeffs = np.polyfit(bx_value, bx_type1, order)
        # для красивой кривой фита
        new_x = np.linspace(float(np.min(bx_value)), float(np.max(bx_value)), num=bx_value.size)
        new_line = np.polyval(coeffs, new_x)

        # --- HDF5 структура как в старом коде ---
        # output_base/{fill}/hd5/type1_{offset}.h5
        output_dir = os.path.join(type1_dir, str(fill))
        os.makedirs(output_dir, exist_ok=True)

        hd5_dir = os.path.join(output_dir, "hd5")
        os.makedirs(hd5_dir, exist_ok=True)

        hd5_path = os.path.join(hd5_dir, f"type1_{offset}.h5")

        fill_arr_scatter = np.full(bx_value.size, int(fill), dtype=np.int64)
        fill_arr_binned  = np.full(x_bins.size,   int(fill), dtype=np.int64)
        fill_arr_fit     = np.full(new_x.size,    int(fill), dtype=np.int64)

        with h5py.File(hd5_path, "a") as h5:
            h5.attrs["fill"] = int(fill)
            h5.attrs["type"] = int(offset)
            h5.attrs["order"] = int(order)

            # --- scatter ---
            ds_fill = _h5_get_or_create(h5, "scatter/fill", dtype=np.int64)
            ds_x    = _h5_get_or_create(h5, "scatter/x",    dtype=np.float64)
            ds_y    = _h5_get_or_create(h5, "scatter/y",    dtype=np.float64)
            _h5_append_1d(ds_fill, fill_arr_scatter)
            _h5_append_1d(ds_x, bx_value)
            _h5_append_1d(ds_y, bx_type1)

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
            "[analyze_type1_step] fill %d offset %d: saved debug HDF5 to %s",
            fill,
            offset,
            hd5_path,
        )

        # --- PNG (опционально, как раньше) ---
        if make_plots:
            fig = plt.figure(figsize=(7, 5))
            plt.plot(
                bx_value,
                bx_type1,
                ".",
                alpha=0.2,
                label=f"Type1 fraction for BCID [i+{offset}]",
            )
            plt.errorbar(
                x_bins,
                y_bins,
                yerr=s_bins,
                fmt="o",
                linestyle="",
                markersize=4,
                lw=1,
                zorder=10,
                capsize=3,
                capthick=1,
                label="Binned",
            )
            if order == 1:
                plt.plot(
                    new_x,
                    new_line,
                    label=f"Linear fit: {coeffs[0]:.5f} x + {coeffs[1]:.5f}",
                )
            else:
                poly_str = " + ".join(
                    f"{c:.5f} x^{i}"
                    for i, c in zip(range(order, -1, -1), coeffs)
                )
                plt.plot(new_x, new_line, label=f"Poly{order} fit: {poly_str}")

            plt.xlabel("Instantaneous luminosity [Hz/μb]")
            plt.ylabel("Type1 fraction")
            plt.title(f"Fill {fill}, offset={offset}")
            plt.legend(loc="upper right", frameon=False)
            plt.tight_layout()

            png_path = os.path.join(output_dir, f"type1_{offset}.png")
            plt.savefig(png_path, dpi=300)
            plt.close(fig)

            log.info(
                "[analyze_type1_step] fill %d offset %d: saved PNG to %s",
                fill,
                offset,
                png_path,
            )