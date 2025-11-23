# src/hfcore/pipeline.py
from __future__ import annotations

import os
from typing import List

import numpy as np
import logging

from .decorators import log_step, timeit
from .io import load_hd5_to_arrays, arrays_to_rows, save_to_hd5
from .hd5schema import BX_LEN
from .afterglow_lsq import build_afterglow_solver_from_file
from .type1_fit import compute_type1_coeffs, save_type1_coeffs
from .type1_apply import apply_type1_batch

from .config import PipelineConfig


log = logging.getLogger("hfpipe")

# ------------------ Шаг 1: afterglow / восстановление mu_true ------------------

def calculate_dynamic_pedestal(mu_hist: np.ndarray) -> np.ndarray:
    """
    Полная копия CMS-логики:
    Берём 13*4 последних BX (3500..3500+4*13-1 = 3500..3551),
    группируем по поддетекторам HF (0..3),
    возвращаем pedestal[4].
    """
    n_sample = 13
    pedestal = np.zeros(4, dtype=np.float32)

    # последние 52 BX (3500..3551)
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
    Шаг 1: восстановление истинных рейтов (mu_true) через LSQ,
    затем ВЫЧИТАНИЕ ДИНАМИЧЕСКОГО ПЬЕДЕСТАЛА (как в оригинале CMS!),
    затем пересчёт bx, avg.
    """
    fills = np.unique(data["fillnum"])
    if fills.size != 1:
        raise ValueError(f"Expected exactly one fill in file, got {fills}")
    fill = int(fills[0])

    # --- загрузка HFSBR ---
    hfsbr_path = cfg.afterglow.hfsbr_pattern.format(fill=fill)
    if not os.path.exists(hfsbr_path):
        raise FileNotFoundError(f"HFSBR file not found: {hfsbr_path}")

    bx_to_clean = cfg.afterglow.bx_to_clean or []
    lambda_reg = cfg.afterglow.lambda_reg
    lambda_nonactive = cfg.afterglow.lambda_nonactive
    n_jobs = cfg.afterglow.n_jobs

    # --- строим LSQ-солвер ---
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

    # --- Основной LSQ-проход ---
    mu_true, ped_lsq = solver.apply_batch(
        bxraw_obs,
        n_jobs=n_jobs,
        desc=f"LSQ afterglow (fill {fill})",
    )

    # ------------------------------------------------------------------
    # ★ Новое: динамическое вычитание пьедестала (CMS-логика) ★
    # ------------------------------------------------------------------
    mu_corr = np.empty_like(mu_true, dtype=np.float32)

    for i in range(mu_true.shape[0]):
        hist = mu_true[i]
        ped = calculate_dynamic_pedestal(hist)

        # вычитаем pedestal для BX 0..3551 по схеме HF (0..3)
        # BX принадлежит поддету № (bx % 4)
        corr = hist - ped[np.arange(BX_LEN) % 4]
        mu_corr[i] = corr.astype(np.float32)

    # --- обновляем bxraw ---
    data["bxraw"] = mu_corr

    # ------------------------------------------------------------------
    # Пересчитываем bx и avg (эти величины уже должны быть ПОСЛЕ pedestal)
    # ------------------------------------------------------------------
    sigvis = cfg.afterglow.sigvis
    scale = 1.0 if not sigvis else 11245.6 / float(sigvis)

    bx_lumi = (mu_corr * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    return data

# ------------------ Шаг 2: fit Type1 ------------------

@log_step("compute_type1_step")
@timeit("compute_type1_step")
def compute_type1_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Шаг пайплайна: оценка коэффициентов Type1 для заданного fill.

    - Использует bxraw (уже восстановленные от afterglow/pedestal).
    - В качестве "avg" (SBIL) берёт:
        * data["sbil"], если есть,
        * иначе быстренько считает sbil = sum(bxraw * mask) / Nactive.
    - Для каждого offset из cfg.type1.offsets:
        * offset == 1 -> квадратичный фит (order=2)
        * offset >  1 -> линейный фит (order=1)
    - Сохраняет p0,p1,p2,offsets,orders в HDF5.
    """
    if "bxraw" not in data:
        raise KeyError("compute_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"compute_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # --- SBIL / avg для Type1 ---
    if "sbil" in data:
        avg = np.asarray(data["sbil"], dtype=np.float64)
    elif "avg" in data:
        avg = np.asarray(data["avg"], dtype=np.float64)
    else:
        # fallback: считаем SBIL сами
        mask = np.asarray(active_mask, dtype=np.int32)
        n_active = int(mask.sum())
        if n_active == 0:
            raise ValueError("compute_type1_step: active_mask has zero active BX")
        avg = (bxraw * mask[None, :]).sum(axis=1) / float(n_active)

    offsets = list(getattr(cfg.type1, "offsets", [1, 2, 3, 4]))
    sbil_min = float(getattr(cfg.type1, "sbil_min", 0.1))

    # --- вычисляем коэффициенты ---
    p0, p1, p2, orders = compute_type1_coeffs(
        bxraw=bxraw,
        avg=avg,
        active_mask=active_mask,
        offsets=offsets,
        sbil_min=sbil_min,
    )

    # --- куда сохраняем ---
    # если в конфиге есть явный путь — используем его
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        # по умолчанию кладём в поддиректорию "type1" рядом с репроцессингом,
        # но НЕ внутрь каждого фила
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

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
        "[compute_type1_step] fill %d: Type1 coeffs saved to %s (offsets=%s)",
        fill,
        path,
        offsets,
    )

    # сами данные пока не меняем
    return data


# ------------------ Шаг 3: apply Type1 ------------------

@log_step("apply_type1_step")
@timeit("apply_type1_step")
def apply_type1_step(data, cfg, active_mask: np.ndarray, fill: int):
    """
    Шаг пайплайна: применяет вычитание Type1 к bxraw.

    - читает коэффициенты из type1_coeffs_fill{fill}.h5
    - вызывает apply_type1_batch(bxraw, active_mask, p0, p1, p2)
    - обновляет data["bxraw"]
    """
    if "bxraw" not in data:
        raise KeyError("apply_type1_step: 'bxraw' not found in data")

    bxraw = np.asarray(data["bxraw"], dtype=np.float64)
    if bxraw.ndim != 2 or bxraw.shape[1] != BX_LEN:
        raise ValueError(
            f"apply_type1_step: bxraw has shape {bxraw.shape}, expected (T, {BX_LEN})"
        )

    # --- где искать Type1-coeffs ---
    type1_dir = getattr(cfg.io, "type1_dir", None)
    if type1_dir is None:
        type1_dir = os.path.join(cfg.io.output_dir, "type1")

    coeff_path = os.path.join(type1_dir, f"type1_coeffs_fill{fill}.h5")
    if not os.path.exists(coeff_path):
        raise FileNotFoundError(
            f"apply_type1_step: Type1 coeff file not found: {coeff_path}"
        )

    # --- читаем коэффициенты ---
    with h5py.File(coeff_path, "r") as h5:
        p0 = h5["p0"][:]
        p1 = h5["p1"][:]
        p2 = h5["p2"][:]
        # offsets и orders можно использовать для логов / sanity-чеков
        offsets = h5["offsets"][:]
        orders = h5["orders"][:]

    log.info(
        "[apply_type1_step] fill %d: loaded Type1 coeffs from %s (offsets=%s, orders=%s)",
        fill,
        coeff_path,
        list(offsets),
        list(orders),
    )

    # --- применяем вычитание Type1 ---
    corrected = apply_type1_batch(
        bxraw=bxraw,
        active_mask=active_mask,
        p0=p0,
        p1=p1,
        p2=p2,
    )

    data["bxraw"] = corrected
    return data


# ------------------ Основная точка входа для одного fill ------------------

@log_step("run_fill")
@timeit("run_fill")
def run_fill(fill: int, cfg: PipelineConfig) -> None:
    """
    Полный проход по одному fill:
      - читаем исходный HD5 (возможно, несколько файлов и несколько fillnum)
      - фильтруем строки только с fillnum == fill
      - загружаем activeBXMask
      - последовательно применяем включённые шаги
      - сохраняем результат в новый HD5
    """
    input_name = cfg.io.input_pattern.format(fill=fill)
    output_name = cfg.io.output_pattern.format(fill=fill)

    # --- activeBXMask из npy-файла ---
    if not cfg.io.active_mask_pattern:
        raise ValueError("io.active_mask_pattern is not set in config")

    mask_path = cfg.io.active_mask_pattern.format(fill=fill)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"ActiveBX mask not found: {mask_path}")

    active_mask = np.load(mask_path)
    active_mask = np.asarray(active_mask, dtype=np.int32)
    if active_mask.shape[0] != BX_LEN:
        raise ValueError(f"active_mask len={active_mask.shape[0]} != BX_LEN={BX_LEN}")

    # --- 1) загрузить исходные данные (могут быть несколько файлов и несколько fillnum) ---
    data = load_hd5_to_arrays(cfg.io.input_dir, input_name, node=cfg.io.node)

    # --- 1a) отфильтровать только текущий fill ---
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

        # применяем маску ко всем полям, у которых первая размерность совпадает
        for key, arr in data.items():
            if isinstance(arr, np.ndarray) and arr.shape[0] == n_before:
                data[key] = arr[mask]
    else:
        log.warning(
            "[run_fill] fill %d: 'fillnum' column not found in data; proceeding without fill filtering",
            fill,
        )

    # --- 2) шаги пайплайна ---
    if cfg.steps.restore_rates:
        data = restore_rates_step(data, cfg, active_mask)

    if cfg.steps.compute_type1:
        data = compute_type1_step(data, cfg, active_mask, fill)

    if cfg.steps.apply_type1:
        data = apply_type1_step(data, cfg, active_mask, fill)

    # --- 3) сохранить результат ---
    rows = arrays_to_rows(data)
    save_to_hd5(rows, node=cfg.io.node, path=cfg.io.output_dir, name=output_name)


def run_many_fills(cfg: PipelineConfig, fills: list[int]):
    """
    Запускает run_fill() для каждого fill.
    Любые исключения внутри run_fill — не критичны:
    - сохраняем fill в список failed_fills
    - идём дальше
    В конце выводим аккуратный отчёт.
    """
    failed = []

    for fill in fills:
        try:
            run_fill(fill, cfg)

        except FileNotFoundError as e:
            # Частый случай: нет маски, нет hd5, нет beam файла
            print(f"[WARN] Fill {fill} skipped: {e}")
            failed.append(fill)

        except Exception as e:
            # Любые другие ошибки — тоже пропускаем
            print(f"[ERROR] Fill {fill} failed with exception:")
            print(e)
            failed.append(fill)

    # Финальный отчёт
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