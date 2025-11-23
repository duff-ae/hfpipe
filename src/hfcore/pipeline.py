# src/hfcore/pipeline.py
from __future__ import annotations

import os
from typing import List

import numpy as np

from .decorators import log_step, timeit
from .io import load_hd5_to_arrays, arrays_to_rows, save_to_hd5
from .hd5schema import BX_LEN
from .afterglow_lsq import build_afterglow_solver_from_file
from .type1_fit import compute_type1_coeffs, save_type1_coeffs
from .type1_apply import apply_type1_batch, load_type1_coeffs
from .config import PipelineConfig


# ------------------ Шаг 1: afterglow / восстановление mu_true ------------------

@log_step("restore_rates")
@timeit("restore_rates")
def restore_rates_step(data: dict, cfg: PipelineConfig, active_mask: np.ndarray) -> dict:
    """
    Шаг 1: восстановление истинных рейтов по BX (mu_true) из наблюдаемых bxraw.

    В выходном файле:
      - data["bxraw"] = восстановленный mu_true (после afterglow LSQ)
      - data["bx"]    = data["bxraw"] * (11245.6 / sigvis)
      - data["avg"]   = средняя светимость по BX (по bx)
    """
    fills = np.unique(data["fillnum"])
    if fills.size != 1:
        raise ValueError(f"Expected exactly one fill in file, got {fills}")
    fill = int(fills[0])

    if not cfg.afterglow.hfsbr_pattern:
        raise ValueError("afterglow.hfsbr_pattern is not set in config")
    hfsbr_path = cfg.afterglow.hfsbr_pattern.format(fill=fill)
    if not os.path.exists(hfsbr_path):
        raise FileNotFoundError(f"HFSBR file not found: {hfsbr_path}")

    bx_to_clean = cfg.afterglow.bx_to_clean or []
    lambda_reg = cfg.afterglow.lambda_reg
    lambda_nonactive = cfg.afterglow.lambda_nonactive
    n_jobs = cfg.afterglow.n_jobs

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

    mu_true, ped = solver.apply_batch(
        bxraw_obs,
        n_jobs=n_jobs,
        desc=f"LSQ afterglow (fill {fill})",
    )

    # обновляем bxraw (это теперь mu_true)
    data["bxraw"] = mu_true.astype(np.float32, copy=False)

    # пересчитываем bx и avg
    sigvis = cfg.afterglow.sigvis
    if sigvis is None or sigvis == 0.0:
        scale = 1.0
    else:
        scale = 11245.6 / float(sigvis)

    bx_lumi = (data["bxraw"] * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    return data


# ------------------ Шаг 2: fit Type1 ------------------

@log_step("compute_type1")
@timeit("compute_type1")
def compute_type1_step(data: dict, cfg: PipelineConfig, active_mask: np.ndarray, fill: int) -> dict:
    """
    Шаг 2: расчёт остаточных Type1 и сохранение коэффициентов в type1_coeffs_fill{fill}.h5.
    """
    offsets = cfg.type1.offsets or []
    if not offsets:
        return data

    bxraw = data["bxraw"]
    avg = data["avg"]

    p0, p1, p2 = compute_type1_coeffs(
        bxraw=bxraw,
        avg=avg,
        active_mask=active_mask,
        offsets=offsets,
        sbil_min=cfg.type1.sbil_min,
        order=cfg.type1.order,
    )

    save_type1_coeffs(
        fill=fill,
        output_dir=cfg.io.output_dir,
        p0=p0,
        p1=p1,
        p2=p2,
        offsets=offsets,
        order=cfg.type1.order,
    )

    return data


# ------------------ Шаг 3: apply Type1 ------------------

@log_step("apply_type1")
@timeit("apply_type1")
def apply_type1_step(data: dict, cfg: PipelineConfig, active_mask: np.ndarray, fill: int) -> dict:
    """
    Шаг 3: применение Type1 к bxraw и пересчёт bx/avg.
    """
    p0, p1, p2 = load_type1_coeffs(fill, cfg.io.output_dir)

    bxraw = data["bxraw"]
    bxraw_corr = apply_type1_batch(
        bxraw=bxraw,
        active_mask=active_mask,
        p0=p0,
        p1=p1,
        p2=p2,
    )

    data["bxraw"] = bxraw_corr.astype(np.float32, copy=False)

    sigvis = cfg.afterglow.sigvis
    if sigvis is None or sigvis == 0.0:
        scale = 1.0
    else:
        scale = 11245.6 / float(sigvis)

    bx_lumi = (data["bxraw"] * scale).astype(np.float32, copy=False)
    data["bx"] = bx_lumi
    data["avg"] = bx_lumi.mean(axis=1).astype(np.float32)

    return data


# ------------------ Основная точка входа для одного fill ------------------

@log_step("run_fill")
@timeit("run_fill")
def run_fill(fill: int, cfg: PipelineConfig) -> None:
    """
    Полный проход по одному fill:
      - читаем исходный HD5
      - загружаем activeBXMask
      - последовательно применяем включённые шаги
      - сохраняем результат в новый HD5
    """
    input_name = cfg.io.input_pattern.format(fill=fill)
    output_name = cfg.io.output_pattern.format(fill=fill)

    # activeBXMask из npy-файла
    if not cfg.io.active_mask_pattern:
        raise ValueError("io.active_mask_pattern is not set in config")
    mask_path = cfg.io.active_mask_pattern.format(fill=fill)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"ActiveBX mask not found: {mask_path}")
    active_mask = np.load(mask_path)
    active_mask = np.asarray(active_mask, dtype=np.int32)
    if active_mask.shape[0] != BX_LEN:
        raise ValueError(f"active_mask len={active_mask.shape[0]} != BX_LEN={BX_LEN}")

    # 1) загрузить исходные данные
    data = load_hd5_to_arrays(cfg.io.input_dir, input_name, node=cfg.io.node)

    # 2) шаги пайплайна
    if cfg.steps.restore_rates:
        data = restore_rates_step(data, cfg, active_mask)

    if cfg.steps.compute_type1:
        data = compute_type1_step(data, cfg, active_mask, fill)

    if cfg.steps.apply_type1:
        data = apply_type1_step(data, cfg, active_mask, fill)

    # 3) сохранить результат
    rows = arrays_to_rows(data)
    save_to_hd5(rows, node=cfg.io.node, path=cfg.io.output_dir, name=output_name)


@log_step("run_many_fills")
@timeit("run_many_fills")
def run_many_fills(cfg: PipelineConfig, fills: List[int] | None = None) -> None:
    """
    Запуск пайплайна для нескольких fills.
    Если fills=None, берём cfg.fills из YAML.
    """
    if fills is None:
        fills = cfg.fills

    for fill in fills:
        run_fill(fill, cfg)