# src/hfcore/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class StepsConfig:
    # 1) восстановление истинных bxraw (afterglow LSQ)
    restore_rates: bool = False

    # 2) расчёт коэффициентов Type1 (компактные p0/p1/p2)
    compute_type1: bool = False

    # 3) толстый анализ Type1 (HDF5 + PNG, как старый calculate_type1)
    analyze_type1: bool = False

    # 4) вычитание Type1 из bxraw (когда допишем apply_type1_step)
    apply_type1: bool = False


@dataclass
class IOConfig:
    input_dir: str
    beam_dir: str
    input_pattern: str
    output_dir: str
    output_pattern: str
    node: str = "hfetlumi"
    active_mask_pattern: str = ""  # "/path/to/activeBXMask_fill{fill}.npy"
    type1_dir: Optional[str] = None


@dataclass
class AfterglowConfig:
    lambda_reg: float = 0.01
    lambda_nonactive: float = 0.05
    bx_to_clean: Optional[List[int]] = None
    hfsbr_pattern: Optional[str] = None
    n_jobs: int = -1
    sigvis: Optional[float] = None



@dataclass
class Type1Config:
    # Порог по SBIL (avg/sbil) для отбора точек
    sbil_min: float = 0.1

    # Базовый порядок для компактных коэффициентов (compute_type1_coeffs).
    # В анализаторе мы всё равно делаем: offset=1 -> 2, остальные -> 1.
    order: int = 1

    # Список сдвигов BX: j = i + offset
    offsets: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Рисовать PNG в analyze_type1_step
    make_plots: bool = False

    debug: bool = False
    debug_after_apply: bool = False
    save_hd5: bool = False

@dataclass
class PipelineConfig:
    io: IOConfig
    steps: StepsConfig
    afterglow: AfterglowConfig
    type1: Type1Config
    fills: List[int]


def load_config(path: str) -> PipelineConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    io = IOConfig(**cfg_dict["io"])
    steps = StepsConfig(**cfg_dict.get("steps", {}))
    afterglow = AfterglowConfig(**cfg_dict.get("afterglow", {}))
    type1 = Type1Config(**cfg_dict.get("type1", {}))
    fills = cfg_dict.get("fills", [])

    return PipelineConfig(
        io=io,
        steps=steps,
        afterglow=afterglow,
        type1=type1,
        fills=fills,
    )
