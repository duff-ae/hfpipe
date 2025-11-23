# src/hfcore/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class StepsConfig:
    restore_rates: bool = True
    compute_type1: bool = True
    apply_type1: bool = True


@dataclass
class IOConfig:
    input_dir: str
    input_pattern: str
    output_dir: str
    output_pattern: str
    node: str = "lumi"
    active_mask_pattern: str = ""  # "/path/to/activeBXMask_fill{fill}.npy"


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
    sbil_min: float = 0.1
    order: int = 1
    offsets: Optional[List[int]] = None


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
