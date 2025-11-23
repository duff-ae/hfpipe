from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class DataCfg:
    file_path: str
    beam_path: str
    node: str
    fills: List[int]

@dataclass
class RevertCfg:
    enable: bool = True
    afterglow_node: str = "hfafterglowfrac"
    pedestal_node:  str = "hfEtPedestal"

@dataclass
class BatchLSQCfg:
    enable: bool = True
    hfsbr_file: str = ""
    lambda_reg: float = 0.01
    lambda_nonactive: float = 0.05
    lambda_bx1: float = 1.0
    use_cubic: bool = True
    n_jobs: int = -1
    backend: str = "processes"     # "processes" | "threads"
    p0_guess: List[float] = field(default_factory=lambda: [0,0,0,0])

@dataclass
class PipelineCfg:
    revert_online: RevertCfg = RevertCfg()
    batch_afterglow_lsq: BatchLSQCfg = BatchLSQCfg()

@dataclass
class SbrCfg:
    enable: bool = False

@dataclass
class PlotCfg:
    per_bx: bool = True
    instant: bool = True

@dataclass
class IOCfg:
    save_hd5: bool = False
    out_dir: str = "./out/"

@dataclass
class FiltersCfg:
    status_in: Optional[List[str]] = None  # e.g. ["STABLE BEAMS"] or ["ADJUST","STABLE BEAMS"]

@dataclass
class AppCfg:
    data: DataCfg
    pipeline: PipelineCfg
    sbr: SbrCfg = SbrCfg()
    plot: PlotCfg = PlotCfg()
    io: IOCfg = IOCfg()
    filters: FiltersCfg = FiltersCfg()

def load_config(path: str) -> AppCfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    revert = RevertCfg(**raw["pipeline"].get("revert_online", {}))
    batch  = BatchLSQCfg(**raw["pipeline"].get("batch_afterglow_lsq", {}))

    return AppCfg(
        data=DataCfg(**raw["data"]),
        pipeline=PipelineCfg(revert_online=revert, batch_afterglow_lsq=batch),
        sbr=SbrCfg(**raw.get("sbr", {})),
        plot=PlotCfg(**raw.get("plot", {})),
        io=IOCfg(**raw.get("io", {})),
        filters=FiltersCfg(**raw.get("filters", {})),
    )
