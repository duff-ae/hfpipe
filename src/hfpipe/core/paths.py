import os
from pathlib import Path
import numpy as np
from importlib import resources

def resolve_input(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    local = Path("data") / rel
    if local.exists():
        return local
    try:
        pkg_root = resources.files("hfpipe") / "data" / "reference"
        cand = pkg_root / rel
        if cand.exists():
            return Path(cand)
    except Exception:
        pass
    base = os.getenv("HFPIPE_DATA")
    if base:
        cand = Path(base) / rel
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Input not found: {rel}")

def load_hfsbr(path_or_rel: str) -> np.ndarray:
    p = resolve_input(path_or_rel)
    return np.loadtxt(p, delimiter=",", dtype=np.float32)

