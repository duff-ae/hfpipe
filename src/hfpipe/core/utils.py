import numpy as np
from ..io.beam import BX

def ensure(a, shape=None, dtype=None):
    out = np.ascontiguousarray(a, dtype=dtype) if dtype is not None else np.ascontiguousarray(a)
    if shape is not None and tuple(out.shape) != tuple(shape):
        raise ValueError(f"expected {shape}, got {out.shape}")
    return out

def nan_to_num_inplace(a: np.ndarray):
    np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def derive_mask_from_lumi(bxraw: np.ndarray) -> np.ndarray:
    """
    Грубая, но устойчивая оценка маски из самих lumi-гистограмм:
      1) берём средний по всем записям ряд;
      2) оцениваем базовую 'тёмную' компоненту как медиану вне хвоста;
      3) активные BX = где mean > baseline + k*IQR (k ~ 6).
    """
    if bxraw.size == 0:
        return np.zeros(BX, np.int32)
    mean_hist = bxraw.mean(axis=0)
    # робастная базовая линия
    q1, q3 = np.quantile(mean_hist, [0.25, 0.75])
    iqr = max(q3 - q1, 1e-9)
    baseline = q1
    k = 6.0
    mask = (mean_hist > baseline + k * iqr).astype(np.int32)
    # немного «зачистим» лазер/хвост, если вдруг в них всплески
    for bx in [3488, 3489, 3490, 3491, 3553, 3554, 3555, 3556, 3557]:
        mask[bx] = 0
    return mask

