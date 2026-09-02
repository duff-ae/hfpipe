from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

from .hd5schema import BX_LEN

log = logging.getLogger("hfpipe")

try:
    from cffi import FFI
except Exception as e:
    raise RuntimeError(
        "online_recovery.py requires cffi. Install it with `pip install cffi`."
    ) from e


# ----------------------------------------------------------------------
# Exact online afterglow inverse
# ----------------------------------------------------------------------

_ffi = FFI()
_ffi.cdef(
    """
    void revert_afterglow(const int * activeBXMask, float * muHistPerBX, const float * HFSBR);
    """
)

_C = _ffi.verify(
    r"""
    void revert_afterglow(const int * activeBXMask, float * muHistPerBX, const float * HFSBR) {
        int ibx, jbx;
        int bx_len = 3564;
        int sbr_len = 3564;

        for (ibx = bx_len; ibx-- > 0; ) {
            if (activeBXMask[ibx] == 1) {
                for (jbx = ibx + 1; jbx < ibx + sbr_len; jbx++) {
                    if (jbx < bx_len) {
                        muHistPerBX[jbx] += muHistPerBX[ibx] * HFSBR[jbx - ibx];
                    } else {
                        muHistPerBX[jbx - bx_len] += muHistPerBX[ibx] * HFSBR[jbx - ibx];
                    }
                }
            }
        }
    }
    """,
    extra_compile_args=["-O3"],
)


@dataclass
class OnlineRecoveryResult:
    """
    Common output of both online-recovery methods.

    recovered_raw:
        Histogram before the online afterglow/pedestal corrections.
        Shape (T, BX_LEN).

    pedestal:
        Four per-row pedestal components, indexed by BX % 4.
        Shape (T, 4).
    """

    recovered_raw: np.ndarray
    pedestal: np.ndarray


# ----------------------------------------------------------------------
# Validation / conversion helpers
# ----------------------------------------------------------------------


def _validate_bx_hist_2d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != BX_LEN:
        raise ValueError(f"{name} has shape {arr.shape}, expected (T, {BX_LEN})")
    return arr


def _validate_bx_hist_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.shape[0] != BX_LEN:
        raise ValueError(f"{name} has shape {arr.shape}, expected ({BX_LEN},)")
    return arr


def _validate_active_mask(active_mask: np.ndarray) -> np.ndarray:
    active = np.asarray(active_mask, dtype=np.int32).ravel()
    if active.shape[0] != BX_LEN:
        raise ValueError(
            f"active_mask has length {active.shape[0]}, expected {BX_LEN}"
        )
    if not np.all((active == 0) | (active == 1)):
        raise ValueError("active_mask must contain only 0/1 values")
    return np.ascontiguousarray(active, dtype=np.int32)


def _validate_hfsbr(hfsbr: np.ndarray) -> np.ndarray:
    hfsbr = np.asarray(hfsbr, dtype=np.float32).ravel()
    if hfsbr.shape[0] < BX_LEN:
        raise ValueError(
            f"hfsbr length {hfsbr.shape[0]} is smaller than BX_LEN={BX_LEN}"
        )
    return np.ascontiguousarray(hfsbr, dtype=np.float32)


def _validate_zero_bx(zero_bx) -> np.ndarray:
    zero = np.asarray(tuple(zero_bx), dtype=np.int64).ravel()
    if zero.size < 4:
        raise ValueError(
            f"At least 4 artificial zero BX are required, got {zero.size}"
        )
    if np.any((zero < 0) | (zero >= BX_LEN)):
        raise ValueError(f"zero_bx contains values outside [0, {BX_LEN - 1}]")
    if np.unique(zero).size != zero.size:
        raise ValueError("zero_bx contains duplicates")
    return zero


def _as_c_float32_1d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return _validate_bx_hist_1d(name, arr)


def _call_revert_afterglow_inplace(
    active_mask_i32: np.ndarray,
    mu_hist_f32: np.ndarray,
    hfsbr_f32: np.ndarray,
) -> None:
    _C.revert_afterglow(
        _ffi.cast("const int *", _ffi.from_buffer(active_mask_i32)),
        _ffi.cast("float *", _ffi.from_buffer(mu_hist_f32)),
        _ffi.cast("const float *", _ffi.from_buffer(hfsbr_f32)),
    )


def apply_revert_afterglow_batch(
    hist: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Apply the exact C-style afterglow inverse independently to each row."""

    hist = _validate_bx_hist_2d("hist", np.asarray(hist, dtype=np.float32))
    hfsbr_f32 = _validate_hfsbr(hfsbr)
    active_i32 = _validate_active_mask(active_mask)

    out = np.empty_like(hist, dtype=np.float32)
    for i in range(hist.shape[0]):
        row = _as_c_float32_1d("hist row", hist[i]).copy()
        _call_revert_afterglow_inplace(active_i32, row, hfsbr_f32)
        out[i] = row
    return out


# ----------------------------------------------------------------------
# HFSBR loading
# ----------------------------------------------------------------------


def load_hfsbr_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HFSBR file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        return _validate_hfsbr(arr)

    if suffix in (".txt", ".dat"):
        text = path.read_text()
        text = text.replace("[", " ").replace("]", " ")
        values = []
        for tok in re.split(r"[,\s]+", text):
            tok = tok.strip()
            if not tok:
                continue
            try:
                values.append(float(tok))
            except ValueError:
                continue
        if not values:
            raise RuntimeError(f"HFSBR file {path} did not contain numeric values")
        return _validate_hfsbr(np.asarray(values, dtype=np.float32))

    if suffix in (".h5", ".hd5"):
        with h5py.File(path, "r") as h5:
            if "hfsbr" in h5:
                return _validate_hfsbr(np.asarray(h5["hfsbr"][:], dtype=np.float32))
            for obj in h5.values():
                if hasattr(obj, "shape"):
                    return _validate_hfsbr(np.asarray(obj[...], dtype=np.float32))
        raise RuntimeError(f"Could not find an HFSBR dataset in {path}")

    raise RuntimeError(f"Unsupported HFSBR file format: {path}")


# ----------------------------------------------------------------------
# Method A: authoritative recovery from saved online tables
# ----------------------------------------------------------------------


def reconstruct_from_tables_batch(
    bxraw_final: np.ndarray,
    pedestal_4: np.ndarray,
    afterglow_frac: np.ndarray,
    *,
    zero_bx=(3553, 3554, 3555, 3556, 3557),
) -> OnlineRecoveryResult:
    """
    Authoritative table-based recovery.

      pre_pedestal = bxraw_final + pedestal[BX % 4]
      recovered_raw = pre_pedestal / afterglow_frac

    The artificial zero BX have no valid afterglow fraction in the saved
    table. They are known by construction to be exactly zero in recovered_raw.
    Any other invalid fraction is treated as an error rather than silently
    creating a corrupted histogram.
    """

    bxraw_final = _validate_bx_hist_2d(
        "bxraw_final", np.asarray(bxraw_final, dtype=np.float64)
    )
    pedestal_4 = np.asarray(pedestal_4, dtype=np.float64)
    afterglow_frac = np.asarray(afterglow_frac, dtype=np.float64)

    T = bxraw_final.shape[0]
    if pedestal_4.shape != (T, 4):
        raise ValueError(f"pedestal_4 shape {pedestal_4.shape}, expected ({T}, 4)")
    if afterglow_frac.shape != (T, BX_LEN):
        raise ValueError(
            f"afterglow_frac shape {afterglow_frac.shape}, expected ({T}, {BX_LEN})"
        )

    zero = _validate_zero_bx(zero_bx)
    zero_mask = np.zeros(BX_LEN, dtype=bool)
    zero_mask[zero] = True

    idx_mod4 = np.arange(BX_LEN) % 4
    pre_pedestal = bxraw_final + pedestal_4[:, idx_mod4]

    valid = np.isfinite(afterglow_frac) & (afterglow_frac > 0.0)
    invalid_outside_zero = (~valid) & (~zero_mask[None, :])
    if np.any(invalid_outside_zero):
        rows, bx = np.where(invalid_outside_zero)
        preview = list(zip(rows[:10].tolist(), bx[:10].tolist()))
        raise RuntimeError(
            "Invalid hfafterglowfrac outside configured artificial zero BX. "
            f"Count={len(rows)}, first (row, BX)={preview}"
        )

    recovered = np.zeros_like(pre_pedestal, dtype=np.float64)
    recovered[valid] = pre_pedestal[valid] / afterglow_frac[valid]
    recovered[:, zero] = 0.0

    return OnlineRecoveryResult(
        recovered_raw=recovered.astype(np.float32),
        pedestal=pedestal_4.astype(np.float32),
    )


# ----------------------------------------------------------------------
# Method B: recovery without saved tables
# ----------------------------------------------------------------------


class OnlineRecoverySolver:
    """
    Recover the four per-row pedestal values from the artificial zero BX,
    then invert the online afterglow operation.

    The response matrix depends only on HFSBR, active mask and zero-BX
    positions, so it is constructed once per fill. The RHS and therefore the
    fitted pedestal are recomputed independently for every histogram row.
    """

    def __init__(
        self,
        hfsbr: np.ndarray,
        active_mask: np.ndarray,
        *,
        zero_bx=(3553, 3554, 3555, 3556, 3557),
    ) -> None:
        self.hfsbr = _validate_hfsbr(hfsbr)
        self.active_mask = _validate_active_mask(active_mask)
        self.zero_bx = _validate_zero_bx(zero_bx)

        self.response = self._build_pedestal_response().astype(np.float64)
        self.A = self.response[:, self.zero_bx].T

        self.rank = int(np.linalg.matrix_rank(self.A))
        self.condition = float(np.linalg.cond(self.A))
        if self.rank < 4:
            raise RuntimeError(
                "Artificial-zero pedestal system has rank < 4: "
                f"shape={self.A.shape}, rank={self.rank}"
            )

        self.A_pinv = np.linalg.pinv(self.A)

        log.info(
            "[online_recovery] zero-BX pedestal system shape=%s rank=%d condition=%.6e",
            self.A.shape,
            self.rank,
            self.condition,
        )

    def _build_pedestal_response(self) -> np.ndarray:
        response = np.empty((4, BX_LEN), dtype=np.float32)
        bx_mod4 = np.arange(BX_LEN) % 4

        for k in range(4):
            row = np.ascontiguousarray((bx_mod4 == k).astype(np.float32))
            _call_revert_afterglow_inplace(self.active_mask, row, self.hfsbr)
            response[k] = row

        return response

    def recover_batch(self, bxraw_final: np.ndarray) -> OnlineRecoveryResult:
        bxraw_final = _validate_bx_hist_2d(
            "bxraw_final", np.asarray(bxraw_final, dtype=np.float32)
        )

        # R(final) is row dependent and therefore evaluated for every row.
        base = apply_revert_afterglow_batch(
            bxraw_final,
            hfsbr=self.hfsbr,
            active_mask=self.active_mask,
        ).astype(np.float64)

        # For every row i solve
        #     A p_i = -R(final_i)[zero_bx]
        rhs = -base[:, self.zero_bx]
        pedestal = rhs @ self.A_pinv.T

        # By linearity:
        # R(final + pedestal_pattern) = R(final) + p @ R(pattern_k)
        recovered = base + pedestal @ self.response

        # These bins are software-injected zeros before any online operation;
        # enforce the known exact state explicitly in the recovered histogram.
        recovered[:, self.zero_bx] = 0.0

        return OnlineRecoveryResult(
            recovered_raw=recovered.astype(np.float32),
            pedestal=pedestal.astype(np.float32),
        )
