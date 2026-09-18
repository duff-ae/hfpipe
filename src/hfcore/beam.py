from __future__ import annotations

import glob
import os
from typing import Iterable, Optional

import numpy as np
import tables

from .hd5schema import open_hd5


def decode_status(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def iter_beam_files_for_fill(beam_path: str, fill: int) -> list[str]:
    """Return sorted ``<beam_path>/<fill>/*.hd5`` files."""
    return sorted(glob.glob(os.path.join(beam_path, str(fill), "*.hd5")))


def load_beam_fill(
    beam_path: str,
    fill: int,
    node: str = "beam",
    columns: Optional[Iterable[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Load the beam table for one fill.

    Beam data stays completely outside the normal HF production pipeline; only
    standalone diagnostics that need beam intensity pay the I/O cost.
    """
    paths = iter_beam_files_for_fill(beam_path, fill)
    if not paths:
        raise FileNotFoundError(
            f"No beam files found for fill {fill}: {beam_path}/{fill}/*.hd5"
        )

    wanted = set(columns) if columns is not None else None
    pieces: dict[str, list[np.ndarray]] = {}
    found_node = False

    for path in paths:
        h5 = open_hd5(path, mode="r")
        try:
            if not hasattr(h5.root, node):
                continue
            found_node = True
            table: tables.Table = getattr(h5.root, node)
            names = list(table.coldescrs.keys())
            if wanted is not None:
                names = [name for name in names if name in wanted]
            if not names:
                continue
            # Read requested columns individually.  In particular this avoids
            # pulling large per-BX fields such as `collidable` when an
            # after-dump diagnostic only needs timestamps + total intensities.
            for name in names:
                pieces.setdefault(name, []).append(np.asarray(table.col(name)))
        finally:
            h5.close()

    if not found_node:
        raise RuntimeError(f"Node '/{node}' was not found in beam files for fill {fill}")
    if not pieces:
        available = []
        for path in paths[:1]:
            with open_hd5(path, mode="r") as h5:
                if hasattr(h5.root, node):
                    available = list(getattr(h5.root, node).coldescrs.keys())
        raise RuntimeError(
            f"None of the requested beam columns were found. Available columns: {available}"
        )

    out = {name: np.concatenate(parts, axis=0) for name, parts in pieces.items()}
    if "fillnum" in out:
        sel = np.asarray(out["fillnum"], dtype=np.int64) == int(fill)
        if np.any(sel) and not np.all(sel):
            out = {k: v[sel] for k, v in out.items()}
    return out


def timestamp_seconds(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build floating-point UNIX-like seconds from timestampsec/msec columns."""
    if "timestampsec" not in data:
        raise KeyError("data has no 'timestampsec' column")
    sec = np.asarray(data["timestampsec"], dtype=np.float64)
    if "timestampmsec" in data:
        sec = sec + np.asarray(data["timestampmsec"], dtype=np.float64) / 1000.0
    return sec


def _scalar_series(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float64)
    raise ValueError(
        f"Beam column {name!r} has shape {arr.shape}; expected one scalar value per row"
    )


def align_beam_columns_to_lumi(
    lumi: dict[str, np.ndarray],
    beam: dict[str, np.ndarray],
    columns: Iterable[str] = ("intensity1", "intensity2"),
    max_dt: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Nearest-time alignment of beam scalars to HF rows.

    The beam and HF tables are intentionally loaded independently.  Alignment
    uses their timestamps and therefore does not require merging beam columns
    into the large luminosity dataset.
    """
    lumi_t = timestamp_seconds(lumi)
    beam_t = timestamp_seconds(beam)
    if beam_t.size == 0:
        raise ValueError("Beam table is empty")

    order = np.argsort(beam_t)
    bt = beam_t[order]
    pos = np.searchsorted(bt, lumi_t, side="left")
    right = np.clip(pos, 0, bt.size - 1)
    left = np.clip(pos - 1, 0, bt.size - 1)
    choose_right = np.abs(bt[right] - lumi_t) < np.abs(bt[left] - lumi_t)
    nearest = np.where(choose_right, right, left)
    dt = np.abs(bt[nearest] - lumi_t)

    out: dict[str, np.ndarray] = {"dt": dt}
    for name in columns:
        if name not in beam:
            raise KeyError(f"Beam table has no {name!r} column")
        values = _scalar_series(beam[name], name)[order]
        aligned = values[nearest]
        if max_dt is not None:
            aligned = aligned.astype(np.float64, copy=True)
            aligned[dt > max_dt] = np.nan
        out[name] = aligned
    return out