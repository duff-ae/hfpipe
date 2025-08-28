import tables as tb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from typing import Tuple, Sequence

BX = 3564
META_COLS = (
    "fillnum","runnum","lsnum","nbnum",
    "timestampsec","timestampmsec",
    "totsize","publishnnb","datasourceid","algoid","channelid",
    "avgraw","avg","maskhigh","masklow","calibtag"
)

def read_fill_arrays(path: str, fill: int, node: str) -> Tuple[pd.DataFrame, np.ndarray]:
    metas, bx_list = [], []
    for f in sorted(Path(path, str(fill)).glob("*.hd5")):
        with tb.open_file(f, "r") as h5:
            tab = h5.root[node]
            rec = tab.read()             # structured ndarray
            if rec.size == 0: continue
            bx = np.asarray(rec["bxraw"], dtype=np.float32)  # (N, 3564)
            bx_list.append(bx)
            meta = {c: rec[c] for c in META_COLS if c in rec.dtype.names}
            metas.append(pd.DataFrame(meta))
    if not bx_list:
        return pd.DataFrame(columns=META_COLS), np.empty((0, BX), np.float32)
    meta = pd.concat(metas, ignore_index=True)
    bxraw = np.ascontiguousarray(np.vstack(bx_list), dtype=np.float32)
    return meta, bxraw

def save_fill_arrays(meta: pd.DataFrame, bxraw: np.ndarray, node: str, out_path: str, name: str, desc_cls):
    import os
    os.makedirs(out_path, exist_ok=True)
    filters = tb.Filters(complevel=9, complib="blosc")
    with tb.open_file(f"{out_path}/{name}", mode="w") as h5:
        tab = h5.create_table("/", node, desc_cls, filters=filters, chunkshape=(1024,))
        dtype = tab.dtype
        N = bxraw.shape[0]
        arr = np.zeros(N, dtype=dtype)
        for col in dtype.names:
            if col == "bxraw":
                arr[col] = bxraw
            elif col in meta.columns:
                arr[col] = meta[col].to_numpy()
        tab.append(arr)

META_KEYS = ("fillnum","runnum","lsnum","nbnum","timestampsec","timestampmsec")

def _read_node(h5: tb.File, node: str):
    return h5.root[node].read()

def read_fill_field(path: str,
                    fill: int,
                    node: str,
                    field_candidates: Sequence[str] = ("bxraw","data","bx")
                   ) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Читает любой узел и возвращает:
      meta: DataFrame с ключами (fillnum, runnum, lsnum, nbnum, ...)
      arr:  np.ndarray shape (N, M), где M=3564 или 4 (или другое)
      field: имя найденного поля (напр. "data" или "bxraw")
    """
    metas, arrays = [], []
    chosen_field = None

    for f in sorted(Path(path, str(fill)).glob("*.hd5")):
        with tb.open_file(f, "r") as h5:
            rec = _read_node(h5, node)
            if rec.size == 0:
                continue
            # выбрать существующее поле
            if chosen_field is None:
                for cand in field_candidates:
                    if cand in rec.dtype.names:
                        chosen_field = cand
                        break
                if chosen_field is None:
                    raise ValueError(f"No expected field {field_candidates} in node '{node}' of file {f}")

            arr = np.asarray(rec[chosen_field])
            # гарантировать 2D (N, M)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)

            # собрать метаданные
            meta_cols = {k: rec[k] for k in META_KEYS if k in rec.dtype.names}
            metas.append(pd.DataFrame(meta_cols))

    if not arrays:
        return pd.DataFrame(columns=META_KEYS), np.empty((0, 0)), chosen_field or ""

    meta = pd.concat(metas, ignore_index=True)
    arr  = np.ascontiguousarray(np.vstack(arrays))
    return meta, arr, chosen_field

def align_to_lumi(lumi_meta: pd.DataFrame,
                  other_meta: pd.DataFrame,
                  other_arr: np.ndarray,
                  default_row: np.ndarray) -> np.ndarray:
    """
    Возвращает other_arr, выровненный по порядку lumi_meta (left-join по ключам).
    Если строка не нашлась — подставляет default_row.
    """
    keys = ["fillnum","runnum","lsnum","nbnum"]
    idx = np.arange(len(other_meta), dtype=np.int64)
    other_idxed = other_meta[keys].copy()
    other_idxed["_ix"] = idx
    merged = lumi_meta[keys].merge(other_idxed, on=keys, how="left")

    out = np.repeat(default_row[None, :], len(lumi_meta), axis=0)
    val_mask = merged["_ix"].notna().to_numpy()
    if val_mask.any():
        take = merged.loc[val_mask, "_ix"].to_numpy(dtype=int)
        out[val_mask] = other_arr[take]
    return out