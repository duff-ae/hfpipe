import tables as tb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

BX = 3564
KEYS = ["fillnum","runnum","lsnum","nbnum"]

# ---------- utils ----------
def _decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x

def _find_beam_table(h5: tb.File, preferred: Optional[str]) -> Optional[tb.table.Table]:
    # 1) если явно указан узел
    if preferred:
        try:
            node = h5.get_node(f"/{preferred}")
            if isinstance(node, tb.table.Table) and "collidable" in node.colnames:
                return node
        except tb.NoSuchNodeError:
            pass
    # 2) поиск таблицы с полем collidable
    cand = None
    best_score = -1
    for t in h5.walk_nodes("/", classname="Table"):
        cols = set(t.colnames)
        if "collidable" in cols:
            score = 0
            if "ncollidable" in cols or "nCollidable" in cols: score += 2
            if "status" in cols: score += 1
            if score > best_score:
                best_score, cand = score, t
    return cand

def _read_all_rows(table: tb.table.Table) -> pd.DataFrame:
    rec = table.read()
    if rec.size == 0:
        return pd.DataFrame()
    cols = set(rec.dtype.names)
    # собрать ключи, статус, ncollidable, timestampsec, collidable
    data: Dict[str, Any] = {}
    for k in KEYS:
        data[k] = rec[k] if k in cols else np.full(rec.size, -1, dtype=np.int64)
    data["timestampsec"] = rec["timestampsec"] if "timestampsec" in cols else np.full(rec.size, -1, dtype=np.int64)
    data["status"] = np.array([_decode(s) for s in rec["status"]]) if "status" in cols else np.array([""]*rec.size)
    if "ncollidable" in cols: data["ncollidable"] = rec["ncollidable"]
    elif "nCollidable" in cols: data["ncollidable"] = rec["nCollidable"]
    else: data["ncollidable"] = np.full(rec.size, -1, dtype=np.int64)
    # collidable: (N,3564) или список объектов
    coll = rec["collidable"]
    coll = np.asarray([np.asarray(c).astype(np.int32, copy=False).reshape(-1) for c in coll])
    df = pd.DataFrame({k: data[k] for k in data if k != "status"})
    df["status"] = data["status"]
    df["collidable"] = list(coll)  # как список векторов длины BX
    return df

def read_beam_fill(beam_path: str, fill: int, beam_node: Optional[str] = None) -> pd.DataFrame:
    """Собрать beam DF по всему fill (объединить все файлы)."""
    dfs = []
    for f in sorted(Path(beam_path, str(fill)).glob("*.hd5")):
        with tb.open_file(f, "r") as h5:
            t = _find_beam_table(h5, preferred=beam_node)
            if t is None: continue
            df = _read_all_rows(t)
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=KEYS+["timestampsec","status","ncollidable","collidable"])
    beam = pd.concat(dfs, ignore_index=True)
    # оставить только валидные векторы
    beam = beam[beam["collidable"].map(lambda v: isinstance(v, np.ndarray) and v.shape==(BX,))]
    return beam.reset_index(drop=True)

# ---------- глобальная маска fill ----------
def pick_fill_mask(beam: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Выбрать одну 'лучшее догадку' маску для всего fill."""
    if beam.empty:
        return np.zeros(BX, np.int32), 0
    # 1) STABLE BEAMS
    stable = beam[beam["status"]=="STABLE BEAMS"]
    if not stable.empty:
        row = stable.iloc[0]
        m = row["collidable"].astype(np.int32, copy=False)
        return m, int(m.sum())
    # 2) максимум ncollidable
    if (beam["ncollidable"]>=0).any():
        row = beam.iloc[int(beam["ncollidable"].argmax())]
        m = row["collidable"].astype(np.int32, copy=False)
        return m, int(m.sum())
    # 3) «бОльшинство голосов»
    arr = np.stack(beam["collidable"].to_numpy().tolist()).astype(np.float32)
    m = (arr.mean(axis=0) >= 0.5).astype(np.int32)
    return m, int(m.sum())

# ---------- выравнивание к lumi ----------
def align_masks_to_lumi(lumi_meta: pd.DataFrame, beam: pd.DataFrame,
                        fallback_mask: np.ndarray) -> np.ndarray:
    """
    Вернёт маски per-record (N,3564). При отсутствии подходящего beam — просто повторит fallback.
    Алгоритм:
      1) Мерж по ключам (fill,run,ls,nb).
      2) Где не нашли — ищем ближайший по timestampsec (нестрогий).
      3) Остальное — fallback.
    """
    N = len(lumi_meta)
    if N == 0:
        return np.zeros((0,BX), np.int32)
    out = np.repeat(fallback_mask[None,:], N, axis=0)

    if beam.empty:
        return out

    # 1) мерж по ключам
    k = KEYS
    b_small = beam[k + ["timestampsec","collidable"]].copy()
    b_small["__ix"] = np.arange(len(b_small))
    merged = lumi_meta[k].merge(b_small[k+["__ix"]], on=k, how="left")
    exact = merged["__ix"].notna().to_numpy()
    if exact.any():
        take = merged.loc[exact, "__ix"].to_numpy(int)
        out[exact] = np.stack(b_small.loc[take, "collidable"].to_numpy()).astype(np.int32)

    # 2) ближайший по timestampsec (если есть у обоих)
    have_ts = ("timestampsec" in lumi_meta.columns) and ("timestampsec" in beam.columns)
    if have_ts:
        lumi_ts = lumi_meta["timestampsec"].to_numpy()
        beam_ts = beam["timestampsec"].to_numpy()
        if np.all(lumi_ts >= 0) and np.all(beam_ts >= 0) and beam_ts.size > 0:
            # для каждой незаполненной строки возьмём ближайший timestamp
            miss = ~exact
            if miss.any():
                ts = lumi_ts[miss]
                # быстрее всего через argmin по |ts - beam_ts|
                # (для больших N можно заменить на бинарный поиск по отсортированному beam_ts)
                diffs = np.abs(ts[:,None] - beam_ts[None,:])
                j = diffs.argmin(axis=1)
                out[miss] = np.stack(beam.iloc[j]["collidable"].to_numpy()).astype(np.int32)

    return out
