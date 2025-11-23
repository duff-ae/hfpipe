import numpy as np
import pandas as pd
from ..io.hd5 import read_fill_field, BX

_KEYS = ["fillnum", "runnum", "lsnum", "nbnum"]


def _inner_match(
    lumi_meta: pd.DataFrame,
    other_meta: pd.DataFrame,
    *,
    keep: str = "last",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Strict one-to-one key matching between lumi_meta and other_meta.

    This function returns a boolean mask over lumi_meta rows indicating which rows
    have a corresponding match in other_meta (after deduplication), as well as an
    index array pointing into the *original* other_meta rows for those matches.

    Parameters
    ----------
    lumi_meta : pd.DataFrame
        Left-side metadata (N rows). Must contain columns _KEYS.
    other_meta : pd.DataFrame
        Right-side metadata (M rows). Must contain columns _KEYS.
    keep : {'first', 'last'}, default 'last'
        Deduplication policy for duplicates in other_meta with the same key.

    Returns
    -------
    keep_mask : np.ndarray, dtype=bool, shape (N,)
        True for lumi_meta rows that found a unique match in other_meta.
    take_idx : np.ndarray, dtype=int, shape (K,)
        Indices into the *original* other_meta that correspond to the True entries
        in keep_mask (K == keep_mask.sum()).
    stats : dict
        {'other_dupes': int, 'other_unique': int} where
        - 'other_dupes'  is the number of duplicate rows removed from other_meta
        - 'other_unique' is the number of unique keys remaining in other_meta

    Notes
    -----
    - Uses MultiIndex + reindex to avoid row blow-up that can happen with merges.
    - The mapping is guaranteed to be one-to-one after deduplication, so
      keep_mask and take_idx align cleanly with lumi_meta.
    """
    if not set(_KEYS).issubset(lumi_meta.columns):
        raise KeyError(f"lumi_meta must contain columns: {_KEYS}")
    if not set(_KEYS).issubset(other_meta.columns):
        raise KeyError(f"other_meta must contain columns: {_KEYS}")

    # Remember original row indices of other_meta so we can select arrays by them later.
    other_idx = np.arange(len(other_meta), dtype=np.int64)

    # Build left keys as a MultiIndex (shape N).
    left_key = pd.MultiIndex.from_frame(lumi_meta[_KEYS], names=_KEYS)

    # Build a compact view of other_meta keys with the original row index attached.
    other_view = other_meta[_KEYS].copy()
    other_view["_ix"] = other_idx

    # Deduplicate other_meta on the join key, keeping either the first or last occurrence.
    other_dedup = other_view.drop_duplicates(subset=_KEYS, keep=keep)
    stats = {
        "other_dupes": int(len(other_view) - len(other_dedup)),
        "other_unique": int(len(other_dedup)),
    }

    # Build right keys and align to left via reindex (no row blow-up).
    right_key = pd.MultiIndex.from_frame(other_dedup[_KEYS], names=_KEYS)
    aligned = pd.Series(other_dedup["_ix"].to_numpy(), index=right_key).reindex(left_key)

    # Rows with NaN in 'aligned' have no match.
    keep_mask = aligned.notna().to_numpy()
    take_idx = aligned[keep_mask].to_numpy(dtype=np.int64)

    return keep_mask, take_idx, stats


def revert_online(
    file_path: str,
    fill: int,
    lumi_meta: pd.DataFrame,
    bxraw: np.ndarray,
    afterglow_node: str,
    pedestal_node: str,
    ped_mod4_perm: tuple[int, int, int, int] = (0, 1, 2, 3),
    *,
    dedup_keep: str = "last",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Strict rollback of online corrections in reverse order:
      (1) + pedestal   (inner join only; rows without pedestal are DROPPED)
      (2) / afterglow  (inner join only; rows without afterglow are DROPPED)

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    fill : int
        Fill number used by read_fill_field to select the dataset group.
    lumi_meta : pd.DataFrame
        Metadata aligned with bxraw, must contain _KEYS columns.
    bxraw : np.ndarray
        Raw BX data, shape (N, BX).
    afterglow_node : str
        HDF5 node for afterglow multiplicative factors (expected shape (*, BX)).
    pedestal_node : str
        HDF5 node for pedestal additive offsets (expected shape (*, 4) or (*, BX)).
    ped_mod4_perm : tuple[int,int,int,int], default (0,1,2,3)
        Optional permutation to reorder the 4-value pedestal pattern before expansion.
    dedup_keep : {'first', 'last'}, default 'last'
        Deduplication policy for duplicates in pedestal/afterglow metadata.

    Returns
    -------
    new_meta : pd.DataFrame
        Filtered metadata after dropping rows lacking pedestal and/or afterglow.
    new_bxraw : np.ndarray
        BX data after applying +pedestal and /afterglow, aligned with new_meta.
    stats : dict
        {
          'dropped_no_ped'      : int,
          'dropped_no_afterglow': int,
          'ped_other_dupes'     : int,
          'ag_other_dupes'      : int,
        }

    Raises
    ------
    ValueError
        If shapes are invalid (e.g., bxraw is not (N, BX), or nodes have unexpected shape).

    Notes
    -----
    - Pedestal may come either as 4 numbers cycling over BX (shape (*,4)) or a full BX array (*, BX).
      In the 4-wide case, the code expands via index % 4.
    - Afterglow is multiplicative; division by zero will produce inf/NaN, which are set to 0 afterward.
      If you prefer to drop rows that contain zeros in afterglow, this can be added easily.
    """
    if bxraw.ndim != 2 or bxraw.shape[1] != BX:
        raise ValueError(f"`bxraw` must be (N, {BX}), got {bxraw.shape}")
    if len(lumi_meta) != bxraw.shape[0]:
        raise ValueError("lumi_meta and bxraw must have the same number of rows")

    N0 = len(lumi_meta)
    stats = {
        "dropped_no_ped": 0,
        "dropped_no_afterglow": 0,
        "ped_other_dupes": 0,
        "ag_other_dupes": 0,
    }

    # -------------------- (1) P E D E S T A L --------------------
    ped_meta, ped_arr, ped_field = read_fill_field(
        file_path, fill, pedestal_node, field_candidates=("data", "bxraw", "bx")
    )

    # If there are no pedestal rows at all, drop everything.
    if ped_arr.size == 0:
        stats["dropped_no_ped"] = N0
        return lumi_meta.iloc[0:0].copy(), bxraw[0:0], stats

    # Pedestal must be either (*, 4) or (*, BX).
    if ped_arr.ndim != 2 or ped_arr.shape[1] not in (4, BX):
        raise ValueError(
            f"Node '{pedestal_node}' field '{ped_field}' has shape {ped_arr.shape}, "
            f"expected (*,4) or (*,{BX})"
        )

    keep_ped, take_ped, ped_match_stats = _inner_match(lumi_meta, ped_meta, keep=dedup_keep)
    stats["ped_other_dupes"] = ped_match_stats["other_dupes"]

    # If nothing matches, drop everything.
    if not keep_ped.any():
        stats["dropped_no_ped"] = N0
        return lumi_meta.iloc[0:0].copy(), bxraw[0:0], stats

    # Keep only rows with a pedestal match.
    stats["dropped_no_ped"] = int((~keep_ped).sum())
    meta1 = lumi_meta.loc[keep_ped].reset_index(drop=True)
    bx1 = np.ascontiguousarray(bxraw[keep_ped])  # contiguous for safe in-place ops

    # Apply pedestal (+)
    if ped_arr.shape[1] == 4:
        ped4 = ped_arr.take(take_ped, axis=0).astype(bx1.dtype, copy=False)  # (K, 4)
        if ped_mod4_perm != (0, 1, 2, 3):
            ped4 = ped4[:, ped_mod4_perm]
        idx = np.arange(BX, dtype=np.int64) % 4
        bx1 += ped4[:, idx]  # broadcast add
    else:
        ped = ped_arr.take(take_ped, axis=0).astype(bx1.dtype, copy=False)  # (K, BX)
        bx1 += ped

    # -------------------- (2) A F T E R G L O W --------------------
    ag_meta, ag_arr, ag_field = read_fill_field(
        file_path, fill, afterglow_node, field_candidates=("data", "bxraw", "bx")
    )

    # If there are no afterglow rows at all, drop all rows that survived step (1).
    if ag_arr.size == 0:
        stats["dropped_no_afterglow"] = len(meta1)
        return meta1.iloc[0:0].copy(), bx1[0:0], stats

    if ag_arr.ndim != 2 or ag_arr.shape[1] != BX:
        raise ValueError(
            f"Node '{afterglow_node}' field '{ag_field}' has shape {ag_arr.shape}, "
            f"expected (*,{BX})"
        )

    keep_ag, take_ag, ag_match_stats = _inner_match(meta1, ag_meta, keep=dedup_keep)
    stats["ag_other_dupes"] = ag_match_stats["other_dupes"]

    # If nothing matches, drop everything that survived step (1).
    if not keep_ag.any():
        stats["dropped_no_afterglow"] = len(meta1)
        return meta1.iloc[0:0].copy(), bx1[0:0], stats

    # Keep only rows that also have afterglow.
    stats["dropped_no_afterglow"] = int((~keep_ag).sum())
    meta2 = meta1.loc[keep_ag].reset_index(drop=True)
    bx2 = np.ascontiguousarray(bx1[keep_ag])  # (M, BX)

    # Apply afterglow division (/)
    ag = ag_arr.take(take_ag, axis=0).astype(bx2.dtype, copy=False)  # (M, BX)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(bx2, ag, out=bx2)
        # Replace non-finite results (inf, -inf, NaN) with 0 to avoid contaminating data.
        bad = ~np.isfinite(bx2)
        if bad.any():
            bx2[bad] = 0

    return meta2, bx2, stats