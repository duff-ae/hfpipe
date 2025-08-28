import numpy as np
import pandas as pd
from ..io.hd5 import read_fill_field, align_to_lumi, BX

def revert_online(file_path: str, fill: int,
                  lumi_meta: pd.DataFrame, bxraw: np.ndarray,
                  afterglow_node: str, pedestal_node: str):
    """
    Отменяем online-коррекции:
      1) bxraw += online_pedestal   (field: data|bxraw, shape (N,4) ИЛИ (N,3564))
      2) bxraw /= online_afterglow  (field: data|bxraw, shape (N,3564))
    С выравниванием по ключам (fillnum, runnum, lsnum, nbnum).
    """

    # --------- PEDESTAL (сложение) ---------
    try:
        ped_meta, ped_arr, ped_field = read_fill_field(file_path, fill, pedestal_node,
                                                       field_candidates=("data","bxraw","bx"))
        if ped_arr.size == 0:
            return

        M = ped_arr.shape[1]
        if M == 4:
            # выровнять (N,4), затем развернуть по mod4 → (N,3564)
            ped_default4 = np.zeros((4,), dtype=bxraw.dtype)
            ped4 = align_to_lumi(lumi_meta, ped_meta, ped_arr.astype(bxraw.dtype, copy=False), ped_default4)
            idx = np.arange(BX) % 4
            bxraw += ped4[:, idx]  # broadcasting
        elif M == BX:
            # выровнять (N,3564) и просто сложить
            ped_default3564 = np.zeros((BX,), dtype=bxraw.dtype)
            ped = align_to_lumi(lumi_meta, ped_meta, ped_arr.astype(bxraw.dtype, copy=False), ped_default3564)
            bxraw += ped
        else:
            raise ValueError(f"Node '{pedestal_node}' field '{ped_field}' has shape {ped_arr.shape}, expected (*,4) or (*,{BX})")
    except Exception as e:
        print(f"[revert_online] pedestal skipped: {e}")

    '''
    # --------- AFTERGLOW (деление) ---------
    try:
        ag_meta, ag_arr, ag_field = read_fill_field(file_path, fill, afterglow_node,
                                                    field_candidates=("data","bxraw","bx"))
        # ожидаем (N,3564)
        if ag_arr.size and ag_arr.shape[1] != BX:
            raise ValueError(f"Node '{afterglow_node}' field '{ag_field}' has shape {ag_arr.shape}, expected (*, {BX})")

        # default: все единицы (ничего не делим)
        ag_default = np.ones((BX,), dtype=bxraw.dtype)
        ag_aligned = align_to_lumi(lumi_meta, ag_meta, ag_arr.astype(bxraw.dtype, copy=False), ag_default)

        denom = np.where(ag_aligned == 0.0, 1.0, ag_aligned)
        # безопасное деление in-place
        np.divide(bxraw, denom, out=bxraw)
    except Exception as e:
        print(f"[revert_online] afterglow skipped: {e}")
    '''
