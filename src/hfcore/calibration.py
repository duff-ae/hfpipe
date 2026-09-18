from __future__ import annotations

import logging

import numpy as np

from .config import PipelineConfig
from .io import align_aux_by_keys, load_hd5_to_arrays
from .online_recovery import (
    OnlineRecoverySolver,
    load_hfsbr_file,
    reconstruct_from_tables_batch,
)

log = logging.getLogger("hfpipe.calibration")


def recover_online_full_fill(
    data: dict[str, np.ndarray],
    cfg: PipelineConfig,
    active_mask: np.ndarray,
    fill: int,
) -> dict[str, np.ndarray]:
    """
    Undo online HF pedestal/afterglow corrections for standalone calibration.

    This is the in-memory counterpart of the production pipeline's
    ``_recover_online_chunk`` step.  It intentionally stops before the offline
    fixed-pedestal and LSQ afterglow corrections so Type-2/after-dump helpers can
    inspect the same pre-LSQ quantity used by the normal pipeline.
    """
    if not cfg.steps.online_recovery:
        return dict(data)

    method = str(cfg.online_recovery.method).lower()
    input_name = cfg.io.input_pattern.format(fill=fill)

    if method == "tables":
        ped_data = load_hd5_to_arrays(
            cfg.io.input_dir,
            input_name,
            node=cfg.online_recovery.pedestal_node,
        )
        aft_data = load_hd5_to_arrays(
            cfg.io.input_dir,
            input_name,
            node=cfg.online_recovery.afterglow_node,
        )
        pedestal_4 = align_aux_by_keys(data, ped_data, "bxraw").astype(np.float32)
        afterglow_frac = align_aux_by_keys(data, aft_data, "bxraw").astype(np.float32)
        result = reconstruct_from_tables_batch(
            bxraw_final=np.asarray(data["bxraw"], dtype=np.float32),
            pedestal_4=pedestal_4,
            afterglow_frac=afterglow_frac,
        )

    elif method == "online":
        pattern = cfg.online_recovery.hfsbr_pattern or cfg.afterglow.hfsbr_pattern
        if not pattern:
            raise ValueError("No HFSBR pattern configured for online recovery")
        path = pattern.format(fill=fill)
        hfsbr = load_hfsbr_file(path)
        solver = OnlineRecoverySolver(hfsbr=hfsbr, active_mask=active_mask)
        result = solver.recover_batch(np.asarray(data["bxraw"], dtype=np.float32))

    else:
        raise ValueError(
            f"Unknown online_recovery.method={cfg.online_recovery.method!r}; "
            "expected 'tables' or 'online'"
        )

    out = dict(data)
    out["bxraw"] = result.recovered_raw
    log.info("fill %d: online recovery undone with method=%s", fill, method)
    return out