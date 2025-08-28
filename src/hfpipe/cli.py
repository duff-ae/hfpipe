# src/hfpipe/cli.py
from __future__ import annotations

import numpy as np
import typer
from pathlib import Path
from tables import IsDescription, UInt32Col, UInt8Col, StringCol, Float32Col

from .config import load_config
from .io.hd5 import read_fill_arrays, save_fill_arrays
from .io.beam import read_beam_fill, pick_fill_mask, align_masks_to_lumi
from .core.utils import nan_to_num_inplace, derive_mask_from_lumi
from .core.paths import load_hfsbr
from .core.revert import revert_online
from .core.lsq import batch_afterglow_lsq_matrix
from .plot.plots import per_bx, instant


# ======== HD5 schema for saving =========
class DefaultLumitable(IsDescription):
    fillnum = UInt32Col()
    runnum = UInt32Col()
    lsnum = UInt32Col()
    nbnum = UInt32Col()
    timestampsec = UInt32Col()
    timestampmsec = UInt32Col()
    totsize = UInt32Col()
    publishnnb = UInt8Col()
    datasourceid = UInt8Col()
    algoid = UInt8Col()
    channelid = UInt8Col()
    payloadtype = UInt8Col()
    calibtag = StringCol(itemsize=32)
    avgraw = Float32Col()
    avg = Float32Col()
    bxraw = Float32Col(shape=(3564,))
    bx = Float32Col(shape=(3564,))
    maskhigh = UInt32Col()
    masklow = UInt32Col()


app = typer.Typer(add_completion=False, help="HF pipeline: read -> revert online -> LSQ -> plots -> save")

@app.command(help="Run pipeline on a YAML config")
def run(config: str = typer.Argument(..., help="Path to YAML config file")):
    cfg = load_config(config)

    plots_root = Path(cfg.io.out_dir) / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(cfg.io.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for fill in cfg.data.fills:
        typer.echo(f"[hfpipe] Processing fill {fill}")

        # ---- Read lumi as (meta, matrix) ----
        meta, bxraw = read_fill_arrays(cfg.data.file_path, fill, cfg.data.node)
        if len(meta) == 0 or bxraw.size == 0:
            typer.echo(f"[hfpipe] No lumi rows for fill {fill}, skipping.")
            continue

        nan_to_num_inplace(bxraw)

        # ---- Read beam (may be partial/missing) & build masks ----
        beam_df = read_beam_fill(cfg.data.beam_path, fill, getattr(cfg.data, "beam_node", None))

        if beam_df.empty:
            typer.echo("[hfpipe:beam] beam table not found; deriving mask from lumi mean()")
            global_mask = derive_mask_from_lumi(bxraw)
            n_active = int(global_mask.sum())
        else:
            global_mask, n_active = pick_fill_mask(beam_df)

        record_masks = align_masks_to_lumi(meta, beam_df, global_mask).astype(np.float64, copy=False)
        global_mask_i32 = global_mask.astype(np.int32, copy=False)

        # ---- Revert online corrections (if enabled) ----
        rev = cfg.pipeline.revert_online
        if rev.enable:
            typer.echo("[hfpipe] Reverting online afterglow/pedestal …")
            revert_online(cfg.data.file_path, fill, meta, bxraw, rev.afterglow_node, rev.pedestal_node)
            nan_to_num_inplace(bxraw)

        # ---- BEFORE series (for instant plot) ----
        sbil_before = (bxraw * record_masks).sum(axis=1)

        # ---- Per-BX plot (before) ----
        if cfg.plot.per_bx:
            per_bx(
                bxraw.mean(axis=0),
                "Before LSQ",
                fill,
                out=str(plots_root / f"perbx_{fill}_before.png"),
            )

        # ---- Batch LSQ (single fill-wide mask) ----
        blsq = cfg.pipeline.batch_afterglow_lsq
        if blsq.enable:
            typer.echo("[hfpipe] Running batch LSQ …")
            HFSBR = load_hfsbr(blsq.hfsbr_file)
            out, ped, q1, c1 = batch_afterglow_lsq_matrix(
                bxraw,
                HFSBR,
                global_mask_i32,
                p0_guess=np.array(blsq.p0_guess, np.float64),
                lambda_reg=blsq.lambda_reg,
                lambda_nonactive=blsq.lambda_nonactive,
                lambda_bx1=blsq.lambda_bx1,
                use_cubic=blsq.use_cubic,
                n_jobs=blsq.n_jobs,
                backend=blsq.backend,
            )
            bxraw[:] = out  # in-place replace
            nan_to_num_inplace(bxraw)

        # ---- AFTER series (for instant plot) ----
        sbil_after = (bxraw * record_masks).sum(axis=1)

        # ---- Plots ----
        if cfg.plot.per_bx:
            per_bx(
                bxraw.mean(axis=0),
                "After LSQ",
                fill,
                out=str(plots_root / f"perbx_{fill}_after.png"),
            )

        if cfg.plot.instant:
            instant(
                meta,
                sbil_before,
                sbil_after,
                fill,
                out=str(plots_root / f"instant_{fill}.png"),
            )

        # ---- Save processed HDF5 ----
        if cfg.io.save_hd5:
            out_name = f"{fill}_processed.hdf5"
            typer.echo(f"[hfpipe] Saving to {out_root / out_name}")
            save_fill_arrays(meta, bxraw, cfg.data.node, str(out_root), out_name, DefaultLumitable)

        typer.echo(f"[hfpipe] Done fill {fill}\n")


if __name__ == "__main__":
    app()
