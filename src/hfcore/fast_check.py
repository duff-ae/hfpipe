#!/usr/bin/env python3
# python3 src/hfcore/fast_check.py --config configs/analysis_25_physics.yaml --fill 10709 --save-npz online_recovery.npz

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from hfcore.config import load_config
from hfcore.hd5schema import BX_LEN
from hfcore.io import load_hd5_to_arrays
from hfcore.online_recovery import (
    OnlineRecoverySolver,
    apply_revert_afterglow_batch,
    load_hfsbr_file,
    reconstruct_from_tables_batch,
)
from hfcore.plotter import create_figure


KEYS = ("fillnum", "runnum", "lsnum", "nbnum")
ARTIFICIAL_ZERO_BX = (3553, 3554, 3555, 3556, 3557)


def plot_online_recovery_closure_per_bx(
    raw_ref: np.ndarray,
    recovered: np.ndarray,
    valid: np.ndarray,
    active_mask: np.ndarray,
    cfg,
    fill: int,
    year: int = 2025,
    zero_bx=ARTIFICIAL_ZERO_BX,
) -> str:
    """Zoomed mean per-BX deviation from the authoritative table reference."""
    raw_ref = np.asarray(raw_ref, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    active = np.asarray(active_mask, dtype=bool).ravel()

    mask = valid & np.isfinite(raw_ref) & np.isfinite(recovered)
    norm_mask = mask & active[None, :]
    if not np.any(norm_mask):
        raise RuntimeError("No valid colliding-BX values for closure normalization")

    mean_colliding = raw_ref[norm_mask].mean()
    diff = np.where(mask, recovered - raw_ref, 0.0)
    count = mask.sum(axis=0)

    mean_diff = np.full(BX_LEN, np.nan, dtype=np.float64)
    good = count > 0
    mean_diff[good] = diff[:, good].sum(axis=0) / count[good]
    deviation_pct = 100.0 * mean_diff / mean_colliding

    max_abs = float(np.nanmax(np.abs(deviation_pct)))
    y_lim = max(0.01, 1.20 * max_abs)

    fig = create_figure(
        "BCID",
        "Mean deviation from table reference [%]",
        fill,
        year=year,
        plot_type="Preliminary",
    )
    ax = fig.gca()
    bx = np.arange(BX_LEN)
    ax.plot(bx, deviation_pct, linewidth=1.0, label="Zero-BX/HFSBR recovery")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, BX_LEN - 1)
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.text(
        0.02, 0.95,
        f"max |mean deviation| = {max_abs:.4f}%",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=11,
    )

    zero = np.asarray(tuple(zero_bx), dtype=np.int64)
    if zero.size:
        ax.axvspan(zero.min() - 0.5, zero.max() + 0.5, alpha=0.06)

    fig.tight_layout()
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")
    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"online_recovery_closure_per_bx_fill_{fill}.png"
    )
    fig.savefig(output_path, dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)

    print(
        "Online-recovery per-BX closure: "
        f"max |mean deviation| = {max_abs:.6e}%"
    )
    print(f"Saved: {output_path}")
    return output_path


def select_fill(data: dict, fill: int) -> dict:
    fillnum = np.asarray(data["fillnum"])
    sel = fillnum == fill
    n = len(fillnum)

    out = {}
    for name, value in data.items():
        arr = np.asarray(value)
        out[name] = arr[sel] if arr.ndim > 0 and arr.shape[0] == n else arr
    return out


def strict_align_aux(main: dict, aux: dict, colname: str) -> np.ndarray:
    for key in KEYS:
        if key not in main:
            raise KeyError(f"main is missing key column {key!r}")
        if key not in aux:
            raise KeyError(f"aux is missing key column {key!r}")
    if colname not in aux:
        raise KeyError(f"aux is missing requested column {colname!r}")

    main_keys = np.stack(
        [np.asarray(main[k], dtype=np.int64) for k in KEYS], axis=1
    )
    aux_keys = np.stack(
        [np.asarray(aux[k], dtype=np.int64) for k in KEYS], axis=1
    )

    index = {}
    for i, key_arr in enumerate(aux_keys):
        key = tuple(key_arr.tolist())
        if key in index:
            raise RuntimeError(
                f"Duplicate auxiliary key {key}: rows {index[key]} and {i}"
            )
        index[key] = i

    source = np.asarray(aux[colname])
    out = np.empty((len(main_keys),) + source.shape[1:], dtype=source.dtype)
    missing = []

    for i, key_arr in enumerate(main_keys):
        key = tuple(key_arr.tolist())
        j = index.get(key)
        if j is None:
            missing.append(key)
        else:
            out[i] = source[j]

    if missing:
        raise RuntimeError(
            f"{len(missing)} main rows have no matching auxiliary row. "
            f"First missing keys: {missing[:10]}"
        )
    return out


def print_metric(
    name: str,
    test: np.ndarray,
    ref: np.ndarray,
    valid: np.ndarray | None = None,
) -> None:
    test = np.asarray(test, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    if test.shape != ref.shape:
        raise ValueError(f"{name}: shape mismatch {test.shape} vs {ref.shape}")

    mask = np.isfinite(test) & np.isfinite(ref)
    if valid is not None:
        mask &= np.broadcast_to(valid, test.shape)

    if not np.any(mask):
        print(f"{name:48s}: NO VALID VALUES")
        return

    d = test[mask] - ref[mask]
    r = ref[mask]
    rms = np.sqrt(np.mean(d * d))
    ref_rms = np.sqrt(np.mean(r * r))
    rel_rms = rms / ref_rms if ref_rms > 0 else np.nan

    print(
        f"{name:48s} "
        f"N={len(d):10d}  "
        f"bias={np.mean(d): .6e}  "
        f"RMS={rms: .6e}  "
        f"MAE={np.mean(np.abs(d)): .6e}  "
        f"P95={np.percentile(np.abs(d), 95): .6e}  "
        f"max={np.max(np.abs(d)): .6e}  "
        f"RMS/refRMS={rel_rms: .6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fill", required=True, type=int)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--save-npz", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fill = args.fill
    zero_bx = ARTIFICIAL_ZERO_BX
    zero_idx = np.asarray(zero_bx, dtype=np.int64)
    input_name = cfg.io.input_pattern.format(fill=fill)

    print("============================================================")
    print(f"ONLINE RECOVERY VALIDATION -- fill {fill}")
    print("============================================================")
    print(f"Input: {os.path.join(cfg.io.input_dir, input_name)}")
    print()

    main_data = select_fill(
        load_hd5_to_arrays(cfg.io.input_dir, input_name, node=cfg.io.node),
        fill,
    )
    final_online = np.asarray(main_data["bxraw"], dtype=np.float64)
    T = final_online.shape[0]
    print(f"Main rows: {T}")

    ped_data = select_fill(
        load_hd5_to_arrays(
            cfg.io.input_dir, input_name, node=cfg.online_recovery.pedestal_node
        ),
        fill,
    )
    aft_data = select_fill(
        load_hd5_to_arrays(
            cfg.io.input_dir, input_name, node=cfg.online_recovery.afterglow_node
        ),
        fill,
    )

    pedestal_4 = strict_align_aux(main_data, ped_data, "bxraw").astype(np.float64)
    afterglow_frac = strict_align_aux(main_data, aft_data, "bxraw").astype(np.float64)

    valid = np.isfinite(afterglow_frac) & (afterglow_frac > 0.0)
    reference_valid = valid.copy()
    reference_valid[:, zero_idx] = False

    print()
    print("TABLE COVERAGE")
    print("--------------")
    print(
        f"afterglow_frac > 0: {np.count_nonzero(valid)}/{valid.size} "
        f"({100.0 * np.mean(valid):.3f}%)"
    )

    table = reconstruct_from_tables_batch(
        final_online,
        pedestal_4,
        afterglow_frac,
        zero_bx=zero_bx,
    )
    raw_ref = table.recovered_raw.astype(np.float64)

    idx_mod4 = np.arange(BX_LEN) % 4
    pre_pedestal = final_online + pedestal_4[:, idx_mod4]
    roundtrip = raw_ref * afterglow_frac - pedestal_4[:, idx_mod4]

    print()
    print("1. TABLE SELF-CLOSURE")
    print("---------------------")
    print_metric(
        "raw_table * frac - pedestal -> final",
        roundtrip,
        final_online,
        valid=reference_valid,
    )

    # 13x4 dynamic pedestal definition used online.
    ped13 = np.empty_like(pedestal_4)
    for k in range(4):
        bx = 3500 + k + 4 * np.arange(13)
        ped13[:, k] = pre_pedestal[:, bx].mean(axis=1)

    print()
    print("2. PEDESTAL DEFINITION")
    print("-----------------------")
    print_metric("D(pre-pedestal), 13 samples vs table", ped13, pedestal_4)

    if T <= args.max_rows:
        sample_idx = np.arange(T)
    else:
        sample_idx = np.linspace(0, T - 1, args.max_rows, dtype=int)

    final_s = final_online[sample_idx]
    ped_s = pedestal_4[sample_idx]
    frac_s = afterglow_frac[sample_idx]
    valid_s = reference_valid[sample_idx]
    raw_ref_s = raw_ref[sample_idx]
    pre_ped_s = pre_pedestal[sample_idx]

    mask_path = cfg.io.active_mask_pattern.format(fill=fill)
    with open(mask_path, "r") as f:
        active_mask = np.asarray(json.load(f), dtype=np.int32)
    active_bool = active_mask.astype(bool)

    hfsbr_pattern = cfg.online_recovery.hfsbr_pattern or cfg.afterglow.hfsbr_pattern
    if not hfsbr_pattern:
        raise ValueError("No HFSBR pattern configured")
    hfsbr = load_hfsbr_file(hfsbr_pattern.format(fill=fill))

    print()
    print(f"HFSBR validation rows: {len(sample_idx)} / {T}")

    print()
    print("3. HFSBR-ONLY CLOSURE (TRUE TABLE PEDESTAL)")
    print("-------------------------------------------")
    hfsbr_only = apply_revert_afterglow_batch(pre_ped_s, hfsbr, active_mask)
    print_metric("HFSBR only vs table raw [all]", hfsbr_only, raw_ref_s, valid_s)
    print_metric(
        "HFSBR only vs table raw [active]",
        hfsbr_only,
        raw_ref_s,
        valid_s & active_bool[None, :],
    )

    print()
    print("4. ZERO-BX FULL RECOVERY")
    print("------------------------")
    solver = OnlineRecoverySolver(hfsbr, active_mask)
    online = solver.recover_batch(final_s)

    print(
        f"pedestal system: shape={solver.A.shape} "
        f"rank={solver.rank} condition={solver.condition:.6e}"
    )
    print_metric("fitted pedestal vs table pedestal", online.pedestal, ped_s)
    print_metric("recovered raw vs table raw [all]", online.recovered_raw, raw_ref_s, valid_s)
    print_metric(
        "recovered raw vs table raw [active]",
        online.recovered_raw,
        raw_ref_s,
        valid_s & active_bool[None, :],
    )
    print_metric(
        "recovered raw vs table raw [non-active]",
        online.recovered_raw,
        raw_ref_s,
        valid_s & (~active_bool)[None, :],
    )

    print()
    print("5. EXACT ARTIFICIAL-ZERO CHECK")
    print("------------------------------")
    z_true_ped = hfsbr_only[:, zero_idx]
    for j, bcid in enumerate(zero_idx):
        vals = z_true_ped[:, j].astype(np.float64)
        print(
            f"TABLE pedestal -> R(...) BX {bcid}: "
            f"mean={np.mean(vals): .8e}  "
            f"RMS={np.sqrt(np.mean(vals**2)): .8e}  "
            f"max|x|={np.max(np.abs(vals)): .8e}"
        )

    z_fit = np.asarray(online.recovered_raw[:, zero_idx], dtype=np.float64)
    print(
        "FITTED pedestal -> recovered artificial-zero BX: "
        f"mean={np.mean(z_fit): .8e}  "
        f"RMS={np.sqrt(np.mean(z_fit**2)): .8e}  "
        f"max|x|={np.max(np.abs(z_fit)): .8e}"
    )

    print()
    print("6. PER-BX CLOSURE PLOT")
    print("----------------------")
    plot_online_recovery_closure_per_bx(
        raw_ref=raw_ref_s,
        recovered=online.recovered_raw,
        valid=valid_s,
        active_mask=active_mask,
        cfg=cfg,
        fill=fill,
        year=args.year,
        zero_bx=zero_bx,
    )

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            sample_idx=sample_idx,
            final_online=final_s,
            pedestal_table=ped_s,
            pedestal_fitted=online.pedestal,
            afterglow_frac_table=frac_s,
            raw_table=raw_ref_s,
            recovered_online=online.recovered_raw,
            hfsbr_only=hfsbr_only,
            active_mask=active_mask,
            hfsbr=hfsbr,
        )
        print(f"Saved detailed validation arrays to {args.save_npz}")


if __name__ == "__main__":
    main()
