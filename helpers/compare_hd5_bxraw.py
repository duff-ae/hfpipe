#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import numpy as np
import tables as pt
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    HAS_MPLHEP = True
    hep.style.use(hep.style.ROOT)
    hep.style.use(hep.style.CMS)
except Exception:
    HAS_MPLHEP = False


# ============================================================
# USER SETTINGS
# ============================================================

REF_BASE = "/cephfs/brilshare/alshevel/hf_reprocessed/hfet/test_tables"
TEST_BASE = "/cephfs/brilshare/alshevel/hf_reprocessed/hfet/test_remove"

FILLS = ["11094"]          # e.g. ["11094", "11095"] ; or None to scan all ref-base subdirs
FILENAME_PATTERN = None    # e.g. "11094*.hd5"; if None -> "{fill}*.hd5"

MASK_PATH = None           # e.g. "/path/to/active_mask_11094.npy" ; or None
OUTDIR = "bxraw_compare_plots"

REF_LABEL = "origin"
TEST_LABEL = "test"

LOWER_MODE = "ratio"       # "ratio" or "pull"
TIME_REDUCE = "sum"        # "sum" or "mean"
SCALE = 1.0                # e.g. 11245.6 / sigvis if needed

BX_LEN = 3564
KEYS = ("fillnum", "runnum", "lsnum", "nbnum")


# ============================================================
# HELPERS
# ============================================================

def find_table(h5: pt.File):
    for node in ("hfetlumi", "hfet"):
        if hasattr(h5.root, node):
            return getattr(h5.root, node), node
    raise RuntimeError(
        f"No known table node found. Root contains: {[x._v_name for x in h5.root._f_list_nodes()]}"
    )


def table_to_dict(path: str) -> dict[str, np.ndarray]:
    with pt.open_file(path, "r") as h5:
        tab, node_name = find_table(h5)
        arr = tab.read()

    if arr.dtype.names is None:
        raise RuntimeError(f"Table in {path} has no named dtype")

    out = {}
    for name in arr.dtype.names:
        new_name = "bxraw" if name == "data" else name
        out[new_name] = np.array(arr[name])

    if "bxraw" not in out:
        raise RuntimeError(f"No 'bxraw' or 'data' column found in {path}")

    return out


def load_hd5_pattern(directory: str, pattern: str) -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching pattern: {os.path.join(directory, pattern)}")

    all_data = None

    for path in paths:
        local = table_to_dict(path)

        if all_data is None:
            all_data = local
            continue

        for key, arr in local.items():
            if key not in all_data:
                all_data[key] = arr
                continue

            if all_data[key].ndim != arr.ndim:
                raise RuntimeError(
                    f"Column '{key}' ndim mismatch: {all_data[key].ndim} vs {arr.ndim}"
                )
            if all_data[key].shape[1:] != arr.shape[1:]:
                raise RuntimeError(
                    f"Column '{key}' shape mismatch: {all_data[key].shape} vs {arr.shape}"
                )

            all_data[key] = np.concatenate([all_data[key], arr], axis=0)

    return all_data


def build_key_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    missing = [k for k in KEYS if k not in data]
    if missing:
        raise KeyError(f"Missing key columns: {missing}")
    return np.stack([np.asarray(data[k], dtype=np.int64) for k in KEYS], axis=1)


def align_bxraw_by_keys(
    ref: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    ref_key = build_key_matrix(ref)
    test_key = build_key_matrix(test)

    ref_map = {tuple(ref_key[i]): i for i in range(ref_key.shape[0])}
    test_map = {tuple(test_key[i]): i for i in range(test_key.shape[0])}

    common = [k for k in ref_map if k in test_map]
    if not common:
        raise RuntimeError("No common rows found between reference and test")

    ref_idx = np.array([ref_map[k] for k in common], dtype=np.int64)
    test_idx = np.array([test_map[k] for k in common], dtype=np.int64)

    ref_bxraw = np.asarray(ref["bxraw"])[ref_idx]
    test_bxraw = np.asarray(test["bxraw"])[test_idx]

    return ref_bxraw, test_bxraw


def load_mask(mask_path: str | None) -> np.ndarray | None:
    if mask_path is None:
        return None

    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = np.loadtxt(mask_path)

    mask = np.asarray(mask).ravel()
    if mask.shape[0] != BX_LEN:
        raise ValueError(f"Mask length {mask.shape[0]} != {BX_LEN}")

    return mask.astype(bool)


def compute_ratio(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return test / np.where(np.abs(ref) > eps, ref, np.nan)


def compute_pull(ref: np.ndarray, test: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return (test - ref) / np.sqrt(np.abs(ref) + eps)


def make_double_figure(fill: str, y1: str, y2: str):
    fig, ax = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.rcParams.update({"font.size": 14})

    if HAS_MPLHEP:
        hep.cms.label("Preliminary", loc=0, data=True, year=2025, rlabel=f"Fill {fill} (13.6 TeV)", ax=ax[0])

    ax[0].set_ylabel(y1)
    ax[1].set_ylabel(y2)
    ax[1].set_xlabel("BCID / row index")
    ax[1].minorticks_on()
    ax[1].tick_params(direction="in", which="both")

    return fig, ax


def plot_mean_bx(
    ref_bxraw: np.ndarray,
    test_bxraw: np.ndarray,
    outdir: str,
    fill: str,
):
    ref_mean = np.mean(ref_bxraw, axis=0).astype(np.float64) * SCALE
    test_mean = np.mean(test_bxraw, axis=0).astype(np.float64) * SCALE

    if LOWER_MODE == "pull":
        lower = compute_pull(ref_mean, test_mean)
        lower_label = "Pull"
    else:
        lower = compute_ratio(ref_mean, test_mean)
        lower_label = f"{TEST_LABEL}/{REF_LABEL}"

    bx = np.arange(BX_LEN)

    fig, ax = make_double_figure(fill, "Mean bxraw", lower_label)

    ax[0].step(bx, ref_mean, where="mid", label=REF_LABEL)
    ax[0].step(bx, test_mean, where="mid", label=TEST_LABEL)
    ax[0].legend(frameon=False, loc="upper right")

    ax[1].step(bx, lower, where="mid", label=lower_label)
    ax[1].legend(frameon=False, loc="upper right")

    if LOWER_MODE == "ratio":
        ax[1].set_ylim(0.95, 1.05)
    else:
        ax[1].set_ylim(-5.0, 5.0)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"mean_bx_{LOWER_MODE}.png"), dpi=300)
    plt.close(fig)


def plot_time_summary(
    ref_bxraw: np.ndarray,
    test_bxraw: np.ndarray,
    outdir: str,
    fill: str,
    active_mask: np.ndarray | None,
):
    if active_mask is None:
        mask = np.ones(BX_LEN, dtype=bool)
    else:
        mask = active_mask

    if TIME_REDUCE == "mean":
        ref_red = ref_bxraw[:, mask].mean(axis=1).astype(np.float64) * SCALE
        test_red = test_bxraw[:, mask].mean(axis=1).astype(np.float64) * SCALE
        ylabel = "Mean over selected BX"
    else:
        ref_red = ref_bxraw[:, mask].sum(axis=1).astype(np.float64) * SCALE
        test_red = test_bxraw[:, mask].sum(axis=1).astype(np.float64) * SCALE
        ylabel = "Sum over selected BX"

    ratio = compute_ratio(ref_red, test_red)
    idx = np.arange(ref_red.shape[0])

    fig, ax = make_double_figure(fill, ylabel, f"{TEST_LABEL}/{REF_LABEL}")

    ax[0].plot(idx, ref_red, ".", markersize=3, label=REF_LABEL)
    ax[0].plot(idx, test_red, ".", markersize=3, label=TEST_LABEL)
    ax[0].legend(frameon=False, loc="upper right")

    ax[1].plot(idx, ratio, ".", markersize=3, label=f"{TEST_LABEL}/{REF_LABEL}")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_ylim(0.95, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"time_summary_{TIME_REDUCE}.png"), dpi=300)
    plt.close(fig)


def compare_one_fill(fill: str, active_mask: np.ndarray | None):
    ref_dir = os.path.join(REF_BASE, fill)
    test_dir = os.path.join(TEST_BASE, fill)

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Reference fill dir not found: {ref_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test fill dir not found: {test_dir}")

    pattern = FILENAME_PATTERN if FILENAME_PATTERN is not None else f"{fill}*.hd5"

    ref_data = load_hd5_pattern(ref_dir, pattern)
    test_data = load_hd5_pattern(test_dir, pattern)

    ref_bxraw, test_bxraw = align_bxraw_by_keys(ref_data, test_data)

    if ref_bxraw.ndim != 2 or ref_bxraw.shape[1] != BX_LEN:
        raise RuntimeError(f"Bad ref_bxraw shape: {ref_bxraw.shape}")
    if test_bxraw.ndim != 2 or test_bxraw.shape[1] != BX_LEN:
        raise RuntimeError(f"Bad test_bxraw shape: {test_bxraw.shape}")

    outdir = os.path.join(OUTDIR, fill)
    os.makedirs(outdir, exist_ok=True)

    plot_mean_bx(ref_bxraw, test_bxraw, outdir, fill)
    plot_time_summary(ref_bxraw, test_bxraw, outdir, fill, active_mask)

    diff = test_bxraw.astype(np.float64) - ref_bxraw.astype(np.float64)

    print({
        "fill": fill,
        "n_common_rows": int(ref_bxraw.shape[0]),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    })


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    active_mask = load_mask(MASK_PATH)

    if FILLS is None:
        fills = sorted(
            d for d in os.listdir(REF_BASE)
            if os.path.isdir(os.path.join(REF_BASE, d))
        )
    else:
        fills = [str(x) for x in FILLS]

    failed = []
    for fill in fills:
        try:
            compare_one_fill(fill, active_mask)
        except Exception as e:
            print(f"[ERROR] fill {fill}: {e}")
            failed.append(fill)

    if failed:
        print("\nFailed fills:")
        for fill in failed:
            print(" -", fill)


if __name__ == "__main__":
    main()