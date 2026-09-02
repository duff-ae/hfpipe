import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep as hep
import h5py

from .hd5schema import BX_LEN

hep.style.use(hep.style.ROOT)
hep.style.use(hep.style.CMS)


def create_figure(x_axis, y_axis, fill, year=2025, plot_type='Preliminary'):
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})

    rlabel = f"Fill {fill} ({year}, 13.6 TeV)"
    cms_status = "Preliminary"
    petroff_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler('color', petroff_10)
    pad_inches = 0.5

    hep.cms.label(cms_status, loc=0, data=True, year=year, rlabel=rlabel)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    return fig

def create_double_figure(x_axis, y_axis1, y_axis2, fill, ratio=2, year=2025, plot_type='Preliminary'):
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [ratio, 1]})
    plt.rcParams.update({"font.size": 14})

    rlabel = f"Fill {fill} ({year}, 13.6 TeV)"
    cms_status = plot_type
    petroff_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler('color', petroff_10)
    pad_inches = 0.5

    hep.cms.label(cms_status, loc=0, data=True, year=year, rlabel=rlabel, ax = ax[0])

    # Second plot
    ax[1].set_xlabel(x_axis, fontsize=14, fontname='Helvetica')
    ax[0].set_ylabel(y_axis1, fontsize=14, fontname='Helvetica')
    ax[1].set_ylabel(y_axis2, fontsize=14, fontname='Helvetica')

    ax[1].minorticks_on()
    ax[1].tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both', labelsize=14)
    ax[1].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in', which='both', labelsize=14)

    return fig, ax


def _get_output_dir(cfg, fill) -> str:
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")
    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# plot_hist_bx: per-BX mean profile over the whole fill.
#
# The only thing this plot ever needed from `data` is
# `np.stack(data['bxraw']).mean(axis=0)` -- a per-BX mean across all rows.
# That is exactly `sum(bxraw, axis=0) / T`, which streams trivially and
# exactly: BxProfileAccumulator below never holds more than one
# (BX_LEN,) sum vector, independent of how many rows the fill has.
# ---------------------------------------------------------------------------
class BxProfileAccumulator:
    """
    Streaming, EXACT accumulator for the per-BX mean profile used by
    `plot_hist_bx`. Memory cost is O(BX_LEN), independent of the number
    of rows streamed through `add_chunk`.
    """

    def __init__(self, n_bx: int = BX_LEN):
        self.n_bx = n_bx
        self._sum = np.zeros(n_bx, dtype=np.float64)
        self._count = 0

    def add_chunk(self, bxraw_chunk: np.ndarray) -> None:
        arr = np.asarray(bxraw_chunk, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_bx:
            raise ValueError(
                f"BxProfileAccumulator.add_chunk: expected (T, {self.n_bx}), got {arr.shape}"
            )
        self._sum += arr.sum(axis=0)
        self._count += arr.shape[0]

    @property
    def count(self) -> int:
        return self._count

    def mean_profile(self) -> np.ndarray:
        """Mean bxraw per BX (mu-space, NOT yet scaled by sigvis)."""
        if self._count == 0:
            return np.zeros(self.n_bx, dtype=np.float64)
        return self._sum / self._count


"""
    Plot the instant lumi per bcid
"""
def _plot_hist_bx_core(mean_profile: np.ndarray, cfg, fill, label) -> None:
    """
    Shared plotting/saving core for plot_hist_bx, taking an already
    computed per-BX mean profile (mu-space) instead of the full
    (T, BX_LEN) array.
    """
    fig, ax = create_double_figure('BCID', 'Instantaneous luminosity [Hz/ub]', '', fill)

    hist = list(mean_profile * (11245.6 / cfg.afterglow.sigvis))

    ax[0].bar(list(range(3564)), hist, label=label)
    ax[1].bar(list(range(3564)), hist, label=label)

    ax[0].legend(loc='upper right', frameon=False, fontsize=12)
    ax[1].legend(loc='upper right', frameon=False, fontsize=12)

    plt.tight_layout()

    if 'Corr. Luminosity' in label:
        ax[1].set_ylim(-0.01, 0.01)
    else:
        ax[1].set_ylim(-0.01, 0.01)

    #ax[1].set_xlim(0, 500) # TODO remove

    output_dir = _get_output_dir(cfg, fill)
    png_path = os.path.join(output_dir, f"per_bcid_hist_{label}.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_hist_bx_from_profile(mean_profile: np.ndarray, cfg, fill, label) -> None:
    """
    Chunked entry point: call this once you've streamed the whole fill
    through a `BxProfileAccumulator` and have its `.mean_profile()`.
    """
    _plot_hist_bx_core(mean_profile, cfg, fill, label)


def plot_hist_bx(data, cfg, fill, label):
    """
    Backward-compatible, in-memory entry point (unchanged behaviour):
    computes the mean profile directly from a full `data['bxraw']`
    array and defers to the shared core.
    """
    # TODO I do not understand the reason for this (just copied from the old code)
    #hist = [x * 11245.6/cfg.afterglow.sigvis if abs(x) < 1e3 else 0 for x in np.stack(data['bxraw']).mean(axis=0)]
    mean_profile = np.stack(data['bxraw']).astype(np.float64).mean(axis=0)
    _plot_hist_bx_core(mean_profile, cfg, fill, label)


# ---------------------------------------------------------------------------
# plot_lumi_comparison: only ever used `avg` and `avg_origin`, two
# per-row (per-LS) scalars, both computed as
#   sum(bxraw[i] * active_mask) * scale
# Streamable as a plain O(T) row series -- never needs the full bxraw.
# ---------------------------------------------------------------------------
def compute_scaled_active_sum_chunk(
    bxraw_chunk: np.ndarray, active_mask: np.ndarray, scale: float
) -> np.ndarray:
    """
    Per-row `sum(bxraw * active_mask) * scale` for one chunk -- the
    quantity `plot_lumi_comparison` / `plot_lasers` call `avg`.
    Returns a 1D array of length T (one value per row in the chunk).
    """
    bxraw_chunk = np.asarray(bxraw_chunk, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=np.float64)
    return (bxraw_chunk * mask[None, :]).sum(axis=1) * scale


def _plot_lumi_comparison_core(avg: np.ndarray, avg_origin: np.ndarray, cfg, fill) -> None:
    index = np.arange(avg.shape[0])

    fig, ax = create_double_figure('Fill duration [s]', 'Instantenious luminosity [Hz/µb]', '', fill)

    ax[0].plot(index, avg, '.', label='Corr. instantenious luminosity')
    ax[0].plot(index, avg_origin, '.', label='Uncorr. instantenious luminosity')
    ax[0].legend(loc='upper right', frameon=False, fontsize=12)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = avg / avg_origin
    ax[1].plot(index, ratio, '.', label='Corr. instantenious luminosity')

    plt.tight_layout()
    plt.ylim(0.95, 1.05)

    output_dir = _get_output_dir(cfg, fill)
    png_path = os.path.join(output_dir, "instantaneous.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_lumi_comparison_from_series(avg: np.ndarray, avg_origin: np.ndarray, cfg, fill) -> None:
    """
    Chunked entry point: `avg`/`avg_origin` are the full-fill,
    row-aligned series accumulated via `compute_scaled_active_sum_chunk`
    during the "restored" pass and the final pass respectively.
    """
    _plot_lumi_comparison_core(np.asarray(avg), np.asarray(avg_origin), cfg, fill)


def plot_lumi_comparison(data, data_origin, cfg, active_mask, fill):
    """
    Backward-compatible, in-memory entry point.
    """
    scale = 11245.6 / cfg.afterglow.sigvis
    avg = compute_scaled_active_sum_chunk(np.stack(data['bxraw']), active_mask, scale)
    avg_origin = compute_scaled_active_sum_chunk(np.stack(data_origin['bxraw']), active_mask, scale)
    _plot_lumi_comparison_core(avg, avg_origin, cfg, fill)


# ---------------------------------------------------------------------------
# plot_residuals: only ever used avg_col[i], avg_type1[i], avg_type2[i]
# (three per-row scalars, one per LS) -- never the full bxraw beyond
# that. Streamable as three O(T) row series.
# ---------------------------------------------------------------------------
def build_residual_masks(active_mask: np.ndarray, bx_to_clean, n_bx: int = BX_LEN):
    """
    Build (active_mask_bool, type1_mask, type2_mask) exactly as
    `plot_residuals` did inline. Compute this ONCE per fill (masks don't
    change chunk to chunk) and reuse across all `compute_residual_row_averages`
    calls.
    """
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.shape[0] != n_bx:
        raise ValueError(
            f"active_mask length {active_mask.shape[0]} does not match BX dimension {n_bx}"
        )

    bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64) % n_bx
    clean_mask = np.zeros(n_bx, dtype=bool)
    clean_mask[bx_to_clean] = True

    prev_is_col = np.roll(active_mask, 1)

    type1_mask = (~active_mask) & prev_is_col
    type1_mask[clean_mask] = False

    type2_mask = (~active_mask) & (~type1_mask) & (~clean_mask)

    return active_mask, type1_mask, type2_mask


def compute_residual_row_averages(
    bxraw_chunk: np.ndarray,
    active_mask: np.ndarray,
    type1_mask: np.ndarray,
    type2_mask: np.ndarray,
    scale: float,
):
    """
    Per-row means over the three BX subsets, for one chunk. Vectorized
    (no python row loop): `hists[:, mask].mean(axis=1)` works directly
    on a 2D chunk. Returns (avg_col, avg_type1, avg_type2), each a 1D
    array of length T (one value per row in the chunk), already scaled
    by `scale` (11245.6 / sigvis).
    """
    hists = np.asarray(bxraw_chunk, dtype=np.float64) * scale

    avg_col = hists[:, active_mask].mean(axis=1)
    avg_type1 = hists[:, type1_mask].mean(axis=1)
    avg_type2 = hists[:, type2_mask].mean(axis=1)

    return avg_col, avg_type1, avg_type2


def _residual_ylim(y):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-0.25, 0.25)
    y_abs = np.max(np.abs(y))
    y_lim = max(0.25, 1.15 * y_abs)
    y_lim = max(y_lim, 0.25)
    return (-y_lim, y_lim)


def _style_residual_plot(fig, ax, ylabel, yvals, fill):
    ax.set_xlabel("Mean SBIL [Hz/µb]")
    ax.set_ylabel(ylabel)

    ymin, ymax = _residual_ylim(yvals)
    ax.set_ylim(ymin, ymax)

    ax.axhline(0.0, linestyle="-", linewidth=1.0)
    ax.axhline(+0.2, linestyle="--", linewidth=1.0)
    ax.axhline(-0.2, linestyle="--", linewidth=1.0)
    ax.axhspan(-0.2, 0.2, alpha=0.08)

    ax.grid(True, alpha=0.3)

    try:
        ax.set_title(f"Fill {fill}")
    except Exception:
        pass

    fig.tight_layout()


def plot_residuals_finalize(
    avg_col: np.ndarray,
    avg_type1: np.ndarray,
    avg_type2: np.ndarray,
    cfg,
    fill,
    label,
    n_col: int,
    n_type1: int,
    n_type2: int,
) -> None:
    """
    Shared plotting/saving core: takes the three full-fill per-row
    average series (already scaled) and does exactly what the tail of
    the original `plot_residuals` did -- percent calc, sbil_min filter,
    HDF5 point-cloud save, PNG plots. No dependency on the full bxraw.
    """
    avg_col = np.asarray(avg_col, dtype=np.float64)
    avg_type1 = np.asarray(avg_type1, dtype=np.float64)
    avg_type2 = np.asarray(avg_type2, dtype=np.float64)

    if n_col == 0:
        raise ValueError("No colliding BX found in active_mask")
    if n_type1 == 0:
        raise ValueError("No Type1 BX found after applying masks")
    if n_type2 == 0:
        raise ValueError("No Type2 BX found after applying masks")

    with np.errstate(divide="ignore", invalid="ignore"):
        type1_pct = 100.0 * avg_type1 / avg_col
        type2_pct = 100.0 * avg_type2 / avg_col

    sbil_min = 0.1

    finite1 = np.isfinite(avg_col) & np.isfinite(type1_pct) & (avg_col > sbil_min)
    finite2 = np.isfinite(avg_col) & np.isfinite(type2_pct) & (avg_col > sbil_min)

    type1_points = np.column_stack([avg_col[finite1], type1_pct[finite1]])
    type2_points = np.column_stack([avg_col[finite2], type2_pct[finite2]])

    output_dir = _get_output_dir(cfg, fill)

    # --- save compact point clouds ---
    h5_path = os.path.join(output_dir, f"residual_points_fill_{fill}.h5")
    with h5py.File(h5_path, "a") as f:
        ds1 = f"type1_{label}"
        ds2 = f"type2_{label}"

        if ds1 in f:
            del f[ds1]
        if ds2 in f:
            del f[ds2]

        dset1 = f.create_dataset(ds1, data=type1_points, compression="gzip")
        dset2 = f.create_dataset(ds2, data=type2_points, compression="gzip")

        dset1.attrs["columns"] = np.array(["mean_colliding_sbil", "residual_type1_pct"], dtype="S32")
        dset2.attrs["columns"] = np.array(["mean_colliding_sbil", "residual_type2_pct"], dtype="S32")

        dset1.attrs["fill"] = int(fill)
        dset2.attrs["fill"] = int(fill)

        dset1.attrs["label"] = str(label)
        dset2.attrs["label"] = str(label)

        dset1.attrs["n_colliding_bx"] = n_col
        dset2.attrs["n_colliding_bx"] = n_col

        dset1.attrs["n_type1_bx"] = n_type1
        dset2.attrs["n_type2_bx"] = n_type2

        dset1.attrs["sbil_min_for_plot"] = sbil_min
        dset2.attrs["sbil_min_for_plot"] = sbil_min

    # --- Type1 plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    if type1_points.size > 0:
        ax.plot(type1_points[:, 0], type1_points[:, 1], ".", markersize=3)
        _style_residual_plot(fig, ax, "Type1 Residual [% of mean SBIL]", type1_points[:, 1], fill)
    else:
        _style_residual_plot(fig, ax, "Type1 Residual [% of mean SBIL]", np.array([]), fill)

    png_path = os.path.join(output_dir, f"type1_residuals_{label}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    # --- Type2 plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    if type2_points.size > 0:
        ax.plot(type2_points[:, 0], type2_points[:, 1], ".", markersize=3)
        _style_residual_plot(fig, ax, "Type2 Residual [% of mean SBIL]", type2_points[:, 1], fill)
    else:
        _style_residual_plot(fig, ax, "Type2 Residual [% of mean SBIL]", np.array([]), fill)

    png_path = os.path.join(output_dir, f"type2_residuals_{label}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_residuals_from_series(
    avg_col: np.ndarray,
    avg_type1: np.ndarray,
    avg_type2: np.ndarray,
    active_mask: np.ndarray,
    bx_to_clean,
    cfg,
    fill,
    label,
) -> None:
    """
    Chunked entry point: `avg_col`/`avg_type1`/`avg_type2` are the
    full-fill row series accumulated via `compute_residual_row_averages`
    (using masks from `build_residual_masks`).
    """
    active_bool, type1_mask, type2_mask = build_residual_masks(active_mask, bx_to_clean)
    plot_residuals_finalize(
        avg_col, avg_type1, avg_type2, cfg, fill, label,
        n_col=int(active_bool.sum()),
        n_type1=int(type1_mask.sum()),
        n_type2=int(type2_mask.sum()),
    )


def plot_residuals(data, cfg, active_mask, fill, label):
    """
    Backward-compatible, in-memory entry point (unchanged behaviour):
    computes avg_col/avg_type1/avg_type2 directly from a full
    `data['bxraw']` array and defers to the shared finalize.
    """
    scale = 11245.6 / cfg.afterglow.sigvis
    active_bool, type1_mask, type2_mask = build_residual_masks(
        active_mask, cfg.afterglow.bx_to_clean, n_bx=np.asarray(active_mask).shape[0]
    )

    bxraw = np.stack(data["bxraw"]).astype(np.float64)
    avg_col, avg_type1, avg_type2 = compute_residual_row_averages(
        bxraw, active_bool, type1_mask, type2_mask, scale
    )

    plot_residuals_finalize(
        avg_col, avg_type1, avg_type2, cfg, fill, label,
        n_col=int(active_bool.sum()),
        n_type1=int(type1_mask.sum()),
        n_type2=int(type2_mask.sum()),
    )


# ---------------------------------------------------------------------------
# plot_lasers: only ever used avg/avg_origin (per-row, same quantity as
# plot_lumi_comparison) plus 4 specific BX columns (laser BCIDs) per
# row, for both corrected and original data. All of it is O(T),
# independent of BX_LEN.
# ---------------------------------------------------------------------------
LASER_BCID = [3489, 3490, 3491, 3492]


def compute_laser_columns_chunk(bxraw_chunk: np.ndarray, scale: float, laser_bcid=LASER_BCID) -> dict:
    """
    Per-row values of the laser BX columns for one chunk, already
    scaled. Returns {bcid: 1D array of length T}.
    """
    bxraw_chunk = np.asarray(bxraw_chunk, dtype=np.float64) * scale
    return {bcid: bxraw_chunk[:, bcid].copy() for bcid in laser_bcid}


def _plot_lasers_core(
    avg: np.ndarray,
    avg_origin: np.ndarray,
    laser_corr: dict,
    laser_uncorr: dict,
    cfg,
    fill,
    laser_bcid=LASER_BCID,
) -> None:
    index = np.arange(avg.shape[0], dtype=np.float64)
    n_active = None  # not needed for the plot itself, kept for h5 attrs below if available

    output_dir = _get_output_dir(cfg, fill)

    # --- laser evolution vs time index ---
    fig, ax = create_double_figure(
        "Fill duration [a.u.]", "Uncorrected", "Corrected", fill, ratio=1,
    )
    for bcid in laser_bcid:
        ax[0].plot(index, laser_uncorr[bcid], ".", label=f"LASER BCID {bcid}")
        ax[1].plot(index, laser_corr[bcid], ".", label=f"LASER BCID {bcid}")

    ax[0].legend(loc="upper right", frameon=False, fontsize=12)
    ax[1].legend(loc="upper right", frameon=False, fontsize=12)

    png_path = os.path.join(output_dir, "laser_evolution.png")
    plt.savefig(png_path, dpi=500)
    plt.close(fig)

    # --- laser vs SBIL (scatter only, no fit) ---
    fig, ax = create_double_figure(
        "SBIL [Hz/µb]", "Uncorr. laser bins", "Corr. laser bins", fill, ratio=1,
    )

    h5_path = os.path.join(output_dir, f"laser_summary_fill_{fill}.h5")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["fill"] = int(fill)
        h5.attrs["n_histograms"] = int(avg.shape[0])

        h5.create_dataset("global/index", data=index, compression="gzip")
        h5.create_dataset("global/avg_uncorr", data=avg_origin, compression="gzip")
        h5.create_dataset("global/avg_corr", data=avg, compression="gzip")

        for bcid in laser_bcid:
            y_unc = laser_uncorr[bcid]
            y_cor = laser_corr[bcid]

            ax[0].plot(avg_origin, y_unc, ".", label=f"LASER BCID {bcid}")
            ax[1].plot(avg, y_cor, ".", label=f"LASER BCID {bcid}")

            grp = h5.create_group(f"bcid_{bcid}")
            grp.attrs["bcid"] = int(bcid)

            grp.create_dataset(
                "uncorr_time_points", data=np.column_stack([index, y_unc]), compression="gzip",
            )
            grp.create_dataset(
                "corr_time_points", data=np.column_stack([index, y_cor]), compression="gzip",
            )
            grp.create_dataset(
                "uncorr_sbil_points", data=np.column_stack([avg_origin, y_unc]), compression="gzip",
            )
            grp.create_dataset(
                "corr_sbil_points", data=np.column_stack([avg, y_cor]), compression="gzip",
            )

            grp.attrs["uncorr_n_points"] = int(len(y_unc))
            grp.attrs["corr_n_points"] = int(len(y_cor))
            grp.attrs["columns_time"] = np.array(["index", "laser_value"], dtype="S32")
            grp.attrs["columns_sbil"] = np.array(["mean_colliding_sbil", "laser_value"], dtype="S32")

    ax[0].legend(loc="upper right", frameon=False, fontsize=12)
    ax[1].legend(loc="upper right", frameon=False, fontsize=12)

    png_path = os.path.join(output_dir, "laser_sbil.png")
    plt.savefig(png_path, dpi=500)
    plt.close(fig)


def plot_lasers_from_series(
    avg: np.ndarray,
    avg_origin: np.ndarray,
    laser_corr: dict,
    laser_uncorr: dict,
    cfg,
    fill,
    laser_bcid=LASER_BCID,
) -> None:
    """
    Chunked entry point: `avg`/`avg_origin` are the full-fill row series
    from `compute_scaled_active_sum_chunk`; `laser_corr`/`laser_uncorr`
    are dicts {bcid: full-fill row series} from
    `compute_laser_columns_chunk`, accumulated during the final pass and
    the raw/original pass respectively.
    """
    _plot_lasers_core(
        np.asarray(avg), np.asarray(avg_origin), laser_corr, laser_uncorr, cfg, fill, laser_bcid,
    )


def plot_lasers(data, data_origin, cfg, active_mask, fill):
    """
    Backward-compatible, in-memory entry point (unchanged behaviour).
    """
    scale = 11245.6 / cfg.afterglow.sigvis
    active_mask = np.asarray(active_mask, dtype=bool)

    n_active = int(np.count_nonzero(active_mask))
    if n_active == 0:
        raise ValueError("plot_lasers: active_mask has zero active BX")

    bxraw = np.stack(data["bxraw"]).astype(np.float64)
    bxraw_origin = np.stack(data_origin["bxraw"]).astype(np.float64)

    avg = compute_scaled_active_sum_chunk(bxraw, active_mask, scale) / n_active
    avg_origin = compute_scaled_active_sum_chunk(bxraw_origin, active_mask, scale) / n_active
    # NOTE: original plot_lasers used hists[:, active_mask].mean(axis=1),
    # i.e. sum/n_active, not the raw active-sum used by
    # plot_lumi_comparison's "avg". Divide by n_active here to match that
    # exactly (compute_scaled_active_sum_chunk gives the sum; mean = sum/n).

    laser_corr = compute_laser_columns_chunk(bxraw, scale)
    laser_uncorr = compute_laser_columns_chunk(bxraw_origin, scale)

    _plot_lasers_core(avg, avg_origin, laser_corr, laser_uncorr, cfg, fill)