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

"""
    Plot the instant lumi per bcid
"""
def plot_hist_bx(data, cfg, fill, label):
    # TODO need to pass the year here
    # TODO can also probably add whether the plots are preliminary to the config
    fig, ax = create_double_figure('BCID', 'Instantaneous luminosity [Hz/ub]', '', fill)

    # TODO I do not understand the reason for this (just copied from the old code)
    #hist = [x * 11245.6/cfg.afterglow.sigvis if abs(x) < 1e3 else 0 for x in np.stack(data['bxraw']).mean(axis=0)]
    hist = [x * 11245.6/cfg.afterglow.sigvis for x in np.stack(data['bxraw']).mean(axis=0)]

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

    # TODO this should be changed to a dedicated plotting directory
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")

    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, f"per_bcid_hist_{label}.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

"""
    Plot a comparison between the corrected and uncorrected inst lumi
"""
def plot_lumi_comparison(data, data_origin, cfg, active_mask, fill):
    hists = np.stack(data['bxraw']) * 11245.6 / cfg.afterglow.sigvis
    hists_origin = np.stack(data_origin['bxraw']) * 11245.6 / cfg.afterglow.sigvis

    avg = np.array([np.multiply(hist, active_mask).sum() for hist in hists])
    avg_origin = np.array([np.multiply(hist, active_mask).sum() for hist in hists_origin])

    index = np.arange(len(hists))

    # TODO need to pass the year here
    # TODO can also probably add whether the plots are preliminary to the config
    fig, ax = create_double_figure('Fill duration [s]', 'Instantenious luminosity [Hz/µb]', '', fill)

    ax[0].plot(index, avg, '.', label='Corr. instantenious luminosity')
    ax[0].plot(index, avg_origin, '.', label='Uncorr. instantenious luminosity')
    ax[0].legend(loc='upper right', frameon=False, fontsize=12)

    ratio = avg / avg_origin
    ax[1].plot(index, ratio, '.', label='Corr. instantenious luminosity')

    plt.tight_layout()
    plt.ylim(0.5, 1.5)

    # TODO this should be changed to a dedicated plotting directory
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")

    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, f"instantaneous.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

"""
    Plot residuals
"""
def plot_residuals(data, cfg, active_mask, fill, label):
    """
    Plot Type1 / Type2 residuals and save compact point clouds for later mega-plots.

    Type1:
        BX+1 only

    Type2:
        all non-colliding BX excluding BX+1 and bx_to_clean

    Residuals are expressed in percent of the mean colliding SBIL:
        residual_pct = 100 * mean(non-colliding selection) / mean(colliding BX)
    """
    hists = np.stack(data["bxraw"]).astype(np.float64) * 11245.6 / cfg.afterglow.sigvis
    active_mask = np.asarray(active_mask, dtype=bool)

    if hists.ndim != 2:
        raise ValueError(f"Expected bx histograms to be 2D, got shape {hists.shape}")

    n_bx = hists.shape[1]
    if active_mask.shape[0] != n_bx:
        raise ValueError(
            f"active_mask length {active_mask.shape[0]} does not match histogram BX dimension {n_bx}"
        )

    bx_to_clean = np.asarray(cfg.afterglow.bx_to_clean, dtype=np.int64) % n_bx
    clean_mask = np.zeros(n_bx, dtype=bool)
    clean_mask[bx_to_clean] = True

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    prev_is_col = np.roll(active_mask, 1)

    # Type1 = BX+1 only
    type1_mask = (~active_mask) & prev_is_col
    type1_mask[clean_mask] = False

    # Type2 = all non-colliding BX excluding BX+1 and bx_to_clean
    type2_mask = (~active_mask) & (~type1_mask) & (~clean_mask)

    n_col = int(active_mask.sum())
    n_type1 = int(type1_mask.sum())
    n_type2 = int(type2_mask.sum())

    if n_col == 0:
        raise ValueError("No colliding BX found in active_mask")
    if n_type1 == 0:
        raise ValueError("No Type1 BX found after applying masks")
    if n_type2 == 0:
        raise ValueError("No Type2 BX found after applying masks")

    # ------------------------------------------------------------------
    # Per-histogram means
    # ------------------------------------------------------------------
    avg_col = np.array([hist[active_mask].mean() for hist in hists], dtype=np.float64)
    avg_type1 = np.array([hist[type1_mask].mean() for hist in hists], dtype=np.float64)
    avg_type2 = np.array([hist[type2_mask].mean() for hist in hists], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        type1_pct = 100.0 * avg_type1 / avg_col
        type2_pct = 100.0 * avg_type2 / avg_col

    # ------------------------------------------------------------------
    # Remove meaningless low-SBIL points
    # ------------------------------------------------------------------
    sbil_min = 0.1

    finite1 = np.isfinite(avg_col) & np.isfinite(type1_pct) & (avg_col > sbil_min)
    finite2 = np.isfinite(avg_col) & np.isfinite(type2_pct) & (avg_col > sbil_min)

    type1_points = np.column_stack([avg_col[finite1], type1_pct[finite1]])
    type2_points = np.column_stack([avg_col[finite2], type2_pct[finite2]])

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")

    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save compact point clouds: one HDF5 file per fill
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Common plot styling
    # ------------------------------------------------------------------
    def _compute_ylim(y):
        """
        Dynamic y-range:
        - always include the allowed corridor ±0.2%
        - also include outliers before correction
        """
        y = np.asarray(y, dtype=np.float64)
        y = y[np.isfinite(y)]

        if y.size == 0:
            return (-0.25, 0.25)

        y_abs = np.max(np.abs(y))
        y_lim = max(0.25, 1.15 * y_abs)

        # avoid absurdly tiny limits
        y_lim = max(y_lim, 0.25)

        return (-y_lim, y_lim)

    def _style_residual_plot(fig, ax, ylabel, yvals):
        ax.set_xlabel("Mean SBIL [Hz/µb]")
        ax.set_ylabel(ylabel)

        ymin, ymax = _compute_ylim(yvals)
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

    # ------------------------------------------------------------------
    # Type1 plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if type1_points.size > 0:
        ax.plot(type1_points[:, 0], type1_points[:, 1], ".", markersize=3)
        _style_residual_plot(fig, ax, "Type1 Residual [% of mean SBIL]", type1_points[:, 1])
    else:
        _style_residual_plot(fig, ax, "Type1 Residual [% of mean SBIL]", np.array([]))

    png_path = os.path.join(output_dir, f"type1_residuals_{label}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Type2 plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if type2_points.size > 0:
        ax.plot(type2_points[:, 0], type2_points[:, 1], ".", markersize=3)
        _style_residual_plot(fig, ax, "Type2 Residual [% of mean SBIL]", type2_points[:, 1])
    else:
        _style_residual_plot(fig, ax, "Type2 Residual [% of mean SBIL]", np.array([]))

    png_path = os.path.join(output_dir, f"type2_residuals_{label}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_lasers(data, data_origin, cfg, active_mask, fill):
    """
    Diagnostic laser plots + raw point dump for later offline analysis.

    What is saved
    -------------
    For each laser BCID:
      - uncorrected time evolution:   (index, value)
      - corrected time evolution:     (index, value)
      - uncorr laser vs SBIL points:  (mean_colliding_sbil, laser_value)
      - corr   laser vs SBIL points:  (mean_colliding_sbil, laser_value)

    No online fit is performed here.
    """
    hists = np.stack(data["bxraw"]).astype(np.float64) * 11245.6 / cfg.afterglow.sigvis
    hists_origin = np.stack(data_origin["bxraw"]).astype(np.float64) * 11245.6 / cfg.afterglow.sigvis

    active_mask = np.asarray(active_mask, dtype=bool)

    n_active = int(np.count_nonzero(active_mask))
    if n_active == 0:
        raise ValueError("plot_lasers: active_mask has zero active BX")

    # Mean colliding SBIL per histogram
    avg = hists[:, active_mask].mean(axis=1)
    avg_origin = hists_origin[:, active_mask].mean(axis=1)

    # Time-like index inside fill
    index = np.arange(len(hists), dtype=np.float64)

    # Hardcoded laser BCIDs
    laser_bcid = [3489, 3490, 3491, 3492]

    # Output dir
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")

    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Laser evolution vs time index
    # ------------------------------------------------------------
    fig, ax = create_double_figure(
        "Fill duration [a.u.]",
        "Uncorrected",
        "Corrected",
        fill,
        ratio=1,
    )

    for bcid in laser_bcid:
        ax[0].plot(index, hists_origin[:, bcid], ".", label=f"LASER BCID {bcid}")
        ax[1].plot(index, hists[:, bcid], ".", label=f"LASER BCID {bcid}")

    ax[0].legend(loc="upper right", frameon=False, fontsize=12)
    ax[1].legend(loc="upper right", frameon=False, fontsize=12)

    png_path = os.path.join(output_dir, "laser_evolution.png")
    plt.savefig(png_path, dpi=500)
    plt.close(fig)

    # ------------------------------------------------------------
    # Laser vs SBIL (scatter only, no fit)
    # ------------------------------------------------------------
    fig, ax = create_double_figure(
        "SBIL [Hz/µb]",
        "Uncorr. laser bins",
        "Corr. laser bins",
        fill,
        ratio=1,
    )

    h5_path = os.path.join(output_dir, f"laser_summary_fill_{fill}.h5")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["fill"] = int(fill)
        h5.attrs["n_histograms"] = int(len(hists))
        h5.attrs["n_active_bx"] = int(n_active)

        # Save global x-axes once
        h5.create_dataset("global/index", data=index, compression="gzip")
        h5.create_dataset("global/avg_uncorr", data=avg_origin, compression="gzip")
        h5.create_dataset("global/avg_corr", data=avg, compression="gzip")

        for bcid in laser_bcid:
            y_unc = hists_origin[:, bcid]
            y_cor = hists[:, bcid]

            # Scatter plots
            ax[0].plot(avg_origin, y_unc, ".", label=f"LASER BCID {bcid}")
            ax[1].plot(avg,        y_cor, ".", label=f"LASER BCID {bcid}")

            grp = h5.create_group(f"bcid_{bcid}")
            grp.attrs["bcid"] = int(bcid)

            # Time evolution points
            grp.create_dataset(
                "uncorr_time_points",
                data=np.column_stack([index, y_unc]),
                compression="gzip",
            )
            grp.create_dataset(
                "corr_time_points",
                data=np.column_stack([index, y_cor]),
                compression="gzip",
            )

            # SBIL scatter points
            grp.create_dataset(
                "uncorr_sbil_points",
                data=np.column_stack([avg_origin, y_unc]),
                compression="gzip",
            )
            grp.create_dataset(
                "corr_sbil_points",
                data=np.column_stack([avg, y_cor]),
                compression="gzip",
            )

            # Minimal metadata
            grp.attrs["uncorr_n_points"] = int(len(y_unc))
            grp.attrs["corr_n_points"] = int(len(y_cor))
            grp.attrs["columns_time"] = np.array(["index", "laser_value"], dtype="S32")
            grp.attrs["columns_sbil"] = np.array(["mean_colliding_sbil", "laser_value"], dtype="S32")

    ax[0].legend(loc="upper right", frameon=False, fontsize=12)
    ax[1].legend(loc="upper right", frameon=False, fontsize=12)

    png_path = os.path.join(output_dir, "laser_sbil.png")
    plt.savefig(png_path, dpi=500)
    plt.close(fig)

    # TODO find beam information
    # Bunch intensities + luminosities
    """
    fig, ax = create_double_figure('Fill duration [s]', 'Beam intensity [N protons]', 'Instantenious luminosity [Hz/µb]', fill)

    ax[0].plot(index, intensity1, '.', label='B1 Intensity')
    ax[0].plot(index, intensity2, '.', label='B2 Intensity')

    ax[1].plot(index, avg_origin, '.', label='Origin')
    ax[1].plot(index, avg, '.', label='With corrections')

    ax[0].legend(loc='upper right', frameon=False, fontsize=12)
    ax[1].legend(loc='upper right', frameon=False, fontsize=12)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"beam_intensity.png")
    plt.savefig(png_path, dpi=500)
    plt.close(fig)
    """

