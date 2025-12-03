import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep as hep

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
    hist = [x * 11245.6/cfg.afterglow.sigvis if abs(x) < 1e3 else 0 for x in np.stack(data['bxraw']).mean(axis=0)]

    ax[0].bar(list(range(3564)), hist, label=label)
    ax[1].bar(list(range(3564)), hist, label=label)

    ax[0].legend(loc='upper right', frameon=False, fontsize=12)
    ax[1].legend(loc='upper right', frameon=False, fontsize=12)

    plt.tight_layout()

    if label == 'Corr. Luminosity':
        ax[1].set_ylim(-0.5, 0.5)
        ax[1].set_xlim(0, 500)
    else:
        ax[1].set_ylim(-0.5, 2)
        ax[1].set_xlim(0, 500)

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
def plot_residuals(data, cfg, active_mask, fill):
    hists = np.stack(data['bxraw']) * 11245.6 / cfg.afterglow.sigvis
    
    # calculate the inst lumi
    avg_col     = np.array([np.multiply(hist, active_mask).sum() for hist in hists])

    prev_is_col = np.roll(active_mask, 1)
    bxp1_idx    = ((~active_mask) & prev_is_col)
    bxp2_idx    = np.roll(bxp1_idx, 1)
    bxt2_idx    = ((~active_mask) & (~bxp1_idx) & (~bxp2_idx))

    # calculate the residuals
    avg_t1p1 = np.array([np.multiply(hist, bxp1_idx).mean() for hist in hists])
    avg_t1p2 = np.array([np.multiply(hist, bxp2_idx).mean() for hist in hists])
    avg_t2   = np.array([np.multiply(hist, bxt2_idx).mean() for hist in hists])
    
    # get the path to save plots
    plot_dir = getattr(cfg.io, "type1_dir", None)
    if plot_dir is None:
        plot_dir = os.path.join(cfg.io.output_dir, "type1")

    output_dir = os.path.join(plot_dir, str(fill))
    os.makedirs(output_dir, exist_ok=True)

    # p1 residuals plot
    fig = create_figure('Instantenious luminosity [Hz/µb]', 'p1 Residuals [Hz/µb]', fill)
    plt.plot(avg_col, avg_t1p1, '.')

    png_path = os.path.join(output_dir, f"p1_residuals.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

    # p2 residuals plot
    fig = create_figure('Instantenious luminosity [Hz/µb]', 'p2 Residuals [Hz/µb]', fill)
    plt.plot(avg_col, avg_t1p2, '.')

    png_path = os.path.join(output_dir, f"p2_residuals.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

    # t2 residuals plot
    fig = create_figure('Instantenious luminosity [Hz/µb]', 'T2 Residuals [Hz/µb]', fill)
    plt.plot(avg_col, avg_t2, '.')

    png_path = os.path.join(output_dir, f"T2_residuals.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
