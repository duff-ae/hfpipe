"""
Beam centroid Z vs sigvis correlation analysis
CMS style plots via mplhep

python3 centroid_analysis.py \
  --centroid input/centroid/lumi_centroid.csv \
  --lumi input/luminosity/lumi24.csv \
  --emit input/scans/scan24_new.csv \
  --outdir plots_centroid/
"""

"""
Beam centroid Z vs sigvis correlation analysis
CMS style plots via mplhep

python3 centroid_analysis.py \
  --centroid input/centroid/lumi_centroid.csv \
  --lumi input/luminosity/lumi24.csv \
  --emit input/scans/scan24_new.csv \
  --outdir plots_centroid/
"""

import re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplhep as hep
from scipy import stats

hep.style.use(hep.style.ROOT)
hep.style.use(hep.style.CMS)

PETROFF = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
           "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams.update({
    "font.size": 14,
    "axes.prop_cycle": plt.cycler("color", PETROFF),
})

C_SVIS   = "#bd1f01"
C_CZALL  = "#3f90da"
C_CZSCAN = "#832db6"
C_FIT    = "#1a1a1a"
C_ZERO   = "#aaaaaa"
PERIOD_COLORS = ["#3f90da", "#bd1f01", "#832db6", "#1A8C6E"]


def _fig(w=16, h=6):
    return plt.figure(figsize=(w, h))


def _cms(ax, rlabel="2024 (13.6 TeV)"):
    hep.cms.label("Preliminary", ax=ax, data=True, loc=0, rlabel=rlabel)


def _sx(ax):
    ax.yaxis.set_tick_params(which="both", right=True, direction="in")
    ax.xaxis.set_tick_params(which="both", top=True, direction="in")
    ax.grid(axis="y", which="major", color="#EBEBEB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _xlbl(ax, mode="lumi"):
    if mode == "lumi":
        ax.set_xlabel(r"Integrated luminosity [fb$^{-1}$]")
    else:
        ax.set_xlabel("Fill number")


def _xv(df, mode="lumi"):
    return df["lumi_mid"].values if mode == "lumi" else df["fill"].values


# ── PARSERS ──────────────────────────────────────────────────

def parse_centroid(path):
    df = pd.read_csv(path, comment="#", skiprows=1, names=["ts", "cz"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["cz"] = pd.to_numeric(df["cz"], errors="coerce")
    df = df.dropna().sort_values("ts").reset_index(drop=True)
    print(f"  centroid: {len(df)} entries  [{df['cz'].min():.1f}, {df['cz'].max():.1f}] mm")
    return df


def parse_lumi(path):
    df = pd.read_csv(path, comment="#",
                     names=["run_fill", "ls", "time", "beamstatus",
                            "E_GeV", "delivered", "recorded", "avgpu", "source"])
    df[["run", "fill"]] = df["run_fill"].str.split(":", expand=True).astype(int)
    df["ts"] = pd.to_datetime(df["time"], format="%m/%d/%y %H:%M:%S", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def parse_emittance(path, det="HFET", fittype="SG"):
    df = pd.read_csv(path)
    n0 = len(df)
    print(f"  emittance raw: {n0} rows  detectors: {sorted(df['det'].unique())}")

    if "svis" not in df.columns:
        raise SystemExit("Column 'svis' not found in emittance input.")
    n_before_svis = len(df)
    df = df[df["svis"] >= 3200].copy()
    print(f"  svis>=3000 cut: {len(df)}/{n_before_svis}")

    df = df[df["det"]        == det   ].copy()
    #df = df[df["scanName"]   == "emit9"]
    #df = df[df["scanTiming"] == "early"]
    #df = df[df["scanSteps"]  == 9     ]
    #df = df[df["scanBeta"]   == 120   ]
    #df = df[df["nbcid"]      >  400   ]
    if fittype is not None:
        if "fittype" not in df.columns:
            raise SystemExit("Column 'fittype' not found in emittance input.")
        n_before = len(df)
        print(f"  fittype values present: {sorted(df['fittype'].unique())}")
        df = df[df["fittype"] == fittype].copy()
        print(f"  fittype=={fittype} cut: {len(df)}/{n_before}")
    print(f"  after cuts: {len(df)}/{n0}")
    if len(df) == 0:
        raise SystemExit("No rows passed quality cuts.")

    def scan_window(sf):
        tok = re.findall(r'\d{2}[A-Za-z]{3}\d{2}_\d{6}', str(sf))
        if len(tok) >= 2:
            return (pd.to_datetime(tok[0], format='%d%b%y_%H%M%S', utc=True),
                    pd.to_datetime(tok[1], format='%d%b%y_%H%M%S', utc=True))
        tok2 = re.findall(r'\d{12}', str(sf))
        if len(tok2) >= 2:
            return (pd.to_datetime(tok2[0], format='%y%m%d%H%M%S', utc=True),
                    pd.to_datetime(tok2[1], format='%y%m%d%H%M%S', utc=True))
        return pd.NaT, pd.NaT

    times = df["scanFile"].apply(scan_window).tolist()
    df["scan_t0"] = [t[0] for t in times]
    df["scan_t1"] = [t[1] for t in times]
    df = (df.sort_values(["fill", "scan_t0"])
            .drop_duplicates("fill", keep="first")
            .reset_index(drop=True))
    df = df[["fill", "scanFile", "svis", "svisrms", "scan_t0", "scan_t1"]]
    print(f"  after dedup: {len(df)} fills  svis [{df['svis'].min():.1f}, {df['svis'].max():.1f}]")
    return df


# ── CENTROID HELPERS ─────────────────────────────────────────

def robust_median(v, k=3.0):
    v = v.astype(float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    mask = np.abs(v - med) < k * mad if mad > 0 else np.ones(len(v), bool)
    c = v[mask]
    return float(np.median(c)), (float(np.std(c)) if len(c) > 1 else 0.0)


def cz_window(cdf, t0, t1, pad=pd.Timedelta(0)):
    sub = cdf.loc[(cdf["ts"] >= t0 - pad) & (cdf["ts"] <= t1 + pad), "cz"].values
    if len(sub) == 0: return np.nan, np.nan, 0
    if len(sub) == 1: return float(sub[0]), 0.0, 1
    m, s = robust_median(sub)
    return m, s, len(sub)


# ── BUILD ────────────────────────────────────────────────────

def build(cdf, ldf, edf, scan_pad_min, lumi_scale):
    pad = pd.Timedelta(minutes=scan_pad_min)

    sb = ldf[ldf["beamstatus"] == "STABLE BEAMS"].groupby("fill")
    fr = sb["ts"].agg(t0="min", t1="max").reset_index()

    lumi_pf = (ldf.groupby("fill")["delivered"].sum()
                  .reset_index().rename(columns={"delivered": "lumi_fill"})
                  .sort_values("fill").reset_index(drop=True))
    lumi_pf["lumi_fill"] *= lumi_scale
    lumi_pf["lumi_cum"]   = lumi_pf["lumi_fill"].cumsum()
    lumi_pf["lumi_mid"]   = lumi_pf["lumi_cum"] - lumi_pf["lumi_fill"] / 2

    cz_rows = []
    for _, r in fr.iterrows():
        m, s, n = cz_window(cdf, r["t0"], r["t1"], pd.Timedelta(minutes=5))
        cz_rows.append({"fill": r["fill"], "cz_fill": m, "cz_fill_err": s,
                        "cz_fill_n": n, "t0": r["t0"], "t1": r["t1"]})
    # cz_df keeps lumi_mid for plotting (plot2); it is NOT merged into df below
    # to avoid a duplicate lumi_mid column (cz_df already has it via left-join).
    cz_df = (pd.DataFrame(cz_rows)
               .merge(lumi_pf[["fill", "lumi_mid"]], on="fill", how="left"))

    # Drop lumi_mid from cz_df before merging into df — lumi_pf is the single
    # source of truth for lumi_mid / lumi_cum.
    cz_cols = ["fill", "cz_fill", "cz_fill_err", "cz_fill_n", "t0", "t1"]
    df = (edf.merge(cz_df[cz_cols], on="fill", how="inner")
             .merge(lumi_pf[["fill", "lumi_mid", "lumi_cum"]], on="fill", how="left")
             .dropna(subset=["cz_fill", "svis"])
             .sort_values("fill").reset_index(drop=True))

    sl, ic, rv, *_ = stats.linregress(df["lumi_mid"], df["svis"])
    df["svis_trend"]   = ic + sl * df["lumi_mid"]
    df["svis_res"]     = (df["svis"] - df["svis_trend"]) / df["svis_trend"]
    df["svis_res_err"] = df["svisrms"] / df["svis_trend"]

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["scan_t0"]) or pd.isna(r["scan_t1"]):
            rows.append({"cz_scan": np.nan, "cz_scan_err": np.nan, "cz_scan_n": 0})
        else:
            # Clip scan window to fill boundaries using pandas-native comparisons
            # to avoid undefined behaviour of Python max()/min() on Timestamps.
            t0_raw = r["scan_t0"] - pad
            t1_raw = r["scan_t1"] + pad
            t0_query = r["t0"] if t0_raw < r["t0"] else t0_raw
            t1_query = r["t1"] if t1_raw > r["t1"] else t1_raw
            if t0_query >= t1_query:
                rows.append({"cz_scan": np.nan, "cz_scan_err": np.nan, "cz_scan_n": 0})
                continue
            m, s, n = cz_window(cdf, t0_query, t1_query)
            rows.append({"cz_scan": m, "cz_scan_err": s, "cz_scan_n": n})
    # Reset index on both sides before concat to guarantee row alignment.
    df = df.reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame(rows).reset_index(drop=True)], axis=1)

    total = lumi_pf["lumi_fill"].sum()
    n_nan_res = df["svis_res"].isna().sum()
    if n_nan_res > 0:
        print(f"  WARNING: {n_nan_res}/{len(df)} fills have NaN svis_res")
    print(f"Matched {len(df)} fills | fit on lumi: slope={sl:.4f}/fb⁻¹  R²={rv**2:.4f}")
    print(f"Total lumi = {total:.3f} fb⁻¹ | scan_pad=±{scan_pad_min} min")
    print(f"cz_scan: ok={df['cz_scan'].notna().sum()}/{len(df)}  "
          f"mean n_pts={df['cz_scan_n'].mean():.1f}")
    return df, sl, ic, rv, cz_df, lumi_pf


# ── PLOT 1: svis + fit | residual ────────────────────────────

def plot1(df, sl, ic, rv, out, mode="lumi"):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 1, hspace=0.06, height_ratios=[1.6, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax0.tick_params(labelbottom=False)
    _sx(ax0); _sx(ax1)

    x  = _xv(df, mode)
    xl = r"slope $= {:.3f}$ / fb$^{{-1}}$".format(sl) if mode == "lumi" else f"slope $= {sl:.3f}$ / fill"

    ax0.errorbar(x, df["svis"], yerr=df["svisrms"],
                 fmt="o", ms=5, color=C_CZALL, ecolor="#BBBBBB",
                 capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                 label=r"$\sigma_\mathrm{vis}$ (HFET)")
    xf = np.array([x.min(), x.max()])
    ax0.plot(xf, ic + sl * xf, "-", color=C_FIT, lw=1.8, zorder=4,
             label=f"Linear fit  $R^2={rv**2:.4f}$,  {xl}")
    ax0.set_ylabel(r"$\sigma_\mathrm{vis}$")
    ax0.legend(loc="upper right"); _cms(ax0)

    ax1.errorbar(x, df["svis_res"], yerr=df["svis_res_err"],
                 fmt="o", ms=5, color=C_SVIS, ecolor="#BBBBBB",
                 capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                 label=r"$(\sigma_\mathrm{vis}-\sigma^\mathrm{trend})/\sigma^\mathrm{trend}$")
    ax1.axhline(0, color=C_ZERO, lw=1.0, ls="--")
    valid_res = df["svis_res"].dropna()
    lim = max(np.percentile(np.abs(valid_res), 98) * 1.5, 0.01) if len(valid_res) > 0 else 0.05
    ax1.set_ylim(-lim, lim)
    ax1.set_ylabel("Relative residual")
    ax1.legend(loc="upper right"); _xlbl(ax1, mode)

    p = out / f"plot1_svis_{mode}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")


# ── PLOT 2: centroid overview ────────────────────────────────

def plot2(df, cz_df, out, mode="lumi"):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 1, hspace=0.07, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax0.tick_params(labelbottom=False)
    _sx(ax0); _sx(ax1)

    cz_s  = cz_df.dropna(subset=["cz_fill"]).sort_values("fill")
    x0    = _xv(cz_s, mode)
    mean0 = cz_s["cz_fill"].mean()

    ax0.errorbar(x0, cz_s["cz_fill"], yerr=cz_s["cz_fill_err"].fillna(0),
                 fmt="o", ms=5, color=C_CZALL, ecolor="#BBBBBB",
                 capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                 label="Centroid $z$ — per-fill median")
    ax0.axhline(mean0, color=C_FIT, lw=1.4, ls="--", alpha=0.6,
                label=f"Mean = {mean0:.1f} mm")
    ax0.axhline(0, color=C_ZERO, lw=0.8, ls=":")
    sp0 = np.nanpercentile(np.abs(cz_s["cz_fill"] - mean0), 99) * 1.8
    ax0.set_ylim(mean0 - max(sp0, 3), mean0 + max(sp0, 3))
    ax0.set_ylabel(r"Centroid $z$ [mm]")
    ax0.legend(loc="upper right"); _cms(ax0)

    cz_sc = df.dropna(subset=["cz_scan"]).sort_values("fill")
    if len(cz_sc) > 0:
        x1    = _xv(cz_sc, mode)
        mean1 = cz_sc["cz_scan"].mean()
        ax1.errorbar(x1, cz_sc["cz_scan"], yerr=cz_sc["cz_scan_err"].fillna(0),
                     fmt="s", ms=5, color=C_CZSCAN, ecolor="#BBBBBB",
                     capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                     label="Centroid $z$ — scan window")
        ax1.axhline(mean1, color=C_FIT, lw=1.4, ls="--", alpha=0.6,
                    label=f"Mean = {mean1:.1f} mm")
        ax1.axhline(0, color=C_ZERO, lw=0.8, ls=":")
        sp1 = np.nanpercentile(np.abs(cz_sc["cz_scan"] - mean1), 99) * 1.8
        ax1.set_ylim(mean1 - max(sp1, 3), mean1 + max(sp1, 3))
    ax1.set_ylabel(r"Centroid $z$ [mm]")
    ax1.legend(loc="upper right"); _xlbl(ax1, mode)

    p = out / "plot2_centroid.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")


# ── PLOT 3: residual | centroid scan ─────────────────────────

def plot3(df, scan_pad_min, out, mode="lumi"):
    mask = df["cz_scan"].notna()
    x    = _xv(df, mode)
    cz   = df["cz_scan"].values
    cze  = np.where(np.isnan(df["cz_scan_err"].values), 0, df["cz_scan_err"].values)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 1, hspace=0.06, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax0.tick_params(labelbottom=False)
    _sx(ax0); _sx(ax1)

    ax0.errorbar(x, df["svis_res"], yerr=df["svis_res_err"],
                 fmt="o", ms=5, color=C_SVIS, ecolor="#BBBBBB",
                 capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                 label=r"$(\sigma_\mathrm{vis}-\sigma^\mathrm{trend})/\sigma^\mathrm{trend}$")
    ax0.axhline(0, color=C_ZERO, lw=1.0, ls="--")
    valid_res3 = df["svis_res"].dropna()
    lim = max(np.percentile(np.abs(valid_res3), 99) * 1.6, 0.01) if len(valid_res3) > 0 else 0.05
    ax0.set_ylim(-lim, lim)
    ax0.set_ylabel("Relative residual")
    ax0.legend(loc="upper right"); _cms(ax0)

    mean_cz = np.nanmean(cz[mask]) if mask.any() else 0.0
    ax1.errorbar(x[mask], cz[mask], yerr=cze[mask],
                 fmt="s", ms=5, color=C_CZSCAN, ecolor="#BBBBBB",
                 capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                 label=f"Centroid $z$ — scan $\\pm${scan_pad_min} min")
    ax1.axhline(mean_cz, color=C_FIT, lw=1.4, ls="--", alpha=0.6,
                label=f"Mean = {mean_cz:.1f} mm")
    ax1.axhline(0, color=C_ZERO, lw=0.8, ls=":")
    if mask.any():
        sp = np.nanpercentile(np.abs(cz[mask] - mean_cz), 99) * 2.0
        ax1.set_ylim(mean_cz - max(sp, 5), mean_cz + max(sp, 5))
    ax1.set_ylabel(r"Centroid $z$ [mm]")
    ax1.legend(loc="upper right"); _xlbl(ax1, mode)

    p = out / f"plot3_residual_centroid_{mode}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")


# ── PLOT 4a: correlation scatter ─────────────────────────────

def plot4a(df, out, periods=None, mode="lumi"):
    sub = df.dropna(subset=["cz_scan", "svis_res"]).copy()
    if len(sub) < 3:
        print("  plot4a: not enough data"); return

    ncols = 2 if periods else 1
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 9), squeeze=False)
    axes = axes[0]

    def _one(ax, data, color, label=""):
        if len(data) < 3: return
        ax.errorbar(data["cz_scan"], data["svis_res"],
                    xerr=data["cz_scan_err"].fillna(0),
                    yerr=data["svis_res_err"],
                    fmt="o", ms=6, color=color, ecolor="#BBBBBB",
                    capsize=2, elinewidth=0.9, linestyle="none", zorder=3,
                    label=label)
        sl, ic, r, pv, _ = stats.linregress(data["cz_scan"], data["svis_res"])
        xr = np.linspace(data["cz_scan"].min(), data["cz_scan"].max(), 200)
        ax.plot(xr, ic + sl * xr, "-", color=color, lw=1.8, zorder=4, alpha=0.7)
        return r, pv

    ax = axes[0]
    sl, ic, r, pv, _ = stats.linregress(sub["cz_scan"], sub["svis_res"])
    _one(ax, sub, C_CZALL,
         label=f"All fills  $r={r:.3f}$, $p={pv:.3f}$  (n={len(sub)})")
    ax.axhline(0, color=C_ZERO, lw=0.9, ls="--", alpha=0.5)
    ax.axvline(0, color=C_ZERO, lw=0.9, ls="--", alpha=0.5)
    ax.set_xlabel(r"Centroid $z$ — scan window [mm]")
    ax.set_ylabel(r"$(\sigma_\mathrm{vis}-\sigma^\mathrm{trend})/\sigma^\mathrm{trend}$")
    ax.legend(loc="upper left"); _sx(ax); _cms(ax)

    if periods:
        ax2 = axes[1]
        for i, (lmin, lmax, lbl) in enumerate(periods):
            c   = PERIOD_COLORS[i % len(PERIOD_COLORS)]
            sel = sub[(sub["lumi_mid"] >= lmin) & (sub["lumi_mid"] <= lmax)]
            if len(sel) < 3:
                print(f"  plot4a: {lbl} only {len(sel)} pts, skip"); continue
            sl2, ic2, r2, pv2, _ = stats.linregress(sel["cz_scan"], sel["svis_res"])
            _one(ax2, sel, c,
                 label=f"{lbl}  $r={r2:.3f}$, $p={pv2:.3f}$  (n={len(sel)})")
        ax2.axhline(0, color=C_ZERO, lw=0.9, ls="--", alpha=0.5)
        ax2.axvline(0, color=C_ZERO, lw=0.9, ls="--", alpha=0.5)
        ax2.set_xlabel(r"Centroid $z$ — scan window [mm]")
        ax2.set_ylabel(r"$(\sigma_\mathrm{vis}-\sigma^\mathrm{trend})/\sigma^\mathrm{trend}$")
        ax2.legend(loc="upper left"); _sx(ax2); _cms(ax2)

    fig.tight_layout()
    p = out / "plot4a_correlation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")


# ── PLOT 4b: ratio residual / cz_deviation vs lumi ───────────

def plot4b(df, out, mode="lumi"):
    sub = df.dropna(subset=["cz_scan", "svis_res"]).copy()
    if len(sub) < 3:
        print("  plot4b: not enough data"); return

    mean_cz       = sub["cz_scan"].mean()
    sub["cz_dev"] = sub["cz_scan"] - mean_cz
    sub = sub[sub["cz_dev"].abs() > 0.5].copy()
    if len(sub) < 3:
        print("  plot4b: cz deviation too small"); return

    ratio = sub["svis_res"] / sub["cz_dev"]
    rlim  = ratio.abs().quantile(0.97)
    ratio = ratio.clip(-rlim, rlim)
    rerr  = (ratio.abs() * np.sqrt(
        (sub["svis_res_err"] / sub["svis_res"].abs().clip(lower=1e-5))**2 +
        (sub["cz_scan_err"].fillna(0) / sub["cz_dev"].abs().clip(lower=0.5))**2
    )).clip(upper=rlim)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.errorbar(_xv(sub, mode), ratio.values, yerr=rerr.values,
                fmt="D", ms=5, color="#94a4a2", ecolor="#CCCCCC",
                capsize=2.5, elinewidth=0.9, linestyle="none", zorder=3,
                label=r"$\Delta\sigma^\mathrm{rel}/\Delta z$  [mm$^{-1}$]")
    ax.axhline(0, color=C_ZERO, lw=1.2, ls="--")
    ax.axhline(ratio.median(), color="#94a4a2", lw=1.6, ls=":",
               label=f"Median = {ratio.median():.4f} mm$^{{-1}}$")
    ax.set_ylim(-rlim * 1.3, rlim * 1.3)
    ax.set_ylabel(r"$\Delta\sigma^\mathrm{rel}/\Delta z$  [mm$^{-1}$]")
    ax.legend(loc="upper right"); _xlbl(ax, mode); _sx(ax); _cms(ax)
    fig.tight_layout()
    p = out / f"plot4b_ratio_{mode}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")


# ── MAIN ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--centroid",   required=True)
    ap.add_argument("--lumi",       required=True)
    ap.add_argument("--emit",       required=True)
    ap.add_argument("--outdir",     default="plots")
    ap.add_argument("--det",        default="HFET")
    ap.add_argument("--fittype",    default="_SG_",
                    help="Value to filter the 'fittype' column on (set to '' to disable)")
    ap.add_argument("--scan-pad",   type=int,   default=180,
                    help="Minutes to extend scan window each side")
    ap.add_argument("--lumi-scale", type=float, default=1.0,
                    help="Scale factor for luminosity (e.g. 0.001 if input is /pb)")
    ap.add_argument("--plots",      default="1,2,3,4a,4b")
    ap.add_argument("--periods",    default=None,
                    help=r"Per-period ranges for plot4a: 'L0:L1:label,...' in fb⁻¹")
    args = ap.parse_args()

    wanted = set(args.plots.split(","))
    out    = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Parsing...")
    cdf = parse_centroid(args.centroid)
    ldf = parse_lumi(args.lumi)
    fittype = args.fittype if args.fittype else None
    edf = parse_emittance(args.emit, args.det, fittype)
    df, sl, ic, rv, cz_df, lumi_pf = build(cdf, ldf, edf, args.scan_pad, args.lumi_scale)

    periods = None
    if args.periods:
        periods = []
        for tok in args.periods.split(","):
            parts = tok.strip().split(":")
            periods.append((float(parts[0]), float(parts[1]),
                           parts[2] if len(parts) > 2 else tok))

    print("\nPlotting...")
    for mode in ("lumi", "fill"):
        if "1"  in wanted: plot1(df, sl, ic, rv, out, mode)
        if "2"  in wanted: plot2(df, cz_df, out, mode)
        if "3"  in wanted: plot3(df, args.scan_pad, out, mode)
        if "4a" in wanted: plot4a(df, out, periods=periods, mode=mode)
        if "4b" in wanted: plot4b(df, out, mode)
    print("Done.")


if __name__ == "__main__":
    main()