# src/hfpipe/plot/plots.py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

# конверсия, как в твоих скриптах
SCALE = 11245.6 / 3200.0


def _time_axis(meta, n):
    """Вернёт (x, xlabel): либо время, либо индекс записи."""
    if "timestampsec" in getattr(meta, "columns", []) and len(meta) >= n:
        try:
            x = meta["timestampsec"].to_numpy()[:n]
            t0 = int(x[0])
            return x - t0, "Elapsed time [s]"
        except Exception:
            pass
    return np.arange(n), "Record index"


def per_bx(mean_hist: np.ndarray, label: str, fill: int, out: str | None = None):
    """Двухпанельный plot среднего по BX гистограммы."""
    x = np.arange(mean_hist.size)
    y = np.where(np.isfinite(mean_hist), mean_hist, 0.0) * SCALE

    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    hep.cms.label("Preliminary", loc=0, data=True, year=2024, rlabel=f"Fill {fill}", ax=ax[0])

    ax[0].plot(x, y, ".", label=label)
    ax[0].set_ylabel("Instantaneous luminosity [Hz/µb]")
    ax[0].legend(loc="upper right", frameon=False)

    ax[1].plot(x, y, ".", label=label)
    ax[1].set_xlabel("BCID")
    ax[1].set_ylabel("Value")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300)
    return fig


def instant(meta, sbil_before, sbil_after, fill: int, out: str):
    """
    Плот «инстант»:
      верх: суммарная по активным BX мгновенная светимость до/после;
      низ: их отношение (after / before).
    """
    sbil_before = np.asarray(sbil_before, dtype=float)
    sbil_after = np.asarray(sbil_after, dtype=float)
    n = min(len(sbil_before), len(sbil_after))
    sbil_before = sbil_before[:n]
    sbil_after = sbil_after[:n]

    x, xlabel = _time_axis(meta, n)

    y1 = sbil_before * SCALE
    y2 = sbil_after * SCALE
    ratio = np.divide(y2, y1, out=np.ones_like(y1), where=(y1 != 0))

    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    hep.cms.label("Preliminary", loc=0, data=True, year=2024, rlabel=f"Fill {fill}", ax=ax[0])

    ax[0].plot(x, y1, ".", label="Uncorrected (before LSQ)")
    ax[0].plot(x, y2, ".", label="Corrected (after LSQ)")
    ax[0].set_ylabel("Instantaneous luminosity [Hz/µb]")
    ax[0].legend(loc="upper left", frameon=False)

    ax[1].plot(x, ratio, ".", label="after/before")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("Ratio")

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300)
    return fig


__all__ = ["per_bx", "instant"]
