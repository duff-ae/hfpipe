#!/usr/bin/env python3
"""Focused toy-MC validation for HF afterglow / recovery / Type-1 studies.

Install as src/hfcli/toy_mc_afterglow.py and run:
    python -m hfcli.toy_mc_afterglow

One 2448-bunch pattern is used. Active-BX truth PU is drawn independently from
Poisson distributions around PU=65 and PU=200, with truth rate = PU/8 Hz/ub.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from cffi import FFI
from scipy.optimize import curve_fit

from hfcore.hd5schema import BX_LEN
from hfcore.online_recovery import OnlineRecoverySolver
from hfcore.type1_apply import apply_type1_batch

HFSBR_PATH = Path("input/afterglow/hfet_lsq_test.txt")
ACTIVE_MASK_PATH = Path("input/filling_scheme/active_bx_2448.txt")
OUTPUT_DIR = Path("plots/toy_mc_afterglow")

YEAR = 2025
PLOT_TYPE = "Preliminary"
PETROFF_10 = [
    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
    "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd",
]

PU_MODES = (65.0, 200.0)
PLOT_PEDESTALS = (1.0, 3.0)
PEDESTAL_SCAN = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
PU_SCAN = (20.0, 40.0, 65.0, 80.0, 100.0, 120.0, 150.0, 180.0, 200.0, 220.0, 250.0)
N_SAMPLES = 10
SEED = 12345

ABORT_GAP_START = 3500
ABORT_GAP_N_SAMPLE = 13
ZERO_BX = (3553, 3554, 3555, 3556, 3557)
CG_TOL = 1e-8

ONLINE_RESIDUAL_YLIM = (-2.0, 1.0)
FFT_RESIDUAL_YLIM = (-1e-5, 1e-5)

TYPE1_FIXED_CORRECTION = 0.020
TYPE1_MODELS = ("fixed_2pct", "linear_2to3", "quadratic_2to3")

TRAIN_TAIL_FIT_START_OFFSET = 3
TRAIN_TAIL_MAX_POINTS = 200

# Exact forward partner of online_recovery.revert_afterglow.
_ffi = FFI()
_ffi.cdef("""
void subtract_afterglow_batch(const int *activeBXMask, float *hist,
                              const float *HFSBR, int nrows);
""")
_C = _ffi.verify(r"""
void subtract_afterglow_batch(const int *activeBXMask, float *hist,
                              const float *HFSBR, int nrows) {
    const int N = 3564;
    for (int row = 0; row < nrows; ++row) {
        float *h = hist + ((long long)row) * N;
        for (int ibx = 0; ibx < N; ++ibx) {
            if (activeBXMask[ibx] != 1) continue;
            const float source = h[ibx];
            for (int d = 1; d < N; ++d) {
                int j = ibx + d;
                if (j >= N) j -= N;
                h[j] -= source * HFSBR[d];
            }
        }
    }
}
""", extra_compile_args=["-O3"])


@dataclass
class MainCase:
    pu: float
    pedestal: float
    avg_rate: float
    active_mask: np.ndarray
    truth_pu: np.ndarray
    truth: np.ndarray
    raw_physical: np.ndarray
    raw_online_input: np.ndarray
    online: np.ndarray
    fft: np.ndarray
    recovered_raw: np.ndarray


@dataclass
class TailFit:
    offsets: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray
    tau: float
    baseline: float
    norm_offset: int
    inferred_hfsbr: np.ndarray


# -----------------------------------------------------------------------------
# Style / plotting
# -----------------------------------------------------------------------------

def setup_style() -> None:
    hep.style.use(hep.style.ROOT)
    hep.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", PETROFF_10)


def create_figure(x_axis: str, y_axis: str, rlabel: str):
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", PETROFF_10)
    hep.cms.label(PLOT_TYPE, loc=0, data=True, year=YEAR, rlabel=rlabel)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    return fig


def create_double_figure(x_axis: str, y_top: str, y_bottom: str, rlabel: str):
    fig, ax = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", PETROFF_10)
    hep.cms.label(PLOT_TYPE, loc=0, data=True, year=YEAR, rlabel=rlabel, ax=ax[0])
    ax[0].set_ylabel(y_top, fontsize=14)
    ax[1].set_xlabel(x_axis, fontsize=14)
    ax[1].set_ylabel(y_bottom, fontsize=14)
    for a in ax:
        a.minorticks_on()
        a.tick_params(bottom=True, top=True, left=True, right=True,
                      direction="in", which="both", labelsize=13)
    return fig, ax


def save(fig, outbase: Path) -> None:
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".pdf"))
    fig.savefig(outbase.with_suffix(".png"), dpi=250)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Common physics helpers
# -----------------------------------------------------------------------------

def _load(path: Path, dtype) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        x = np.loadtxt(path, dtype=dtype, delimiter=",")
    except ValueError:
        x = np.loadtxt(path, dtype=dtype)
    return np.asarray(x, dtype=dtype).reshape(-1)


def load_inputs() -> tuple[np.ndarray, np.ndarray]:
    h = _load(HFSBR_PATH, np.float64)
    mask = _load(ACTIVE_MASK_PATH, np.int32)
    if h.shape != (BX_LEN,) or mask.shape != (BX_LEN,):
        raise ValueError(f"Bad input shapes: H={h.shape}, mask={mask.shape}")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("active mask must contain 0/1")
    return h, mask


def generate_truth_pu(mask: np.ndarray, n: int, mean_pu: float,
                      rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, BX_LEN), dtype=np.float64)
    active = mask.astype(bool)
    out[:, active] = rng.poisson(mean_pu, size=(n, int(active.sum())))
    return out


def average_rate(truth: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(truth[:, mask.astype(bool)]))


def pedestal_pattern(ped4: np.ndarray) -> np.ndarray:
    ped4 = np.asarray(ped4, dtype=np.float64)
    if ped4.ndim == 1:
        ped4 = ped4[None, :]
    return ped4[:, np.arange(BX_LEN) % 4]


def estimate_pedestal(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float64)
    ped = np.empty((hist.shape[0], 4), dtype=np.float64)
    for phase in range(4):
        idx = ABORT_GAP_START + phase + 4 * np.arange(ABORT_GAP_N_SAMPLE)
        ped[:, phase] = np.mean(hist[:, idx], axis=1)
    return ped


def subtract_pedestal(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ped = estimate_pedestal(hist)
    return np.asarray(hist, dtype=np.float64) - pedestal_pattern(ped), ped


def convolve(hist: np.ndarray, h: np.ndarray) -> np.ndarray:
    fh = np.fft.fft(h)
    return np.fft.ifft(np.fft.fft(hist, axis=1) * fh[None, :], axis=1).real


def fft_inverse(hist: np.ndarray, h: np.ndarray) -> np.ndarray:
    fh = np.fft.fft(h)
    hscale = max(1.0, float(np.max(np.abs(fh))))
    eps = max(1e-12, CG_TOL, 1e-8 * hscale * hscale)
    y = np.fft.fft(hist, axis=1)
    x = np.conj(fh)[None, :] * y / (np.abs(fh)[None, :] ** 2 + eps)
    return np.fft.ifft(x, axis=1).real


def online_iterative(hist: np.ndarray, h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(hist, dtype=np.float32).copy()
    h32 = np.ascontiguousarray(h, dtype=np.float32)
    m32 = np.ascontiguousarray(mask, dtype=np.int32)
    _C.subtract_afterglow_batch(
        _ffi.cast("const int *", _ffi.from_buffer(m32)),
        _ffi.cast("float *", _ffi.from_buffer(out)),
        _ffi.cast("const float *", _ffi.from_buffer(h32)),
        int(out.shape[0]),
    )
    return out.astype(np.float64)


def mean_profile(x: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(x, dtype=np.float64), axis=0)


def integrated_bias(estimate: np.ndarray, truth: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    active = mask.astype(bool)
    e = np.sum(estimate[:, active], axis=1)
    t = np.sum(truth[:, active], axis=1)
    return 100.0 * (e / np.maximum(t, 1e-15) - 1.0)


def closure_rms(estimate: np.ndarray, reference: np.ndarray,
                norm: float) -> np.ndarray:
    d = (estimate - reference) / max(norm, 1e-15)
    return 100.0 * np.sqrt(np.mean(d * d, axis=1))


def build_case(mean_pu: float, pedestal_pct: float, h: np.ndarray,
               mask: np.ndarray, n: int, seed: int) -> MainCase:
    rng = np.random.default_rng(seed)
    truth_pu = generate_truth_pu(mask, n, mean_pu, rng)
    truth = truth_pu / 8.0
    avg = average_rate(truth, mask)

    ped = (pedestal_pct / 100.0) * avg
    raw = convolve(truth, h) + ped

    raw_online = raw.copy()
    raw_online[:, np.asarray(ZERO_BX, dtype=np.int64)] = 0.0

    online_before_ped = online_iterative(raw_online, h, mask)
    online, _ = subtract_pedestal(online_before_ped)

    fft, _ = subtract_pedestal(fft_inverse(raw, h))

    solver = OnlineRecoverySolver(hfsbr=h, active_mask=mask, zero_bx=ZERO_BX)
    recovered = solver.recover_batch(online.astype(np.float32)).recovered_raw.astype(np.float64)

    return MainCase(mean_pu, pedestal_pct, avg, mask, truth_pu, truth,
                    raw, raw_online, online, fft, recovered)


# -----------------------------------------------------------------------------
# Main study plots
# -----------------------------------------------------------------------------

def plot_truth_distribution(mean_pu: float, mask: np.ndarray, n: int,
                            seed: int, outdir: Path) -> None:
    pu = generate_truth_pu(mask, n, mean_pu, np.random.default_rng(seed))
    values = pu[:, mask.astype(bool)].ravel()
    lo = max(0, int(mean_pu - 5 * np.sqrt(mean_pu)))
    hi = int(mean_pu + 5 * np.sqrt(mean_pu)) + 2
    fig = create_figure("Truth pileup per active BX", "Entries",
                        f"Toy MC, PU {int(mean_pu)}")
    fig.gca().hist(values, bins=np.arange(lo, hi + 1) - 0.5,
                   histtype="step", linewidth=1.8)
    save(fig, outdir / f"00_truth_pu_distribution_pu{int(mean_pu)}")


def plot_rate_residual(truth: np.ndarray, estimate: np.ndarray, avg: float,
                       rlabel: str, estimate_label: str, outbase: Path,
                       ylim: tuple[float, float] | None = None) -> None:
    bx = np.arange(BX_LEN)
    t = mean_profile(truth)
    e = mean_profile(estimate)
    r = 100.0 * (e - t) / max(avg, 1e-15)

    fig, ax = create_double_figure(
        "BCID", "Rate [Hz/ub]", "Residual / <truth> [%]", rlabel)
    ax[0].plot(bx, t, ".", ms=3.0, label="Mean truth")
    ax[0].plot(bx, e, ".", ms=2.4, label=estimate_label)
    ax[0].legend(frameon=False, fontsize=11, loc="upper right")
    ax[1].plot(bx, r, ".", ms=2.4)
    ax[1].axhline(0.0, color="black", linewidth=1.0)
    if ylim is not None:
        ax[1].set_ylim(*ylim)
    ax[0].set_xlim(0, BX_LEN - 1)
    save(fig, outbase)


def plot_recovery(case: MainCase, outbase: Path) -> None:
    bx = np.arange(BX_LEN)
    original = mean_profile(case.raw_online_input)
    recovered = mean_profile(case.recovered_raw)
    residual = 100.0 * (recovered - original) / max(case.avg_rate, 1e-15)

    fig, ax = create_double_figure(
        "BCID", "Rate [Hz/ub]", "(recovered-original) / <truth> [%]",
        f"Toy MC, PU {int(case.pu)}, pedestal {case.pedestal:g}%")
    ax[0].plot(bx, original, ".", ms=3.0, label="Original contaminated raw")
    ax[0].plot(bx, recovered, ".", ms=2.4, label="Recovered raw")
    ax[0].legend(frameon=False, fontsize=11, loc="upper right")
    ax[1].plot(bx, residual, ".", ms=2.4)
    ax[1].axhline(0.0, color="black", linewidth=1.0)
    ax[0].set_xlim(0, BX_LEN - 1)
    save(fig, outbase)


def append_pedestal_rows(rows: list[dict], case: MainCase) -> None:
    ob = integrated_bias(case.online, case.truth, case.active_mask)
    fb = integrated_bias(case.fft, case.truth, case.active_mask)
    rr = closure_rms(case.recovered_raw, case.raw_online_input, case.avg_rate)
    for i in range(case.truth.shape[0]):
        rows.append({
            "pu": case.pu,
            "pedestal_percent": case.pedestal,
            "sample": i,
            "online_bias_pct": float(ob[i]),
            "fft_bias_pct": float(fb[i]),
            "recovery_rms_pct": float(rr[i]),
        })


def scan_values(rows: list[dict], pu: float, ped: float, key: str) -> np.ndarray:
    return np.asarray([r[key] for r in rows
                       if r["pu"] == pu and r["pedestal_percent"] == ped], dtype=float)


def plot_pedestal_scan(rows: list[dict], key: str, ylabel: str,
                       rlabel: str, outbase: Path) -> None:
    fig = create_figure("Injected pedestal / <truth> [%]", ylabel, rlabel)
    ax = fig.gca()
    jitter_rng = np.random.default_rng(SEED + 77)
    for color, pu, marker in zip(PETROFF_10, PU_MODES, ("o", "s")):
        for ped in PEDESTAL_SCAN:
            vals = scan_values(rows, pu, ped, key)
            jitter = jitter_rng.uniform(-0.04, 0.04, vals.size)
            ax.plot(ped + jitter, vals, ".", ms=1.8, alpha=0.08, color=color)
            ax.plot([ped], [np.mean(vals)], marker=marker, linestyle="none",
                    ms=7, color=color, label=f"PU {int(pu)}" if ped == 0 else None)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(frameon=False, fontsize=11)
    save(fig, outbase)

# -----------------------------------------------------------------------------
# Type-1 model mismatch
# -----------------------------------------------------------------------------
def type1_fraction(pu: np.ndarray, model: str) -> np.ndarray:
    pu = np.asarray(pu, dtype=float)
    if model == "fixed_2pct":
        return np.full_like(pu, 0.020)
    if model == "linear_2to3":
        return 0.020 + 0.010 * pu / 200.0
    if model == "quadratic_2to3":
        return 0.020 + 0.010 * (pu / 200.0) ** 2
    raise ValueError(model)


def inject_type1_bx1(clean: np.ndarray, truth_pu: np.ndarray,
                     mask: np.ndarray, model: str) -> np.ndarray:
    """Exact reverse-order partner of the +1 Type-1 subtraction."""
    out = np.asarray(clean, dtype=float).copy()
    for ibx in range(BX_LEN - 2, -1, -1):
        if mask[ibx] != 1:
            continue
        y = out[:, ibx]
        out[:, ibx + 1] += y * type1_fraction(truth_pu[:, ibx], model)
    return out


def apply_fixed_type1(contaminated: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p0 = np.array([0.0, TYPE1_FIXED_CORRECTION])
    p1 = np.zeros(2)
    p2 = np.zeros(2)
    return apply_type1_batch(contaminated, mask, p0, p1, p2)


def run_type1_scan(h: np.ndarray, mask: np.ndarray, n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for ipu, mean_pu in enumerate(PU_SCAN):
        rng = np.random.default_rng(seed + 100 * ipu)
        truth_pu = generate_truth_pu(mask, n, mean_pu, rng)
        truth = truth_pu / 8.0

        # Type-1 is tested in the post-afterglow space, matching the input
        # convention of apply_type1_batch. Keep the tiny nominal FFT residual.
        raw = convolve(truth, h)
        post_fft, _ = subtract_pedestal(fft_inverse(raw, h))

        for model in TYPE1_MODELS:
            contaminated = inject_type1_bx1(post_fft, truth_pu, mask, model)
            corrected = apply_fixed_type1(contaminated, mask)
            bias = integrated_bias(corrected, truth, mask)
            for i, value in enumerate(bias):
                rows.append({
                    "mean_pu": mean_pu,
                    "sample": i,
                    "truth_model": model,
                    "integrated_bias_pct": float(value),
                })
    return rows


def plot_type1_models(outdir: Path) -> None:
    pu = np.arange(0.0, 251.0, 5.0)
    labels = {
        "fixed_2pct": "True fixed 2%",
        "linear_2to3": "True linear: 2% to 3% at PU 200",
        "quadratic_2to3": "True quadratic: 2% to 3% at PU 200",
    }
    markers = ("o", "s", "^")
    fig = create_figure("Pileup", "BX(i+1) Type-1 fraction [%]",
                        "Toy MC, Type-1 model")
    ax = fig.gca()
    for color, model, marker in zip(PETROFF_10, TYPE1_MODELS, markers):
        ax.plot(pu, 100 * type1_fraction(pu, model), marker=marker,
                linestyle="none", ms=3.5, color=color, label=labels[model])
    # Applied correction is the fixed model already shown as the closure case.
    ax.legend(frameon=False, fontsize=10)
    save(fig, outdir / "40_type1_truth_models")


def plot_type1_bias(rows: list[dict], outdir: Path) -> None:
    labels = {
        "fixed_2pct": "True fixed 2%",
        "linear_2to3": "True linear",
        "quadratic_2to3": "True quadratic",
    }
    markers = ("o", "s", "^")
    fig = create_figure("Mean pileup",
                        "Integrated bias after fixed 2% Type-1 correction [%]",
                        "Toy MC, Type-1 mismatch")
    ax = fig.gca()
    for color, model, marker in zip(PETROFF_10, TYPE1_MODELS, markers):
        means = []
        for mean_pu in PU_SCAN:
            vals = np.asarray([r["integrated_bias_pct"] for r in rows
                               if r["truth_model"] == model and r["mean_pu"] == mean_pu])
            means.append(float(np.mean(vals)))
            rng = np.random.default_rng(SEED + int(mean_pu) + 1000 * TYPE1_MODELS.index(model))
            ax.plot(mean_pu + rng.uniform(-0.7, 0.7, vals.size), vals,
                    ".", ms=1.5, alpha=0.06, color=color)
        ax.plot(PU_SCAN, means, marker=marker, linestyle="none", ms=7,
                color=color, label=labels[model])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(frameon=False, fontsize=11)
    save(fig, outdir / "41_type1_bias_vs_pu")


def plot_type1_examples(h: np.ndarray, mask: np.ndarray, n: int,
                       seed: int, outdir: Path) -> None:
    model = "quadratic_2to3"
    for imode, mean_pu in enumerate(PU_MODES):
        rng = np.random.default_rng(seed + 7000 + imode)
        truth_pu = generate_truth_pu(mask, n, mean_pu, rng)
        truth = truth_pu / 8.0
        avg = average_rate(truth, mask)
        post_fft, _ = subtract_pedestal(fft_inverse(convolve(truth, h), h))
        contaminated = inject_type1_bx1(post_fft, truth_pu, mask, model)
        corrected = apply_fixed_type1(contaminated, mask)
        plot_rate_residual(
            truth, corrected, avg,
            f"Toy MC, PU {int(mean_pu)}, quadratic Type-1 truth",
            "FFT + fixed 2% Type-1 correction",
            outdir / f"42_type1_full_orbit_pu{int(mean_pu)}",
            None,
        )


# -----------------------------------------------------------------------------
# Train-tail HFSBR inference
# -----------------------------------------------------------------------------
def exp_tail(x: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    return a * np.exp(-x / tau) + c


def fit_train_tail_hfsbr(h: np.ndarray, mask: np.ndarray, n: int,
                         seed: int) -> tuple[TailFit, float]:
    rng = np.random.default_rng(seed)
    truth_pu = generate_truth_pu(mask, n, 65.0, rng)
    truth = truth_pu / 8.0
    avg = average_rate(truth, mask)
    raw_mean = mean_profile(convolve(truth, h))

    last_coll = int(np.flatnonzero(mask)[-1])
    start = last_coll + TRAIN_TAIL_FIT_START_OFFSET
    stop = min(BX_LEN, start + TRAIN_TAIL_MAX_POINTS)
    bx = np.arange(start, stop, dtype=int)
    bx = bx[~np.isin(bx, np.asarray(ZERO_BX, dtype=int))]
    if bx.size < 8:
        raise RuntimeError(f"Only {bx.size} valid BX available after last train")

    offsets = (bx - last_coll).astype(float)
    observed = raw_mean[bx]
    c0 = float(np.median(observed[-min(10, observed.size):]))
    a0 = max(float(observed[0] - c0), 1e-12)
    tau0 = max(3.0, 0.25 * (offsets[-1] - offsets[0] + 1.0))
    popt, _ = curve_fit(
        exp_tail, offsets, observed, p0=(a0, tau0, c0),
        bounds=((0.0, 0.25, -np.inf), (np.inf, 1e6, np.inf)),
        maxfev=20000,
    )
    a, tau, c = map(float, popt)
    fitted = exp_tail(offsets, a, tau, c)

    # Take only the fitted shape. Its absolute scale comes from the measured
    # single-bunch response at the first offset included in the tail fit.
    norm_offset = int(offsets[0])
    inferred = np.empty(BX_LEN, dtype=float)
    inferred[0] = h[0]
    d = np.arange(1, BX_LEN, dtype=float)
    inferred[1:] = h[norm_offset] * np.exp(-(d - norm_offset) / tau)

    return TailFit(offsets, observed, fitted, tau, c, norm_offset, inferred), avg


def plot_train_tail_fit(fit: TailFit, avg: float, outdir: Path) -> None:
    residual = 100.0 * (fit.fitted - fit.observed) / max(avg, 1e-15)
    fig, ax = create_double_figure(
        "BX offset after last colliding BX", "Mean tail rate [Hz/ub]",
        "Fit - tail / <truth> [%]", "Toy MC, train-tail fit")
    ax[0].plot(fit.offsets, fit.observed, ".", ms=4, label="Mean train tail")
    ax[0].plot(fit.offsets, fit.fitted, "o", linestyle="none", ms=2.8,
               label=f"Exponential fit, tau={fit.tau:.2f} BX")
    ax[0].legend(frameon=False, fontsize=11)
    ax[1].plot(fit.offsets, residual, ".", ms=3)
    ax[1].axhline(0.0, color="black", linewidth=1.0)
    save(fig, outdir / "50_train_tail_fit")


def plot_hfsbr_comparison(h: np.ndarray, fit: TailFit, outdir: Path) -> None:
    offsets = np.arange(1, min(200, BX_LEN - 1) + 1)
    true = h[offsets]
    inferred = fit.inferred_hfsbr[offsets]
    norm = max(abs(float(h[fit.norm_offset])), 1e-15)
    residual = 100.0 * (inferred - true) / norm

    fig, ax = create_double_figure(
        "BX offset", "HFSBR", "(train-fit - SBR) / H(ref) [%]",
        f"Toy MC, normalized at BX+{fit.norm_offset}")
    ax[0].plot(offsets, true, ".", ms=3.5, label="Measured single-bunch HFSBR")
    ax[0].plot(offsets, inferred, ".", ms=3.0, label="Train-tail-derived HFSBR")
    ax[0].legend(frameon=False, fontsize=10)
    if np.all(true > 0) and np.all(inferred > 0):
        ax[0].set_yscale("log")
    ax[1].plot(offsets, residual, ".", ms=3)
    ax[1].axhline(0.0, color="black", linewidth=1.0)
    save(fig, outdir / "51_hfsbr_single_bunch_vs_train_tail")


def run_train_tail_scan(h: np.ndarray, inferred: np.ndarray, mask: np.ndarray,
                        n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for ipu, mean_pu in enumerate(PU_SCAN):
        rng = np.random.default_rng(seed + 100 * ipu)
        truth = generate_truth_pu(mask, n, mean_pu, rng) / 8.0
        raw = convolve(truth, h)
        corrected, _ = subtract_pedestal(fft_inverse(raw, inferred))
        bias = integrated_bias(corrected, truth, mask)
        for i, value in enumerate(bias):
            rows.append({"mean_pu": mean_pu, "sample": i,
                         "integrated_bias_pct": float(value)})
    return rows


def plot_train_tail_bias(rows: list[dict], outdir: Path) -> None:
    fig = create_figure("Mean pileup",
                        "Integrated bias using train-tail HFSBR [%]",
                        "Toy MC, train-tail-derived kernel")
    ax = fig.gca()
    color = PETROFF_10[0]
    means = []
    for mean_pu in PU_SCAN:
        vals = np.asarray([r["integrated_bias_pct"] for r in rows
                           if r["mean_pu"] == mean_pu])
        means.append(float(np.mean(vals)))
        rng = np.random.default_rng(SEED + int(mean_pu) + 4444)
        ax.plot(mean_pu + rng.uniform(-0.7, 0.7, vals.size), vals,
                ".", ms=1.5, alpha=0.08, color=color)
    ax.plot(PU_SCAN, means, "o", linestyle="none", ms=7, color=color)
    ax.axhline(0.0, color="black", linewidth=1.0)
    save(fig, outdir / "52_train_tail_hfsbr_bias_vs_pu")


def plot_train_tail_examples(h: np.ndarray, inferred: np.ndarray,
                             mask: np.ndarray, n: int, seed: int,
                             outdir: Path) -> None:
    for imode, mean_pu in enumerate(PU_MODES):
        rng = np.random.default_rng(seed + 9000 + imode)
        truth = generate_truth_pu(mask, n, mean_pu, rng) / 8.0
        avg = average_rate(truth, mask)
        corrected, _ = subtract_pedestal(fft_inverse(convolve(truth, h), inferred))
        plot_rate_residual(
            truth, corrected, avg, f"Toy MC, PU {int(mean_pu)}",
            "FFT with train-tail-derived HFSBR",
            outdir / f"53_train_tail_full_orbit_pu{int(mean_pu)}", None)


# -----------------------------------------------------------------------------
# Output / main
# -----------------------------------------------------------------------------
def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF toy-MC validation")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--outdir", default=str(OUTPUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_style()
    h, mask = load_inputs()

    # 0. Truth distributions.
    for i, pu in enumerate(PU_MODES):
        plot_truth_distribution(pu, mask, args.n_samples, args.seed + i, outdir)

    # 1-3. Full-orbit examples only: two PU modes x two pedestal scenarios.
    for ipu, pu in enumerate(PU_MODES):
        for iped, ped in enumerate(PLOT_PEDESTALS):
            print(f"Representative: PU {pu:g}, pedestal {ped:g}%")
            case = build_case(pu, ped, h, mask, args.n_samples,
                              args.seed + 1000 * ipu + 100 * iped + 10)
            tag = f"pu{int(pu)}_ped{int(ped)}"
            plot_rate_residual(case.truth, case.online, case.avg_rate,
                               f"Toy MC, PU {int(pu)}, pedestal {ped:g}%",
                               "Online corrected", outdir / f"10_online_{tag}",
                               ONLINE_RESIDUAL_YLIM)
            plot_rate_residual(case.truth, case.fft, case.avg_rate,
                               f"Toy MC, PU {int(pu)}, pedestal {ped:g}%",
                               "FFT corrected", outdir / f"20_fft_{tag}",
                               FFT_RESIDUAL_YLIM)
            plot_recovery(case, outdir / f"30_recovery_{tag}")
            del case

    # Pedestal summaries; keep only scalar metrics in memory.
    scan_rows: list[dict] = []
    for ipu, pu in enumerate(PU_MODES):
        for iped, ped in enumerate(PEDESTAL_SCAN):
            print(f"Pedestal scan: PU {pu:g}, pedestal {ped:g}%")
            case = build_case(pu, ped, h, mask, args.n_samples,
                              args.seed + 5000 + 1000 * ipu + 100 * iped)
            append_pedestal_rows(scan_rows, case)
            del case
    write_csv(outdir / "main_pedestal_scan.csv", scan_rows)
    plot_pedestal_scan(scan_rows, "online_bias_pct", "Integrated colliding-BX bias [%]",
                       "Toy MC, online iterative correction",
                       outdir / "11_online_bias_vs_pedestal")
    plot_pedestal_scan(scan_rows, "fft_bias_pct", "Integrated colliding-BX bias [%]",
                       "Toy MC, direct FFT correction",
                       outdir / "21_fft_bias_vs_pedestal")
    plot_pedestal_scan(scan_rows, "recovery_rms_pct", "Recovery closure RMS / <truth> [%]",
                       "Toy MC, inverse reconstruction",
                       outdir / "31_recovery_closure_vs_pedestal")

    # 4. Type-1 mismatch.
    print("Type-1 model mismatch")
    plot_type1_models(outdir)
    type1_rows = run_type1_scan(h, mask, args.n_samples, args.seed + 20000)
    write_csv(outdir / "type1_model_mismatch_scan.csv", type1_rows)
    plot_type1_bias(type1_rows, outdir)
    plot_type1_examples(h, mask, args.n_samples, args.seed, outdir)

    # 5. Train-tail-derived response.
    print("Train-tail HFSBR test")
    tail_fit, tail_avg = fit_train_tail_hfsbr(
        h, mask, args.n_samples, args.seed + 30000)
    plot_train_tail_fit(tail_fit, tail_avg, outdir)
    plot_hfsbr_comparison(h, tail_fit, outdir)
    tail_rows = run_train_tail_scan(
        h, tail_fit.inferred_hfsbr, mask, args.n_samples, args.seed + 40000)
    write_csv(outdir / "train_tail_hfsbr_bias_scan.csv", tail_rows)
    plot_train_tail_bias(tail_rows, outdir)
    plot_train_tail_examples(h, tail_fit.inferred_hfsbr, mask,
                             args.n_samples, args.seed, outdir)

    print(f"Results written to {outdir}")


if __name__ == "__main__":
    main()
