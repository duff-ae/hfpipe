# src/hfcore/afterglow_lsq.py

from __future__ import annotations

from typing import Sequence, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from .hd5schema import BX_LEN


class AfterglowSolver:
    """
    Afterglow reconstruction with explicit tail modelling across the orbit boundary.

    Diagnostic-first pipeline
    -------------------------
    1. Find the last colliding BX and the first colliding BX.
    2. Fit the end-of-orbit tail after the last colliding BX, skipping a small offset.
    3. Extrapolate that tail through the orbit boundary up to the first colliding BX.
    4. Replace only bx_to_clean with the extrapolated tail.
    5. Optionally perform regularized FFT deconvolution.

    Important
    ---------
    In the current diagnostic mode inside _solve_one():

        APPLY_TAIL_PATCH = True
        RUN_DECONVOLUTION = False

    so the returned array is just the original mu_obs with bx_to_clean replaced
    by the extrapolated tail prediction. No deconvolution is performed.
    """

    def __init__(
        self,
        H: np.ndarray,
        active_mask: np.ndarray,
        bx_to_clean: np.ndarray,
        lambda_nonactive: float,
        lambda_reg: float,
        p0_mean: float,
        cg_tol: float = 1e-8,
        cg_maxiter: int = 200,
        use_warm_start: bool = True,
    ) -> None:
        H = np.asarray(H, dtype=np.float64).reshape(-1)
        self.N = H.shape[0]

        if self.N != BX_LEN:
            raise ValueError(f"HFSBR length={self.N} does not match BX_LEN={BX_LEN}")

        self.H = H
        self.FH = np.fft.fft(H)
        self.FH_conj = np.conj(self.FH)

        active_mask = np.asarray(active_mask, dtype=np.int32).reshape(-1)
        if active_mask.shape[0] != self.N:
            raise ValueError("active_mask must have length BX_LEN")

        self.active_mask = active_mask
        self.bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64).reshape(-1) % self.N

        self.lambda_nonactive = float(lambda_nonactive)
        self.lambda_reg = float(lambda_reg)
        self.p0_mean = float(p0_mean)

        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.use_warm_start = bool(use_warm_start)

        self.bad_mask = np.zeros(self.N, dtype=bool)
        if self.bx_to_clean.size > 0:
            self.bad_mask[self.bx_to_clean] = True

        self.nonact_mask = (self.active_mask == 0)
        self.coll_mask = (self.active_mask != 0)

        self.coll_idx = np.flatnonzero(self.coll_mask)
        self.nonact_idx = np.flatnonzero(self.nonact_mask)

        self.first_coll_bx = int(self.coll_idx[0]) if self.coll_idx.size > 0 else 0
        self.last_coll_bx = int(self.coll_idx[-1]) if self.coll_idx.size > 0 else 0

        # Region after last colliding BX through orbit end, then from 0 up to first colliding BX - 1.
        self.wrap_region = self._build_wrap_region(self.last_coll_bx, self.first_coll_bx)

        # Diagnostics from the most recent histogram.
        self.last_tail_debug: Optional[Dict[str, Any]] = None

        # Numerically safe regularization for FFT deconvolution.
        hscale = max(1.0, float(np.max(np.abs(self.FH))))
        self.deconv_eps = max(1e-12, self.cg_tol, 1e-8 * hscale * hscale)

        self.laser_bx = np.array([3489, 3490, 3491, 3492], dtype=np.int64) % self.N
        self.laser_mask = np.zeros(self.N, dtype=bool)
        self.laser_mask[self.laser_bx] = True

    # ------------------------------------------------------------------
    #  Basic utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _exp_model(t: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
        return a * np.exp(-t / tau) + c

    def _build_wrap_region(self, last_coll_bx: int, first_coll_bx: int) -> np.ndarray:
        """
        BX after last colliding BX through orbit end, then from 0 up to first colliding BX - 1.
        """
        if self.N <= 0:
            return np.empty(0, dtype=np.int64)

        part1 = np.arange(last_coll_bx + 1, self.N, dtype=np.int64)
        part2 = np.arange(0, first_coll_bx, dtype=np.int64)
        if part1.size == 0 and part2.size == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([part1, part2])

    def _circular_range_len(self, start: int, length: int) -> np.ndarray:
        """
        Circular sequence of exactly `length` BX starting from `start`.
        """
        if length <= 0:
            return np.empty(0, dtype=np.int64)
        return (start + np.arange(length, dtype=np.int64)) % self.N

    # ------------------------------------------------------------------
    #  Tail fit preparation
    # ------------------------------------------------------------------
    def _tail_fit_indices(
        self,
        offset_bx: int = 3,
        window_bx: int = 30,  # kept only for interface compatibility
    ) -> np.ndarray:
        """
        Fit on all good tail BX from (last_colliding + offset) to the end of orbit,
        excluding bx_to_clean.
        """
        if self.coll_idx.size == 0:
            return np.empty(0, dtype=np.int64)

        start = self.last_coll_bx + offset_bx
        if start >= self.N:
            return np.empty(0, dtype=np.int64)

        fit_idx = np.arange(start, self.N, dtype=np.int64)
        fit_idx = fit_idx[~self.bad_mask[fit_idx]]
        return fit_idx

    def _wrap_patch_indices(self) -> np.ndarray:
        """
        BX to patch across the orbit boundary:
        - from last_colliding+1 to end of orbit
        - from 0 to first_colliding-1
        """
        if self.coll_idx.size == 0:
            return np.empty(0, dtype=np.int64)

        part1 = np.arange(self.last_coll_bx + 1, self.N, dtype=np.int64)
        part2 = np.arange(0, self.first_coll_bx, dtype=np.int64)
        if part1.size == 0:
            return part2
        if part2.size == 0:
            return part1
        return np.concatenate([part1, part2])

    def _fit_tail_exponential(
        self,
        mu_obs: np.ndarray,
        fit_idx: np.ndarray,
        init_params: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit y = a * exp(-t / tau) + c on the selected tail BX,
        using REAL BX distances from the start of the fit region.
        """
        out: Dict[str, Any] = {
            "ok": False,
            "fit_idx": fit_idx.copy(),
            "params": None,
            "cov": None,
            "reason": "",
        }

        if fit_idx.size < 5:
            out["reason"] = "too_few_points"
            return out

        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        yfit = np.asarray(mu_obs[fit_idx], dtype=np.float64)

        finite = np.isfinite(yfit)
        fit_idx = fit_idx[finite]
        yfit = yfit[finite]

        if yfit.size < 5:
            out["reason"] = "too_few_finite_points"
            return out

        # Real BX coordinate: distance from the first fit BX.
        t0_bx = int(fit_idx[0])
        tfit = (fit_idx - t0_bx).astype(np.float64)

        c0 = float(np.median(yfit[-min(5, yfit.size):]))
        a0 = float(max(yfit[0] - c0, 1e-9))
        if not np.isfinite(a0) or a0 <= 0:
            a0 = float(max(np.max(yfit) - c0, 1e-9))

        span = float(max(tfit[-1] - tfit[0], 1.0))
        tau0 = max(3.0, 0.5 * span)

        p0 = np.array([a0, tau0, c0], dtype=np.float64)
        if init_params is not None and len(init_params) == 3 and np.all(np.isfinite(init_params)):
            p0 = np.asarray(init_params, dtype=np.float64)

        lower = np.array([0.0, 0.5, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, 1e6, np.inf], dtype=np.float64)

        try:
            popt, pcov = curve_fit(
                self._exp_model,
                tfit,
                yfit,
                p0=p0,
                bounds=(lower, upper),
                maxfev=10000,
            )
            out["ok"] = True
            out["params"] = popt
            out["cov"] = pcov
            return out
        except Exception as exc:
            out["reason"] = f"curve_fit_failed: {exc}"
            return out

    def _extrapolate_wrap_tail(
        self,
        fit_result: Dict[str, Any],
        fit_idx: np.ndarray,
        target_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Extrapolate fitted tail using REAL circular BX distances
        from the first fit BX, not compressed array indices.
        """
        pred = np.zeros(target_idx.size, dtype=np.float64)

        if target_idx.size == 0:
            return pred

        if not fit_result.get("ok", False):
            return pred

        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        target_idx = np.asarray(target_idx, dtype=np.int64)

        if fit_idx.size == 0:
            return pred

        a, tau, c = map(float, fit_result["params"])

        fit_start_bx = int(fit_idx[0])

        # Real circular BX distance from the first fit BX.
        # This correctly continues through end-of-orbit and across BX=0.
        t = ((target_idx - fit_start_bx) % self.N).astype(np.float64)

        pred[:] = self._exp_model(t, a, tau, c)
        return pred


    def _prepare_tail_patch(
        self,
        mu_obs: np.ndarray,
        prev_tail_params: Optional[np.ndarray] = None,
        fit_offset_bx: int = 3,
        fit_window_bx: int = 30,  # kept only for interface compatibility
    ) -> Dict[str, Any]:
        """
        Fit on the good end-of-orbit tail, then extrapolate through the orbit boundary
        and replace only bx_to_clean in the full wrap region.
        """
        fit_idx = self._tail_fit_indices(offset_bx=fit_offset_bx, window_bx=fit_window_bx)
        fit_res = self._fit_tail_exponential(mu_obs, fit_idx, init_params=prev_tail_params)

        wrap_idx = self._wrap_patch_indices()
        wrap_pred = self._extrapolate_wrap_tail(fit_res, fit_idx, wrap_idx)

        replace_idx = wrap_idx[self.bad_mask[wrap_idx]]
        replace_pred = wrap_pred[self.bad_mask[wrap_idx]]

        patch = {
            "fit_idx": fit_idx,
            "fit_ok": bool(fit_res.get("ok", False)),
            "fit_params": None if fit_res.get("params") is None else np.asarray(fit_res["params"], dtype=np.float64),
            "fit_reason": fit_res.get("reason", ""),
            "wrap_idx": wrap_idx,
            "wrap_pred": wrap_pred,
            "replace_idx": replace_idx,
            "replace_pred": replace_pred,
        }
        return patch


    def _wrap_extrapolation_indices(self) -> np.ndarray:
        """
        Extrapolation region through BX=0:
        from 0 up to first_colliding_bx - 1.

        This is the region where we want to predict the tail after wrapping
        around the orbit boundary.
        """
        if self.coll_idx.size == 0:
            return np.empty(0, dtype=np.int64)

        if self.first_coll_bx <= 0:
            return np.empty(0, dtype=np.int64)

        return np.arange(0, self.first_coll_bx, dtype=np.int64)

    # ------------------------------------------------------------------
    #  Signal patching
    # ------------------------------------------------------------------

    def _apply_tail_patch(
        self,
        mu_obs: np.ndarray,
        patch: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Replace only bx_to_clean inside the wrap region with the extrapolated tail.
        Returns patched signal, replaced indices, and saved original buffer.
        """
        y = np.asarray(mu_obs, dtype=np.float64).copy()

        replace_idx = np.asarray(patch["replace_idx"], dtype=np.int64)
        replace_pred = np.asarray(patch["replace_pred"], dtype=np.float64)

        if replace_idx.size == 0:
            return y, replace_idx, np.empty(0, dtype=np.float64)

        saved = y[replace_idx].copy()
        y[replace_idx] = replace_pred
        return y, replace_idx, saved

    def _restore_buffer_on_mu(
        self,
        mu_true: np.ndarray,
        mu_obs: np.ndarray,
        y_work: np.ndarray,
    ) -> np.ndarray:
        """
        Restore only the removed laser contribution on top of the corrected mu_true.
        """
        out = np.asarray(mu_true, dtype=np.float64).copy()

        delta = mu_obs[self.laser_bx] - y_work[self.laser_bx]
        out[self.laser_bx] += delta

        return out
    
    # ------------------------------------------------------------------
    #  Deconvolution
    # ------------------------------------------------------------------

    def _deconvolve_fft(self, y: np.ndarray) -> np.ndarray:
        """
        Regularized FFT deconvolution:
            X = H* Y / (|H|^2 + eps)
        """
        y = np.asarray(y, dtype=np.float64)
        Y = np.fft.fft(y)
        denom = np.abs(self.FH) ** 2 + self.deconv_eps
        X = self.FH_conj * Y / denom
        x = np.fft.ifft(X).real
        return x

    def _postprocess_mu(self, mu_true: np.ndarray) -> np.ndarray:
        """
        Light suppression on known non-active BX. No hard clipping by default.
        """
        mu_true = np.asarray(mu_true, dtype=np.float64).copy()

        if self.nonact_idx.size > 0 and self.lambda_nonactive > 0.0:
            mu_true[self.nonact_idx] /= (1.0 + self.lambda_nonactive)

        return mu_true

    # ------------------------------------------------------------------
    #  Single solve
    # ------------------------------------------------------------------

    def _solve_one(
        self,
        mu_obs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Keep the original interface:
            returns (mu_true, ped)

        Diagnostic mode right now:
        - replace only bx_to_clean in the original mu_obs by the extrapolated tail,
        - do NOT run deconvolution,
        - return the patched rates directly.
        """
        mu_obs = np.asarray(mu_obs, dtype=np.float64)
        if mu_obs.shape[0] != self.N:
            raise ValueError(f"_solve_one: mu_obs length={mu_obs.shape[0]} != N={self.N}")

        # ------------------------------------------------------------------
        # local control switches
        # ------------------------------------------------------------------
        APPLY_TAIL_PATCH = True
        RUN_DECONVOLUTION = True

        prev_tail_params = None
        if x0 is not None and x0.shape[0] >= 3:
            cand = np.asarray(x0[-3:], dtype=np.float64)
            if np.all(np.isfinite(cand)):
                prev_tail_params = cand

        patch = self._prepare_tail_patch(
            mu_obs=mu_obs,
            prev_tail_params=prev_tail_params,
            fit_offset_bx=3,
            fit_window_bx=30,
        )
        self.last_tail_debug = patch

        if APPLY_TAIL_PATCH and patch["fit_ok"]:
            y_work, replaced_idx, saved_buffer = self._apply_tail_patch(mu_obs, patch)
        else:
            y_work = mu_obs.copy()
            replaced_idx = np.empty(0, dtype=np.int64)
            saved_buffer = np.empty(0, dtype=np.float64)

        if RUN_DECONVOLUTION:
            mu_true = self._deconvolve_fft(y_work)
            mu_true = self._postprocess_mu(mu_true)

            if APPLY_TAIL_PATCH:
                mu_true = self._restore_buffer_on_mu(mu_true, mu_obs, y_work)
        else:
            mu_true = y_work
            
        ped = 0.0
        return mu_true, ped

    # ------------------------------------------------------------------
    #  Batch API
    # ------------------------------------------------------------------

    def apply_batch(
        self,
        hists: np.ndarray,
        n_jobs: int = -1,
        desc: str = "afterglow tail fit + patch",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same external API as before.
        """
        hists = np.asarray(hists, dtype=np.float64)
        if hists.ndim != 2:
            raise ValueError(f"apply_batch: hists must be 2D, got shape {hists.shape}")

        T, N = hists.shape
        if N != self.N:
            raise ValueError(f"apply_batch: hists.shape[1]={N} does not match solver N={self.N}")

        if self.use_warm_start:
            mu_batch = np.empty((T, N), dtype=np.float64)
            ped_arr = np.empty(T, dtype=np.float64)

            prev_tail_params: Optional[np.ndarray] = None

            for t in tqdm(range(T), desc=desc, unit="hist"):
                if prev_tail_params is None:
                    x0 = None
                else:
                    x0 = np.asarray(prev_tail_params, dtype=np.float64)

                mu_true, ped = self._solve_one(hists[t], x0=x0)

                mu_batch[t] = mu_true
                ped_arr[t] = ped

                dbg = self.last_tail_debug
                if dbg is not None and dbg.get("fit_ok", False):
                    prev_tail_params = np.asarray(dbg["fit_params"], dtype=np.float64)
                else:
                    prev_tail_params = None

            return mu_batch, ped_arr

        results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=8)(
            delayed(self._solve_one)(hists[t], x0=None)
            for t in tqdm(range(T), desc=desc, unit="hist")
        )
        mu_list, ped_list = map(list, zip(*results))
        mu_batch = np.stack(mu_list, axis=0).astype(np.float64, copy=False)
        ped_arr = np.asarray(ped_list, dtype=np.float64)
        return mu_batch, ped_arr


def build_afterglow_solver_from_file(
    hfsbr_path: str,
    active_mask: np.ndarray,
    bx_to_clean: Sequence[int],
    p0_guess: Optional[np.ndarray] = None,
    lambda_reg: float = 0.01,
    lambda_nonactive: float = 0.05,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 200,
    use_warm_start: bool = True,
) -> AfterglowSolver:
    """
    Same builder interface as before.
    """
    H = np.loadtxt(hfsbr_path, dtype=np.float64, delimiter=",")
    H = np.asarray(H, dtype=np.float64).reshape(-1)

    if H.shape[0] != BX_LEN:
        raise ValueError(f"HFSBR len={H.shape[0]} does not match BX_LEN={BX_LEN}")

    active_mask = np.asarray(active_mask, dtype=np.int32).reshape(-1)
    if active_mask.shape[0] != BX_LEN:
        raise ValueError("active_mask must match HFSBR length")

    bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64).reshape(-1)

    if p0_guess is not None:
        p0_mean = float(np.mean(np.asarray(p0_guess, dtype=np.float64)))
    else:
        p0_mean = 0.0

    return AfterglowSolver(
        H=H,
        active_mask=active_mask,
        bx_to_clean=bx_to_clean,
        lambda_nonactive=lambda_nonactive,
        lambda_reg=lambda_reg,
        p0_mean=p0_mean,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        use_warm_start=use_warm_start,
    )