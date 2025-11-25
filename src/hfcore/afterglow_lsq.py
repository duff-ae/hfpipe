# src/hfcore/afterglow_lsq.py

from __future__ import annotations

from typing import Sequence, Optional, Tuple

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse.linalg import LinearOperator, cg

from .hd5schema import BX_LEN


class AfterglowSolver:
    """
    Fast iterative afterglow solver based on FFT + Conjugate Gradient.

    The model is:

        mu_obs ≈ M * mu_true + pedestal * 1,

    where
      - mu_obs    : observed per-BX histogram (after raw HFET processing),
      - M         : circulant "afterglow" matrix built from the HFSBR kernel,
      - mu_true   : "true" instantaneous mu (what we want),
      - pedestal  : common offset term.

    We solve the normal equations

        (A^T A) x = A^T b,

    where x = [mu_true, pedestal].

    Public API
    ----------
    - apply_batch(hists, n_jobs=-1, desc="...") -> (mu_batch, ped_array)

    Warm start
    ----------
    If `use_warm_start=True`, solutions for consecutive histograms are used
    as initial guesses for the next one, scaled by the total amplitude
    (sum over BX). This significantly accelerates convergence for time-like
    sequences of histograms from the same fill.
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
        H = np.asarray(H, dtype=np.float64)
        self.N = H.shape[0]

        # Optional sanity check: H should be consistent with BX_LEN.
        if self.N != BX_LEN:
            raise ValueError(f"HFSBR length={self.N} does not match BX_LEN={BX_LEN}")

        self.H = H
        self.FH = np.fft.fft(H)
        self.FH_conj = np.conj(self.FH)

        active_mask = np.asarray(active_mask, dtype=np.int32)
        if active_mask.shape[0] != self.N:
            raise ValueError("active_mask must have length equal to BX length")

        self.bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64)
        self.lambda_nonactive = float(lambda_nonactive)
        self.lambda_reg = float(lambda_reg)
        self.p0_mean = float(p0_mean) if lambda_reg > 0.0 else 0.0

        # Non-active BX (soft constrained to mu_true ~ 0), excluding bx_to_clean
        mask_nonact = (active_mask == 0)
        if self.bx_to_clean.size > 0:
            mask_nonact[self.bx_to_clean % self.N] = False
        self.nonact_idx = np.flatnonzero(mask_nonact)

        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.use_warm_start = bool(use_warm_start)

        # Linear operator representing (A^T A) with an explicit pedestal dimension
        dim = self.N + 1
        self.op = LinearOperator(
            shape=(dim, dim),
            matvec=self._matvec_G,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    #  Circulant operations (M and M^T)
    # ------------------------------------------------------------------

    def _M_dot(self, mu: np.ndarray) -> np.ndarray:
        """
        Apply the circulant afterglow matrix M to mu using FFT.
        """
        mu_fft = np.fft.fft(mu)
        y = np.fft.ifft(self.FH * mu_fft).real
        return y

    def _MT_dot(self, v: np.ndarray) -> np.ndarray:
        """
        Apply the transpose (Hermitian) of M using FFT.
        """
        v_fft = np.fft.fft(v)
        y = np.fft.ifft(self.FH_conj * v_fft).real
        return y

    # ------------------------------------------------------------------
    #  (A^T A) matvec
    # ------------------------------------------------------------------

    def _matvec_G(self, x: np.ndarray) -> np.ndarray:
        """
        Compute y = (A^T A) x for x = [mu, ped].

        A encodes:
          - data term: mu_obs ≈ M mu + ped * 1,
          - hard constraints: mu[bx_to_clean] ≈ 0,
          - soft constraints: mu[non-active BX] ≈ 0,
          - pedestal prior: ped ≈ p0_mean (if lambda_reg > 0).
        """
        x = np.asarray(x, dtype=np.float64)
        mu = x[: self.N]
        ped = float(x[self.N])

        # Data term: y_data = M mu + ped * 1
        y_data = self._M_dot(mu) + ped

        # Gradient of data term
        mu_grad = self._MT_dot(y_data)
        ped_grad = float(np.sum(y_data))

        # Hard constraints: mu[bx_to_clean] ~ 0
        if self.bx_to_clean.size > 0:
            mu_grad[self.bx_to_clean] += mu[self.bx_to_clean]

        # Soft constraints: non-active BX -> 0 (scaled by lambda_nonactive)
        if self.nonact_idx.size > 0 and self.lambda_nonactive > 0.0:
            mu_grad[self.nonact_idx] += self.lambda_nonactive * mu[self.nonact_idx]

        # Pedestal prior: (ped - p0_mean)^2 * lambda_reg
        if self.lambda_reg > 0.0:
            ped_grad += self.lambda_reg * ped

        y = np.empty_like(x, dtype=np.float64)
        y[: self.N] = mu_grad
        y[self.N] = ped_grad
        return y

    # ------------------------------------------------------------------
    #  A^T b (right-hand side)
    # ------------------------------------------------------------------

    def _build_rhs(self, mu_obs: np.ndarray) -> np.ndarray:
        """
        Build the right-hand side vector r = A^T b for the given observation.
        """
        mu_obs = np.asarray(mu_obs, dtype=np.float64)
        if mu_obs.shape[0] != self.N:
            raise ValueError(
                f"_build_rhs: mu_obs length={mu_obs.shape[0]} does not match N={self.N}"
            )

        # Data term: A^T b ≈ M^T mu_obs + sum(mu_obs) for pedestal
        r_mu = self._MT_dot(mu_obs)
        r_ped = float(np.sum(mu_obs))

        # Pedestal prior shifts the RHS towards p0_mean
        if self.lambda_reg > 0.0:
            r_ped += self.lambda_reg * self.p0_mean

        r = np.empty(self.N + 1, dtype=np.float64)
        r[: self.N] = r_mu
        r[self.N] = r_ped
        return r

    # ------------------------------------------------------------------
    #  Single-histogram solve
    # ------------------------------------------------------------------

    def _solve_one(
        self,
        mu_obs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve (A^T A) x = A^T b(mu_obs) for a single histogram using CG.

        Parameters
        ----------
        mu_obs : np.ndarray, shape (N,)
            Observed histogram (afterglow-contaminated).
        x0 : np.ndarray or None, shape (N+1,)
            Optional initial guess [mu_true0, ped0].
            If None, CG starts from zeros.

        Returns
        -------
        mu_true : np.ndarray, shape (N,)
            Reconstructed "true" instantaneous mu.
        ped : float
            Fitted pedestal value.
        """
        rhs = self._build_rhs(mu_obs)

        if x0 is None:
            x0 = np.zeros_like(rhs)

        x, info = cg(
            self.op,
            rhs,
            x0=x0,
            tol=self.cg_tol,
            maxiter=self.cg_maxiter,
        )

        if info != 0:
            # Fallback: retry with zero initial guess
            x0_fallback = np.zeros_like(rhs)
            x, info2 = cg(
                self.op,
                rhs,
                x0=x0_fallback,
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )
            if info2 != 0:
                raise RuntimeError(
                    f"CG did not converge (info={info}, fallback info={info2})"
                )

        mu_true = x[: self.N]
        ped = float(x[self.N])
        return mu_true, ped

    # ------------------------------------------------------------------
    #  Public API: batch solve
    # ------------------------------------------------------------------

    def apply_batch(
        self,
        hists: np.ndarray,
        n_jobs: int = -1,
        desc: str = "LSQ afterglow (CG)",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the CG-based solver to a batch of histograms.

        Parameters
        ----------
        hists : np.ndarray, shape (T, N)
            Batch of observed histograms.
        n_jobs : int, optional
            Number of parallel jobs when `use_warm_start=False`.
            Ignored in warm-start mode (always sequential).
        desc : str, optional
            Description for tqdm progress bar.

        Returns
        -------
        mu_batch : np.ndarray, shape (T, N)
            Reconstructed mu_true for each histogram.
        ped_arr : np.ndarray, shape (T,)
            Pedestal values per histogram.

        Modes
        -----
        - If use_warm_start=True (default):
            Histograms are processed sequentially. For t > 0 the initial
            guess is built from the previous solution scaled by the ratio
            of total amplitudes:

                scale = sum(mu_obs[t]) / (sum(mu_obs[t-1]) + eps)

        - If use_warm_start=False:
            All histograms are solved independently in parallel with
            zero initial guesses.
        """
        hists = np.asarray(hists, dtype=np.float64)
        if hists.ndim != 2:
            raise ValueError(f"apply_batch: hists must be 2D, got shape {hists.shape}")

        T, N = hists.shape
        if N != self.N:
            raise ValueError(
                f"apply_batch: hists.shape[1]={N} does not match solver N={self.N}"
            )

        # Warm-start mode: sequential pass with amplitude scaling
        if self.use_warm_start:
            mu_batch = np.empty((T, N), dtype=np.float64)
            ped_arr = np.empty(T, dtype=np.float64)

            prev_mu_true: Optional[np.ndarray] = None
            prev_ped: Optional[float] = None
            prev_obs: Optional[np.ndarray] = None

            eps = 1e-12

            for t in tqdm(range(T), desc=desc, unit="hist"):
                obs = hists[t]

                if prev_mu_true is None:
                    # First histogram: start from zero
                    x0 = None
                else:
                    s_prev = float(np.sum(prev_obs))
                    s_cur = float(np.sum(obs))
                    scale = s_cur / (s_prev + eps)

                    x0 = np.empty(self.N + 1, dtype=np.float64)
                    x0[: self.N] = scale * prev_mu_true
                    x0[self.N] = scale * prev_ped

                mu_true, ped = self._solve_one(obs, x0=x0)

                mu_batch[t] = mu_true
                ped_arr[t] = ped

                prev_mu_true = mu_true
                prev_ped = ped
                prev_obs = obs

            return mu_batch, ped_arr

        # No warm start: parallel independent solves with zero x0
        results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=4)(
            delayed(self._solve_one)(hists[t], x0=None)
            for t in tqdm(range(T), desc=desc, unit="hist")
        )
        mu_list, ped_list = map(list, zip(*results))
        mu_batch = np.stack(mu_list, axis=0).astype(np.float64, copy=False)
        ped_arr = np.asarray(ped_list, dtype=np.float64)
        return mu_batch, ped_arr


# ----------------------------------------------------------------------
#  Builder from HFSBR file
# ----------------------------------------------------------------------

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
    Convenience builder for AfterglowSolver from a text HFSBR kernel file.

    Parameters
    ----------
    hfsbr_path : str
        Path to the HFSBR kernel file. Expected format: plain text with
        one row (or column) of length N, comma-separated.
    active_mask : np.ndarray
        1D array of length N with active / non-active BX flags.
    bx_to_clean : sequence of int
        Indices of BX that should be hard-constrained to mu_true ≈ 0.
    p0_guess : np.ndarray or None, optional
        Optional initial pedestal estimate (e.g. from a previous run).
        If provided and lambda_reg > 0, its mean is used as p0_mean.
    lambda_reg : float, optional
        Regularization strength for the pedestal prior.
        If <= 0, pedestal prior is disabled.
    lambda_nonactive : float, optional
        Regularization strength for mu_true on non-active BX.
    cg_tol : float, optional
        Tolerance for the Conjugate Gradient solver.
    cg_maxiter : int, optional
        Maximum number of CG iterations.
    use_warm_start : bool, optional
        Whether to enable warm start across histograms in apply_batch().

    Returns
    -------
    AfterglowSolver
        Configured solver instance.
    """
    H = np.loadtxt(hfsbr_path, dtype=np.float64, delimiter=",")
    H = H.astype(np.float64, copy=False)
    N = H.shape[0]

    if N != BX_LEN:
        raise ValueError(f"HFSBR len={N} does not match BX_LEN={BX_LEN}")

    active_mask = np.asarray(active_mask, dtype=np.int32)
    if active_mask.shape[0] != N:
        raise ValueError("active_mask must match HFSBR length")

    bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64)

    if (p0_guess is not None) and (lambda_reg > 0.0):
        p0_mean = float(np.mean(p0_guess))
    else:
        # If no pedestal prior is requested / available, turn it off
        lambda_reg = 0.0
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