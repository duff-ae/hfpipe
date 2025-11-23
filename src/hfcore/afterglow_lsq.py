# src/hfcore/afterglow_lsq.py

from __future__ import annotations

from typing import Sequence, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm

from .hd5schema import BX_LEN


class AfterglowSolver:
    """
    Солвер для линейной задачи восстановления истинного mu_true из наблюдаемого mu_obs,
    используя HFSBR-матрицу и регуляры.

    Строится один раз на fill (HFSBR + active_mask + bx_to_clean фиксированы),
    потом apply_batch можно вызывать на любом количестве гистограмм.
    """

    def __init__(
        self,
        A0: np.ndarray,
        chol: Tuple[np.ndarray, bool],
        N: int,
        reg_row: np.ndarray,
        reg_rhs: float,
    ) -> None:
        self.A0 = A0
        self.chol = chol
        self.N = N
        self.reg_row = reg_row
        self.reg_rhs = reg_rhs

    def _make_rhs(self, mu_obs_vec: np.ndarray) -> np.ndarray:
        """
        Построить правую часть b и посчитать A0^T b.
        """
        b = np.zeros(self.A0.shape[0], dtype=np.float64)
        b[: self.N] = mu_obs_vec
        if self.reg_row.shape[0] == 1:
            # последний ряд в A0 соответствует регуляризации на pedestal
            b[-1] = self.reg_rhs
        return self.A0.T @ b

    def _solve_one(self, mu_obs_vec: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Решить нормальную систему для одного гисто:
            (A0^T A0) x = A0^T b
        где x = [mu_true (N), pedestal (1)].
        """
        rhs = self._make_rhs(mu_obs_vec)
        x = cho_solve(self.chol, rhs, check_finite=False)
        mu_true = x[: self.N]
        ped = float(x[self.N])
        return mu_true, ped

    def apply_batch(
        self,
        hists: np.ndarray,
        n_jobs: int = -1,
        desc: str = "LSQ afterglow",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применить солвер к батчу гистограмм.

        hists: shape (T, BX_LEN), float64/float32
        Возвращает:
          mu_batch: shape (T, BX_LEN)
          ped_arr: shape (T,)
        """
        hists = np.asarray(hists, dtype=np.float64)
        T, N = hists.shape
        assert N == self.N, f"hists.shape[1]={N} != N={self.N}"

        results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=8)(
            delayed(self._solve_one)(hists[t]) for t in tqdm(range(T), desc=desc, unit="hist")
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
) -> AfterglowSolver:
    """
    Построить AfterglowSolver, грузя HFSBR из файла CSV.

    Параметры:
      - hfsbr_path: путь к CSV с HFSBR (одна колонка, длина BX_LEN)
      - active_mask: вектор длины BX_LEN (1 для коллайдерных BX, 0 для неактивных)
      - bx_to_clean: список BX, для которых жёстко ставим mu_true = 0
      - p0_guess: опциональный вектор начальных педесталов, используется только
                  для регуляризации, фактический фит педестала считается LSQ
      - lambda_reg: вес регуляризации по педесталу
      - lambda_nonactive: вес мягкого притягивания неактивных BX к 0
    """
    H = np.loadtxt(hfsbr_path, dtype=np.float64, delimiter=",")
    H = H.astype(np.float64, copy=False)
    N = H.shape[0]
    assert N == BX_LEN, f"HFSBR len={N} != BX_LEN={BX_LEN}"

    active_mask = np.asarray(active_mask, dtype=np.int32)
    assert active_mask.shape[0] == N, "active_mask must match BX length"

    bx_to_clean = np.asarray(bx_to_clean, dtype=np.int64)

    # ---------- build [M | B] ----------
    # M: circulant matrix with columns roll(H, j)
    cols = [np.roll(H, j) for j in range(N)]
    M = np.stack(cols, axis=1).astype(np.float64, copy=False)  # (N x N)
    B = np.ones((N, 1), dtype=np.float64)
    A_data = np.hstack([M, B])  # (N x (N+1))

    # ---------- hard constraints: mu_true[bx] = 0 for bx_to_clean ----------
    C_hard = np.zeros((len(bx_to_clean), N + 1), dtype=np.float64)
    for k, bx in enumerate(bx_to_clean):
        C_hard[k, int(bx % N)] = 1.0

    # ---------- soft constraints: non-active BX -> ~0 ----------
    mask_nonact = (active_mask == 0)
    mask_nonact[bx_to_clean % N] = False
    nonact_idx = np.flatnonzero(mask_nonact)
    R = np.zeros((nonact_idx.size, N + 1), dtype=np.float64)
    for row, i in enumerate(nonact_idx):
        R[row, int(i)] = np.sqrt(lambda_nonactive)

    # ---------- optional pedestal prior ----------
    reg_row = np.zeros((0, N + 1), dtype=np.float64)
    reg_rhs = 0.0
    if (p0_guess is not None) and (lambda_reg > 0.0):
        reg_row = np.zeros((1, N + 1), dtype=np.float64)
        reg_row[0, N] = np.sqrt(lambda_reg)
        reg_rhs = np.sqrt(lambda_reg) * float(np.mean(p0_guess))

    # ---------- full A0 and Cholesky ----------
    A0 = np.vstack([A_data, C_hard, R, reg_row])  # ((N+H+S+[0/1]) x (N+1))
    G0 = A0.T @ A0
    chol = cho_factor(G0, overwrite_a=False, check_finite=False)

    return AfterglowSolver(A0=A0, chol=chol, N=N, reg_row=reg_row, reg_rhs=reg_rhs)
