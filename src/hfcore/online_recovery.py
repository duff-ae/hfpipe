# src/hfcore/online_recovery.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .hd5schema import BX_LEN
import logging

log = logging.getLogger("hfpipe")


@dataclass
class OnlineStates:
    """
    Container for reconstructed online-like states per row:

      - mu_before : BX rates before ComputeAfterglow and pedestal
      - mu_after  : BX rates after ComputeAfterglow, before SubtractPedestal
      - pedestal  : 4-element pedestal values (per row)
    """
    mu_before: np.ndarray  # shape (T, BX_LEN)
    mu_after: np.ndarray   # shape (T, BX_LEN)
    pedestal: np.ndarray   # shape (T, 4)


# ----------------------------------------------------------------------
# 1) Exact implementation of online afterglow loops (forward & inverse)
# ----------------------------------------------------------------------

def apply_afterglow_forward(
    mu_before: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact Python reimplementation of the C++ ComputeAfterglow loop
    on a single BX histogram.
    """
    mu_before = np.asarray(mu_before, dtype=np.float64)
    hfsbr = np.asarray(hfsbr, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)

    if mu_before.shape[0] != BX_LEN:
        raise ValueError(f"mu_before has length {mu_before.shape[0]}, expected {BX_LEN}")
    if hfsbr.shape[0] < BX_LEN:
        raise ValueError(
            f"hfsbr length {hfsbr.shape[0]} is smaller than BX_LEN={BX_LEN}"
        )
    if active.shape[0] != BX_LEN:
        raise ValueError(
            f"active_mask has length {active.shape[0]}, expected {BX_LEN}"
        )

    N = BX_LEN
    mu = mu_before.copy()

    for ibx in range(N):
        if not active[ibx]:
            continue
        src = mu[ibx]
        if src == 0.0:
            continue
        for jbx in range(ibx + 1, ibx + N):
            if jbx < N:
                idx = jbx
            else:
                idx = jbx - N
            diff = jbx - ibx
            mu[idx] -= src * hfsbr[diff]

    return mu.astype(np.float64)


def apply_afterglow_inverse(
    mu_after: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact inverse of `apply_afterglow_forward` for a single BX histogram.

    We reverse the outer loop and flip the sign of the update.
    """
    mu_after = np.asarray(mu_after, dtype=np.float64)
    hfsbr = np.asarray(hfsbr, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)

    if mu_after.shape[0] != BX_LEN:
        raise ValueError(f"mu_after has length {mu_after.shape[0]}, expected {BX_LEN}")
    if hfsbr.shape[0] < BX_LEN:
        raise ValueError(
            f"hfsbr length {hfsbr.shape[0]} is smaller than BX_LEN={BX_LEN}"
        )
    if active.shape[0] != BX_LEN:
        raise ValueError(
            f"active_mask has length {active_mask.shape[0]}, expected {BX_LEN}"
        )

    N = BX_LEN
    mu = mu_after.copy()

    for ibx in range(N - 1, -1, -1):
        if not active[ibx]:
            continue
        src = mu[ibx]
        if src == 0.0:
            continue
        for jbx in range(ibx + 1, ibx + N):
            if jbx < N:
                idx = jbx
            else:
                idx = jbx - N
            diff = jbx - ibx
            mu[idx] += src * hfsbr[diff]

    return mu.astype(np.float64)


# ----------------------------------------------------------------------
# 2) Method A: reconstruction using extra tables
# ----------------------------------------------------------------------

def reconstruct_from_tables_batch(
    bxraw_final: np.ndarray,
    pedestal_4: np.ndarray,
    afterglow_frac: np.ndarray,
) -> OnlineStates:
    """
    Reconstruct online C++-like arrays using hfEtPedestal and hfafterglowfrac.

    C++ logic:

        mu_after[bx]    = mu_before[bx] * hfafterglowfrac[bx]
        bxraw_final[bx] = mu_after[bx] - pedestal[bx % 4]

    Hence:

        mu_after[bx]    = bxraw_final[bx] + pedestal[bx % 4]
        mu_before[bx]   = (hfafterglowfrac[bx] > 0)
                          ? mu_after[bx] / hfafterglowfrac[bx] : 0
    """
    bxraw_final = np.asarray(bxraw_final, dtype=np.float64)
    pedestal_4 = np.asarray(pedestal_4, dtype=np.float64)
    afterglow_frac = np.asarray(afterglow_frac, dtype=np.float64)

    T, nbx = bxraw_final.shape
    if nbx != BX_LEN:
        raise ValueError(f"bxraw_final has BX dimension {nbx}, expected {BX_LEN}")
    if pedestal_4.shape != (T, 4):
        raise ValueError(f"pedestal_4 shape {pedestal_4.shape}, expected (T, 4)")
    if afterglow_frac.shape != (T, BX_LEN):
        raise ValueError(
            f"afterglow_frac shape {afterglow_frac.shape}, expected (T, {BX_LEN})"
        )

    idx_mod4 = np.arange(BX_LEN) % 4

    mu_after = np.empty_like(bxraw_final, dtype=np.float64)
    mu_before = np.zeros_like(bxraw_final, dtype=np.float64)

    for i in range(T):
        ped_pattern = pedestal_4[i][idx_mod4]
        mu_after[i] = bxraw_final[i] + ped_pattern

        mask = afterglow_frac[i] > 0.0
        mu_before[i, mask] = mu_after[i, mask] / afterglow_frac[i, mask]

    return OnlineStates(
        mu_before=mu_before.astype(np.float32),
        mu_after=mu_after.astype(np.float32),
        pedestal=pedestal_4.astype(np.float32),
    )


# ----------------------------------------------------------------------
# 3) Method B: reconstruction via inverse online afterglow + pedestal fit
# ----------------------------------------------------------------------

def _precompute_pedestal_basis(
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """
    Build basis vectors v_i = F^{-1}(q_i) in the online afterglow model,
    where q_i is +1 on BX with (bx % 4 == i) in mu_after-space.
    """
    v_basis = np.zeros((4, BX_LEN), dtype=np.float64)

    for i in range(4):
        q = np.zeros(BX_LEN, dtype=np.float64)
        q[i::4] = 1.0
        v_basis[i] = apply_afterglow_inverse(q, hfsbr, active_mask)

    return v_basis


def recover_single_hist_online(
    bxraw_final: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
    v_basis: np.ndarray,
    zero_bx: tuple[int, ...] = (3553, 3554, 3555, 3556, 3557),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate reconstruction of (mu_before, mu_after, pedestal[4])
    for a single BX histogram using the inverse ONLINE afterglow algorithm.

    We seek p[0..3] such that:

        r(p) = F^{-1}(bxraw_final + P(p))

    satisfies r(p)[z] ≈ 0 for z in zero_bx, with F^{-1} = apply_afterglow_inverse.
    """
    y = np.asarray(bxraw_final, dtype=np.float64)

    # r0 = F^{-1}(y)
    r0 = apply_afterglow_inverse(y, hfsbr, active_mask)

    zero_bx = np.asarray(zero_bx, dtype=int)
    A = np.zeros((zero_bx.size, 4), dtype=np.float64)
    b = np.zeros(zero_bx.size, dtype=np.float64)

    for m, z in enumerate(zero_bx):
        for i in range(4):
            A[m, i] = v_basis[i, z]
        b[m] = -r0[z]

    p, *_ = np.linalg.lstsq(A, b, rcond=None)

    ped_pattern = p[np.arange(BX_LEN) % 4]
    mu_after = y + ped_pattern

    mu_before = apply_afterglow_inverse(mu_after, hfsbr, active_mask)

    return (
        mu_before.astype(np.float32),
        mu_after.astype(np.float32),
        p.astype(np.float32),
    )


def reconstruct_from_online_batch(
    bxraw_final: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
    zero_bx: tuple[int, ...] = (3553, 3554, 3555, 3556, 3557),
) -> OnlineStates:
    """
    Батчевая реконструкция (mu_before, mu_after, pedestal[4]) по онлайн-алгоритму.

    Идея:

      1) r0_all = F^{-1}(y_all)          — F^{-1} применён батчево;
      2) v_basis[i,:] = F^{-1}(q_i)      — 4 базисных вектора для шаблона bx%4 == i;
      3) A[m,i]    = v_basis[i, zero_bx[m]];
         p_t       = argmin || A p + r0_t(zero_bx) ||^2
         => p_t = - pinv(A) @ r0_t(zero_bx)
      4) mu_before_t = r0_t + sum_i p_t[i] * v_basis[i,:]
         mu_after_t  = y_t  + P(p_t), где P разворачивает ped по bx%4.
    """
    bxraw_final = np.asarray(bxraw_final, dtype=np.float32)
    hfsbr = np.asarray(hfsbr, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)

    T, N = bxraw_final.shape
    if N != BX_LEN:
        raise ValueError(f"bxraw_final has BX dimension {N}, expected {BX_LEN}")
    if active_mask.shape[0] != N:
        raise ValueError(
            f"active_mask has length {active_mask.shape[0]}, expected {N}"
        )

    zero_bx = np.asarray(zero_bx, dtype=int)
    M = zero_bx.size

    # --------------------------------------------------------------
    # 1) F^{-1} для всех гистограмм сразу
    # --------------------------------------------------------------
    y_all = bxraw_final.astype(np.float64)
    r0_all = apply_afterglow_inverse_batch(
        mu_after_all=y_all,
        hfsbr=hfsbr,
        active_mask=active_mask,
    )  # shape (T, N)

    # --------------------------------------------------------------
    # 2) F^{-1} для базисов q_i (4 шаблона по bx % 4)
    # --------------------------------------------------------------
    q_all = np.zeros((4, N), dtype=np.float64)
    for i in range(4):
        q_all[i, i::4] = 1.0

    v_all = apply_afterglow_inverse_batch(
        mu_after_all=q_all,
        hfsbr=hfsbr,
        active_mask=active_mask,
    )  # shape (4, N)

    # Матрица A[m, i] = v_i(zero_bx[m])
    A = np.zeros((M, 4), dtype=np.float64)
    for m, z in enumerate(zero_bx):
        for i in range(4):
            A[m, i] = v_all[i, z]

    # Псевдообратная матрица
    A_pinv = np.linalg.pinv(A)  # shape (4, M)

    # --------------------------------------------------------------
    # 3) Решаем для всех строк p_t сразу
    # --------------------------------------------------------------
    # r0_all_zero: shape (T, M)
    r0_all_zero = r0_all[:, zero_bx]  # r0_t(zero_bx[m])

    # B = - r0_all_zero^T: shape (M, T)
    B = -r0_all_zero.T  # (M, T)

    # P_all = A_pinv @ B: shape (4, T)
    P_all = A_pinv @ B

    # p_all: shape (T, 4)
    p_all = P_all.T.astype(np.float32)

    # --------------------------------------------------------------
    # 4) mu_after = y + P(p), mu_before = r0 + P(v_basis)
    # --------------------------------------------------------------
    idx_mod4 = np.arange(N) % 4

    # pedestal как паттерн по BX: shape (T, N)
    ped_pattern = np.zeros((T, N), dtype=np.float64)
    for i in range(4):
        mask = (idx_mod4 == i)
        if not np.any(mask):
            continue
        ped_pattern[:, mask] = p_all[:, i][:, None]

    mu_after_all = (y_all + ped_pattern).astype(np.float32)

    # mu_before_all = r0_all + P_all @ v_all
    # P_all: (T,4), v_all: (4,N) -> (T,N)
    mu_before_all = (r0_all + P_all.T @ v_all).astype(np.float32)

    return OnlineStates(
        mu_before=mu_before_all,
        mu_after=mu_after_all,
        pedestal=p_all,
    )



# ----------------------------------------------------------------------
# 4) Debug utilities: pulls / differences between two methods
# ----------------------------------------------------------------------

def compute_pulls(
    ref: np.ndarray,
    test: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Simple pull definition:

        pull = (test - ref) / sqrt(|ref| + eps)
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    return (test - ref) / np.sqrt(np.abs(ref) + eps)


def compare_recovery_methods(
    states_tables: OnlineStates,
    states_online: OnlineStates,
) -> dict[str, np.ndarray]:
    """
    Compare two recovery methods:

      - 'tables' : exact reconstruction using hfEtPedestal + hfafterglowfrac
      - 'online' : approximate reconstruction using inverse ComputeAfterglow
                   and pedestal fit from zero-BX constraints.
    """
    if states_tables.mu_before.shape != states_online.mu_before.shape:
        raise ValueError("mu_before shapes differ between methods")
    if states_tables.mu_after.shape != states_online.mu_after.shape:
        raise ValueError("mu_after shapes differ between methods")
    if states_tables.pedestal.shape != states_online.pedestal.shape:
        raise ValueError("pedestal shapes differ between methods")

    pull_before = compute_pulls(
        ref=states_tables.mu_before,
        test=states_online.mu_before,
    ).astype(np.float32)

    pull_after = compute_pulls(
        ref=states_tables.mu_after,
        test=states_online.mu_after,
    ).astype(np.float32)

    diff_ped = (states_online.pedestal - states_tables.pedestal).astype(np.float32)

    return {
        "pull_mu_before": pull_before,
        "pull_mu_after": pull_after,
        "diff_pedestal": diff_ped,
    }

def _compute_max_diff(hfsbr: np.ndarray, tol: float = 1e-12) -> int:
    """
    Эффективная длина ядра afterglow: максимальный d,
    для которого |HFSBR[d]| > tol.
    """
    hfsbr = np.asarray(hfsbr, dtype=np.float64)
    nonzero = np.where(np.abs(hfsbr) > tol)[0]
    if nonzero.size == 0:
        return 0
    return int(nonzero.max())

def apply_afterglow_inverse_batch(
    mu_after_all: np.ndarray,
    hfsbr: np.ndarray,
    active_mask: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Батчевая версия онлайн-инверсии:

      mu_before_all = F^{-1}(mu_after_all)

    mu_after_all : shape (T, BX_LEN)
    """
    mu = np.asarray(mu_after_all, dtype=np.float64).copy()
    hfsbr = np.asarray(hfsbr, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)

    T, N = mu.shape
    if N != BX_LEN:
        raise ValueError(f"apply_afterglow_inverse_batch: N={N}, expected {BX_LEN}")
    if active.shape[0] != N:
        raise ValueError(f"active_mask length {active.shape[0]} != {N}")

    max_diff = _compute_max_diff(hfsbr, tol=tol)
    if max_diff <= 0:
        # ядро нулевое -> F ~ Identity
        return mu

    # Ограничиваем длину ядра по BX
    max_diff = min(max_diff, N - 1)

    log.info(
        "[online_recovery] apply_afterglow_inverse_batch: T=%d, N=%d, max_diff=%d",
        T, N, max_diff,
    )

    # Алгоритм: то же, что в apply_afterglow_inverse, но сразу по всем T.
    for ibx in range(N - 1, -1, -1):
        if not active[ibx]:
            continue

        src = mu[:, ibx]        # shape (T,)
        if np.allclose(src, 0.0):
            continue

        max_j = ibx + max_diff

        # Сегмент без wrap-around: j in [ibx+1, min(N-1, max_j)]
        j1_start = ibx + 1
        if j1_start <= N - 1:
            j1_end = min(N - 1, max_j)
            # бежим по diff, их обычно немного (max_diff ~ десятки)
            for j in range(j1_start, j1_end + 1):
                d = j - ibx
                mu[:, j] += src * hfsbr[d]

        # Сегмент с wrap-around, если max_j >= N
        if max_j >= N:
            j2_start = N
            j2_end = max_j
            for j in range(j2_start, j2_end + 1):
                d = j - ibx
                idx = j - N
                mu[:, idx] += src * hfsbr[d]

    return mu.astype(np.float64)
