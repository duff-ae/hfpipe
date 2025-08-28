import numpy as np
from scipy.linalg import cho_factor, cho_solve
from joblib import Parallel, delayed
from tqdm import tqdm

BX_LEN = 3564
BX_TO_CLEAN = [3488, 3489, 3490, 3491, 3553, 3554, 3555, 3556, 3557]

def ensure(a, shape=None, dtype=None):
    out = np.ascontiguousarray(a, dtype=dtype) if dtype is not None else np.ascontiguousarray(a)
    if shape is not None and tuple(out.shape) != tuple(shape):
        raise ValueError(f"expected {shape}, got {out.shape}")
    return out

# ---- fast circulant M (vectorized) ----
def build_afterglow_matrix_fast(HFSBR, active_mask, bx_len=BX_LEN, dtype=np.float64):
    # M[:, j] = roll(HFSBR, +j)
    cols = np.vstack([np.roll(HFSBR, j) for j in range(bx_len)]).T
    return cols.astype(dtype, copy=False)

def build_single_pedestal_block(bx_len=BX_LEN, dtype=np.float64):
    return np.ones((bx_len, 1), dtype=dtype)

def precompute_lsq_context(
    HFSBR, active_mask,
    lambda_nonactive=0.05,
    lambda_reg=0.0,
    p0_guess=None,
    lambda_bx1=1.0
):
    N = len(HFSBR)
    HFSBR = ensure(HFSBR, (N,), np.float64)
    active_mask = ensure(active_mask, (N,), np.int32)

    w = np.ones(N, dtype=np.float64)
    if lambda_bx1 and lambda_bx1 != 1.0:
        bx1 = [ (i+1) % N for i in range(N) if active_mask[i] == 1 and ((i+1) % N) not in BX_TO_CLEAN ]
        w[bx1] = np.sqrt(lambda_bx1)

    M = build_afterglow_matrix_fast(HFSBR, active_mask, N)
    B = build_single_pedestal_block(N)
    A0_data = np.hstack([M, B])          # N x (N+1)
    A0_data *= w[:, None]

    C_hard = np.zeros((len(BX_TO_CLEAN), N+1), dtype=np.float64)
    for k, bx in enumerate(BX_TO_CLEAN):
        C_hard[k, bx] = 1.0

    nonactive_idx = [i for i in range(N) if active_mask[i] == 0 and i not in BX_TO_CLEAN]
    R = np.zeros((len(nonactive_idx), N+1), dtype=np.float64)
    for row, i in enumerate(nonactive_idx):
        R[row, i] = np.sqrt(lambda_nonactive)

    reg_row = np.zeros((1, N+1), dtype=np.float64)
    reg_rhs = 0.0
    if (p0_guess is not None) and (lambda_reg > 0.0):
        reg_row[0, N] = np.sqrt(lambda_reg)
        reg_rhs = np.sqrt(lambda_reg) * float(np.mean(p0_guess))

    A0 = np.vstack([A0_data, C_hard, R, reg_row])
    G0 = A0.T @ A0
    chol = cho_factor(G0, overwrite_a=False, check_finite=False)

    return dict(
        N=N, HFSBR=HFSBR, active_mask=active_mask,
        w=w, A0=A0, A0_data=A0_data,
        reg_rhs=reg_rhs, chol=chol
    )

def _make_QC(mu, active_mask, w, use_cubic=True):
    mu = np.asarray(mu, dtype=np.float64)
    am = active_mask.astype(np.float64)
    q = np.roll((mu**2) * am, +1).reshape(-1,1)
    q *= w.reshape(-1,1)
    if not use_cubic:
        return q, None
    c = np.roll((mu**3) * am, +1).reshape(-1,1)
    c *= w.reshape(-1,1)
    return q, c

def solve_one_hist_with_context(mu_obs, ctx, use_cubic=True, ridge=1e-12):
    N       = ctx['N']
    A0      = ctx['A0']
    A0_data = ctx['A0_data']
    w       = ctx['w']
    chol    = ctx['chol']
    reg_rhs = ctx['reg_rhs']
    actmask = ctx['active_mask']

    mu_obs = ensure(mu_obs, (N,), np.float64)

    b = np.zeros(A0.shape[0], dtype=np.float64)
    b[:N] = w * mu_obs
    if reg_rhs != 0.0:
        b[-1] = reg_rhs

    Atb = A0.T @ b

    Qd, Cd = _make_QC(mu_obs, actmask, w, use_cubic=use_cubic)
    UQ = A0_data.T @ Qd
    UC = A0_data.T @ Cd if use_cubic else None

    y  = cho_solve(chol, Atb, check_finite=False)
    yQ = cho_solve(chol, UQ.squeeze(-1), check_finite=False)
    if use_cubic:
        yC = cho_solve(chol, UC.squeeze(-1), check_finite=False)

    QtQ = float(Qd.ravel() @ Qd.ravel())
    Qtb = float(Qd.ravel() @ (w*mu_obs).ravel())
    UQTy  = float(UQ.ravel()  @ y.ravel())
    UQTyQ = float(UQ.ravel()  @ yQ.ravel())
    rhs1  = Qtb - UQTy

    if use_cubic:
        CtC  = float(Cd.ravel() @ Cd.ravel())
        Ctb  = float(Cd.ravel() @ (w*mu_obs).ravel())
        QtC  = float(Qd.ravel() @ Cd.ravel())
        UQTyC = float(UQ.ravel() @ yC.ravel())
        UCTy  = float(UC.ravel() @ y.ravel())
        UCTyQ = float(UC.ravel() @ yQ.ravel())
        UCTyC = float(UC.ravel() @ yC.ravel())

    if use_cubic:
        A11 = (QtQ - UQTyQ) + ridge
        A12 = (QtC - UQTyC)
        A21 = (QtC - UCTyQ)
        A22 = (CtC - UCTyC) + ridge
        det = A11*A22 - A12*A21
        if abs(det) < 1e-20:
            use_cubic = False

    if not use_cubic:
        A11 = (QtQ - UQTyQ) + ridge
        q1  = rhs1 / A11
        c1  = 0.0
    else:
        q1 = ( rhs1*A22 - A12*(Ctb - UCTy) ) / det
        c1 = ( A11*(Ctb - UCTy) - A21*rhs1 ) / det

    rhs0 = Atb - UQ.squeeze(-1)*q1
    if use_cubic:
        rhs0 -= UC.squeeze(-1)*c1
    x0 = cho_solve(chol, rhs0, check_finite=False)
    mu_true = x0[:N]
    ped     = x0[N]
    return mu_true, ped, q1, c1

def adjust_pedestal(mu_true, ped, bx_len=BX_LEN, tail_start=3500, bx_to_clean=BX_TO_CLEAN):
    tail_idx = [i for i in range(tail_start, bx_len) if i not in bx_to_clean]
    tmean = float(np.mean(mu_true[tail_idx])) if tail_idx else 0.0
    return (mu_true - tmean), (ped + tmean)

def batch_afterglow_lsq_matrix(
    bxraw: np.ndarray,              # (N, 3564) float64/float32
    HFSBR: np.ndarray,              # (3564,)
    active_mask: np.ndarray,        # (3564,)
    p0_guess=None,
    lambda_reg=0.01,
    lambda_nonactive=0.05,
    lambda_bx1=1.0,
    use_cubic=True,
    n_jobs=-1,
    backend="processes"             # "processes" | "threads"
):
    bxraw = ensure(bxraw, dtype=np.float64)
    ctx = precompute_lsq_context(
        HFSBR, active_mask,
        lambda_nonactive=lambda_nonactive,
        lambda_reg=lambda_reg,
        p0_guess=p0_guess,
        lambda_bx1=lambda_bx1
    )

    def _solve_row(row):
        mu_true, ped, q1, c1 = solve_one_hist_with_context(row, ctx, use_cubic=use_cubic)
        mu_true, ped = adjust_pedestal(mu_true, ped)
        return mu_true, ped, q1, c1

    results = Parallel(n_jobs=n_jobs, prefer=backend, batch_size=8)(
        delayed(_solve_row)(bxraw[i]) for i in tqdm(range(bxraw.shape[0]), desc="LSQ", unit="hist")
    )

    mu_list, ped_list, q1_list, c1_list = map(np.asarray, zip(*results))
    out = np.ascontiguousarray(np.vstack(mu_list), dtype=bxraw.dtype)
    ped = ped_list.astype(np.float64)
    q1  = q1_list.astype(np.float64)
    c1  = c1_list.astype(np.float64)
    return out, ped, q1, c1

# (опц) совместимость с DataFrame:
def batch_afterglow_lsq_dataframe(df, hfsbr_path, active_mask, **kwargs):
    HFSBR = np.loadtxt(hfsbr_path, dtype=np.float64, delimiter=',')
    hists = np.stack(df.bxraw).astype(np.float64, copy=False)
    out, ped, q1, c1 = batch_afterglow_lsq_matrix(hists, HFSBR, active_mask, **kwargs)
    df.bxraw = [row for row in out]
    df['pedestal_fit'] = ped
    df['q1_fit'] = q1
    df['c1_fit'] = c1
    return df

