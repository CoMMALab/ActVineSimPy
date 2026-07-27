import warnings
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from .sqrtm import MatrixSquareRoot

# ---- Differentiable QP backend (cvxpylayers 1.2.0 `solver=`) --------------------------------
# The whole backend choice lives in this ONE constant. DIFFCP is the CPU default and the only
# backend whose deps are installed; the others are GPU backends. Switching to a GPU backend is:
# (1) set BACKEND below, (2) install that backend's stack, (3) move the QP tensors + layer to a CUDA
# device (see solve_layers / init_layers). cvxpylayers 1.2.0 backends:
#   DIFFCP      CPU, differentiable via diffcp; solve_method picks the cone solver (see below).
#   MOREAU      CPU+GPU, batched, warm-start.  deps: moreau[cuda12].          COMMERCIAL license.
#   CUCLARABEL  GPU, interior-point (robust).  deps: Julia+juliacall+cupy+diffqcp.       OSS.
#   MPAX        GPU, PDHG (first-order).        deps: jax.                                OSS.
# NOTE: GPU only pays off at LARGE batch. CPU batched throughput is ~1000 solves/s and saturates the
# 24 threads by batch~64 (per-problem cost is flat ~1ms and does NOT improve past that), so a big
# parallel batch (e.g. the SST planner) is exactly the case GPU would accelerate; at batch=1 CPU wins.
BACKEND = "DIFFCP"

# Per-backend solver options. GPU backends fall back to {} (their own defaults) until benchmarked.
# DIFFCP is the differentiable ENGINE; `solve_method` selects the cone solver it wraps and
# differentiates (diffcp supports SCS / Clarabel / ECOS).
#   Clarabel (interior-point) is the DEFAULT: on this ill-conditioned QP it converges cleanly in ~9
#   iterations to ~1e-11, with NO tuning and NO "Solved/Inaccurate" status. SCS (first-order) instead
#   stalls at its residual tolerance -- the solution is accurate (matches a high-accuracy reference)
#   but SCS never self-certifies "solved", so it burns iterations and diffcp flags it inaccurate. The
#   root cause is scaling: cvxpy's direct solve equilibrates (SCS optimal in ~50 iters) but diffcp
#   extracts the cone program without that scaling and absorbs any diagonal pre-scaling we apply, so
#   we cannot fix SCS from here -- Clarabel just sidesteps it. Clarabel is ~0-30% slower per batch.
_SOLVER_ARGS = {
    "DIFFCP": {"solve_method": "CLARABEL"},
    # SCS fallback (first-order): {"solve_method": "SCS", "acceleration_lookback": 0, "max_iters": 2_500}
}

cvxpylayer = None   # built once per process on the first init_layers() for a given problem shape
sqrtm_module = MatrixSquareRoot()


def init_layers(sol_size, Q_size, p_size, G_size, h_size, A_size, b_size):
    """Set up the differentiable cvxpylayer for the QP (inputs are unbatched sizes).
    No-op if already built -- reset the module global `cvxpylayer` to None to rebuild."""
    global cvxpylayer
    if cvxpylayer is not None:
        return

    next_dstate = cp.Variable(sol_size)
    Q_sqrt = cp.Parameter(Q_size)
    p = cp.Parameter(p_size)
    G = cp.Parameter(G_size)
    h = cp.Parameter(h_size)
    A = cp.Parameter(A_size)
    b = cp.Parameter(b_size)

    objective = cp.Minimize(0.5 * cp.sum_squares(Q_sqrt @ next_dstate) + p @ next_dstate)
    constraints = [A @ next_dstate == b, G @ next_dstate <= h]
    problem = cp.Problem(objective, constraints)

    cvxpylayer = CvxpyLayer(problem, parameters=[Q_sqrt, p, G, h, A, b],
                            variables=[next_dstate], solver=BACKEND)
    # GPU: also `.to("cuda")` the layer here once BACKEND is a GPU backend.


# Opt-in: solve the per-step QP on the GPU with qpth (CoMMALab fork) instead of cvxpylayers/diffcp.
# qpth is a batched differentiable interior-point QP solver -- accelerates LARGE batches (many
# parallel rollouts). Requires CUDA torch + qpth. See gpu_qp/ for the standalone build-out/benchmark.
USE_QPTH = False


_QP_MAXITER = 30         # qpth interior-point iteration cap
_QP_NIMP = 10            # qpth notImprovedLim (higher = don't stop early on degenerate problems)


def _solve_qpth(Q, p, G, h, A, b):
    """Batched QP solve via GPU qpth. dvsim's QP is 0.5 v'Q v + p'v s.t. Gv<=h, Av=b with Q = the
    mass matrix M (qpth takes it directly; no sqrt). Runs in fp64.
    """
    import torch
    from qpth.qp import QPFunction
    dev = torch.device("cuda")
    Qd = Q.to(dev, torch.float64)
    pd, Gd, hd, Ad, bd = (t.to(dev, torch.float64) for t in (p, G, h, A, b))
    N = Qd.shape[-1]

    nz_A = Ad.abs() > 1e-12
    g_zero = (Gd.abs().amax(dim=(0, 1)) <= 1e-12 if Gd.shape[1]
              else torch.ones(N, dtype=torch.bool, device=dev))       # (N,) zero in all inequalities
    nnz_row = nz_A.sum(dim=2)                                          # (B, mA) nonzeros per equality row
    in_real = (nz_A & (nnz_row >= 2).unsqueeze(2)).any(dim=1).any(dim=0)   # (N,) col used by a real (multi-var) row
    pinned = (nz_A & (nnz_row == 1).unsqueeze(2)).any(dim=1).any(dim=0)    # (N,) col fixed by an identity pin v_j=0
    p_zero = pd.abs().amax(0) <= 1e-12                                 # (N,) zero objective gradient
    # Trim a column only if its solution is PROVABLY 0
    keep_col = ~(g_zero & ~in_real & (pinned | p_zero))
    trimmed = not bool(keep_col.all())
    if trimmed:
        Qd = Qd[keep_col][:, keep_col]
        pd, Gd, Ad = pd[:, keep_col], Gd[:, :, keep_col], Ad[:, :, keep_col]

    empty = Gd.abs().amax(-1) <= 1e-12                # (B, mG): per-element all-zero (padding) row
    keep = ~empty.all(0)                              # keep any row with a real gradient in ANY element
    if not bool(keep.any()):
        keep[0] = True
    Gk, hk, emk = Gd[:, keep], hd[:, keep], empty[:, keep]
    Gk = torch.where(emk.unsqueeze(-1), torch.zeros_like(Gk), Gk)
    hk = torch.where(emk, torch.ones_like(hk), hk)
    amask = Ad.abs().amax(0).amax(-1) > 1e-12
    z = QPFunction(verbose=-1, maxIter=_QP_MAXITER, eps=1e-10, dense=True, notImprovedLim=_QP_NIMP)(
        Qd, pd, Gk, hk, Ad[:, amask], bd[:, amask])
    if trimmed:                                       # scatter the (0-valued) inactive columns back
        z_full = torch.zeros(z.shape[0], N, dtype=z.dtype, device=z.device)
        z_full[:, keep_col] = z
        z = z_full
    return z.to("cpu", torch.float32)


def solve_layers(Q, p, G, h, A, b):
    """Batched QP solve. Returns the primal solution (batch, N) as float32. Uses cvxpylayers (CPU,
    default) or the robust GPU qpth dense-KKT path when USE_QPTH is set."""
    if USE_QPTH:
        return _solve_qpth(Q, p, G, h, A, b)
    batch_size = p.shape[0]
    Q_batched = sqrtm_module.apply(Q).unsqueeze(0).expand(batch_size, -1, -1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solved/Inaccurate.")
        solution = cvxpylayer(Q_batched, p, G, h, A, b, solver_args=_SOLVER_ARGS.get(BACKEND, {}))
    return solution[0].float()   # diffcp returns float64 under numpy2; keep float32
