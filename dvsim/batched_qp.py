"""Batched QP solve for dvsim's step, in plain torch: the whole batch in one set of tensor ops, on
whatever device the inputs are on (CPU or CUDA). It replaces cvxpylayers/diffcp, which solves the
batch one element at a time on a thread pool. Opt in with solver.BACKEND = "TORCH_IPM".

The problem dynamic_vine.dynamic_solve builds is

    min  1/2 v'Mv + p'v + sum_k || f_k * v_k ||       s.t.  A v = b,  G v <= h

with v_k = (vx, vy, vtheta) of movable object k and f_k its friction weights (w, w, w c). Steps:

1. Friction cone -> linear. || (a, b, c) || <= t is replaced by two nested polygons:
       r >= d . (a, b)      for N_XY directions d around the circle      (r ~ ||(a, b)||)
       t >= e . (r, c)      for N_RC directions e over the half circle   (t ~ ||(r, c)||)
   with cost w * t, two extra variables per object. Inscribed polygons underestimate the norm by
   at most a factor cos(pi / N_XY) * cos(pi / (2 (N_RC - 1))) = 0.96; the weights are divided by
   the mean factor so the friction is unbiased on average.
2. Rows that cannot bind this step are dropped: G rows that are all zero, and rows whose gap h is
   larger than `cull` (a body or box would have to move that far in one step; the step's velocity
   caps allow a few mm). What is left is packed to the batch's largest count, padded with 0 <= 1.
3. Equalities eliminated by variable reduction from a pivoted LU of A': x = x0 + Z u with Z a
   basis of A's null space. The vine's joint and growth rows leave k = max_bodies + 5 * n_obj
   free variables.
4. The inequality-only QP in u is solved with Mehrotra's predictor-corrector interior point
   method, a fixed number of iterations for the whole batch; each iteration is one batched
   Cholesky factorization of a k x k matrix. Elements that do not converge come back as NaN rows
   (dynamic_vine.step then reports them not ok); the others are unaffected.
Everything inside runs in float64.
"""
import math
import torch

N_XY = 16          # directions for || (vx, vy) ||
N_RC = 9           # directions for || (r, c * vtheta) ||, over the half circle r >= 0
ITERS = 30
TOL = 1e-7         # converged when the scaled residuals and the duality gap are below this
CULL = 60.0        # [internal length units, mm] drop rows whose gap is larger than this
GRAPH = False       # graph mode (see the docstring)
STATIC_ROWS = None  # graph mode: pack the culled rows to exactly this many (None: the batch's largest count)
SAFE = 20.0         # [mm] graph mode: dropping a row with a smaller gap than this sets OVERFLOW
OVERFLOW = None     # graph mode: (1,) bool tensor, True when the last step dropped a row within SAFE


def _directions():
    a = torch.arange(N_XY, dtype=torch.float64) * (2 * math.pi / N_XY)
    xy = torch.stack([torch.cos(a), torch.sin(a)], dim=1)                       # (N_XY, 2)
    b = torch.linspace(-math.pi / 2, math.pi / 2, N_RC, dtype=torch.float64)
    rc = torch.stack([torch.cos(b), torch.sin(b)], dim=1)                       # (N_RC, 2)
    return xy, rc


def _mean_polygon_factor(n=20000, seed=0):
    """Mean over random directions of (polygonal norm) / (true norm), for unbiasing the weights."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, generator=g, dtype=torch.float64, device="cpu")
    xy, rc = (t.cpu() for t in _directions())
    r = (x[:, :2] @ xy.T).amax(1)
    t = (torch.stack([r, x[:, 2]], 1) @ rc.T).amax(1)
    return float((t / x.norm(dim=1)).mean())


POLY_FACTOR = _mean_polygon_factor()


def linearize_friction(M, p, G, h, A, b, fric, obj_start):
    """Append the friction variables (r_k, t_k) and their rows; returns (M, p, G, h, A, b) with the
    norm terms gone. fric: (B, n_obj, 3) weights (w, w, w c)."""
    B, n = p.shape
    if fric.dim() == 2:
        fric = fric.unsqueeze(0).expand(B, -1, -1)
    n_obj = fric.shape[1]
    xy, rc = _directions()
    xy, rc = xy.to(p), rc.to(p)
    nx = n + 2 * n_obj
    w = fric[:, :, 0] / POLY_FACTOR                                      # (B, n_obj)
    c = fric[:, :, 2] / fric[:, :, 0].clamp_min(1e-30)                   # rotational factor

    dev = p.device
    rows = torch.zeros(B, n_obj * (N_XY + N_RC), nx, dtype=p.dtype, device=dev)
    for k in range(n_obj):
        o, r_col, t_col = obj_start + 3 * k, n + 2 * k, n + 2 * k + 1
        i = k * (N_XY + N_RC)
        rows[:, i:i + N_XY, o] = xy[:, 0]                                # d . (vx, vy) - r <= 0
        rows[:, i:i + N_XY, o + 1] = xy[:, 1]
        rows[:, i:i + N_XY, r_col] = -1.0
        j = i + N_XY
        rows[:, j:j + N_RC, r_col] = rc[:, 0]                            # e . (r, c vtheta) - t <= 0
        rows[:, j:j + N_RC, o + 2] = rc[:, 1] * c[:, k:k + 1]
        rows[:, j:j + N_RC, t_col] = -1.0

    M = torch.block_diag(M, torch.zeros(2 * n_obj, 2 * n_obj, dtype=M.dtype, device=dev))
    q = torch.zeros(B, 2 * n_obj, dtype=p.dtype, device=dev)
    q[:, 1::2] = w
    p = torch.cat([p, q], dim=1)
    G = torch.cat([torch.cat([G, torch.zeros(B, G.shape[1], 2 * n_obj, dtype=G.dtype, device=dev)], 2), rows], 1)
    h = torch.cat([h, torch.zeros(B, rows.shape[1], dtype=h.dtype, device=dev)], 1)
    A = torch.cat([A, torch.zeros(B, A.shape[1], 2 * n_obj, dtype=A.dtype, device=dev)], 2)
    return M, p, G, h, A, b


def cull_rows(G, h, cull=CULL, rows=None):
    """Drop rows that cannot bind (all-zero, or gap > cull); pack the rest, padding with 0 <= 1.
    rows: pack to exactly this many, nearest first (graph mode); None: to the largest count."""
    global OVERFLOW
    keep = (G.abs().amax(-1) > 0) & (h < cull)
    count = keep.sum(1)
    if rows is None:
        m = max(int(count.max()), 1)
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :m]   # kept rows first
    else:
        m = min(rows, G.shape[1])
        key = torch.where(keep, h, torch.full_like(h, math.inf))
        full = torch.argsort(key, dim=1, stable=True)
        order = full[:, :m]                                                        # nearest kept rows first
        OVERFLOW = (torch.gather(key, 1, full[:, m:m + 1]) < SAFE).any().reshape(1) if G.shape[1] > m \
            else torch.zeros(1, dtype=torch.bool, device=G.device)
        count = count.clamp(max=m)
    Gk = torch.gather(G, 1, order.unsqueeze(-1).expand(-1, -1, G.shape[2]))
    hk = torch.gather(h, 1, order)
    pad = torch.arange(m, device=G.device).unsqueeze(0) >= count.unsqueeze(1)
    Gk = torch.where(pad.unsqueeze(-1), torch.zeros_like(Gk), Gk)
    hk = torch.where(pad, torch.ones_like(hk), hk)
    return Gk, hk


def eliminate_equalities(A, b):
    """x = x0 + Z u for every x with A x = b: (x0 (B, n), Z (B, n, n - me)).
    Variable reduction from a pivoted LU of A' (batched LU is ~60x faster than batched QR on a
    GPU): A' = P L U, L = [L1; L2] with L1 unit lower triangular, so A = U' L' P' and with
    y = P' x, A x = b becomes U' (L1' y1 + L2' y2) = b. Then y2 = u is free,
    y1 = L1^-T (U^-T b - L2' u).  Requires A to have full row rank (dvsim pins the padding rows)."""
    B, me, n = A.shape
    At = A.transpose(1, 2)
    if GRAPH:                                                           # square: one recordable batched call
        At = torch.cat([At, torch.zeros(B, n, n - me, dtype=A.dtype, device=A.device)], 2)
    LU, piv, _ = torch.linalg.lu_factor_ex(At)                          # (_ex: no error check, no sync)
    P, L, U = torch.lu_unpack(LU, piv)
    L, U = L[:, :, :me], U[:, :me, :me]                                 # (B,n,n), (B,n,me), (B,me,me)
    d = torch.diagonal(U, dim1=1, dim2=2)
    bad = d.abs() < 1e-12 * d.abs().amax(1, keepdim=True).clamp_min(1e-300)   # an all-zero row of A
    U = U + torch.diag_embed(bad.to(U.dtype))
    L1t, L2t = L[:, :me, :].transpose(1, 2), L[:, me:, :].transpose(1, 2)
    w = torch.linalg.solve_triangular(U.transpose(1, 2), torch.where(bad, 0.0, b).unsqueeze(-1), upper=False)
    y1 = torch.linalg.solve_triangular(L1t, w, upper=True, unitriangular=True)
    Y1 = -torch.linalg.solve_triangular(L1t, L2t, upper=True, unitriangular=True)    # (B, me, k)
    eye = torch.eye(n - me, dtype=A.dtype, device=A.device).expand(B, -1, -1)
    x0 = (P[:, :, :me] @ y1).squeeze(-1)
    Z = P[:, :, :me] @ Y1 + P[:, :, me:] @ eye
    return x0, Z


def ipm(H, g, G, h, iters=ITERS, tol=TOL):
    """min 1/2 u'Hu + g'u  s.t.  G u <= h, batched. Returns (u (B, k), converged (B,))."""
    B, m, k = G.shape
    dev = H.device
    eye = torch.eye(k, dtype=H.dtype, device=dev)
    reg = 1e-12 * torch.diagonal(H, dim1=1, dim2=2).abs().amax(1).clamp_min(1.0)
    u = torch.zeros(B, k, dtype=H.dtype, device=dev)
    s = torch.clamp(h, min=1.0)
    z = torch.ones(B, m, dtype=H.dtype, device=dev)
    scale = 1.0 + torch.maximum(g.abs().amax(1), h.abs().amax(1))
    ok = torch.ones(B, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    def mv(Mt, x):
        return (Mt @ x.unsqueeze(-1)).squeeze(-1)

    def max_step(x, dx):
        ratio = torch.where(dx < 0, -x / dx, torch.full_like(x, math.inf))
        return ratio.amin(1).clamp(max=1.0)

    for i in range(iters):
        rd = mv(H, u) + g + mv(G.transpose(1, 2), z)
        rp = mv(G, u) + s - h
        mu = (s * z).sum(1) / m
        done = done | ((rd.abs().amax(1) < tol * scale) & (rp.abs().amax(1) < tol * scale) & (mu < tol * scale))
        if not GRAPH and i % 5 == 4 and bool(done.all()):       # (a GPU sync, so not every iteration)
            break
        W = z / s
        K = H + G.transpose(1, 2) @ (W.unsqueeze(-1) * G) + reg[:, None, None] * eye
        L, info = torch.linalg.cholesky_ex(K)
        ok = ok & (info == 0)
        L = torch.where((info == 0)[:, None, None], L, eye.expand_as(L))
        # L^-1 once, then both solves are matrix-vector products (one triangular solve per iteration
        # instead of four: on a GPU each is a slow, latency-bound kernel)
        Li = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)

        def solve(rc):
            rhs = -rd - mv(G.transpose(1, 2), W * rp - rc / s)
            du = mv(Li.transpose(1, 2), mv(Li, rhs))
            dz = W * (mv(G, du) + rp) - rc / s
            ds = -rp - mv(G, du)
            return du, ds, dz

        # predictor (affine) then Mehrotra corrector
        du, ds, dz = solve(s * z)
        a = torch.minimum(max_step(s, ds), max_step(z, dz))
        mu_aff = ((s + a[:, None] * ds) * (z + a[:, None] * dz)).sum(1) / m
        sigma = (mu_aff / mu.clamp_min(1e-300)).clamp(0, 1) ** 3
        du, ds, dz = solve(s * z + ds * dz - (sigma * mu)[:, None])
        a = 0.99 * torch.minimum(max_step(s, ds), max_step(z, dz))
        a = torch.where(done, torch.zeros_like(a), a)          # converged elements stay put
        u = u + a[:, None] * du
        s = (s + a[:, None] * ds).clamp_min(1e-300)
        z = (z + a[:, None] * dz).clamp_min(1e-300)
    return u, ok & done & torch.isfinite(u).all(1)


def solve(M, p, G, h, A, b, fric=None, obj_start=None, cull=CULL, iters=None):
    """dvsim's QP for the whole batch (see the module docstring). M (n, n) shared or (B, n, n);
    p (B, n); G (B, mi, n); h (B, mi); A (B, me, n); b (B, me); fric (B, n_obj, 3) or None.
    Returns the primal v (B, n) in the inputs' dtype, NaN rows where the solve failed."""
    out_dtype = p.dtype
    B, n = p.shape
    with torch.device(p.device):
        M, p, G, h, A, b = (t.to(torch.float64) for t in (M, p, G, h, A, b))
        if M.dim() == 2:
            M = M.unsqueeze(0).expand(B, -1, -1)
        if fric is not None and fric.numel() > 0:
            M, p, G, h, A, b = linearize_friction(M[0], p, G, h, A, b, fric.to(torch.float64), obj_start)
            M = M.unsqueeze(0).expand(B, -1, -1)
        G, h = cull_rows(G, h, cull, STATIC_ROWS)
        x0, Z = eliminate_equalities(A, b)
        Zt = Z.transpose(1, 2)
        H = Zt @ M @ Z
        g = (Zt @ ((M @ x0.unsqueeze(-1)).squeeze(-1) + p).unsqueeze(-1)).squeeze(-1)
        Gr = G @ Z
        hr = h - (G @ x0.unsqueeze(-1)).squeeze(-1)
        u, ok = ipm(H, g, Gr, hr, iters=iters or ITERS)
        x = x0 + (Z @ u.unsqueeze(-1)).squeeze(-1)
        x = torch.where(ok[:, None], x, torch.full_like(x, math.nan))
        return x[:, :n].to(out_dtype)
