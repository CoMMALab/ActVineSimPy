'''
Is a JAX QP solver better than dvsim/batched_qp.py (torch)? Solves the same dvsim QPs, recorded
from a push in env 1a (friction cone linearized, far rows culled, as batched_qp does), replicated
to a batch, on the GPU:
    torch   dvsim/batched_qp.py: equalities eliminated by LU, Mehrotra interior point, float64
    qpax    qpax.solve_qp (primal-dual interior point in JAX), jit(vmap), float64
    mpax    MPAX r2HPDHG (first-order PDHG in JAX; what cvxpylayers' MPAX backend runs), jit(vmap)
Accuracy is max |x - x_clarabel| / max |x_clarabel| over the recorded QPs; time is per batched
solve after compilation. Memory: torch capped at 15% of the GPU, JAX at 15%, no preallocation.
Run from the repo root (needs pip install qpax mpax):
    python xnew_rrt/bench_jax_qp.py --batch 256 1024
'''

import argparse, dataclasses, json, os, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
import dvsim.batched_qp as bq
import check_batched_qp
import run_envs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepared(qps, obj_start):
    '''The recorded QPs as batched_qp sees them after linearizing and culling (float64, CPU),
    padded to common row counts.'''
    out = []
    for qp in qps:
        M, p, G, h, A, b = bq.linearize_friction(qp["M"].double(), qp["p"].double(), qp["G"].double(),
                                                 qp["h"].double(), qp["A"].double(), qp["b"].double(),
                                                 qp["fric"].double(), obj_start)
        G, h = bq.cull_rows(G, h)
        out.append((M, p[0], G[0], h[0], A[0], b[0]))
    m = max(o[2].shape[0] for o in out)
    padded = []
    for M, p, G, h, A, b in out:
        k = m - G.shape[0]
        G = torch.cat([G, torch.zeros(k, G.shape[1], dtype=G.dtype)])
        h = torch.cat([h, torch.ones(k, dtype=h.dtype)])
        padded.append((M, p, G, h, A, b))
    return padded


def batch_of(problems, B):
    idx = [i % len(problems) for i in range(B)]
    M = problems[0][0]
    return M, *(torch.stack([problems[i][j] for i in idx]) for j in range(1, 6)), idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="out/bench_jax_qp.json")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.15)

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import qpax
    from mpax import create_qp, r2HPDHG

    e, scene, cfg, _ = run_envs.load_env(os.path.join(REPO, "envs", "1a.json"))
    sim = Sim(scene, cfg)
    qps, _ = check_batched_qp.record_qps(sim, sim.initial_state(), 330, 10)
    obj_start = 3 * cfg.max_bodies
    problems = prepared(qps, obj_start)
    refs = [check_batched_qp.clarabel_linearized(qp, obj_start).numpy() for qp in qps]
    n = qps[0]["p"].shape[1]

    def accuracy(x, idx):
        return max(float(np.abs(x[i, :n] - refs[j]).max() / max(np.abs(refs[j]).max(), 1.0))
                   for i, j in enumerate(idx[:len(problems)]))

    def timed(fn, sync):
        fn(); sync()
        t = time.perf_counter()
        for _ in range(args.reps):
            out = fn()
        sync()
        return out, (time.perf_counter() - t) / args.reps

    rows = []
    for B in args.batch:
        M, p, G, h, A, b, idx = batch_of(problems, B)

        # torch: batched_qp's own path from the already linearized/culled problem
        dev = "cuda"
        Mt, pt, Gt, ht, At, bt = (t.to(dev) for t in (M, p, G, h, A, b))

        def torch_solve():
            x0, Z = bq.eliminate_equalities(At, bt)
            Mb = Mt.expand(B, -1, -1)
            Zt = Z.transpose(1, 2)
            H = Zt @ Mb @ Z
            g = (Zt @ ((Mb @ x0.unsqueeze(-1)).squeeze(-1) + pt).unsqueeze(-1)).squeeze(-1)
            u, ok = bq.ipm(H, g, Gt @ Z, ht - (Gt @ x0.unsqueeze(-1)).squeeze(-1))
            return (x0 + (Z @ u.unsqueeze(-1)).squeeze(-1)).cpu().numpy(), ok
        (x, ok), secs = timed(torch_solve, torch.cuda.synchronize)
        rows.append(dict(solver="torch batched_qp", batch=B, ms=1000 * secs, accuracy=accuracy(x, idx),
                         converged=float(ok.float().mean())))

        # qpax
        Mj, pj, Gj, hj, Aj, bj = (jnp.asarray(t.numpy()) for t in (M, p, G, h, A, b))
        qp_fn = jax.jit(jax.vmap(lambda q, G_, h_, A_, b_: qpax.solve_qp(Mj, q, A_, b_, G_, h_, solver_tol=1e-8, max_iter=40)))

        def qpax_solve():
            out = qp_fn(pj, Gj, hj, Aj, bj)
            return out
        out, secs = timed(qpax_solve, lambda: None)
        jax.block_until_ready(out)
        out, secs = timed(lambda: jax.block_until_ready(qp_fn(pj, Gj, hj, Aj, bj)), lambda: None)
        rows.append(dict(solver="qpax (jax)", batch=B, ms=1000 * secs, accuracy=accuracy(np.asarray(out[0]), idx),
                         converged=float(np.asarray(out[4]).mean())))

        # MPAX r2HPDHG: min 1/2 x'Qx + c'x  s.t.  A x = b,  G x >= h,  l <= x <= u
        solver = r2HPDHG(eps_abs=1e-6, eps_rel=1e-6, iteration_limit=20000, verbose=False)
        lo, hi = -jnp.inf * jnp.ones(Mj.shape[0]), jnp.inf * jnp.ones(Mj.shape[0])

        def one(q, G_, h_, A_, b_):
            prob = create_qp(Mj, q, A_, b_, -G_, -h_, lo, hi, use_sparse_matrix=False)
            prob = dataclasses.replace(prob, is_lp=False)   # computed from the data, which vmap cannot branch on
            res = solver.optimize(prob)
            return res.primal_solution
        mp_fn = jax.jit(jax.vmap(one))
        try:
            out, secs = timed(lambda: jax.block_until_ready(mp_fn(pj, Gj, hj, Aj, bj)), lambda: None)
            rows.append(dict(solver="mpax r2HPDHG (jax)", batch=B, ms=1000 * secs, accuracy=accuracy(np.asarray(out), idx),
                             converged=None))
        except Exception as err:                        # report rather than crash
            rows.append(dict(solver="mpax r2HPDHG (jax)", batch=B, error=f"{type(err).__name__}: {str(err)[:200]}"))
        for r in rows[-3:]:
            print(r, flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
