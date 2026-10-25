'''
Checks dvsim/batched_qp.py (the batched torch QP solver, solver.BACKEND = "TORCH_IPM") against
cvxpylayers/diffcp/Clarabel on QPs recorded from a real push (env 1a, the vine grown straight into
the box), and compares whole rollouts.
    1. same linearized-friction QP, solved by cvxpy + Clarabel  -> is the interior point correct?
    2. exact friction cone (cvxpylayers, the sim's default)      -> what does linearizing cost?
    3. no row culling                                           -> does dropping far rows change it?
    4. rollouts of 4 curvatures, 330 steps, both backends        -> how far apart do they end up?
    5. the head-on push alone with the curvature nudged by 0.02  -> how sensitive is it anyway?
Run from the repo root:
    python xnew_rrt/check_batched_qp.py [--device cuda]
'''

import argparse, math, os, sys

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
import dvsim.dynamic_vine as dv
import dvsim.batched_qp as bq
import dvsim.solver as solver
import run_envs
import viz
from tip_block_rrt import box_positions_mm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def record_qps(sim, start, steps, every):
    recorded, original = [], dv.solve_layers

    def spy(M, p, G, h, A, b, fric=None):
        out = original(M, p, G, h, A, b, fric=fric)
        if spy.k % every == 0:
            recorded.append(dict(M=M.clone(), p=p.clone(), G=G.clone(), h=h.clone(), A=A.clone(), b=b.clone(),
                                 fric=None if fric is None else fric.clone(), diffcp=out.clone()))
        spy.k += 1
        return out
    spy.k = 0
    dv.solve_layers = spy
    try:
        states, ok = sim.rollout(start, 0.0, steps)
    finally:
        dv.solve_layers = original
    return recorded, states


def clarabel_linearized(qp, obj_start):
    '''The exact problem batched_qp solves (friction linearized, all rows), via cvxpy + Clarabel.'''
    import cvxpy as cp
    M, p, G, h, A, b = bq.linearize_friction(qp["M"].double(), qp["p"].double(), qp["G"].double(), qp["h"].double(),
                                             qp["A"].double(), qp["b"].double(), qp["fric"].double(), obj_start)
    M, p, G, h, A, b = (t.numpy() for t in (M, p[0], G[0], h[0], A[0], b[0]))
    x = cp.Variable(M.shape[0])
    L = np.linalg.cholesky(M + 1e-12 * np.eye(M.shape[0]))
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(L.T @ x) + p @ x), [A @ x == b, G @ x <= h])
    prob.solve(solver=cp.CLARABEL)
    return torch.tensor(x.value[:qp["p"].shape[1]], device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=os.path.join(REPO, "envs", "1a.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    e, scene, cfg, _ = run_envs.load_env(args.env)
    sim = Sim(scene, cfg)
    qps, _ = record_qps(sim, sim.initial_state(), 330, 10)
    obj_start = 3 * cfg.max_bodies
    print(f"{len(qps)} QPs recorded: {qps[0]['p'].shape[1]} variables, {qps[0]['G'].shape[1]} inequality rows, "
          f"{qps[0]['A'].shape[1]} equality rows")

    def vine_obj(x):     # internal units: vine velocities, object velocities
        return x[..., :obj_start], x[..., obj_start:obj_start + 3 * sim.num_objs]

    worst = {"vs Clarabel, same QP": 0.0, "vs exact cone (vine)": 0.0, "vs exact cone (box)": 0.0, "culled vs not": 0.0}
    for qp in qps:
        ins = [qp[k].to(args.device) for k in ("M", "p", "G", "h", "A", "b")]
        fric = qp["fric"].to(args.device)
        ours = bq.solve(*ins, fric=fric, obj_start=obj_start).cpu().double()
        full = bq.solve(*ins, fric=fric, obj_start=obj_start, cull=math.inf).cpu().double()
        if True:
            ref = clarabel_linearized(qp, obj_start)
            scale = ref.abs().max().clamp_min(1.0)
            worst["vs Clarabel, same QP"] = max(worst["vs Clarabel, same QP"], float((ours[0] - ref).abs().max() / scale))
            worst["culled vs not"] = max(worst["culled vs not"], float((ours - full).abs().max() / scale))
            v1, o1 = vine_obj(ours[0]); v2, o2 = vine_obj(qp["diffcp"][0].double())
            worst["vs exact cone (vine)"] = max(worst["vs exact cone (vine)"], float((v1 - v2).abs().max() / v2.abs().max().clamp_min(1.0)))
            worst["vs exact cone (box)"] = max(worst["vs exact cone (box)"], float((o1 - o2).abs().max() / v2.abs().max().clamp_min(1.0)))
    for k, v in worst.items():
        print(f"  max |difference| / max |velocity|, {k}: {v:.2e}")

    # whole rollouts
    curvs = [-2.0, 0.0, 1.0, 3.0]
    ends = {}
    for backend, device in (("DIFFCP", "cpu"), ("TORCH_IPM", args.device)):
        cfg.qp_backend, cfg.device = backend, device
        s = Sim(scene, cfg)
        res = s.rollout_batch([s.initial_state()] * len(curvs), curvs, 330)
        ends[backend] = [(viz._tip_mm(st[-1]), box_positions_mm(st[-1])[0], ok, len(st)) for st, ok in res]
    for c, a, b in zip(curvs, ends["DIFFCP"], ends["TORCH_IPM"]):
        print(f"  curvature {c:+.1f}: tip {math.dist(a[0], b[0]):6.1f} mm apart, box {math.dist(a[1], b[1]):6.1f} mm apart "
              f"after {a[3]}/{b[3]} steps (box moved {math.dist(a[1], box_positions_mm(sim.initial_state())[0]):.0f} mm "
              f"with cvxpylayers), ok {a[2]}/{b[2]}")
    # how sensitive is the head-on push by itself? the reference backend, curvature nudged
    cfg.qp_backend, cfg.device = "DIFFCP", "cpu"
    s = Sim(scene, cfg)
    nudged = [0.0, 0.02, -0.02]
    res = s.rollout_batch([s.initial_state()] * len(nudged), nudged, 330)
    boxes = [box_positions_mm(st[-1])[0] for st, _ in res]
    print("  cvxpylayers alone, curvature 0 vs +-0.02 1/m: box ends "
          + ", ".join(f"{math.dist(boxes[0], b):.1f}" for b in boxes[1:]) + " mm apart")
    solver.BACKEND = "DIFFCP"


if __name__ == "__main__":
    main()
