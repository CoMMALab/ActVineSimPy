'''
Mixed precision in the QP's interior point method (dvsim/batched_qp.FACTOR_DTYPE = float32: the
Newton matrix built and factored in float32, residuals, iterates and refinement in float64) against
the all-float64 solve.

Part 1, one step: along float64 rollouts every QP is also solved mixed, from the same inputs; reports
the relative difference of the solutions (max |x_mixed - x_64| / max |x_64| per element), elements
that fail in one and not the other, and the interior point iterations each needed.
Part 2, rollouts: time per step and how far whole rollouts drift apart (chaotic: one different
contact choice and two rollouts part ways, so read the median, not the max).

    xnew_rrt/py.sh xnew_rrt/check_mixed_qp.py envs/1b.json envs/3a.json --checkpoint checkpoints/wrinkle_mu03.json
'''

import argparse, random, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
import run_envs, checkpoint
import dvsim.batched_qp as bq
from check_graph_rollout import final_points


def setup(path, ck):
    e, scene, cfg, _ = run_envs.load_env(path)
    if ck:
        checkpoint.apply(cfg, ck)
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    cfg.cuda_graphs = False            # eager: the interior point stops when converged, as at the planner's batches
    return e, Sim(scene, cfg)


def one_step(sim, B, steps, refine):
    '''Solve every QP of a float64 rollout both ways.'''
    orig = bq.solve
    err, fail64, failmix, it64, itmix = [], 0, 0, [], []

    def both(*a, **k):
        nonlocal fail64, failmix
        bq.FACTOR_DTYPE, bq.STATS = None, []
        x = orig(*a, **k)
        it64.append(torch.cat(bq.STATS))
        bq.FACTOR_DTYPE, bq.REFINE, bq.STATS = torch.float32, refine, []
        y = orig(*a, **k)
        itmix.append(torch.cat(bq.STATS))
        bq.FACTOR_DTYPE, bq.STATS = None, None
        okx, oky = torch.isfinite(x).all(1), torch.isfinite(y).all(1)
        fail64 += int((~okx & oky).sum())
        failmix += int((okx & ~oky).sum())
        both_ok = okx & oky
        if both_ok.any():
            d = (x - y)[both_ok].abs().amax(1) / x[both_ok].abs().amax(1).clamp_min(1e-12)
            err.append(d.cpu())
        return x

    bq.solve = both
    try:
        rng = random.Random(B)
        curv = [rng.uniform(-3.33, 3.33) for _ in range(B)]
        sim.rollout_batch([sim.initial_state()] * B, curv, steps)
    finally:
        bq.solve = orig
    err = torch.cat(err).numpy()
    i64, imx = torch.cat(it64).numpy(), torch.cat(itmix).numpy()
    q = lambda a: "  ".join(f"p{p} {np.percentile(a, p):.2g}" for p in (50, 99, 99.9, 100))
    print(f"  one step (refine {refine}): {len(err)} element-solves; relative difference {q(err)}")
    print(f"    fails only in float64 {fail64}, only mixed {failmix}; iterations float64 "
          f"{q(i64[i64 >= 0])} (never {int((i64 < 0).sum())}), mixed {q(imx[imx >= 0])} (never {int((imx < 0).sum())})")


def rollouts(sim, B, steps, refine):
    rng = random.Random(B)
    curv = [rng.uniform(-3.3, 3.3) for _ in range(B)]
    starts = [sim.initial_state()] * B
    out = {}
    for name, fd in (("float64", None), ("mixed", torch.float32)):
        bq.FACTOR_DTYPE, bq.REFINE = fd, refine
        sim.rollout_batch(starts, curv, 3)
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = sim.rollout_batch(starts, curv, steps)
        torch.cuda.synchronize()
        out[name] = (res, time.perf_counter() - t)
    bq.FACTOR_DTYPE = None
    (ra, ta), (rb, tb) = out["float64"], out["mixed"]
    ta_, ba_ = final_points(ra)
    tb_, bb_ = final_points(rb)
    d_tip = (ta_ - tb_).norm(dim=1)
    d_box = (ba_ - bb_).reshape(B, -1, 2).norm(dim=2).amax(1) if ba_.numel() else torch.zeros(B)
    print(f"  rollouts B={B}: float64 {1000 * ta / steps:.1f} ms/step, mixed {1000 * tb / steps:.1f} ms/step "
          f"({ta / tb:.2f}x); final tip diff median {float(d_tip.median()):.2f} mm, 90% {float(d_tip.quantile(0.9)):.1f}; "
          f"box diff median {float(d_box.median()):.2f} mm, 90% {float(d_box.quantile(0.9)):.1f}; "
          f"ok {sum(o for _, o in ra)}/{sum(o for _, o in rb)} of {B}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--refine", type=int, nargs="+", default=[2])
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.7)
    for path in args.envs:
        e, sim = setup(path, args.checkpoint)
        print(f"== {e['name']}", flush=True)
        for r in args.refine:
            one_step(sim, 128, args.steps, r)
        for B in args.batch:
            rollouts(sim, B, args.steps, args.refine[-1])


if __name__ == "__main__":
    main()
