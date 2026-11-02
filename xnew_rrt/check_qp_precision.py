'''
float64 vs float32 QP (dvsim/batched_qp.DTYPE): time per step and how far the rollouts drift apart.
Same starts and curvatures; reports the final tip and box differences and failed (not ok) rollouts.
    python xnew_rrt/check_qp_precision.py envs/3a.json envs/5c.json --batch 32 128 512 --steps 300
'''

import argparse, random, time

import torch

from scene import Sim  # (puts the repo root on sys.path)
import run_envs
import dvsim.batched_qp as bq
from check_graph_rollout import final_points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--batch", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.7)
    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        sim = Sim(scene, cfg)
        for B in args.batch:
            rng = random.Random(B)
            curv = [rng.uniform(-3.3, 3.3) for _ in range(B)]
            starts = [sim.initial_state()] * B
            out = {}
            for dtype in (torch.float64, torch.float32):
                bq.DTYPE = dtype
                sim.__dict__.pop("_graph_steps", None)                   # recapture for this dtype
                sim.rollout_batch(starts, curv, 3)
                torch.cuda.synchronize()
                t = time.perf_counter()
                res = sim.rollout_batch(starts, curv, args.steps)
                torch.cuda.synchronize()
                out[dtype] = (res, time.perf_counter() - t)
            bq.DTYPE = torch.float64
            (ra, ta), (rb, tb) = out[torch.float64], out[torch.float32]
            ta_, ba_ = final_points(ra)
            tb_, bb_ = final_points(rb)
            d_tip = (ta_ - tb_).norm(dim=1)
            print(f"{e['name']} B={B}: float64 {1000 * ta / args.steps:.1f} ms/step, float32 "
                  f"{1000 * tb / args.steps:.1f} ms/step ({ta / tb:.1f}x); final tip diff median "
                  f"{float(d_tip.median()):.2f} mm, 90% {float(d_tip.quantile(0.9)):.2f} mm, max "
                  f"{float(d_tip.max()):.1f} mm; box diff max {float((ba_ - bb_).abs().max()):.2f} mm; "
                  f"ok {sum(o for _, o in ra)}/{sum(o for _, o in rb)} of {B}", flush=True)


if __name__ == "__main__":
    main()
