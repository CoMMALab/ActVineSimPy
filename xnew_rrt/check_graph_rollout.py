'''
Checks and times the GPU-resident, CUDA-graph rollout (batched_rollout.py) against the older loop
(states on the CPU between eager batched steps). Same starts and curvatures; reports the largest
difference in final tip and box positions, and the time per step.
    python xnew_rrt/check_graph_rollout.py envs/1a.json envs/4c.json envs/5c.json --batch 64 256 --steps 200
'''

import argparse, os, random, time

import torch

from scene import Sim  # (puts the repo root on sys.path)
import run_envs


def final_points(results):
    tips, boxes = [], []
    for states, ok in results:
        s = states[-1]
        tips.append(torch.tensor(s.tip()[:2].tolist()))
        boxes.append(s.moveable_obj_pose[0, :, :2].reshape(-1))
    return torch.stack(tips), torch.stack(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--batch", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        sim = Sim(scene, cfg)
        for B in args.batch:
            rng = random.Random(B)
            curv = [rng.uniform(-3.3, 3.3) for _ in range(B)]
            starts = [sim.initial_state()] * B
            out = {}
            for mode in ("loop", "graph"):
                sim.cfg.cuda_graphs = mode == "graph"
                sim.rollout_batch(starts, curv, 3)                       # warm up / capture
                torch.cuda.synchronize()
                t = time.perf_counter()
                res = sim.rollout_batch(starts, curv, args.steps)
                torch.cuda.synchronize()
                out[mode] = (res, time.perf_counter() - t)
            (ra, ta), (rb, tb) = out["loop"], out["graph"]
            ta_, ba_ = final_points(ra)
            tb_, bb_ = final_points(rb)
            lens = [(len(a[0]), len(b[0])) for a, b in zip(ra, rb)]
            print(f"{e['name']} B={B}: loop {1000 * ta / args.steps:.0f} ms/step, graph {1000 * tb / args.steps:.0f} ms/step "
                  f"({ta / tb:.1f}x); final tip diff max {float((ta_ - tb_).norm(dim=1).max()):.2f} mm, "
                  f"median {float((ta_ - tb_).norm(dim=1).median()):.3f} mm; box diff max "
                  f"{float((ba_ - bb_).abs().max()):.2f} mm; same lengths {all(a == b for a, b in lens)}; "
                  f"ok {sum(o for _, o in ra)}/{sum(o for _, o in rb)}", flush=True)


if __name__ == "__main__":
    main()
