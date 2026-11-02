'''
Where a planner run's time goes, on the real code path (run_envs settings, GPU, batch 256).

Part 1 (cProfile over kino.plan for --edges edges): wall time split into the sim stepping
(BatchRollout.step, which waits on the GPU at its syncs), bringing the history back and building
StateInfos (finish/states), the edge checks and the tree bookkeeping.
Part 2 (--kernels): torch.profiler over a few planner iterations: GPU kernel time vs wall time, i.e.
how busy the GPU is (busy ~100%: compute bound; much less: launch or CPU bound).
Part 3: the interior point iteration at which each solve converged (batched_qp.STATS; eager steps
only, so with --no-graphs every step is counted), and with --iters the run at a lower cap.

    python xnew_rrt/profile_planner.py envs/1b.json --checkpoint checkpoints/wrinkle_mu03.json --edges 5120
    python xnew_rrt/profile_planner.py envs/1b.json --checkpoint checkpoints/wrinkle_mu03.json --iters 15 --no-graphs
'''

import argparse, cProfile, pstats, random, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
import run_envs, checkpoint, kino
import dvsim.batched_qp as bq

PARTS = [  # (label, function name, file suffix)
    ("rollout total", "_rollout_batch_device", "scene.py"),
    ("  sim steps", "step", "batched_rollout.py"),
    ("    graph replay", "__call__", "batched_rollout.py"),
    ("  stop test", "test", "scene.py"),
    ("  history to CPU", "finish", "batched_rollout.py"),
    ("  StateInfos", "states", "batched_rollout.py"),
    ("edge_valid (border)", "valid", "run_envs.py"),
    ("choose (nearest)", "choose", "kino.py"),
    ("register", "register", "kino.py"),
    ("tree.add_node", "add_node", "rrt_util.py"),
    ("goal_distance", "box_goal_distance_m", "scene.py"),
]


def setup(args):
    e, scene, cfg, planner = run_envs.load_env(args.env)
    if args.checkpoint:
        checkpoint.apply(cfg, args.checkpoint)
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    cfg.cuda_graphs = not args.no_graphs
    if scene.box_goal_m is not None:          # a goal of radius 0, never reached: every run simulates exactly its edges
        scene.box_goal_m = (*scene.box_goal_m[:2], 0.0)
    else:
        scene.goal_radius_m = -1.0
    return e, Sim(scene, cfg), planner


def run(e, sim, planner, args, edges):
    goal_test, goal_distance = run_envs.goal_fns(sim, e)
    return kino.plan(sim, planner["planner"], goal_test, goal_distance, random.Random(args.seed),
                     batch_size=args.batch, steps_per_edge=planner["steps_per_edge"],
                     max_curvature=planner["max_curvature"], goal_bias=planner["goal_bias"], max_edges=edges,
                     cell_mm=planner["cell_mm"], heading_deg=planner["heading_deg"], coarse=planner["coarse"],
                     box_weight=planner["box_weight"], node_every=args.node_every,
                     edge_valid=None if e.get("border") == "wall" else run_envs.off_border(sim, e))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--edges", type=int, default=5120)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--iters", type=int, default=None, help="interior point iteration cap (default batched_qp.ITERS)")
    parser.add_argument("--graph-max", type=int, default=None, help="CUDA graph steps up to this batch (default batched_rollout.GRAPH_MAX_BATCH)")
    parser.add_argument("--no-graphs", action="store_true", help="every step eager (all of them in the iteration stats)")
    parser.add_argument("--kernels", type=int, default=0, help="also profile GPU kernels over this many iterations (at most 4)")
    parser.add_argument("--node-every", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the part of a step before the QP (compiled_forward.py)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.7)
    if args.iters:
        bq.ITERS = args.iters
    if args.graph_max:
        import batched_rollout
        batched_rollout.GRAPH_MAX_BATCH = args.graph_max
    if args.compile:
        import compiled_forward
        compiled_forward.enable()
    e, sim, planner = setup(args)
    run(e, sim, planner, args, 2 * args.batch)                    # warm up: graph capture, allocator
    torch.cuda.synchronize()

    bq.STATS = []
    prof = cProfile.Profile()
    t = time.perf_counter()
    prof.enable()
    tree, best, stats = run(e, sim, planner, args, args.edges)
    prof.disable()
    wall = time.perf_counter() - t
    print(f"\n{e['name']}  batch {args.batch}  iters cap {bq.ITERS}  graphs {'off' if args.no_graphs else 'up to batch %d' % __import__('batched_rollout').GRAPH_MAX_BATCH}: "
          f"{stats['edges']} edges, {stats['nodes']} nodes, {stats['failed_rollouts']} failed rollouts, "
          f"{stats['invalid']} border-invalid in {wall:.1f} s ({stats['edges'] / wall:.0f} edges/s)")
    ps = pstats.Stats(prof)
    for label, name, file in PARTS:
        rows = [(v[3], v[1]) for k, v in ps.stats.items() if k[2] == name and k[0].endswith(file)]
        if rows:
            ct, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
            print(f"  {label:22s} {ct:7.1f} s  {100 * ct / wall:5.1f}%   ({n} calls)")
    print("  top functions by own time:")
    for k, v in sorted(ps.stats.items(), key=lambda kv: -kv[1][2])[:12]:
        print(f"    {v[2]:6.1f} s  {v[1]:8d}  {k[2]}  ({k[0].split('/')[-1]}:{k[1]})")

    if bq.STATS:
        at = torch.cat(bq.STATS).numpy()
        conv = at[at >= 0]
        print(f"  interior point iterations to converge ({len(at)} element-solves in eager steps, "
              f"{100 * (at < 0).mean():.2f}% did not converge in {bq.ITERS}):")
        print("    " + "  ".join(f"p{q} {np.percentile(conv, q):.0f}" for q in (50, 90, 99, 99.9, 100)))
        h = np.bincount(conv, minlength=bq.ITERS)
        print("    count by iteration: " + " ".join(f"{i}:{c}" for i, c in enumerate(h) if c))
    bq.STATS = None

    if args.kernels:
        # GPU activity only, and few iterations: the profiler keeps every event in RAM (CPU-side
        # tracing of a 16-iteration batch-256 run exhausted the machine's 43 GB)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            torch.cuda.synchronize()
            t = time.perf_counter()
            run(e, sim, planner, args, min(args.kernels, 4) * args.batch)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        avg = p.key_averages()
        busy = sum(a.self_device_time_total for a in avg) / 1e6
        print(f"  GPU kernels over {min(args.kernels, 4)} iterations: {busy:.1f} s of kernel time "
              f"in {wall:.1f} s wall ({100 * busy / wall:.0f}% busy, profiler overhead included)")
        print(avg.table(sort_by="self_cuda_time_total", row_limit=15))

if __name__ == "__main__":
    main()
