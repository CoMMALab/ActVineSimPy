'''
Planner ablation: every env x planner (kino.PLANNERS) x seed, with the same edge budget, on the
batched GPU sim. One json line per run in <out>/results.jsonl; tree.gif, solution.mp4 and
result.json per run in <out>/<planner>/<env>/seed<k>/.
Run from the repo root:
    python xnew_rrt/ablation.py --max-edges 16384 --seeds 0 --mem-fraction 0.25        # all envs, all planners
    python xnew_rrt/ablation.py envs/1a.json envs/4b.json --planners rrt kinopax --seeds 0 1   # envs first
    python xnew_rrt/ablation.py --summarize                                            # table of results.jsonl
Several of these can run side by side (split the envs); each appends to results.jsonl and skips
runs already in it. --mem-fraction caps each process's share of the GPU's memory.
'''

import argparse, json, os, random, sys, time
from dataclasses import asdict

import torch

from scene import Sim, scene_to_dict  # (puts the repo root on sys.path)
import kino
import run_envs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def done_runs(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["env"], r["planner"], r["seed"]) for r in map(json.loads, f) if r}


def summarize(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "results.jsonl")) if l.strip()]
    envs = sorted({r["env"] for r in rows})
    planners = [p for p in kino.PLANNERS if any(r["planner"] == p for r in rows)]
    lines = ["| Env | " + " | ".join(planners) + " |", "|---|" + "---|" * len(planners)]
    for e in envs:
        cells = []
        for p in planners:
            rs = [r for r in rows if r["env"] == e and r["planner"] == p]
            if not rs:
                cells.append("")
                continue
            solved = [r for r in rs if r["reached"]]
            if solved:
                edges = sorted(r["edges_to_goal"] for r in solved)
                secs = sorted(r["seconds_to_goal"] for r in solved)
                cells.append(f"{len(solved)}/{len(rs)}: {edges[len(edges) // 2]} edges, {secs[len(secs) // 2] / 60:.1f} min")
            else:
                cells.append(f"0/{len(rs)} ({min(r['goal_distance_mm'] for r in rs):.0f} mm)")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    solved = {p: sum(r["reached"] for r in rows if r["planner"] == p) for p in planners}
    total = {p: sum(r["planner"] == p for r in rows) for p in planners}
    lines.append("| solved | " + " | ".join(f"{solved[p]}/{total[p]}" for p in planners) + " |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="*", help="envs/*.json (default: all)")
    parser.add_argument("--planners", nargs="+", default=list(kino.PLANNERS), choices=kino.PLANNERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-edges", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--time-limit", type=float, default=30.0, help="per run [min]")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-fraction", type=float, default=0.25, help="of the GPU's memory, per process")
    parser.add_argument("--tree-frames", type=int, default=16)
    parser.add_argument("--render-seeds", type=int, nargs="*", default=[0], help="draw tree.gif / solution.mp4 for these")
    parser.add_argument("--out", default="out/ablation")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.out)
        return
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    paths = args.envs or sorted(os.path.join(REPO, "envs", f) for f in os.listdir(os.path.join(REPO, "envs"))
                                if f.endswith(".json"))
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "results.jsonl")
    for path in paths:
        for planner_name in args.planners:
            for seed in args.seeds:
                e, scene, cfg, planner = run_envs.load_env(path)
                if (e["name"], planner_name, seed) in done_runs(log_path):
                    continue
                planner.update(planner=planner_name, max_edges=args.max_edges, batch_size=args.batch_size,
                               time_limit_min=args.time_limit)
                cfg.device, cfg.qp_backend = args.device, "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
                sim = Sim(scene, cfg)
                if args.device.startswith("cuda"):
                    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
                out = os.path.join(args.out, planner_name, e["name"], f"seed{seed}")
                os.makedirs(out, exist_ok=True)
                print(f"=== {e['name']} {planner_name} seed {seed}", flush=True)
                t0 = time.time()
                result = run_envs.run(e, sim, planner, out, seed, args.tree_frames, render=seed in args.render_seeds)
                result.update(env=e["name"], category=e["category"], seed=seed, wall_s=time.time() - t0,
                              planner_cfg=planner, sim=asdict(cfg))
                with open(os.path.join(out, "result.json"), "w") as f:
                    json.dump(dict(result, scene=scene_to_dict(scene)), f, indent=1)
                with open(log_path, "a") as f:
                    f.write(json.dumps({k: v for k, v in result.items() if k != "controls"}) + "\n")
                print(f"{e['name']} {planner_name} seed {seed}: {'SOLVED' if result['reached'] else 'failed'}, "
                      f"{result['goal_distance_mm']:.0f} mm, {result['edges']} edges, {result['nodes']} nodes, "
                      f"{result['pruned']} pruned, {result['seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
