'''
Render the solution video of a run from its result.json alone (runs made with --no-render keep no video).

The controls in result.json are replayed in one sim: each control grows for steps_per_edge steps (the
planner's edges; a vine that is fully grown stops early, as in the planner) and the last one stops at the
goal; a "reset" puts the initial vine back with the boxes where they are. The replay is a single
(unbatched) sim, so it can differ slightly from the planner's batched one: the title says if the replay
misses the goal. Runs on the CPU by default, to leave the GPU to the benchmark.

Run from the repo root:
    python xnew_rrt/render_result.py envs/3a.json out/bench_rrt_mu03/runs/3a/s000/results/3a.result.json \
        --checkpoint checkpoints/wrinkle_mu03.json --out out/videos_mu03/3a_s000
    -> <out>/solution.gif, solution.mp4 (+ .webm), solution_final.png
'''

import argparse, json, os

import torch

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import run_envs
from reset_action_rrt import RESET, RESET_FRAMES
from reset_rrt import reset_state
from rollout import render_states


def replay(sim, controls, steps_per_edge, goal_test):
    '''States and labels along the controls, as reset_action_rrt.branch_states draws a branch.'''
    state = sim.initial_state()
    states, labels, push = [state], ["initial state"], 1
    for k, (c, _) in enumerate(controls):
        if c == RESET:
            push += 1
            state = reset_state(sim, state)
            states += [state] * RESET_FRAMES
            labels += [f"reset: the vine is retracted (not modeled), push {push} starts"] * RESET_FRAMES
            continue
        new, ok = sim.rollout(state, float(c), steps_per_edge, stop_fn=goal_test if k == len(controls) - 1 else None)
        states += new
        labels += [f"push {push}: curvature {float(c):+.2f} 1/m"] * len(new)
        if not ok:
            print(f"  replay failed at control {k}")
            break
        state = new[-1] if new else state
    return states, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("result", help="result.json written by run_envs.py")
    parser.add_argument("--checkpoint", required=True, help="the checkpoint the run used")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.chdir(run_envs.REPO)

    e, scene, cfg, planner = run_envs.load_env(args.env)
    with open(args.result) as f:
        result = json.load(f)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.max_bodies = result["sim"]["max_bodies"] if "sim" in result else cfg.max_bodies
    cfg.device = args.device
    cfg.qp_backend = "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.7)
    sim = Sim(scene, cfg)
    goal_test, goal_distance = run_envs.goal_fns(sim, e)

    states, labels = replay(sim, result["controls"], result["planner"]["steps_per_edge"], goal_test)
    reached = goal_test(states[-1])
    dist = goal_distance(states[-1]) * 1000
    tag = "SOLVED" if reached else f"replay misses the goal by {dist:.0f} mm"
    if not result["reached"]:
        tag = f"FAILED run, closest {result['goal_distance_mm']:.0f} mm"
    print(f"{e['name']}: {len(states) - 1} steps, {result['pushes']} pushes, replay {'reaches' if reached else 'misses'} "
          f"the goal ({dist:.0f} mm)")
    os.makedirs(args.out, exist_ok=True)
    render_states(sim, states, [f"{e['name']} {result['planner'].get('planner', 'rrt')} {tag} | {l}" for l in labels],
                  args.out, name="solution", every=1, hold_last_ms=int(1000 * run_envs.HOLD_LAST_S))
    run_envs.gif_to_mp4(os.path.join(args.out, "solution.gif"), os.path.join(args.out, "solution.mp4"))
    print("wrote", os.path.join(args.out, "solution.mp4"))


if __name__ == "__main__":
    main()
