'''
Kinodynamic RRT for a growing vine that can push movable boxes, on top of the dvsim simulator.

    1. Sample a random tip position in the workspace (sometimes the goal)        => x_rand
    2. Find the tree node whose vine tip is nearest to it                        => x_nearest
    3. Pick a random curvature and grow the vine from x_nearest with it for a
       fixed number of sim steps                                                 => x_new
    4. Add x_new to the tree; stop if its tip is inside the goal region
    5. Repeat

A tree node is a full sim state (vine, boxes, and the curvature every existing body was grown with),
so pushing a box is just part of the state that gets propagated.

Headless: while it runs, <out>/tree_live.png is rewritten every --viz-every iterations. Keep it open
in a VS Code tab (it reloads on change) to watch the tree grow. At the end it writes
    <out>/tree_final.png    the whole tree
    <out>/tree_growth.gif   the live snapshots as an animation
    <out>/plan.json         controls of the best branch (replay: python xnew_rrt/rollout.py --plan ...)
    <out>/best_path.gif     the simulation along the best branch

Run from the repo root:
    python xnew_rrt/sim_rrt.py
    python xnew_rrt/sim_rrt.py --env envs/env_dynamics.txt --iters 500 --seed 1
'''

import argparse, json, os, random, shutil, time
from dataclasses import asdict

from scene import Sim, scene_to_dict  # (puts the repo root on sys.path)
import viz
from rollout import add_sim_args, render_states, sim_from_args
from rrt_util import Node, RRTTree, StateInfo


#----------------------------------------------------------------------- Sampling / distance (tip based)

def sample_tip_position(sim: Sim, rng: random.Random, goal_bias):
    '''Random tip position (x, y) in mm inside the scene bounds; the goal center with prob. goal_bias.'''
    if rng.random() < goal_bias:
        return sim.scene.goal_m[0] * 1000, sim.scene.goal_m[1] * 1000

    xmin, xmax, ymin, ymax = [v * 1000 for v in sim.scene.bounds_m]
    return rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)


def tip_distance(state: StateInfo, point_mm):
    '''Distance (mm) between the vine's tip and a point. The boxes are not part of the metric.'''
    x, y = viz._tip_mm(state)
    return ((x - point_mm[0]) ** 2 + (y - point_mm[1]) ** 2) ** 0.5


#----------------------------------------------------------------------- RRT

def run_rrt(sim: Sim, iters, steps_per_edge, max_curvature, goal_bias, rng: random.Random,
            on_iteration=None, sample_fn=sample_tip_position, distance_fn=tip_distance,
            goal_test=None, goal_distance=None):
    '''
    Returns (tree, best_node): best_node is the goal node if one was found, otherwise the node
    that got closest to the goal.
    on_iteration(it, tree, best_node, sample_mm, new_node) is called after every iteration.

    The state space the tree explores is set by sample_fn(sim, rng, goal_bias) -> sample and
    distance_fn(state, sample); the goal by goal_test(state) and goal_distance(state) [m].
    The defaults plan for the tip only (see tip_block_rrt.py for tip + box positions).
    '''
    goal_test = goal_test or sim.reached_goal
    goal_distance = goal_distance or sim.goal_distance_m

    start_state = sim.initial_state()
    tree = RRTTree(start_state, goal_test=goal_test)
    tree.root.expandable = sim.can_grow(start_state)

    best_node = tree.root
    best_dist = goal_distance(start_state)
    num_failed = 0
    MAX_FAILURES_PER_NODE = 3

    for it in range(1, iters + 1):

        sample_mm = sample_fn(sim, rng, goal_bias)
        nearest_node = tree.find_nearest_to(sample_mm, distance_fn)

        if nearest_node is None:
            print("\nEvery node is fully grown or keeps failing: nothing left to expand.")
            break

        # Propagate from the nearest node with a random control:
        curvature = rng.uniform(-max_curvature, max_curvature)
        states, ok = sim.rollout(nearest_node.info, curvature, steps_per_edge, stop_fn=goal_test)

        new_node = None
        if ok and states:
            new_node = tree.add_node(nearest_node, states[-1], control=curvature, edge=states)
            new_node.expandable = sim.can_grow(new_node.info)

            dist = goal_distance(new_node.info)
            if dist < best_dist:
                best_dist, best_node = dist, new_node
        else:
            # the sim could not finish this edge (QP infeasible / non-finite): drop it
            num_failed += 1
            nearest_node.failures += 1
            if nearest_node.failures >= MAX_FAILURES_PER_NODE:
                nearest_node.expandable = False

        print(f"\r iter {it}/{iters}  nodes {len(tree.all_nodes)}  failed rollouts {num_failed}  "
              f"closest to goal {best_dist * 1000:.0f} mm   ", end="", flush=True)

        if on_iteration is not None:
            on_iteration(it, tree, best_node, sample_mm, new_node)

        if tree.goal_node is not None:
            best_node = tree.goal_node
            break

    print()
    return tree, best_node


def branch_controls(node: Node):
    '''[(curvature, num_steps)] along the branch root -> node, for replaying with rollout.py.'''
    return [(n.control, len(n.edge)) for n in node.path_from_root() if n.parent is not None]


def branch_states(node: Node):
    '''Every sim state along the branch root -> node, plus a label per state.'''
    path = node.path_from_root()
    states, labels = [path[0].info], ["initial state"]
    for k, n in enumerate(path[1:]):
        states += n.edge
        labels += [f"edge {k + 1}/{len(path) - 1}: curvature {n.control:+.2f} 1/m"] * len(n.edge)
    return states, labels


#---------------------------------------------------------------------- Main

def main(description=__doc__, default_out="out/rrt", add_args=None, configure=None, run=None):
    '''
    Command line entry point, shared with tip_block_rrt.py and batch_tip_block_rrt.py.
    add_args(parser) adds extra flags; configure(sim, args) -> dict of extra run_rrt() keyword
    arguments (sample_fn, distance_fn, goal_test, goal_distance); run replaces run_rrt itself.
    '''
    run = run or run_rrt

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sim_args(parser)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--steps-per-edge", type=int, default=25,
                        help="sim steps per tree edge (dt = 1/90 s; ~83 mm of growth at 0.3 m/s)")
    parser.add_argument("--max-curvature", type=float, default=3.33,
                        help="controls are sampled from [-max, max] 1/m (3.33 = 0.3 m bend radius)")
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz-every", type=int, default=10, help="redraw tree_live.png every N iterations (0 = never)")
    parser.add_argument("--out", default=default_out)
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()

    sim = sim_from_args(args)
    rng = random.Random(args.seed)
    rrt_kwargs = configure(sim, args) if configure is not None else {}
    goal_distance = rrt_kwargs.get("goal_distance", sim.goal_distance_m)

    os.makedirs(args.out, exist_ok=True)
    live_path = os.path.join(args.out, "tree_live.png")
    snap_dir = os.path.join(args.out, "_tree_frames")
    shutil.rmtree(snap_dir, ignore_errors=True)
    os.makedirs(snap_dir)
    snapshots = []

    def draw_tree_to(path, it, tree, best_node, sample_mm=None, new_node=None, status=""):
        fig, ax = viz.new_figure(sim.scene, max_dim_in=13)
        title = (f"{sim.scene.name}  |  RRT iter {it}  |  {len(tree.all_nodes)} nodes  |  "
                 f"closest to goal: {goal_distance(best_node.info) * 1000:.0f} mm{status}")
        viz.draw_tree(ax, sim, tree, highlight=best_node, sample_mm=sample_mm, new_node=new_node, title=title)
        viz.save_figure(fig, path)

    def on_iteration(it, tree, best_node, sample_mm, new_node):
        if args.viz_every > 0 and (it % args.viz_every == 0 or tree.goal_node is not None):
            draw_tree_to(live_path, it, tree, best_node, sample_mm, new_node)
            snap = os.path.join(snap_dir, f"t{it:06d}.png")
            shutil.copyfile(live_path, snap)
            snapshots.append(snap)

    print(f"Scene '{sim.scene.name}', sim config {asdict(sim.cfg)}")
    print(f"Starting RRT ({args.iters} iterations). Watch {live_path}")

    t0 = time.time()
    tree, best_node = run(sim, args.iters, args.steps_per_edge, args.max_curvature,
                          args.goal_bias, rng, on_iteration, **rrt_kwargs)
    elapsed = time.time() - t0

    found = tree.goal_node is not None
    print(f"RRT finished in {elapsed:.1f} s: " +
          ("found a path to the goal!" if found else
           f"did NOT reach the goal; closest is {goal_distance(best_node.info) * 1000:.0f} mm "
           f"from the goal center"))
    tree.print_diagnostics()

    # Outputs:

    final_path = os.path.join(args.out, "tree_final.png")
    draw_tree_to(final_path, len(tree.all_nodes) - 1, tree, best_node,
                 status="  |  GOAL REACHED" if found else "  |  goal not reached")
    shutil.copyfile(final_path, live_path)
    print("wrote", final_path)

    if snapshots:
        growth_path = os.path.join(args.out, "tree_growth.gif")
        viz.write_gif(snapshots + [final_path], growth_path, frame_ms=250, hold_last_ms=3000)
        print("wrote", growth_path)
    shutil.rmtree(snap_dir)

    plan_path = os.path.join(args.out, "plan.json")
    with open(plan_path, "w") as f:
        json.dump({"scene": scene_to_dict(sim.scene), "sim": asdict(sim.cfg),
                   "controls": branch_controls(best_node),
                   "reached_goal": found,
                   "goal_distance_m": goal_distance(best_node.info)}, f, indent=1)
    print("wrote", plan_path)

    states, labels = branch_states(best_node)
    outputs = render_states(sim, states, labels, args.out, name="best_path")
    print("wrote", outputs["gif"])


if __name__ == "__main__":
    main()
