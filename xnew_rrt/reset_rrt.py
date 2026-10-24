'''
RRT with vine resets: several pushes in a row, for long-horizon nonprehensile manipulation.

A vine can only grow, so one vine gives one push. Here the planner may reset the vine: it
retracts to its starting stub and regrows from the base, while the boxes stay where the last
push left them. Retraction itself is not modeled; the reset is instantaneous and the boxes are
taken to be at rest when it happens.

The plan is a sequence of episodes. Each one is an RRT over (vine tip, box positions), the same
search as tip_block_rrt.py, batched like batch_tip_block_rrt.py, from a fresh vine with the boxes
where they are now:

    1. Grow a tree for up to --episode-iters iterations. Stop early at the goal, or once the
       best box-to-goal distance has not improved for --patience iterations.
    2. Choose to reset: commit the branch to the node that brought box 0 closest to its goal.
       An episode that improves it by less than --min-progress mm ends the plan. (Patience only
       counts once the episode has moved box 0 closer.)
    3. Reset: the next episode starts from the initial vine with that node's box poses.

What the next episode's tree starts with (--reuse):
    none        a fresh tree, just the reset state
    untouched   also every node of the previous tree whose vine stayed clear of the boxes, both
                where they were and where they are now. Such a branch never touched a box, and
                vine-box contact is frictionless, so its motion is the same with the boxes moved:
                it is copied with the box poses replaced. Nodes where the vine touched a box are
                not reused, even if the box did not move, since the contact force shaped the vine.

Run from the repo root:
    python xnew_rrt/reset_rrt.py
    python xnew_rrt/reset_rrt.py --reuse none --max-episodes 6 --episode-iters 10 --batch-size 32
    python xnew_rrt/reset_rrt.py --base 0.4 0.43 0 --box-goal 1.05 0.40 0.04     # pushes along x
Outputs under out/reset_rrt/:
    episode_<k>/tree_final.png, tree_stages.png/.gif, tree_log.pkl    each episode's tree
    plan.png             box 0's path over all episodes, with every committed push
    best_path.gif        the whole plan, episode after episode (the vine jumps back at each reset)
    plan.json            per episode: controls of the committed branch, box poses before and after
'''

import argparse, json, math, os, random, time
from dataclasses import asdict

import numpy as np
import torch

from scene import Sim, scene_to_dict  # (puts the repo root on sys.path)
from dvsim import si
import sim_rrt
import tip_block_rrt
import tree_stages
import viz
from rollout import add_sim_args, render_states, sim_from_args
from rrt_util import RRTTree, StateInfo


#----------------------------------------------------------------------- reset and reuse

def reset_state(sim: Sim, state: StateInfo):
    '''The initial vine, with the boxes where `state` has them (at rest): the state after a reset.'''
    s = sim.initial_state()
    s.moveable_obj_pose = state.moveable_obj_pose.clone()
    s.moveable_obj_dstate = torch.zeros_like(state.moveable_obj_dstate)
    return s


def with_boxes(state: StateInfo, obj_pose):
    '''The same vine with the boxes at obj_pose, at rest.'''
    return StateInfo(state.state, state.dstate, state.bodies, obj_pose, torch.zeros_like(state.moveable_obj_dstate),
                     state.p_control, state.l0_control, state.moment_control, state.bend_target, state.base)


def vine_clearance_mm(sim: Sim, state: StateInfo, obj_pose):
    '''Smallest gap [mm] between the vine's bodies and the boxes at obj_pose ((1, n, 3), internal units).'''
    xs, ys = viz.vine_xy_mm(state)
    pts = np.column_stack([xs, ys])
    r = sim.cfg.vine_radius_m * 1000
    gap = math.inf
    for k in range(sim.num_objs):
        cx, cy, th = [float(v) for v in obj_pose[0, k]]
        cx, cy = viz.MM(cx), viz.MM(cy)
        hw, hh = viz.MM(float(sim.params.obj_hw[k])), viz.MM(float(sim.params.obj_hh[k]))
        c, s = math.cos(th), math.sin(th)
        local = (pts - [cx, cy]) @ np.array([[c, -s], [s, c]])     # into the box frame
        outside = np.maximum(np.abs(local) - [hw, hh], 0.0)
        gap = min(gap, float((np.linalg.norm(outside, axis=1) - r).min()))
    return gap


def boxes_moved(a: StateInfo, b: StateInfo, tol_mm=0.5, tol_rad=0.01):
    d = (a.moveable_obj_pose - b.moveable_obj_pose)[0]
    return bool((viz.MM(1.0) * d[:, :2].norm(dim=1) > tol_mm).any() or (d[:, 2].abs() > tol_rad).any())


def reusable_nodes(sim: Sim, tree: RRTTree, new_pose, clearance_mm):
    '''
    Nodes of `tree` (root excluded) whose whole branch kept the vine at least clearance_mm from
    the boxes, both at the tree's own box poses and at new_pose. Returned parents first.
    '''
    old_pose = tree.root.info.moveable_obj_pose
    keep = []
    ok = {tree.root.node_id}
    for node in tree.all_nodes[1:]:            # node ids grow with depth along every branch
        if node.parent.node_id not in ok:
            continue
        clear = all(not boxes_moved(s, tree.root.info)
                    and vine_clearance_mm(sim, s, old_pose) >= clearance_mm
                    and vine_clearance_mm(sim, s, new_pose) >= clearance_mm for s in node.edge)
        if clear:
            ok.add(node.node_id)
            keep.append(node)
    return keep


def seed_tree(sim: Sim, root_state, goal_test, reuse_from=None, clearance_mm=5.0):
    '''A tree from root_state, with the reusable part of reuse_from (another episode's tree) copied in.'''
    tree = RRTTree(root_state, goal_test=goal_test)
    tree.root.expandable = sim.can_grow(root_state)
    if reuse_from is None:
        return tree, 0
    pose = root_state.moveable_obj_pose
    copy_of = {reuse_from.root.node_id: tree.root}
    for node in reusable_nodes(sim, reuse_from, pose, clearance_mm):
        new = tree.add_node(copy_of[node.parent.node_id], with_boxes(node.info, pose), control=node.control,
                            edge=[with_boxes(s, pose) for s in node.edge])
        new.expandable = node.expandable
        new.reused = True
        copy_of[node.node_id] = new
    return tree, len(copy_of) - 1


#----------------------------------------------------------------------- one episode

def run_episode(sim: Sim, tree: RRTTree, iters, steps_per_edge, max_curvature, goal_bias, rng,
                sample_fn, distance_fn, goal_test, goal_distance, batch_size, patience, on_iteration=None,
                straight_prob=0.0):
    '''
    Batched RRT on an existing (possibly seeded) tree. Returns the node that brought box 0 closest
    to its goal (the goal node if one was reached).
    '''
    best_node = min(tree.all_nodes, key=lambda n: goal_distance(n.info))
    best_dist = goal_distance(best_node.info)
    since_improved, num_failed = 0, 0
    MAX_FAILURES_PER_NODE = 3

    for it in range(1, iters + 1):
        nearest_nodes = []
        for _ in range(batch_size):
            node = tree.find_nearest_to(sample_fn(sim, rng, goal_bias), distance_fn)
            if node is not None:
                nearest_nodes.append(node)
        if not nearest_nodes:
            print("\n  every node is fully grown or keeps failing")
            break

        # a push needs the vine to keep growing straight: uniform curvatures almost never do that
        curvatures = [0.0 if rng.random() < straight_prob else rng.uniform(-max_curvature, max_curvature)
                      for _ in nearest_nodes]
        results = sim.rollout_batch([n.info for n in nearest_nodes], curvatures, steps_per_edge, stop_fn=goal_test)

        improved = False
        for node, curvature, (states, ok) in zip(nearest_nodes, curvatures, results):
            if ok and states:
                new = tree.add_node(node, states[-1], control=curvature, edge=states)
                new.expandable = sim.can_grow(new.info)
                d = goal_distance(new.info)
                if d < best_dist - 1e-4:
                    best_dist, best_node, improved = d, new, True
            else:
                num_failed += 1
                node.failures += 1
                if node.failures >= MAX_FAILURES_PER_NODE:
                    node.expandable = False

        # patience only runs once this episode has moved box 0 closer: before that, the vine is
        # still growing out to the boxes
        since_improved = 0 if improved or best_node is tree.root or getattr(best_node, "reused", False) \
            else since_improved + 1
        print(f"\r  iter {it}/{iters}  nodes {len(tree.all_nodes)}  failed {num_failed}  "
              f"box 0 to goal {best_dist * 1000:.0f} mm   ", end="", flush=True)
        if on_iteration is not None:
            on_iteration(it, tree)
        if tree.goal_node is not None:
            best_node = tree.goal_node
            break
        if patience and since_improved >= patience:
            break
    print()
    return best_node


#----------------------------------------------------------------------- the plan

def plan_with_resets(sim: Sim, args, rng, sample_fn, distance_fn, goal_test, goal_distance, out_dir):
    '''Episodes until the goal, --max-episodes, or an episode without progress. Returns the episodes.'''
    episodes = []
    root_state = sim.initial_state()
    previous_tree = None
    for k in range(args.max_episodes):
        reuse = previous_tree if args.reuse == "untouched" else None
        tree, num_reused = seed_tree(sim, root_state, goal_test, reuse, args.clearance)
        start_dist = goal_distance(root_state)
        print(f"episode {k}: box 0 is {start_dist * 1000:.0f} mm from its goal; "
              f"{num_reused} nodes reused from the last tree")

        ep_dir = os.path.join(out_dir, f"episode_{k}")
        os.makedirs(ep_dir, exist_ok=True)
        log = tree_stages.TreeLog(sim, goal_distance)
        log.update(tree, 0)
        t0 = time.time()
        best = run_episode(sim, tree, args.episode_iters, args.steps_per_edge, args.max_curvature,
                           args.goal_bias, rng, sample_fn, distance_fn, goal_test, goal_distance,
                           args.batch_size, args.patience, on_iteration=lambda it, t: log.update(t, it),
                           straight_prob=args.straight_prob)
        log.save(os.path.join(ep_dir, "tree_log.pkl"))
        tree_stages.render_stages(log, ep_dir, args.stages, args.box_spread)

        end_dist = goal_distance(best.info)
        episodes.append({"tree": tree, "best": best, "start_dist_m": start_dist, "end_dist_m": end_dist,
                         "nodes": len(tree.all_nodes), "reused": num_reused, "seconds": time.time() - t0})
        print(f"  -> committed a push to {end_dist * 1000:.0f} mm "
              f"({(start_dist - end_dist) * 1000:+.0f} mm) in {time.time() - t0:.0f} s")

        if goal_test(best.info):
            print("box 0 reached its goal")
            break
        if (start_dist - end_dist) * 1000 < args.min_progress:
            print(f"no push improved it by {args.min_progress} mm: stopping")
            episodes[-1]["no_progress"] = True
            break
        root_state = reset_state(sim, best.info)
        previous_tree = tree
    return episodes


#----------------------------------------------------------------------- outputs

def plan_figure(sim: Sim, episodes, path):
    '''Every committed push: the vine at its end, box 0 before and after, and box 0's path.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = viz.new_figure(sim.scene, max_dim_in=13)
    viz.draw_scene(ax, sim, episodes[0]["tree"].root.info)
    cmap = plt.get_cmap("viridis")
    trail = []
    for k, ep in enumerate(episodes):
        color = cmap(k / max(len(episodes) - 1, 1))
        branch = ep["best"].path_from_root()
        for node in branch[1:]:
            trail += [box_positions(s)[0] for s in node.edge]
        viz.draw_vine(ax, sim, ep["best"].info, color=color, alpha=0.25)
        viz.draw_objects(ax, sim, ep["best"].info, alpha=0.6, outline_only=True)
        x, y = box_positions(ep["best"].info)[0]
        ax.annotate(str(k + 1), (x, y), color=color, fontsize=12, weight="bold", ha="center", va="center")
    if trail:
        t = np.array(trail)
        ax.plot(t[:, 0], t[:, 1], "-", color="tab:orange", lw=2, zorder=8)
    ax.set_title(f"{len(episodes)} pushes; box 0 path in orange, the vine at the end of each push "
                 f"(numbers: box after push k)")
    viz.save_figure(fig, path)


def box_positions(state):
    return tip_block_rrt.box_positions_mm(state)


def write_outputs(sim: Sim, episodes, out_dir, goal_distance):
    plan_figure(sim, episodes, os.path.join(out_dir, "plan.png"))
    states, labels, plan = [], [], []
    for k, ep in enumerate(episodes):
        branch = ep["best"].path_from_root()
        states.append(branch[0].info)
        labels.append(f"push {k + 1}: reset" if k else "push 1: start")
        for j, node in enumerate(branch[1:]):
            states += node.edge
            labels += [f"push {k + 1}/{len(episodes)}, edge {j + 1}/{len(branch) - 1}: "
                       f"curvature {node.control:+.2f} 1/m"] * len(node.edge)
        plan.append({"controls": sim_rrt.branch_controls(ep["best"]),
                     "boxes_before_mm": box_positions(branch[0].info),
                     "boxes_after_mm": box_positions(ep["best"].info),
                     "goal_distance_m": [ep["start_dist_m"], ep["end_dist_m"]],
                     "tree_nodes": ep["nodes"], "reused_nodes": ep["reused"], "seconds": ep["seconds"]})
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump({"scene": scene_to_dict(sim.scene), "sim": asdict(sim.cfg),
                   "reached_goal": bool(episodes[-1]["end_dist_m"] < sim.scene.box_goal_m[2]),
                   "episodes": plan}, f, indent=1)
    render_states(sim, states, labels, out_dir, name="best_path", every=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sim_args(parser)
    tip_block_rrt.add_args(parser)
    parser.add_argument("--reuse", choices=("none", "untouched"), default="none",
                        help="what the next episode's tree starts with (see above)")
    parser.add_argument("--clearance", type=float, default=5.0,
                        help="mm between the vine and every box for a node to count as untouched")
    parser.add_argument("--max-episodes", type=int, default=5, help="pushes (vine resets + 1) at most")
    parser.add_argument("--episode-iters", type=int, default=12, help="batched iterations per episode at most")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4,
                        help="end an episode after this many iterations without getting box 0 any closer, "
                             "counted once it has (0 = never)")
    parser.add_argument("--min-progress", type=float, default=5.0,
                        help="mm an episode has to bring box 0 closer to its goal, else the plan stops")
    parser.add_argument("--steps-per-edge", type=int, default=25)
    parser.add_argument("--max-curvature", type=float, default=3.33)
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--straight-prob", type=float, default=0.3,
                        help="probability that an edge grows straight (curvature 0) instead of a uniform curvature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--box-spread", type=float, default=tree_stages.BOX_SPREAD_MM)
    parser.add_argument("--base", type=float, nargs=3, default=None, metavar=("X_M", "Y_M", "HEADING_RAD"),
                        help="vine base instead of the scene's start")
    parser.add_argument("--out", default="out/reset_rrt")
    args = parser.parse_args()
    if args.tip_goal:
        raise SystemExit("reset_rrt plans for a box goal")

    sim = sim_from_args(args)
    if args.base is not None:
        sim.scene.start = tuple(args.base)
        sim.base0 = torch.tensor([si.len_to_nd(args.base[0]), si.len_to_nd(args.base[1]), float(args.base[2])])
    kw = tip_block_rrt.configure(sim, args)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"Scene '{sim.scene.name}', box goal {sim.scene.box_goal_m}, reuse '{args.reuse}'")

    t0 = time.time()
    episodes = plan_with_resets(sim, args, rng, kw["sample_fn"], kw["distance_fn"], kw["goal_test"],
                                kw["goal_distance"], args.out)
    print(f"{len(episodes)} episodes in {time.time() - t0:.0f} s; box 0 ends "
          f"{episodes[-1]['end_dist_m'] * 1000:.0f} mm from its goal "
          f"(started {episodes[0]['start_dist_m'] * 1000:.0f} mm)")
    write_outputs(sim, episodes, args.out, kw["goal_distance"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
