'''
Batched RRT over (vine tip, movable box positions).

tip_block_rrt.py with the expansion done in parallel: every iteration draws --batch-size
samples, finds the nearest node for each of them, picks a random curvature for each, and
simulates all those edges together. Each sim step is one batched dvsim step, i.e. the constraint
Jacobians are evaluated for the whole batch at once and the QPs are solved side by side.

It is the naive fine-grained version: the samples of one iteration do not see each other's new
nodes (so the same node can be expanded several times in one iteration, with different
curvatures), and nearest-neighbour search is still a Python loop over the tree.

Run from the repo root:
    python xnew_rrt/batch_tip_block_rrt.py
    python xnew_rrt/batch_tip_block_rrt.py --batch-size 64 --iters 20 --box-goal 0.85 0.60 0.06
--iters counts batched iterations, so the tree gets up to iters * batch_size nodes.
Outputs are the same as sim_rrt.py, under out/batch_tip_block_rrt/.
'''

import random

from scene import Sim  # (puts the repo root on sys.path)
import sim_rrt
import tip_block_rrt
from rrt_util import RRTTree


def run_rrt_batched(sim: Sim, iters, steps_per_edge, max_curvature, goal_bias, rng: random.Random,
                    on_iteration=None, sample_fn=sim_rrt.sample_tip_position,
                    distance_fn=sim_rrt.tip_distance, goal_test=None, goal_distance=None,
                    batch_size=32):
    '''Same contract as sim_rrt.run_rrt, expanding batch_size nodes per iteration.'''
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

        # Sample, and pick what to expand, for the whole batch:
        samples, nearest_nodes = [], []
        for _ in range(batch_size):
            sample_mm = sample_fn(sim, rng, goal_bias)
            nearest_node = tree.find_nearest_to(sample_mm, distance_fn)
            if nearest_node is not None:
                samples.append(sample_mm)
                nearest_nodes.append(nearest_node)

        if not nearest_nodes:
            print("\nEvery node is fully grown or keeps failing: nothing left to expand.")
            break

        curvatures = [rng.uniform(-max_curvature, max_curvature) for _ in nearest_nodes]

        # Propagate all of them together:
        results = sim.rollout_batch([node.info for node in nearest_nodes], curvatures,
                                    steps_per_edge, stop_fn=goal_test)

        new_node = None
        for nearest_node, curvature, (states, ok) in zip(nearest_nodes, curvatures, results):
            if ok and states:
                new_node = tree.add_node(nearest_node, states[-1], control=curvature, edge=states)
                new_node.expandable = sim.can_grow(new_node.info)

                dist = goal_distance(new_node.info)
                if dist < best_dist:
                    best_dist, best_node = dist, new_node
            else:
                num_failed += 1
                nearest_node.failures += 1
                if nearest_node.failures >= MAX_FAILURES_PER_NODE:
                    nearest_node.expandable = False

        print(f"\r iter {it}/{iters}  nodes {len(tree.all_nodes)}  failed rollouts {num_failed}  "
              f"closest to goal {best_dist * 1000:.0f} mm   ", end="", flush=True)

        if on_iteration is not None:
            on_iteration(it, tree, best_node, samples[-1], new_node)

        if tree.goal_node is not None:
            best_node = tree.goal_node
            break

    print()
    return tree, best_node


def add_args(parser):
    tip_block_rrt.add_args(parser)
    parser.add_argument("--batch-size", type=int, default=32, help="edges simulated together per iteration")
    parser.set_defaults(iters=20, viz_every=1)


def configure(sim: Sim, args):
    kwargs = tip_block_rrt.configure(sim, args)
    kwargs["batch_size"] = args.batch_size
    return kwargs


if __name__ == "__main__":
    sim_rrt.main(description=__doc__, default_out="out/batch_tip_block_rrt", add_args=add_args,
                 configure=configure, run=run_rrt_batched)
