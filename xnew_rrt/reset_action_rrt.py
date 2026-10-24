'''
RRT for the retractable vine in which resetting the vine is an action inside ONE tree.

reset_rrt.py grows a fresh tree for every push and commits greedily to one push at a time, so a
set-up push that first moves the box away from the goal is never kept. Here there is a single tree
over (vine, box poses) with two kinds of edge:
    grow    the batched RRT's edge: from the node nearest a sample, grow --steps-per-edge sim steps
            at a random curvature (straight with probability straight_prob)
    reset   the vine goes back to its stub at the base (retraction itself is not modeled); the
            boxes stay where the node left them, at rest
A node unlocks its reset when its boxes are at least unlock_mm from the boxes of every reset taken
so far (the root counts as one), and the reset is added right away. So every box configuration,
at unlock_mm resolution, gets one fresh vine. Resets cannot multiply unless the boxes really move,
nothing is thrown away, and any later push can start from any box configuration reached before.

Distance between two box configurations: the largest over the boxes of
    center shift + half diagonal * turn          (turn modulo the box's symmetry: 90 deg if square)

With unlock_mm=None (no resets) this is the batched RRT of batch_tip_block_rrt.py, with the
nearest-neighbor search vectorized and an optional time limit.
Run it through the environment runner (from the repo root):
    python xnew_rrt/run_envs.py envs/3a.json
'''

import math, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
from rrt_util import RRTTree
from reset_rrt import reset_state
import tip_block_rrt
import tree_stages
import viz

RESET = "reset"
MAX_FAILURES_PER_NODE = 3


class NodeIndex:
    '''(tip, box centers) of every node in mm, for a vectorized nearest-node search.'''

    def __init__(self, num_objs, box_weight):
        self.dim = 2 + 2 * num_objs
        self.weights = np.array([1.0, 1.0] + [box_weight] * (2 * num_objs))
        self.features = np.zeros((1024, self.dim))
        self.expandable = np.zeros(1024, dtype=bool)
        self.nodes = []

    def add(self, node):
        i = len(self.nodes)
        assert node.node_id == i, "every tree node must be added, in order"
        if i == len(self.features):
            self.features = np.concatenate([self.features, np.zeros_like(self.features)])
            self.expandable = np.concatenate([self.expandable, np.zeros_like(self.expandable)])
        self.features[i] = list(viz._tip_mm(node.info)) + [v for xy in tip_block_rrt.box_positions_mm(node.info) for v in xy]
        self.expandable[i] = node.expandable
        self.nodes.append(node)

    def set_expandable(self, node, value):
        node.expandable = value
        self.expandable[node.node_id] = value

    def nearest(self, sample):
        n = len(self.nodes)
        d = (((self.features[:n] - np.asarray(sample)) * self.weights) ** 2).sum(axis=1)
        d[~self.expandable[:n]] = np.inf
        i = int(np.argmin(d))
        return None if not np.isfinite(d[i]) else self.nodes[i]


class ResetIndex:
    '''The box configurations at which the vine has been reset.'''

    def __init__(self, sim: Sim):
        hw = np.array([viz.MM(float(v)) for v in sim.params.obj_hw])
        hh = np.array([viz.MM(float(v)) for v in sim.params.obj_hh])
        self.half_diag = np.hypot(hw, hh)
        self.symmetry = np.where(np.abs(hw - hh) < 1e-6, math.pi / 2, math.pi)
        self.configs = np.zeros((0, len(hw), 3))

    @staticmethod
    def config(state):
        '''(n_obj, 3) box poses: x, y in mm, theta in rad.'''
        pose = state.moveable_obj_pose[0].detach().numpy().astype(float).copy()
        pose[:, :2] = np.vectorize(viz.MM)(pose[:, :2]) if len(pose) else pose[:, :2]
        return pose

    def add(self, state):
        self.configs = np.concatenate([self.configs, self.config(state)[None]])

    def distance(self, state):
        '''Distance (mm) from this state's boxes to the nearest configuration already reset at.'''
        c = self.config(state)
        shift = np.linalg.norm(self.configs[:, :, :2] - c[None, :, :2], axis=2)
        turn = np.mod(self.configs[:, :, 2] - c[None, :, 2], self.symmetry)
        turn = np.minimum(turn, self.symmetry - turn)
        return float((shift + self.half_diag * turn).max(axis=1).min())


def plan(sim: Sim, iters, batch_size, steps_per_edge, max_curvature, goal_bias, rng, goal_test, goal_distance,
         unlock_mm=None, straight_prob=0.0, box_weight=1.0, time_limit_s=None, on_iteration=None):
    '''
    Grow one tree for at most `iters` batched iterations (or time_limit_s seconds), stopping at the
    goal. Returns (tree, best node, stats). on_iteration(it, tree, best) is called after each one.
    '''
    start = sim.initial_state()
    tree = RRTTree(start, goal_test=goal_test)
    tree.root.expandable = sim.can_grow(start)
    index = NodeIndex(sim.num_objs, box_weight)
    index.add(tree.root)
    resets = ResetIndex(sim)
    resets.add(start)

    best, best_dist = tree.root, goal_distance(start)
    stats = {"iterations": 0, "failed_rollouts": 0, "resets": 0, "stopped": "iterations"}
    t0 = time.time()

    for it in range(1, iters + 1):
        nearest = [index.nearest(tip_block_rrt.sample_tip_and_boxes(sim, rng, goal_bias)) for _ in range(batch_size)]
        nearest = [n for n in nearest if n is not None]
        if not nearest:
            stats["stopped"] = "nothing left to expand"
            break
        curvatures = [0.0 if rng.random() < straight_prob else rng.uniform(-max_curvature, max_curvature)
                      for _ in nearest]
        results = sim.rollout_batch([n.info for n in nearest], curvatures, steps_per_edge, stop_fn=goal_test)

        new = []
        for node, curvature, (states, ok) in zip(nearest, curvatures, results):
            if ok and states:
                child = tree.add_node(node, states[-1], control=curvature, edge=states)
                child.expandable = sim.can_grow(child.info)
                index.add(child)
                new.append(child)
            else:
                stats["failed_rollouts"] += 1
                node.failures += 1
                if node.failures >= MAX_FAILURES_PER_NODE:
                    index.set_expandable(node, False)

        if unlock_mm is not None and tree.goal_node is None:
            for child in list(new):
                if resets.distance(child.info) >= unlock_mm:
                    r = reset_state(sim, child.info)
                    reset_node = tree.add_node(child, r, control=RESET, edge=[r])
                    reset_node.expandable = True
                    index.add(reset_node)
                    resets.add(r)
                    stats["resets"] += 1

        for child in new:
            d = goal_distance(child.info)
            if d < best_dist:
                best, best_dist = child, d
        stats["iterations"] = it
        print(f"\r iter {it}/{iters}  nodes {len(tree.all_nodes)}  resets {stats['resets']}  "
              f"failed rollouts {stats['failed_rollouts']}  closest to goal {best_dist * 1000:.0f} mm  "
              f"{time.time() - t0:.0f} s   ", end="", flush=True)
        if on_iteration is not None:
            on_iteration(it, tree, best)

        if tree.goal_node is not None:
            best = tree.goal_node
            stats["stopped"] = "goal"
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            stats["stopped"] = "time limit"
            break
    print()
    stats["nodes"] = len(tree.all_nodes)
    stats["seconds"] = time.time() - t0
    return tree, best, stats


#----------------------------------------------------------------------- outputs

class ResetTreeLog(tree_stages.TreeLog):
    '''TreeLog that marks reset edges (they have no tip path).'''

    def _summarize(self, node, it):
        d = super()._summarize(node, it)
        d["reset"] = node.control == RESET
        if d["reset"]:
            d["tip_trace"] = np.zeros((0, 2), dtype=np.float32)
        return d


RESET_FRAMES = 12   # states shown for a reset in the solution video


def branch_states(node):
    '''Every sim state along the branch root -> node, with a label per state.'''
    path = node.path_from_root()
    states, labels = [path[0].info], ["initial state"]
    push = 1
    for n in path[1:]:
        if n.control == RESET:
            push += 1
            states += n.edge * RESET_FRAMES
            labels += [f"reset: the vine is retracted (not modeled), push {push} starts"] * RESET_FRAMES
        else:
            states += n.edge
            labels += [f"push {push}: curvature {n.control:+.2f} 1/m"] * len(n.edge)
    return states, labels


def branch_controls(node):
    '''[(curvature, num_steps) or ("reset", 0)] along the branch root -> node.'''
    return [(RESET, 0) if n.control == RESET else (n.control, len(n.edge))
            for n in node.path_from_root() if n.parent is not None]
