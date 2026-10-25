'''
Batched kinodynamic planners for the vine. Each iteration they pick a batch of (node, control)
pairs, simulate all of those edges in one batched rollout (scene.Sim.rollout_batch; on the GPU with
SimConfig(qp_backend="TORCH_IPM", device="cuda")), and insert the results. They differ in two
choices, which makes a 2 x 2 ablation:

                          keep every new node     keep a new node only if its fine cell is empty
  nearest-neighbor pick   rrt                     rrt_grid
  cell-weighted pick      kpiece                  kinopax

  rrt       sample (tip, box centers) uniformly (goal with prob goal_bias), expand the nearest
            node: the batched RRT of batch_tip_block_rrt.py / reset_action_rrt.py
  rrt_grid  the same, but the tree keeps one node per fine cell: SST's pruning with grid cells
            instead of witnesses and without costs (first come, first kept)
  kpiece    no nearest neighbor. Pick a coarse cell with probability proportional to its
            importance (KPIECE, Sucan & Kavraki 2009, one level):
                (1 + log(1 + iteration it was made)) / ((1 + times picked) (1 + neighbors) (1 + nodes))
            with neighbors = occupied cells next to it in tip x, y (fewer: on the frontier), then a
            node in it at random. Keeps every node.
  kinopax   Kino-PAX-like (Perrault, Ho, Lahijanian 2024): coarse regions weighted by how often an
            expansion from them reached a new fine cell and by how little they have been tried,
                ((reached + 1) / (tried + 2)) / sqrt((1 + tried) (1 + nodes)),
            a batch of nodes drawn from the regions by weight, new nodes kept only in empty fine cells.
  *_shape   rrt_grid / kinopax whose fine cells also cut two points along the vine's body
With probability goal_bias the cell planners instead expand one of the 16 nodes closest to the goal.

Cells: the projection (tip x, tip y, tip heading, every box center) is cut into a grid, tip and
boxes at cell_mm and the heading at heading_deg (fine cells), and tip and boxes at
coarse * cell_mm without the heading (coarse cells). Keys are hashed tuples, so many boxes cost
nothing. Why prune: the vine funnels. Edges whose curvatures differ by > 1 1/m end within 2 mm of
each other in 42-84% of sibling pairs when the tip is in contact (funneling.py), so most of a dense
tree is copies.

Resets (b2g_retractable): as in reset_action_rrt.py, a node whose boxes are unlock_mm from every
earlier reset gets a reset child right away (fresh vine, boxes kept), in any of the planners.
'''

import math, time

import numpy as np

from scene import Sim  # (puts the repo root on sys.path)
from rrt_util import RRTTree
from reset_action_rrt import NodeIndex, ResetIndex, RESET, MAX_FAILURES_PER_NODE
from reset_rrt import reset_state
import tip_block_rrt
import viz

PLANNERS = ("rrt", "rrt_grid", "kpiece", "kinopax", "rrt_grid_shape", "kinopax_shape")
EDGE_STRIDE = 3        # keep every 3rd sim state along an edge (and the last); enough to draw and replay


class Projection:
    '''shape: the fine cells also cut the vine's body points at 1/3 and 2/3 of its length (at
    2 * cell_mm), so states with the same tip but a different body path are not duplicates.'''

    def __init__(self, sim: Sim, cell_mm=15.0, heading_deg=30.0, coarse=5, shape=False):
        self.cell, self.heading, self.coarse_cell = cell_mm, math.radians(heading_deg), coarse * cell_mm
        self.shape = shape

    def features(self, state):
        '''(tip x, tip y [mm], tip heading [rad], box centers [mm]..., [body points [mm]...])'''
        x, y, th = (float(v) for v in state.tip())
        f = [viz.MM(x), viz.MM(y), th, *(v for xy in tip_block_rrt.box_positions_mm(state) for v in xy)]
        if self.shape:
            n = state.num_bodies
            for i in (n // 3, (2 * n) // 3):
                f += [viz.MM(float(state.state[0, 3 * i])), viz.MM(float(state.state[0, 3 * i + 1]))]
        return tuple(f)

    def fine(self, f):
        nb = 4 if self.shape else 0
        rest = f[3:len(f) - nb]
        return (math.floor(f[0] / self.cell), math.floor(f[1] / self.cell),
                math.floor(math.remainder(f[2], 2 * math.pi) / self.heading),
                *(math.floor(v / self.cell) for v in rest),
                *(math.floor(v / (2 * self.cell)) for v in f[len(f) - nb:]))

    def coarse(self, f):
        nb = 4 if self.shape else 0
        return (math.floor(f[0] / self.coarse_cell), math.floor(f[1] / self.coarse_cell),
                *(math.floor(v / self.coarse_cell) for v in f[3:len(f) - nb]))


class Region:
    __slots__ = ("nodes", "picked", "tried", "reached", "neighbors", "made")

    def __init__(self, made):
        self.nodes, self.picked, self.tried, self.reached, self.neighbors, self.made = [], 0, 0, 0, 0, made


class Planner:
    def __init__(self, name, sim, proj, rng, goal_distance, goal_bias, box_weight=1.0):
        assert name in PLANNERS, name
        name = name.replace("_shape", "")         # the _shape variants differ only in the projection
        self.name, self.sim, self.proj, self.rng = name, sim, proj, rng
        self.np_rng = np.random.default_rng(rng.randrange(2 ** 32))
        self.goal_distance, self.goal_bias = goal_distance, goal_bias
        self.prune = name in ("rrt_grid", "kinopax")
        self.index = NodeIndex(sim.num_objs, box_weight) if name.startswith("rrt") else None
        self.fine = {}          # fine cell -> its node (pruning planners)
        self.regions = {}       # coarse cell -> Region
        self.region_of = {}     # node id -> coarse key
        self.best = []          # (goal distance, node id), the closest few
        self.nodes = {}
        self.iteration = 0
        self.pruned = 0

    # ---- bookkeeping
    def register(self, node):
        f = self.proj.features(node.info)
        self.nodes[node.node_id] = node
        if self.index is not None:
            # NodeIndex needs every tree node in order; pruned ones are never added to the tree
            self.index.add(node)
        if self.prune:
            self.fine[self.proj.fine(f)] = node
        key = self.proj.coarse(f)
        region = self.regions.get(key)
        if region is None:
            region = self.regions[key] = Region(self.iteration)
            for dx in (-1, 0, 1):          # neighbors in tip x, y with the rest of the key the same
                for dy in (-1, 0, 1):
                    other = self.regions.get((key[0] + dx, key[1] + dy, *key[2:]))
                    if other is not None and other is not region:
                        other.neighbors += 1
                        region.neighbors += 1
        region.nodes.append(node)
        self.region_of[node.node_id] = key
        d = self.goal_distance(node.info)
        self.best.append((d, node.node_id))
        if len(self.best) > 64:
            self.best.sort()
            del self.best[16:]

    def accept(self, state):
        '''False if pruning and this state's fine cell is taken.'''
        if not self.prune:
            return True
        return self.proj.fine(self.proj.features(state)) not in self.fine

    def set_expandable(self, node, value):
        if self.index is not None:
            self.index.set_expandable(node, value)
        else:
            node.expandable = value

    def expanded(self, parent, kept):
        region = self.regions[self.region_of[parent.node_id]]
        region.tried += 1
        region.reached += bool(kept)

    # ---- choosing what to expand
    def choose(self, n):
        if self.index is not None:
            return [self.index.nearest(tip_block_rrt.sample_tip_and_boxes(self.sim, self.rng, self.goal_bias))
                    for _ in range(n)]
        keys = [k for k, r in self.regions.items() if any(nd.expandable for nd in r.nodes)]
        if not keys:
            return []
        regs = [self.regions[k] for k in keys]
        if self.name == "kpiece":
            w = np.array([(1 + math.log1p(r.made)) / ((1 + r.picked) * (1 + r.neighbors) * (1 + len(r.nodes)))
                          for r in regs])
        else:
            w = np.array([((r.reached + 1) / (r.tried + 2)) / math.sqrt((1 + r.tried) * (1 + len(r.nodes)))
                          for r in regs])
        picks = self.np_rng.choice(len(regs), size=n, p=w / w.sum())
        out = []
        self.best.sort()
        top = [self.nodes[i] for _, i in self.best[:16] if self.nodes[i].expandable]
        for j in picks:
            if top and self.rng.random() < self.goal_bias:
                out.append(self.rng.choice(top))
                continue
            r = regs[j]
            r.picked += 1
            out.append(self.rng.choice([nd for nd in r.nodes if nd.expandable]))
        return out


def plan(sim: Sim, planner_name, goal_test, goal_distance, rng, batch_size=256, steps_per_edge=25,
         max_curvature=3.33, goal_bias=0.1, straight_prob=0.0, max_edges=20000, time_limit_s=None,
         unlock_mm=None, cell_mm=15.0, heading_deg=30.0, coarse=5, on_iteration=None, box_weight=1.0):
    '''
    Grow a tree with one of PLANNERS until the goal, max_edges simulated edges or time_limit_s.
    Returns (tree, best node, stats). on_iteration(it, tree, best) is called after every batch.
    '''
    proj = Projection(sim, cell_mm, heading_deg, coarse, shape=planner_name.endswith("_shape"))
    planner = Planner(planner_name, sim, proj, rng, goal_distance, goal_bias, box_weight)
    start = sim.initial_state()
    tree = RRTTree(start, goal_test=goal_test)
    tree.root.expandable = sim.can_grow(start)
    planner.register(tree.root)
    resets = ResetIndex(sim)
    resets.add(start)

    best, best_dist = tree.root, goal_distance(start)
    stats = dict(planner=planner_name, iterations=0, edges=0, failed_rollouts=0, pruned=0, resets=0,
                 stopped="edge budget", edges_to_goal=None, seconds_to_goal=None)
    t0 = time.time()
    it = 0
    while stats["edges"] < max_edges:
        it += 1
        planner.iteration = it
        parents = [p for p in planner.choose(min(batch_size, max_edges - stats["edges"])) if p is not None]
        if not parents:
            stats["stopped"] = "nothing left to expand"
            break
        controls = [0.0 if rng.random() < straight_prob else rng.uniform(-max_curvature, max_curvature)
                    for _ in parents]
        results = sim.rollout_batch([p.info for p in parents], controls, steps_per_edge, stop_fn=goal_test)
        stats["edges"] += len(parents)

        new = []
        for parent, c, (states, ok) in zip(parents, controls, results):
            if not (ok and states):
                stats["failed_rollouts"] += 1
                parent.failures += 1
                if parent.failures >= MAX_FAILURES_PER_NODE:
                    planner.set_expandable(parent, False)
                if planner.index is None:
                    planner.expanded(parent, False)
                continue
            kept = planner.accept(states[-1])
            if planner.index is None:
                planner.expanded(parent, kept)
            if not kept:
                stats["pruned"] += 1
                continue
            edge = states[EDGE_STRIDE - 1::EDGE_STRIDE]
            if not edge or edge[-1] is not states[-1]:
                edge = edge + [states[-1]]
            child = tree.add_node(parent, states[-1], control=c, edge=edge)
            child.expandable = sim.can_grow(child.info)
            planner.register(child)
            new.append(child)
            if tree.goal_node is not None:
                break

        if unlock_mm is not None and tree.goal_node is None:
            for child in list(new):
                if resets.distance(child.info) >= unlock_mm:
                    r = reset_state(sim, child.info)
                    reset_node = tree.add_node(child, r, control=RESET, edge=[r])
                    reset_node.expandable = True
                    planner.register(reset_node)
                    resets.add(r)
                    stats["resets"] += 1

        for child in new:
            d = goal_distance(child.info)
            if d < best_dist:
                best, best_dist = child, d
        stats["iterations"] = it
        print(f"\r {planner_name} iter {it}  edges {stats['edges']}  nodes {len(tree.all_nodes)}  pruned {stats['pruned']}  "
              f"resets {stats['resets']}  closest {best_dist * 1000:.0f} mm  {time.time() - t0:.0f} s   ", end="", flush=True)
        if on_iteration is not None:
            on_iteration(it, tree, best)
        if tree.goal_node is not None:
            best = tree.goal_node
            stats.update(stopped="goal", edges_to_goal=stats["edges"], seconds_to_goal=time.time() - t0)
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            stats["stopped"] = "time limit"
            break
    print()
    stats.update(nodes=len(tree.all_nodes), seconds=time.time() - t0, best_mm=best_dist * 1000,
                 regions=len(planner.regions))
    return tree, best, stats
