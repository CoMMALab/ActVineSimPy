'''
RRT over (vine tip, movable box positions).

Same planner loop as sim_rrt.py; the only difference is the space the tree is spread over.
sim_rrt.py samples a tip position and extends the node whose tip is nearest, so boxes only
move when they happen to be in the way. Here a sample is

    (tip x, tip y, box0 x, box0 y, box1 x, box1 y, ...)        [mm]

and the nearest node is the one minimizing

    sqrt( |tip - tip_sample|^2 + box_weight^2 * sum_k |box_k - box_sample_k|^2 )

so nodes that have moved a box somewhere new count as different states and get expanded too.
This is the naive version: box positions are sampled uniformly over the workspace (most of
which no box can reach), and box orientation and velocities are not part of the metric.

Goal: the tip goal of the scene by default. With --box-goal the goal is instead that movable
box 0's center ends up inside the given disc.

Run from the repo root:
    python xnew_rrt/tip_block_rrt.py
    python xnew_rrt/tip_block_rrt.py --box-goal 0.85 0.60 0.06 --iters 400
Outputs are the same as sim_rrt.py, under out/tip_block_rrt/.
'''

import random

from scene import Sim  # (puts the repo root on sys.path)
import sim_rrt
import viz
from rrt_util import StateInfo


def box_positions_mm(state: StateInfo):
    '''[(x, y)] of the movable box centers, in mm.'''
    return [(viz.MM(float(pose[0])), viz.MM(float(pose[1]))) for pose in state.moveable_obj_pose[0]]


def sample_tip_and_boxes(sim: Sim, rng: random.Random, goal_bias):
    '''
    Random (tip x, tip y, box0 x, box0 y, ...) in mm inside the scene bounds.
    With prob. goal_bias the sample is the goal: the tip goal (and box 0's goal, if there is one).
    '''
    xmin, xmax, ymin, ymax = [v * 1000 for v in sim.scene.bounds_m]
    sample = [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]
    for _ in range(sim.num_objs):
        sample += [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]

    if rng.random() < goal_bias:
        if sim.scene.box_goal_m is None:
            sample[0:2] = [sim.scene.goal_m[0] * 1000, sim.scene.goal_m[1] * 1000]
        elif sim.num_objs > 0:
            # where the tip should be for this is unknown, so keep its random sample
            sample[2:4] = [sim.scene.box_goal_m[0] * 1000, sim.scene.box_goal_m[1] * 1000]

    return tuple(sample)


def make_tip_block_distance(box_weight):
    def tip_block_distance(state: StateInfo, sample_mm):
        tip_x, tip_y = viz._tip_mm(state)
        dist_sq = (tip_x - sample_mm[0]) ** 2 + (tip_y - sample_mm[1]) ** 2

        for k, (box_x, box_y) in enumerate(box_positions_mm(state)):
            dist_sq += box_weight ** 2 * ((box_x - sample_mm[2 + 2 * k]) ** 2 +
                                          (box_y - sample_mm[3 + 2 * k]) ** 2)
        return dist_sq ** 0.5

    return tip_block_distance


def add_args(parser):
    parser.add_argument("--box-weight", type=float, default=1.0,
                        help="weight of box displacement relative to tip displacement in the distance")
    parser.add_argument("--box-goal", type=float, nargs=3, default=None, metavar=("X_M", "Y_M", "RADIUS_M"),
                        help="plan for movable box 0's center to end inside this disc (instead of the tip goal)")


def configure(sim: Sim, args):
    kwargs = {"sample_fn": sample_tip_and_boxes,
              "distance_fn": make_tip_block_distance(args.box_weight)}

    if args.box_goal is not None:
        if sim.num_objs == 0:
            raise SystemExit("--box-goal needs a scene with at least one movable box")
        sim.scene.box_goal_m = tuple(args.box_goal)
        kwargs["goal_test"] = sim.box_reached_goal
        kwargs["goal_distance"] = sim.box_goal_distance_m

    return kwargs


if __name__ == "__main__":
    sim_rrt.main(description=__doc__, default_out="out/tip_block_rrt", add_args=add_args, configure=configure)
