'''
Scene description + a thin wrapper around the dvsim simulator, shared by the planner
(sim_rrt.py) and the single-rollout viewer (rollout.py).

Units at this level: scene geometry in METERS, headings in RADIANS, curvature in 1/m.
States coming out of the sim are in dvsim's internal units (mm, rad) -- see rrt_util.StateInfo.
'''

import math, os, sys
from dataclasses import dataclass, asdict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from dvsim import si
import dvsim.solver as solver
from dvsim.dynamic_vine import step
from dvsim.vine import create_state_batched, init_state_batched

from rrt_util import StateInfo

B = 1  # everything here is unbatched


@dataclass
class Scene:
    walls_m: list                       # static boxes [x1, y1, x2, y2], or oriented [cx, cy, theta_rad, hw, hh]
    objects_m: list                     # movable boxes [x1, y1, x2, y2]
    masses_kg: list                     # one per movable box
    start: tuple                        # (x_m, y_m, heading_rad) of the vine base
    goal_m: tuple                       # (x_m, y_m)
    goal_radius_m: float
    bounds_m: tuple = None              # (xmin, xmax, ymin, ymax); derived from the walls if None
    box_goal_m: tuple = None            # optional (x_m, y_m, radius_m): where movable box 0 should end up
    name: str = "scene"
    objects_theta: list = None          # optional initial heading [rad] per movable box (default 0)
    box_goal_theta: tuple = None        # optional (theta_rad, tol_rad): box 0's goal orientation too

    def __post_init__(self):
        if self.bounds_m is None:
            aabb = [w for w in self.walls_m if len(w) == 4]
            xs = [w[0] for w in aabb] + [w[2] for w in aabb]
            ys = [w[1] for w in aabb] + [w[3] for w in aabb]
            self.bounds_m = (min(xs), max(xs), min(ys), max(ys))


def default_scene():
    '''The table-top scene used so far: 4 boundary walls, one fixed block, one pushable block.'''
    walls = [(0.0, 0.0, 1.525, 0.1),
             (0.0, 0.9, 1.525, 1.0),
             (0.0, 0.0, 0.1, 1.0),
             (1.425, 0.0, 1.525, 1.0)]
    static_objs = [(0.7125, 0.2, 0.8125, 0.3)]
    return Scene(walls_m=walls + static_objs,
                 objects_m=[(0.7125, 0.4, 0.8125, 0.5)], masses_kg=[0.2],
                 start=(0.4, 0.4, 0.0),
                 goal_m=(1.225, 0.3), goal_radius_m=0.2,
                 name="default")


def load_scene(env_path):
    '''Build a Scene from an envs/*.txt file (mm in the file; `dynamic_obstacles:` are the
    movable boxes: x1 y1 x2 y2 mass_kg inertia -- the inertia column is ignored, dvsim derives it).'''
    from kinodynamic.env_loader import load_box_config
    cfg = load_box_config(env_path)
    mm = 1e-3
    return Scene(walls_m=[[float(v) * mm for v in ob] for ob in cfg['obstacles']],
                 objects_m=[[float(v) * mm for v in ob] for ob in cfg['dynamic_obstacles']],
                 masses_kg=[float(m[0]) for m in cfg['dynamic_object_masses']],
                 start=(cfg['start'][0] * mm, cfg['start'][1] * mm, cfg['start'][2]),
                 goal_m=(cfg['goal'][0] * mm, cfg['goal'][1] * mm),
                 goal_radius_m=cfg['goal_radius'] * mm,
                 bounds_m=(0.0, cfg['bound_x'] * mm, 0.0, cfg['bound_y'] * mm),
                 name=os.path.splitext(os.path.basename(env_path))[0])


STIFFNESS_MODES = ("curvature", "linear", "spam")


@dataclass
class SimConfig:
    max_bodies: int = 60
    grow_rate_mps: float = 0.3
    vine_radius_m: float = 0.05
    seg_len_m: float = 0.018
    stiffness_mode: str = "curvature"
    grow_force_n: float = 5.0   # stall force of growth [N] (a placeholder); None = hard growth constraint
    # Segment mass from a sweep over (inertia, mass, stiffness, damping) scored on following a
    # commanded curvature (sweep_vine_params.py; dvsim's own 0.02 kg oscillates). Joint stiffness
    # (VineParams.stiffness_val) and damping fitted to master's PBD vine with the 5 N growth force
    # (fit_to_master.py). Not measured values.
    seg_mass_kg: float = 0.005
    joint_stiffness: float = 0.2
    ang_damp: float = 1.24e-4
    seg_inertia_kgm2: float = 1.0e-5
    obj_friction: float = 0.5           # box-table Coulomb friction coefficient


class Sim:
    '''
    One vine + movable boxes in one scene. The control of a rollout is the curvature [1/m] of the
    part of the vine grown during it; bodies that already exist keep what they were grown with,
    like a fabricated vine keeps its actuators. What the control does depends on cfg.stiffness_mode:
        'curvature'  spring toward that curvature (dvsim 'curvature' mode)
        'spam'       sPAM actuators whose (constant) moment balances the vine's stiffness there
        'linear'     passive vine, spring toward straight; the control is ignored
    '''

    def __init__(self, scene: Scene, cfg: SimConfig = None):
        self.scene = scene
        self.cfg = cfg or SimConfig()
        assert self.cfg.stiffness_mode in STIFFNESS_MODES, self.cfg.stiffness_mode

        solver.cvxpylayer = None  # the cached QP layer depends on the scene's shapes

        self.params = si.vine_params_si(max_bodies=self.cfg.max_bodies, obstacles_m=scene.walls_m,
                                        grow_rate_mps=self.cfg.grow_rate_mps,
                                        radius_m=self.cfg.vine_radius_m, seg_len_m=self.cfg.seg_len_m,
                                        grow_force_n=self.cfg.grow_force_n,
                                        seg_mass_kg=self.cfg.seg_mass_kg, ang_damp=self.cfg.ang_damp,
                                        seg_inertia_kgm2=self.cfg.seg_inertia_kgm2)
        self.params.obj_friction = self.cfg.obj_friction
        self.params.stiffness_mode = self.cfg.stiffness_mode
        self.params.stiffness_val = torch.tensor(self.cfg.joint_stiffness, dtype=torch.float32)

        self.num_objs = len(scene.objects_m)
        if self.num_objs > 0:
            self._obj_pose0 = si.set_objects_si(self.params, scene.objects_m, scene.masses_kg)
            if scene.objects_theta is not None:
                self._obj_pose0[0, :, 2] = torch.tensor([float(t) for t in scene.objects_theta])
        else:
            self._obj_pose0 = torch.zeros(B, 0, 3)

        x0, y0, heading0 = scene.start
        self.base0 = torch.tensor([si.len_to_nd(x0), si.len_to_nd(y0), float(heading0)])

    #------------------------------------------------------------ geometry for drawing (mm)

    def static_boxes(self):
        '''[(pose(cx, cy, theta), hw, hh)] of the static obstacles, internal units.'''
        p = self.params
        return [([float(v) for v in p.obstacle_pose[k]], float(p.obstacle_hw[k]), float(p.obstacle_hh[k]))
                for k in range(p.obstacle_pose.shape[0])]

    def object_boxes(self, state: StateInfo):
        p = self.params
        return [([float(v) for v in state.moveable_obj_pose[0, k]], float(p.obj_hw[k]), float(p.obj_hh[k]))
                for k in range(self.num_objs)]

    def base_of(self, state: StateInfo):
        '''(x, y, heading) of the vine base for this state, internal units.'''
        return self.base0 if state.base is None else state.base

    #------------------------------------------------------------ simulation

    def initial_state(self, base_m=None):
        '''Start state. base_m = (x_m, y_m, heading_rad) launches from there instead of scene.start.'''
        base = None
        if base_m is not None:
            base = torch.tensor([si.len_to_nd(base_m[0]), si.len_to_nd(base_m[1]), float(base_m[2])])
        x, y, heading = self.base0 if base is None else base

        state, dstate = create_state_batched(B, self.cfg.max_bodies)
        bodies = torch.full((B, 1), 2)
        init_state_batched(self.params, state, bodies, heading.reshape(B, 1), x.reshape(B, 1), y.reshape(B, 1))

        mb = self.cfg.max_bodies
        return StateInfo(state, dstate, bodies, self._obj_pose0.clone(),
                         torch.zeros(B, self.num_objs, 3),
                         p_control=torch.zeros(mb), l0_control=torch.zeros(mb),
                         moment_control=torch.zeros(mb),
                         bend_target=torch.zeros(mb), base=base)

    def can_grow(self, state: StateInfo):
        # growing writes body index `bodies`, so the last slot has to stay free
        return state.num_bodies < self.cfg.max_bodies - 1

    def apply_control(self, start: StateInfo, curvature):
        '''
        The per-body design (p_control, l0_control, moment_control, bend_target) after commanding
        `curvature` [1/m] for everything grown from `start` on.
        '''
        n = start.num_bodies
        design = [start.p_control, start.l0_control, start.moment_control, start.bend_target]

        if self.cfg.stiffness_mode == "curvature":
            design[3] = design[3].clone()
            design[3][n:] = float(curvature) * self.cfg.seg_len_m

        elif self.cfg.stiffness_mode == "spam":
            from dvsim.spam import actuation_moment, design_actuator
            from dvsim.vine import STIFF_FACTOR
            p = l0 = moment = 0.0
            if abs(curvature) > 1e-6:
                # the moment that holds a joint at the commanded angle against the vine's stiffness,
                # and the actuator (pressure, rest length) that produces it at that bend radius
                stiffness_si = STIFF_FACTOR * float(self.params.stiffness_val) / self.params.spam_moment_scale
                p, l0 = design_actuator(stiffness_si * float(curvature) * self.cfg.seg_len_m, 1.0 / curvature)
                moment = actuation_moment(p, l0, 1.0 / curvature)
            for k, value in enumerate((p, l0, moment)):
                design[k] = design[k].clone()
                design[k][n:] = value

        return tuple(design)

    def rollout(self, start: StateInfo, curvature, num_steps, stop_fn=None):
        '''
        Grow for up to num_steps with `curvature` [1/m] applied to the newly grown part.
        Returns (states, ok): every state along the way (start excluded). ok is False if the QP
        failed or the state went non-finite; the offending step is not included.
        Stops early when the vine is fully grown or stop_fn(state) is True.
        '''
        return self.rollout_batch([start], [curvature], num_steps, stop_fn)[0]

    def rollout_batch(self, starts, curvatures, num_steps, stop_fn=None):
        '''
        rollout() for several (start state, curvature) pairs at once: each sim step is ONE batched
        dvsim step over all rollouts that are still running. Returns [(states, ok)], one per pair.
        '''
        N = len(starts)
        designs = [self.apply_control(s, c) for s, c in zip(starts, curvatures)]
        current = list(starts)
        results = [([], True) for _ in range(N)]
        running = [i for i in range(N) if self.can_grow(starts[i])]

        for _ in range(num_steps):
            if not running:
                break

            def stack(get):
                return torch.cat([get(i) for i in running], dim=0)

            bases = torch.stack([self.base_of(current[i]) for i in running])
            state, dstate, bodies, obj_pose, obj_dstate, ok = step(
                self.params, bases[:, 2:3], bases[:, 0:1], bases[:, 1:2],
                stack(lambda i: current[i].state), stack(lambda i: current[i].dstate),
                stack(lambda i: current[i].bodies),
                stack(lambda i: current[i].moveable_obj_pose), stack(lambda i: current[i].moveable_obj_dstate),
                return_ok=True,
                per_batch={"spam_moment": torch.stack([designs[i][2] for i in running]),
                           "bend_target": torch.stack([designs[i][3] for i in running])})

            still_running = []
            for k, i in enumerate(running):
                if not ok[k] or not torch.isfinite(state[k]).all() or not torch.isfinite(obj_pose[k]).all():
                    results[i] = (results[i][0], False)
                    continue

                current[i] = StateInfo(state[k:k + 1], dstate[k:k + 1], bodies[k:k + 1],
                                       obj_pose[k:k + 1], obj_dstate[k:k + 1],
                                       *designs[i], base=current[i].base)
                results[i][0].append(current[i])

                if self.can_grow(current[i]) and not (stop_fn is not None and stop_fn(current[i])):
                    still_running.append(i)

            running = still_running

        return results

    #------------------------------------------------------------ queries (meters)

    def tip_m(self, state: StateInfo):
        x, y, _ = state.tip()
        return si.len_to_si(float(x)), si.len_to_si(float(y))

    def reached_goal(self, state: StateInfo):
        '''True if the tip body overlaps the goal disc.'''
        x, y = self.tip_m(state)
        gx, gy = self.scene.goal_m
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 < self.scene.goal_radius_m + self.cfg.vine_radius_m

    def box_goal_distance_m(self, state: StateInfo):
        '''
        Distance from the center of movable box 0 to the center of its goal. With a goal
        orientation, plus how far its corners are turned from it (half-diagonal x angle error).
        '''
        gx, gy, _ = self.scene.box_goal_m
        x, y = [si.len_to_si(float(v)) for v in state.moveable_obj_pose[0, 0, :2]]
        d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
        if self.scene.box_goal_theta is not None:
            half_diag = si.len_to_si(float((self.params.obj_hw[0] ** 2 + self.params.obj_hh[0] ** 2) ** 0.5))
            d += half_diag * self.box_angle_error(state)
        return d

    def box_angle_error(self, state: StateInfo):
        '''|box 0's heading - its goal heading| [rad], modulo the box's symmetry (90 deg if square).'''
        sym = math.pi / 2 if abs(float(self.params.obj_hw[0] - self.params.obj_hh[0])) < 1e-6 else math.pi
        e = (float(state.moveable_obj_pose[0, 0, 2]) - self.scene.box_goal_theta[0]) % sym
        return min(e, sym - e)

    def box_reached_goal(self, state: StateInfo):
        gx, gy, r = self.scene.box_goal_m
        x, y = [si.len_to_si(float(v)) for v in state.moveable_obj_pose[0, 0, :2]]
        if ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 >= r:
            return False
        return self.scene.box_goal_theta is None or self.box_angle_error(state) <= self.scene.box_goal_theta[1]

    def goal_distance_m(self, state: StateInfo):
        x, y = self.tip_m(state)
        gx, gy = self.scene.goal_m
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5


def scene_to_dict(scene: Scene):
    return asdict(scene)


def scene_from_dict(d):
    return Scene(**d)
