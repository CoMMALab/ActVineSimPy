import math
import torch
from functools import partial

# DiffVine's baked-in ad-hoc unit factors. They keep the QP well-conditioned in the base's native
# units; dvsim.si absorbs them into the stored non-dim parameter values. Named here so vine.py,
# dynamic_vine.py and si.py share ONE definition instead of duplicating the bare literals (which
# previously had to be kept in sync by hand -- a latent desync bug).
GROW_FACTOR = 1000.0      # growth equality target = GROW_FACTOR * grow_rate
I_FACTOR = 100.0          # create_M scales moment of inertia by I_FACTOR
DAMP_FACTOR = 100.0       # bending_energy scales angular damping by DAMP_FACTOR
STIFF_FACTOR = 100_000.0  # bending_energy 'linear' elastic stiffness scale


def wrap_angle(a):
    """Wrap an angle (difference) to [-pi, pi]. Body headings are stored UNWRAPPED so the per-joint
    relative angle stays continuous when the vine curls past +-180 deg; anything derived from atan2
    must be wrapped relative to a stored heading before it is compared with / written into the state."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def finite_changes(h, init_val):
    h_rel = torch.empty_like(h)
    h_rel[0] = h[0] - init_val
    h_rel[1:] = h[1:] - h[:-1]

    return h_rel


# ------------------------------------------------------------------ geometry (oriented boxes)
# Everything collidable -- static walls AND movable objects -- is an oriented box (OBB), so these
# two primitives are the ONLY collision geometry in the sim. point_obb_gap handles vine<->box
# (the vine bodies are circles); obb_obb_sep handles box<->box (object<->wall, object<->object).
def point_obb_gap(px, py, pose, hw, hh, radius):
    """Signed gap between a circle (center (px,py), radius) and an oriented box.
    pose = (cx, cy, theta). >0 separated, <0 penetrating."""
    cx, cy, th = pose[0], pose[1], pose[2]
    c, s = torch.cos(th), torch.sin(th)
    dx, dy = px - cx, py - cy
    lx = c * dx + s * dy          # world -> box-local (rotate by -theta)
    ly = -s * dx + c * dy
    qx = lx - torch.clamp(lx, -hw, hw)
    qy = ly - torch.clamp(ly, -hh, hh)
    return torch.sqrt(qx * qx + qy * qy + 1e-9) - radius


def obb_obb_sep(poseA, hwA, hhA, poseB, hwB, hhB):
    """2D separating-axis separation between two oriented boxes.
    >=0 separated (gap on the max-separating axis); <0 penetration depth."""
    dx = poseB[0] - poseA[0]
    dy = poseB[1] - poseA[1]
    cA, sA = torch.cos(poseA[2]), torch.sin(poseA[2])
    cB, sB = torch.cos(poseB[2]), torch.sin(poseB[2])

    def gap(nx, ny):   # projection gap along unit axis (nx, ny)
        d = torch.abs(dx * nx + dy * ny)
        rA = hwA * torch.abs(cA * nx + sA * ny) + hhA * torch.abs(-sA * nx + cA * ny)
        rB = hwB * torch.abs(cB * nx + sB * ny) + hhB * torch.abs(-sB * nx + cB * ny)
        return d - rA - rB

    # candidate separating axes = the 4 face normals (2 per box)
    return torch.stack([gap(cA, sA), gap(-sA, cA), gap(cB, sB), gap(-sB, cB)]).max()


def box_mass(density, hw, hh):
    """Mass of a solid rectangle from an areal density (mass per unit area) and half-extents.
    Lets object mass scale physically with size (a 2x box is 4x heavier)."""
    return density * 4.0 * hw * hh


def box_friction_radius(hw, hh, n=64):
    """Mean distance of a rectangle's points from its center. A box spinning in place on a table,
    pressed down evenly, feels a friction torque of mu * m * g * (this radius)."""
    u = (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2 - 1          # cell centers in [-1, 1]
    x = hw.reshape(-1, 1, 1) * u.reshape(1, n, 1)
    y = hh.reshape(-1, 1, 1) * u.reshape(1, 1, n)
    return torch.sqrt(x ** 2 + y ** 2).mean(dim=(1, 2))


def box_inertia(mass, hw, hh):
    """Physical 2D moment of inertia of a uniform rectangle about its center:
    m*(w^2 + h^2)/12 with w=2hw, h=2hh  ->  m*(hw^2 + hh^2)/3."""
    return mass * (hw ** 2 + hh ** 2) / 3.0


def _obstacle_to_obb(o):
    """Normalize one obstacle spec to (cx, cy, theta, hw, hh). Accepts either an axis-aligned
    box [x1, y1, x2, y2] (theta=0) or an oriented box [cx, cy, theta, hw, hh]."""
    if len(o) == 5:
        return list(o)
    if len(o) == 4:
        x1, y1, x2, y2 = o
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return [(x1 + x2) / 2, (y1 + y2) / 2, 0.0, (x2 - x1) / 2, (y2 - y1) / 2]
    raise ValueError(f"obstacle must be [x1,y1,x2,y2] or [cx,cy,theta,hw,hh], got {o}")


class StateTensor:
    '''
    Helpful wrapper around a position or velocity state vector, providing
    convenience functions for getting x, y, theta data
    '''

    def __init__(self, tensor):
        if isinstance(tensor, StateTensor):
            raise ValueError("You passed a StateTensor to construct a StateTensor")

        self.tensor = tensor
        if tensor.shape[-1] % 3 != 0:
            raise ValueError(f"Tensor size {tensor.shape} must be 3N")

    # Note the following are views
    @property
    def x(self):
        return self.tensor[..., 0::3]

    @property
    def y(self):
        return self.tensor[..., 1::3]

    @property
    def theta(self):
        return self.tensor[..., 2::3]


class VineParams:
    '''
    Time indepedent parameters.
    TODO make tensors and differentiable
    '''

    def __init__(
            self,
            max_bodies,
            obstacles = [],
            grow_rate = 10.0 / 1000,
            stiffness_mode = 'linear',
            stiffness_val = None
        ):
        self.max_bodies = max_bodies
        self.dt = 1 / 90               # 1/90  # Time step
        self.radius = 25.0 / 2         # Uncompressed collision radius of each body
        self.grow_rate = torch.tensor(grow_rate, dtype=torch.float32)     # Length grown per unit time
        # How growth is imposed. None: the tip link MUST lengthen at grow_rate (a hard equality, as
        # in DiffVine). A number: growth is a force along the tip link that falls off linearly with
        # speed, from this stall force when the tip is stuck to zero at grow_rate (internal force
        # units; set through dvsim.si.vine_params_si(grow_force_n=...)).
        self.grow_force = None

        # Robot parameters
        self.m = torch.tensor([0.02], dtype=torch.float32)  # 0.002  # Mass of each body
        self.I = torch.tensor([10.0 / 100], dtype=torch.float32)    # Moment of inertia of each body
        self.half_len = torch.tensor(9.0, dtype=torch.float32)
        # Stiffness and damping coefficients
        self.damping = torch.tensor(50.0 / 100, dtype=torch.float32)        # angular damping (too large is unstable!)
        self.vel_damping = torch.tensor(0.1, dtype=torch.float32)           # linear velocity damping

        if stiffness_val is None:
            stiffness_val = torch.tensor(30_000.0 / STIFF_FACTOR, dtype=torch.float32)
        self.stiffness_val = stiffness_val

        # Environment obstacles, stored as ORIENTED boxes (the sim's one collision geometry).
        # Each input spec is either an axis-aligned box [x1,y1,x2,y2] or an oriented box
        # [cx,cy,theta,hw,hh]; both normalize to a pose (cx,cy,theta) + half-extents (hw,hh).
        obbs = [_obstacle_to_obb(o) for o in obstacles]
        t = torch.tensor(obbs, dtype=torch.float32).reshape(-1, 5)
        self.obstacle_pose = t[:, :3].contiguous()   # (n_obs, 3) = (cx, cy, theta)
        self.obstacle_hw = t[:, 3].contiguous()      # (n_obs,)
        self.obstacle_hh = t[:, 4].contiguous()      # (n_obs,)

        # --- Movable objects (oriented boxes). Empty by default = "no objects"; set via
        #     dvsim.si.set_objects_si. Inertia is derived physically downstream (create_M_obj). ---
        self.obj_hw = torch.zeros(0)                 # (n_obj,) half-width  per object
        self.obj_hh = torch.zeros(0)                 # (n_obj,) half-height per object
        self.obj_mass = torch.zeros(0)               # (n_obj,) mass        per object

        # --- Post-solve velocity regularization (see step()). Caps derived from grow_rate; the
        #     factors are tuned to catch the body-promotion / free-growth spikes only. ---
        gr = float(self.grow_rate)
        self.vine_vel_cap = 1.5 * GROW_FACTOR * gr   # per-component vine velocity cap
        self.obj_vel_cap = 2.0 * GROW_FACTOR * gr    # object linear (vx, vy) velocity cap
        self.obj_ang_vel_cap = 5.0                   # object angular (vtheta) cap [rad/s]
        self.obj_damp = 1.0                          # fraction of an object's velocity carried into the
                                                     # next step (1 = momentum is kept; friction slows it)
        self.obj_friction = 0.5                      # Coulomb friction coefficient object <-> table
        self.gravity = 9810.0                        # [mm/s^2] in the base's native units (si sets it)

        # --- sPAM actuation, used when stiffness_mode == 'spam' (see dvsim/spam.py): a constant
        #     moment per joint on top of the linear vine spring. ---
        self.spam_moment = torch.zeros(max_bodies)   # (max_bodies,) actuator moment per joint [N*m], signed
        self.spam_moment_scale = 1.0e6               # N*m -> internal torque (= dvsim.si.spam_moment_to_nd())
        self.promote_factor = 2.0                    # tip-link length (in half_len) that triggers growth

        # --- 'curvature' stiffness mode: per-joint REST angle [rad] (joint i = between body i-1 and i).
        #     The linear elastic model, but relaxing to bend_target instead of straight -- a stable
        #     stand-in for actuation that planners can use as a control (rest angle = curvature * seg len).
        self.bend_target = torch.zeros(max_bodies)   # (max_bodies,)

        self.si: dict | None = None                  # dvsim.si scale record (set by vine_params_si)

        # 'linear' (passive), 'spam' (linear + constant actuator moment) or 'curvature'
        self.stiffness_mode = stiffness_mode

    def requires_grad_(self):
        """Mark the fittable physical parameters as differentiable (for system identification)."""
        for t in (self.m, self.I, self.damping, self.vel_damping, self.grow_rate, self.stiffness_val):
            t.requires_grad_()

    def opt_params(self):
        """Fittable physical parameters as an optimizer param-group (for system identification)."""
        return [{'params': [self.m, self.I, self.damping, self.vel_damping, self.grow_rate,
                            self.stiffness_val], 'weight_decay': 0}]


def create_M(m, I, max_bodies):
    # Update mass matrix M (block diagonal)
    diagonal_elements = torch.cat([m, m, I * I_FACTOR]).repeat(max_bodies)
    return torch.diag(diagonal_elements)   # Shape: (nq, nq))


def create_state_batched(batch_size, max_bodies):
    state = torch.zeros(batch_size, max_bodies * 3)
    dstate = torch.zeros(batch_size, max_bodies * 3)
    return state, dstate


def init_state_batched(params: VineParams, state, bodies, init_headings,
                       init_x, init_y) -> StateTensor:
    '''
    Create vine state vectors from params
    '''
    batch_size = state.shape[0]
    state = StateTensor(state)

    # Init first body position
    for i in range(batch_size):
        state.theta[i, :] = init_headings[i]

    # (init_x, init_y) is the BASE of the vine; body 0's center sits half a segment along the
    # heading, which is what joint_deviation's base constraint expects.
    state.x[:, 0] = init_x[:, 0] + params.half_len * torch.cos(state.theta[:, 0])
    state.y[:, 0] = init_y[:, 0] + params.half_len * torch.sin(state.theta[:, 0])

    # Init all body positions
    for batch_i in range(batch_size):
        for i in range(1, int(bodies[batch_i])):
            lastx = state.x[batch_i, i - 1]
            lasty = state.y[batch_i, i - 1]
            lasttheta = state.theta[batch_i, i - 1]
            thistheta = state.theta[batch_i, i]

            state.x[
                batch_i,
                i] = lastx + params.half_len * torch.cos(lasttheta) + params.half_len * torch.cos(thistheta)
            state.y[
                batch_i,
                i] = lasty + params.half_len * torch.sin(lasttheta) + params.half_len * torch.sin(thistheta)


def zero_out(state, bodies):
    state = StateTensor(state)

    idx = torch.arange(state.x.shape[-1])
    mask = idx >= bodies   # Mask of uninited values

    state.x[:] = torch.where(mask, 0, state.x)
    state.y[:] = torch.where(mask, 0, state.y)
    state.theta[:] = torch.where(mask, 0, state.theta)


def zero_out_custom(state, bodies):
    idx = torch.arange(state.shape[-1])
    mask = idx >= bodies   # Mask of uninited values
    state[:] = torch.where(mask, 0, state)


def sdf(params: VineParams, state, bodies):
    """Signed gap from each vine body collider to the nearest obstacle (oriented box).
    Args:
        state: vine state (max_bodies*3,)
        bodies: num active bodies
    Returns:
        min_dist (torch.Tensor): (max_bodies,) gap to nearest obstacle, minus body radius
            (negative when the incompressible body would penetrate). Inactive bodies zeroed.
    """
    st = StateTensor(state)

    def body_gap(px, py):   # min gap of one body circle over all obstacle boxes
        gaps = torch.vmap(lambda pose, hw, hh: point_obb_gap(px, py, pose, hw, hh, params.radius))(
            params.obstacle_pose, params.obstacle_hw, params.obstacle_hh)
        return gaps.min()

    min_dist = torch.vmap(body_gap)(st.x, st.y)
    zero_out_custom(min_dist, bodies)
    return min_dist


def joint_deviation(params: VineParams, init_x, init_y, state: torch.Tensor, bodies):

    # Vector of deviation per-joint, [x y x2 y2 x3 y3],
    # where each coordinate pair is the deviation with the last body
    length = state.shape[-1] // 3
    constraints = state.new_zeros(length * 2)

    state = StateTensor(state)

    x = state.x
    y = state.y
    theta = state.theta

    constraints[0] = (x[0] - init_x) - params.half_len * torch.cos(theta[0])
    constraints[1] = (y[0] - init_y) - params.half_len * torch.sin(theta[0])

    constraints[2::2] = (x[1:] - x[:-1]) - params.half_len * torch.cos(theta[1:]) \
                                         - params.half_len * torch.cos(theta[:-1])

    constraints[3::2] = (y[1:] - y[:-1]) - params.half_len * torch.sin(theta[1:]) \
                                         - params.half_len * torch.sin(theta[:-1])

    # Last body is special. It has a sliding AND rotation joint with the second-last body
    endx = x[bodies - 2] + params.half_len * torch.cos(theta[bodies - 2])
    endy = y[bodies - 2] + params.half_len * torch.sin(theta[bodies - 2])

    angle_diff = wrap_angle(torch.atan2(y[bodies - 1] - endy, x[bodies - 1] - endx) - theta[bodies - 1])

    # So if we have 4 bodies, we have constraints: [x y x y x y dtheta]
    #                                               0 1 2 3 4 5 6
    # So the last constraint index is 6 = (bodies-1)*2
    constraints[(bodies - 1) * 2] = angle_diff

    zero_out_custom(constraints, bodies * 2 - 1)

    return constraints

def bending_energy(params: VineParams, theta_rel, dtheta_rel, bodies):
    """Bending torque per joint = elastic restoring moment + angular velocity damping.
    theta_rel / dtheta_rel are the per-joint relative angle / angular velocity."""
    if params.stiffness_mode == 'spam':
        # constant actuator moment (does not depend on the joint angle) minus the vine's own linear
        # bending moment, plus damping
        bend = params.spam_moment_scale * params.spam_moment \
            - STIFF_FACTOR * params.stiffness_val.abs() * theta_rel - DAMP_FACTOR * params.damping.abs() * dtheta_rel
        zero_out_custom(bend, bodies)
        return bend

    if params.stiffness_mode == 'curvature':
        # linear elastic stiffness about a per-joint rest angle (bend_target) + damping
        dev = theta_rel - params.bend_target
        bend = -STIFF_FACTOR * params.stiffness_val.abs() * dev - DAMP_FACTOR * params.damping.abs() * dtheta_rel
        zero_out_custom(bend, bodies)
        return bend

    # 'linear' elastic stiffness + angular velocity damping
    stiffness_response = params.stiffness_val.abs() * theta_rel.abs()
    zero_out_custom(stiffness_response, bodies)
    bend = -STIFF_FACTOR * theta_rel.sign() * stiffness_response - DAMP_FACTOR * params.damping.abs() * dtheta_rel
    zero_out_custom(bend, bodies)
    return bend


def extend(params: VineParams, state, dstate, bodies):
    state = StateTensor(state)
    dstate = StateTensor(dstate)

    new_i = bodies
    last_i = bodies - 1
    penult_i = bodies - 2

    # Compute position of second last seg
    endingx = state.x[penult_i] + params.half_len * torch.cos(state.theta[penult_i])
    endingy = state.y[penult_i] + params.half_len * torch.sin(state.theta[penult_i])

    # Compute last body's distance
    last_link_distance = ((state.x[last_i] - endingx)**2 + \
                          (state.y[last_i] - endingy)**2).sqrt().squeeze(-1)

    # x2 to prevent 0-len segments
    extend_needed = last_link_distance > params.half_len * params.promote_factor

    # Compute location of new seg
    last_link_theta = torch.atan2(state.y[last_i] - state.y[penult_i], state.x[last_i] - state.x[penult_i])

    new_seg_x = endingx + params.half_len * torch.cos(last_link_theta)
    new_seg_y = endingy + params.half_len * torch.sin(last_link_theta)
    # keep the stored heading continuous with the previous body (atan2 alone would jump by 2*pi
    # once the vine has curled past +-180 deg, which reads as a huge joint angle)
    new_seg_theta = (state.theta[penult_i] + wrap_angle(last_link_theta - state.theta[penult_i])).squeeze()

    # Copy last body one forward
    state.x[new_i] = torch.where(extend_needed, state.x[last_i], state.x[new_i])
    state.y[new_i] = torch.where(extend_needed, state.y[last_i], state.y[new_i])
    state.theta[new_i] = torch.where(extend_needed, state.theta[last_i], state.theta[new_i])

    # Copy last body vel too
    # dstate.x[new_i] = torch.where(extend_needed, dstate.x[last_i], dstate.x[new_i])
    # dstate.y[new_i] = torch.where(extend_needed, dstate.y[last_i], dstate.y[new_i])
    # dstate.theta[new_i] = torch.where(extend_needed, dstate.theta[last_i], dstate.theta[new_i])

    dstate.x[new_i] = 0
    dstate.y[new_i] = 0
    dstate.theta[new_i] = 0

    # Set the new segment position
    state.x[last_i] = torch.where(extend_needed, new_seg_x, state.x[last_i])
    state.y[last_i] = torch.where(extend_needed, new_seg_y, state.y[last_i])
    state.theta[last_i] = torch.where(extend_needed, new_seg_theta, state.theta[last_i])

    # Set the new segment to have velocity of the former tip
    # dstate.x[last_i] = torch.where(extend_needed, dstate.x[penult_i], dstate.x[last_i])
    # dstate.y[last_i] = torch.where(extend_needed, dstate.y[penult_i], dstate.y[last_i])
    # dstate.theta[last_i] = torch.where(extend_needed, dstate.theta[penult_i], dstate.theta[last_i])

    bodies = bodies + extend_needed

    zero_out(state.tensor, bodies)
    zero_out(dstate.tensor, bodies)

    # FIXME return copies, don't mutate
    return bodies


def growth_rate(params: VineParams, state, dstate, bodies):

    state = StateTensor(state)
    dstate = StateTensor(dstate)
    id1 = bodies - 2
    id2 = bodies - 1

    assert state.tensor.shape == dstate.tensor.shape

    # Now return the constraints for growing the last segment
    x1 = state.x[id1]
    y1 = state.y[id1]
    vx1 = dstate.x[id1]
    vy1 = dstate.y[id1]

    x2 = state.x[id2]
    y2 = state.y[id2]
    vx2 = dstate.x[id2]
    vy2 = dstate.y[id2]

    constraint = ((x2 - x1) * (vx2 - vx1) + (y2 - y1) * (vy2 - vy1)) / \
                  torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return constraint


def forward_batched_part(params: VineParams, init_heading, init_x, init_y, state, dstate, bodies):
    '''
    Compute some jacobians about this state wrt forces, growth, sdf_now
    
    This function is separated from the QP solving stuff in forward() because it later
    gets compiled into a batched function using vmap, but we can't also compile the QP stuff
    '''

    bodies = extend(params, state, dstate, bodies)

    # Jacobian of SDF with respect to x and y
    L = torch.func.jacrev(partial(sdf, params))(state, bodies)
    sdf_now = sdf(params, state, bodies)

    # Jacobian of joint deviation wrt state
    J = torch.func.jacrev(partial(joint_deviation, params, init_x, init_y))(state, bodies)
    deviation_now = joint_deviation(params, init_x, init_y, state, bodies)
    
    # print('J isnan', torch.isnan(J).any())
    # print('deviation_now isnan', torch.isnan(deviation_now).any())
    
    # Jacobian of growth rate wrt state
    growth_wrt_state, growth_wrt_dstate = torch.func.jacrev(partial(growth_rate, params), argnums=(0, 1))(state, dstate, bodies)
    growth = growth_rate(params, state, dstate, bodies)
    
    # print('growth_wrt_state isnan', torch.isnan(growth_wrt_state).any())
    # print('growth_wrt_dstate isnan', torch.isnan(growth_wrt_dstate).any())
    
    # Stiffness forces
    theta_rel = finite_changes(StateTensor(state).theta, init_heading)
    dtheta_rel = finite_changes(StateTensor(dstate).theta, 0.0)

    bend_energy = bending_energy(params, theta_rel, dtheta_rel, bodies)
    
    # print('deviation_now', deviation_now.mean())
    # print('sdf_now', sdf_now.mean())
    # print('growth', growth.mean())
    # print('bend_energy', bend_energy.mean())

    forces = StateTensor(torch.zeros_like(state))

    # Apply unbending forces
    forces.theta[:] += -bend_energy        # Apply bend energy as torque to own joint
    forces.theta[:-1] += bend_energy[1:]   # Apply bend energy as torque to joint before

    # FIXME sign flipped somewhere
    forces.x[:] += params.vel_damping * StateTensor(dstate).x
    forces.y[:] += params.vel_damping * StateTensor(dstate).y

    forces = forces.tensor

    return bodies, forces, growth, sdf_now, deviation_now, L, J, growth_wrt_state, growth_wrt_dstate
