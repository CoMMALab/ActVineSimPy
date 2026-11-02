"""Movable-obstacle contact on top of the stable dvsim vine base — ORIENTED boxes.

Reuses DiffVine's vine physics (forward_batched_part) untouched, and adds to the QP:
  - object pose DOFs (cx, cy, theta) with mass + rotational inertia
  - TWO-SIDED vine<->object contact (the vine can translate AND rotate objects)
  - object<->static-wall contact (one-sided) and object<->object contact (two-sided), each with
    optional Coulomb friction (params.wall_friction, params.box_friction; 0 = frictionless)

Objects are oriented boxes: state is a pose (cx, cy, theta); half-extents (hw, hh) are fixed
per object (params.obj_hw / obj_hh). Because every collision measure is a differentiable
function of the pose, jacrev w.r.t. the pose yields the full (vx, vy, vtheta) coupling --
so off-center pushes impart spin automatically (torque = autodiff of the geometry).

Decision variable: [vine_velocities (max_bodies*3), object_velocities (n_obj*3)].

UNITS (inherited from the DiffVine base): lengths in mm (vine radius 12.5, half_len 9),
time in seconds (dt = 1/90). Object mass is in the same DiffVine-fitted mass unit as a vine
body (m = 0.02); object rotational inertia is DERIVED PHYSICALLY from mass + size via
create_M_obj -> box_inertia (I = m*(hw^2+hh^2)/3), not hand-set. box_mass(density, hw, hh)
makes object mass scale with size from an areal density. The vine's own m/I keep DiffVine's
fitted values (do not "physicalize" them -- they are calibrated to real vine data).
"""
import torch
from .vine import (StateTensor, create_M, forward_batched_part,
                   point_obb_gap, obb_obb_sep, rounded_obb_sep, box_inertia, box_friction_radius, GROW_FACTOR)
from .solver import init_layers, solve_layers


# ------------------------------------------------------------------ object mass
def create_M_obj(obj_mass, obj_hw, obj_hh):
    """Block-diagonal object mass matrix [m, m, I] per object, with the rotational inertia I
    DERIVED PHYSICALLY from the object's mass and size (uniform rectangle) -- so bigger/heavier
    boxes resist rotation correctly. No hand-set inertia, no magic factor: I/m = (hw^2+hh^2)/3
    is the true physical ratio. Same [m, m] scale as the vine's create_M (units are consistent)."""
    per = torch.stack([obj_mass, obj_mass, box_inertia(obj_mass, obj_hw, obj_hh)], dim=1)
    return torch.diag(per.reshape(-1))


def object_friction_weights(params, mu=None, spin=None):
    """(n_obj, 3) weights of the per-object ground friction term in the QP objective:

        dt * mu * m * g * || (vx, vy, c * vtheta) ||        c = box_friction_radius

    This is Coulomb friction between the object and the table in impulse form. The object's
    weight m*g is carried by the table, so friction can resist with a force of at most mu*m*g
    (or a torque of mu*m*g*c, or a mix of the two). While the push on the object is below that,
    the cheapest velocity is exactly zero and the object does not move; above it, the object
    slides and friction keeps taking mu*m*g away. A heavier object therefore needs a
    proportionally harder push, on top of being harder to accelerate.
    mu: (B, 1) friction coefficients, one per batch element, instead of params.obj_friction ->
    (B, n_obj, 3) weights. spin: (B, 1) scales on c, instead of params.obj_spin (1: a box pressed down
    evenly; larger resists turning more, e.g. a block whose weight sits near its rim)."""
    mu = params.obj_friction if mu is None else mu
    spin = getattr(params, "obj_spin", 1.0) if spin is None else spin
    w = params.dt * mu * params.obj_mass.abs() * params.gravity
    return torch.stack([w, w, w * spin * box_friction_radius(params.obj_hw, params.obj_hh)], dim=-1)


# ------------------------------------------------------------------ measures
def proximity_measure(params, state, obj_pose, bodies):
    """Per (body, object) gap between the vine body circle and the oriented box.
    Phantom bodies (>= bodies) forced separated so they never bind."""
    st = StateTensor(state)
    cx, cy = st.x, st.y

    r = params.obj_round
    def gap_one(pose, hw, hh):
        return torch.vmap(lambda px, py: point_obb_gap(px, py, pose, hw - r, hh - r, params.radius + r))(cx, cy)

    gaps = torch.vmap(gap_one)(obj_pose, params.obj_hw, params.obj_hh).transpose(0, 1)   # (max_bodies, n_obj)
    idx = torch.arange(gaps.shape[0]).unsqueeze(-1)
    return torch.where(idx >= bodies, torch.full_like(gaps, 1e3), gaps)


def proximity_jacobians(params, state, obj_pose, bodies):
    """Jacobians of proximity_measure w.r.t. the vine state (max_bodies, n_obj, state) and the
    object poses (max_bodies, n_obj, n_obj, 3), built pair by pair. Gap (j, o) depends only on body
    j's (x, y) and object o's pose, so one small gradient per pair gives the same numbers as
    jacrev, which runs one backward pass per output over every pair: (bodies * objects)^2 work and
    memory instead of bodies * objects."""
    st = StateTensor(state)
    mb, n_obj = params.max_bodies, obj_pose.shape[0]

    r = params.obj_round
    def gap(px, py, pose, hw, hh):
        return point_obb_gap(px, py, pose, hw - r, hh - r, params.radius + r)

    d = torch.func.grad(gap, argnums=(0, 1, 2))
    d = torch.vmap(d, in_dims=(None, None, 0, 0, 0))                    # over objects
    d = torch.vmap(d, in_dims=(0, 0, None, None, None))                 # over bodies
    gx, gy, gpose = d(st.x, st.y, obj_pose, params.obj_hw, params.obj_hh)  # (mb, n_obj), .., (mb, n_obj, 3)
    active = (torch.arange(mb) < bodies).to(state.dtype).unsqueeze(-1)  # phantom bodies: constant gap
    gx, gy, gpose = gx * active, gy * active, gpose * active.unsqueeze(-1)
    body = torch.eye(mb, dtype=state.dtype)[:, None, :]                 # (mb, 1, mb): pair (j, o) -> body j
    J_state = torch.stack([gx[:, :, None] * body, gy[:, :, None] * body, torch.zeros_like(gx)[:, :, None] * body], -1)
    J_pose = gpose[:, :, None, :] * torch.eye(n_obj, dtype=state.dtype)[None, :, :, None]
    return J_state.reshape(mb, n_obj, 3 * mb), J_pose


def proximity_tangents(params, state, obj_pose, bodies):
    """Jacobians of the TANGENTIAL relative velocity at each (body, object) contact, times
    params.vine_friction: (max_bodies, n_obj, 3 * max_bodies) and (max_bodies, n_obj, n_obj, 3).
    The tangent is the gap normal turned 90 deg; the box's side of the contact is its material point
    nearest the body, so a spinning box drags the vine too. The vine body's own spin is left out."""
    st = StateTensor(state)
    mb, n_obj = params.max_bodies, obj_pose.shape[0]
    r = params.obj_round

    def one(px, py, pose, hw, hh):
        gap = lambda x, y: point_obb_gap(x, y, pose, hw - r, hh - r, params.radius + r)
        nx, ny = torch.func.grad(gap, argnums=(0, 1))(px, py)
        norm = torch.sqrt(nx * nx + ny * ny).clamp_min(1e-9)
        nx, ny = nx / norm, ny / norm
        reach = gap(px, py) + params.radius
        dx, dy = px - nx * reach - pose[0], py - ny * reach - pose[1]       # box center -> contact point
        tx, ty = -ny, nx
        return tx, ty, torch.stack([-tx, -ty, -(dx * ty - dy * tx)])

    f = torch.vmap(torch.vmap(one, in_dims=(None, None, 0, 0, 0)), in_dims=(0, 0, None, None, None))
    tx, ty, tpose = f(st.x, st.y, obj_pose, params.obj_hw, params.obj_hh)   # (mb, n_obj), .., (mb, n_obj, 3)
    mu = params.vine_friction * (torch.arange(mb) < bodies).to(state.dtype).unsqueeze(-1)
    tx, ty, tpose = tx * mu, ty * mu, tpose * mu.unsqueeze(-1)
    body = torch.eye(mb, dtype=state.dtype)[:, None, :]
    T_state = torch.stack([tx[:, :, None] * body, ty[:, :, None] * body, torch.zeros_like(tx)[:, :, None] * body], -1)
    T_pose = tpose[:, :, None, :] * torch.eye(n_obj, dtype=state.dtype)[None, :, :, None]
    return T_state.reshape(mb, n_obj, 3 * mb), T_pose


def object_env_constraints(params, obj_pose):
    """Per-batch object<->wall and object<->object separations + Jacobians w.r.t. pose.
    Walls are the same oriented boxes the vine collides with (params.obstacle_*). Returns empty
    (zero-object) tensors when there are no objects (see dynamic_forward_part on why n_obj==0
    is handled explicitly rather than via vmap)."""
    n_obj, n_walls = obj_pose.shape[0], params.obstacle_pose.shape[0]
    if n_obj == 0:
        z = obj_pose.new_zeros
        return z((0, n_walls)), z((0, n_walls, 0, 3)), z((0, 0)), z((0, 0, 0, 3))
    wall_pose, wall_hw, wall_hh = params.obstacle_pose, params.obstacle_hw, params.obstacle_hh
    ro, rw = params.obj_round, params.wall_round
    if ro == 0 and rw == 0:
        sep_w = sep_o = obb_obb_sep
    else:                                                          # rounded corners
        sep_w = lambda pA, hwA, hhA, pB, hwB, hhB: rounded_obb_sep(pA, hwA, hhA, ro, pB, hwB, hhB, rw)
        sep_o = lambda pA, hwA, hhA, pB, hwB, hhB: rounded_obb_sep(pA, hwA, hhA, ro, pB, hwB, hhB, ro)

    def owall(pose):                                               # (n_obj, n_walls)
        def one(p, hw, hh):
            return torch.vmap(lambda wp, whw, whh: sep_w(p, hw, hh, wp, whw, whh))(wall_pose, wall_hw, wall_hh)
        return torch.vmap(one)(pose, params.obj_hw, params.obj_hh)

    def oobj(pose):                                                # (n_obj, n_obj)
        def one(pA, hwA, hhA):
            return torch.vmap(lambda pB, hwB, hhB: sep_o(pA, hwA, hhA, pB, hwB, hhB))(pose, params.obj_hw, params.obj_hh)
        return torch.vmap(one)(pose, params.obj_hw, params.obj_hh)

    ow_now = owall(obj_pose); ow_J = torch.func.jacrev(owall)(obj_pose)   # (n_obj, n_walls, n_obj, 3)
    oo_now = oobj(obj_pose); oo_J = torch.func.jacrev(oobj)(obj_pose)     # (n_obj, n_obj, n_obj, 3)
    return ow_now, ow_J, oo_now, oo_J


def env_tangents(params, obj_pose, ow_J, oo_J):
    """Tangential-velocity rows of the object<->wall and object<->object contacts, for their Coulomb
    friction (params.wall_friction, params.box_friction), batched: (B, n_obj * n_walls, n_obj * 3) and
    (B, n_obj * n_obj, n_obj * 3). The contact normal n is the separation's gradient w.r.t. the object's
    position (unit, pointing away from the other body); the tangent is n turned 90 deg. The contact point
    lies on the object's side facing the other body, at distance s = its half-extent along n from the
    center (r.n = -s), so its tangential velocity is t.v + omega * (r.n) = t.v - s * omega. For an
    object pair (a, b) the row is the relative velocity: a's point minus b's (b's point is at r.n = +s_b)."""
    B, n_obj = obj_pose.shape[0], obj_pose.shape[1]
    eye = torch.eye(n_obj, dtype=obj_pose.dtype, device=obj_pose.device)
    c, s = torch.cos(obj_pose[..., 2]), torch.sin(obj_pose[..., 2])                # (B, n_obj)
    hw, hh = params.obj_hw.to(obj_pose), params.obj_hh.to(obj_pose)

    def frame(g):                                       # (..., 3) own-pose gradient -> unit n, t
        n = g[..., :2] / g[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return n, torch.stack([-n[..., 1], n[..., 0]], -1)

    def extent(n, k):                                   # half-extent of object k along n
        ck, sk = c[:, k, None], s[:, k, None]                # (B, 1) against n: (B, pairs, 2)
        return hw[k] * (ck * n[..., 0] + sk * n[..., 1]).abs() + hh[k] * (-sk * n[..., 0] + ck * n[..., 1]).abs()

    # object<->wall: (B, n_obj, n_walls, 3) gradients of each pair w.r.t. its own object
    g = (ow_J * eye[None, :, None, :, None]).sum(3)
    n, t = frame(g)
    ext = torch.stack([extent(n[:, k], k) for k in range(n_obj)], 1)            # (B, n_obj, n_walls)
    row = torch.cat([t, -ext.unsqueeze(-1)], -1)                                # (B, n_obj, n_walls, 3)
    T_ow = row.unsqueeze(3) * eye[None, :, None, :, None]                       # slot = the pair's object
    # object<->object: pair (a, b), gradient w.r.t. a
    g = (oo_J * eye[None, :, None, :, None]).sum(3)                             # (B, n_obj, n_obj, 3)
    n, t = frame(g)
    ext_a = torch.stack([extent(n[:, a], a) for a in range(n_obj)], 1)          # (B, a, b)
    ext_b = torch.stack([extent(n[:, :, b], b) for b in range(n_obj)], 2)       # (B, a, b)
    row_a = torch.cat([t, -ext_a.unsqueeze(-1)], -1)
    row_b = torch.cat([-t, -ext_b.unsqueeze(-1)], -1)
    T_oo = row_a.unsqueeze(3) * eye[None, :, None, :, None] + row_b.unsqueeze(3) * eye[None, None, :, :, None]
    T_oo = T_oo * (1 - eye)[None, :, :, None, None]                             # no self pairs
    return T_ow.reshape(B, -1, n_obj * 3), T_oo.reshape(B, -1, n_obj * 3)


def dynamic_forward_part(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose):
    """Per-batch vine measures + vine<->object proximity measures/Jacobians (jacrev w.r.t. pose
    gives the object velocity coupling directly, torque included). Degrades to vine-only when
    there are no objects (n_obj == 0): torch.vmap can't map over a zero-length object axis, so
    the empty proximity tensors are built directly instead of via vmap/jacrev."""
    bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd = \
        forward_batched_part(params, init_heading, init_x, init_y, state, dstate, bodies)
    n_obj, mb = obj_pose.shape[0], params.max_bodies
    if n_obj == 0:
        prox_now = state.new_full((mb, 0), 1e3)
        prox_J_state = state.new_zeros((mb, 0, state.shape[0]))
        prox_J_pose = state.new_zeros((mb, 0, 0, 3))
    else:
        prox_now = proximity_measure(params, state, obj_pose, bodies)
        prox_J_state, prox_J_pose = proximity_jacobians(params, state, obj_pose, bodies)
    return bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd, prox_now, prox_J_state, prox_J_pose


# ------------------------------------------------------------------ solve
def dynamic_solve(params, bodies, dstate, obj_dstate, forces, growth, sdf_now, dev_now, L, J, gws, gwd,
                  prox_now, prox_J_state, prox_J_pose, ow_now, ow_J, oo_now, oo_J, obj_mu=None, tangents=None,
                  obj_spin=None, env_t=None, wall_mu=None, box_mu=None):
    B = forces.shape[0]
    dt = params.dt
    mb = params.max_bodies
    Nv = mb * 3
    n_obj = int(params.obj_mass.shape[0])
    No = n_obj * 3
    N = Nv + No

    def pad_vine(G_obj):                          # (B, rows, No) -> (B, rows, N)
        return torch.cat([torch.zeros(B, G_obj.shape[1], Nv), G_obj], dim=2)

    # objective 0.5 v'Mv + p'v
    Mv = create_M(params.m.abs(), params.I.abs(), mb)
    Mo = create_M_obj(params.obj_mass.abs(), params.obj_hw, params.obj_hh)
    M = torch.block_diag(Mv, Mo)
    p = torch.cat([forces * dt - torch.matmul(dstate, Mv),
                   -params.obj_damp * torch.matmul(obj_dstate.reshape(B, No), Mo)], dim=1)

    Gs, hs = [], []
    # vine self-collision sdf (object cols zero)
    Gs.append(torch.cat([-L * dt, torch.zeros(B, sdf_now.shape[1], No)], dim=2))
    hs.append(sdf_now)
    # two-sided vine<->object proximity. prox_J_pose reshaped to No is the object-velocity
    # coupling directly (measure(j,o) depends only on pose[o], so only o's 3 slots are nonzero).
    Gp = torch.cat([-dt * prox_J_state, (-dt * prox_J_pose).reshape(B, mb, n_obj, No)], dim=3)
    if tangents is None:
        Gs.append(Gp.reshape(B, mb * n_obj, N))
        hs.append(prox_now.reshape(B, mb * n_obj))
    else:
        # Vine-object friction (Anitescu's convex model): the contact row  -dt n.v <= gap  becomes
        # -dt (n.v -+ mu t.v) <= gap. Its two multipliers l+, l- give a normal impulse l+ + l- and a
        # tangential one mu (l+ - l-), i.e. exactly the Coulomb cone |f_t| <= mu f_n, with no
        # extra variables. (Sliding contacts separate by ~mu |v_t| dt per step, a fraction of a mm.)
        T_state, T_pose = tangents
        Gt = torch.cat([dt * T_state, (dt * T_pose).reshape(B, mb, n_obj, No)], dim=3)
        Gs += [(Gp + Gt).reshape(B, mb * n_obj, N), (Gp - Gt).reshape(B, mb * n_obj, N)]
        hs += [prox_now.reshape(B, mb * n_obj)] * 2
    # object<->wall (one-sided). With wall friction, each row splits in two like the vine-object rows
    # above (Anitescu): -dt (n.v -+ mu t.v) <= gap, env_t's rows being t.v (env_tangents).
    n_walls = ow_now.shape[2]
    if n_walls > 0:
        Gw = pad_vine((-dt * ow_J).reshape(B, n_obj * n_walls, No))
        hw_ = ow_now.reshape(B, n_obj * n_walls)
        if env_t is not None and wall_mu is not None:
            Gt = pad_vine(dt * wall_mu.reshape(-1, 1, 1) * env_t[0])
            Gs += [Gw + Gt, Gw - Gt]
            hs += [hw_] * 2
        else:
            Gs.append(Gw)
            hs.append(hw_)
    # object<->object (two-sided); mask self-pairs (i==j) to slack; box_mu: the same friction split
    Goo = (-dt * oo_J).reshape(B, n_obj, n_obj, No)
    self_mask = torch.eye(n_obj, dtype=torch.bool)[None, :, :, None]
    Goo = torch.where(self_mask, torch.zeros_like(Goo), Goo).reshape(B, n_obj * n_obj, No)
    hoo = torch.where(torch.eye(n_obj, dtype=torch.bool)[None], torch.full_like(oo_now, 1e3), oo_now)
    hoo = hoo.reshape(B, n_obj * n_obj)
    if env_t is not None and box_mu is not None:
        Gt = pad_vine(dt * box_mu.reshape(-1, 1, 1) * env_t[1])
        Gs += [pad_vine(Goo) + Gt, pad_vine(Goo) - Gt]
        hs += [hoo] * 2
    else:
        Gs.append(pad_vine(Goo))
        hs.append(hoo)

    G = torch.cat(Gs, dim=1)
    h = torch.cat(hs, dim=1)

    # equalities A v == b (vine only; object cols zero)
    A_joint = torch.cat([J * dt, torch.zeros(B, J.shape[1], No)], dim=2)
    # Reformulate the max_bodies padding so A is full ROW-rank in EVERY batch element
    bodies_i = bodies.reshape(B, 1).long()
    empty = A_joint.abs().amax(-1) < 1e-12                 # (B, 2*mb) all-zero joint rows
    rank = torch.cumsum(empty.long(), dim=1) - 1           # k-th empty row of the element (0-indexed)
    pin_var = 3 * bodies_i + rank                          # -> the k-th inactive vine velocity var
    valid = empty & (pin_var < Nv)                         # (rare) if the vine fills max_bodies, leave
    # (dense one-hot, not index lists: no data-dependent shapes, so the step can be a CUDA graph)
    pin = torch.nn.functional.one_hot(pin_var.clamp(max=Nv - 1), A_joint.shape[2]).to(A_joint.dtype)
    A_joint = torch.where(valid.unsqueeze(-1), pin, A_joint)
    g_con = (growth.squeeze(1) - GROW_FACTOR * params.grow_rate -
             torch.bmm(gwd, dstate.unsqueeze(2)).squeeze(2).squeeze(1))
    A_growth = torch.cat([gws * dt + gwd, torch.zeros(B, 1, No)], dim=2)
    A = torch.cat([A_joint, A_growth], dim=1)
    b = torch.cat([-dev_now, -g_con.unsqueeze(1)], dim=1)

    if params.grow_force is not None:
        # Growth as a force, not a prescribed speed. One extra variable s = how far the tip link's
        # lengthening speed falls short of the target (growth row becomes  growth(v) + s = target),
        # charged 1/2 * c * s^2. The multiplier of that row, c * s, is the growth impulse: zero in
        # free growth, rising linearly with the shortfall, and dt * grow_force when the tip is
        # stopped dead. So a blocked vine pushes with a bounded force instead of making the QP
        # infeasible, and whatever gives first (a box, a bend, the vine itself) moves.
        c = dt * params.grow_force / (GROW_FACTOR * params.grow_rate)
        M = torch.block_diag(M, c.reshape(1, 1))
        p = torch.cat([p, torch.zeros(B, 1)], dim=1)
        G = torch.cat([G, torch.zeros(B, G.shape[1], 1)], dim=2)
        slack_col = torch.zeros(B, A.shape[1], 1)
        slack_col[:, -1] = 1.0                                 # the growth row is the last equality
        A = torch.cat([A, slack_col], dim=2)

    init_layers(M.shape[0], M.shape, p.shape[1:], G.shape[1:], h.shape[1:], A.shape[1:], b.shape[1:],
                n_obj=n_obj, obj_start=Nv)
    sol = solve_layers(M, p, G, h, A, b, fric=object_friction_weights(params, obj_mu, obj_spin) if n_obj > 0 else None)
    return sol[:, :N]


class _Overridden:
    """params with some attributes replaced (one batch element's per_batch values). Used instead of
    copy.copy + setattr, which torch.compile cannot trace."""

    def __init__(self, base, overrides):
        self._base, self._overrides = base, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def forward_all(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose, per_batch):
    """Everything a step needs before the QP, for the whole batch: the vine measures and Jacobians,
    the vine-object proximities, the object-wall/object separations and (with vine friction) the
    contact tangents. Returns the 16 tensors of dynamic_solve's inputs that depend on the state,
    then the tangents (or None)."""
    n_obj = int(params.obj_mass.shape[0])
    names = list(per_batch)

    def forward_one(ih, ix, iy, s, ds, b, op, *own):
        # this element's own values for the per-batch parameters
        p = _Overridden(params, dict(zip(names, own))) if names else params
        return dynamic_forward_part(p, ih, ix, iy, s, ds, b, op)

    out = torch.func.vmap(forward_one)(init_heading, init_x, init_y, state, dstate, bodies, obj_pose,
                                       *[per_batch[name] for name in names])
    env = torch.func.vmap(lambda op: object_env_constraints(params, op))(obj_pose)
    tangents = None
    if n_obj > 0 and float(params.vine_friction) > 0:     # (per-element values may go in per_batch)
        def tangents_one(s, b, op, *own):
            return proximity_tangents(_Overridden(params, dict(zip(names, own))), s, op, b)
        tangents = torch.func.vmap(tangents_one)(state, bodies, obj_pose, *[per_batch[name] for name in names])
    return (*out, *env, tangents)


FORWARD = forward_all   # what step() calls; may be replaced by a compiled version (xnew_rrt/compiled_forward.py)


def step(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose=None, obj_dstate=None,
         return_ok=False, per_batch=None):
    """One full physics step for the vine plus any movable objects -- the sim's single entry point.
    obj_pose is (B, n_obj, 3) = (cx, cy, theta); obj_dstate is (B, n_obj, 3) = (vx, vy, vtheta).
    Both default to the params' object count at zero pose/velocity, so a vine with NO objects is
    just step(params, ih, ix, iy, state, dstate, bodies) and everything downstream degrades to the
    vine-only QP. Returns (new_state, new_dstate, bodies, new_obj_pose, new_obj_dstate), plus an
    `ok` (B,) bool tensor when return_ok is set (False = that element's QP failed and it was frozen).
    The inputs are NOT modified (growth promotion happens on a private copy).
    per_batch: optional {VineParams attribute name: (B, ...) tensor} of parameters that differ
    between batch elements (e.g. bend_target, spam_moment, obj_friction); everything else in params
    is shared."""
    B = state.shape[0]
    # extend() promotes the tip body in place; without the copy that rewrites the caller's tensors
    # while their `bodies` count stays stale (corrupts any state kept around, e.g. planner tree nodes).
    state, dstate = state.clone(), dstate.clone()
    n_obj = int(params.obj_mass.shape[0])
    if obj_pose is None:
        obj_pose = torch.zeros(B, n_obj, 3)
    if obj_dstate is None:
        obj_dstate = torch.zeros(B, n_obj, 3)

    bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd, prox_now, pjs, pjp, ow_now, ow_J, oo_now, oo_J, \
        tangents = FORWARD(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose, per_batch or {})
    qp_inputs = (bodies, dstate, obj_dstate, forces, growth, sdf_now, dev_now,
                 L, J, gws, gwd, prox_now, pjs, pjp, ow_now, ow_J, oo_now, oo_J)
    # the box friction enters the QP's objective, outside the per-element part above
    mu = per_batch["obj_friction"].reshape(B, 1) if per_batch and "obj_friction" in per_batch else None
    spin = per_batch["obj_spin"].reshape(B, 1) if per_batch and "obj_spin" in per_batch else None
    # object-wall and object-object friction (rows only when the coefficient is > 0; per-element values
    # may go in per_batch, with params' value > 0 to build the rows)
    env_t, wall_mu, box_mu = None, None, None
    if n_obj > 0 and (float(params.wall_friction) > 0 or float(params.box_friction) > 0):
        env_t = env_tangents(params, obj_pose, ow_J, oo_J)
        pb = per_batch or {}
        own = lambda name: pb[name].reshape(B, 1) if name in pb else torch.full((B, 1), float(getattr(params, name)))
        wall_mu = own("wall_friction") if float(params.wall_friction) > 0 else None
        box_mu = own("box_friction") if float(params.box_friction) > 0 else None
    ok = torch.ones(B, dtype=torch.bool)
    try:
        sol = dynamic_solve(params, *qp_inputs, obj_mu=mu, tangents=tangents, obj_spin=spin,
                            env_t=env_t, wall_mu=wall_mu, box_mu=box_mu)
    except Exception:
        # QP infeasible (vine fully blocked, hard growth equality unmeetable): freeze this step
        # rather than crash, so a planner rollout survives. (A softer growth model would buckle.)
        # The batched solve fails as a whole, so re-solve one by one and freeze only the bad elements.
        sol = torch.zeros(B, params.max_bodies * 3 + n_obj * 3)
        ok[:] = False
        if B > 1:
            for i in range(B):
                try:
                    sol[i] = dynamic_solve(params, *[t[i:i + 1] for t in qp_inputs],
                                           obj_mu=None if mu is None else mu[i:i + 1],
                                           obj_spin=None if spin is None else spin[i:i + 1],
                                           tangents=None if tangents is None else
                                           tuple(t[i:i + 1] for t in tangents),
                                           env_t=None if env_t is None else tuple(t[i:i + 1] for t in env_t),
                                           wall_mu=None if wall_mu is None else wall_mu[i:i + 1],
                                           box_mu=None if box_mu is None else box_mu[i:i + 1])[0]
                    ok[i] = True
                except Exception:
                    pass

    Nv = params.max_bodies * 3
    # Cap vine velocities: fast FREE growth (no contact to regularize) drives the sliding tip
    # joint unstable above ~1.5x the growth target; clamping to a growth-relative bound catches
    # that spike without limiting normal growth (same technique as DiffVine's solve_layers_new).
    v_vine = sol[:, :Nv].clamp(-params.vine_vel_cap, params.vine_vel_cap)
    v_obj = sol[:, Nv:].reshape(B, -1, 3)                         # (B, n_obj, 3) = (vx, vy, vtheta)
    # Clamp linear (vx,vy) and angular (vtheta) velocities SEPARATELY -- they have different
    # units/scales, and the promotion spike must be capped for both (a shared cap of ~600 would
    # allow ~600 rad/s of spin). These caps only bite on spikes; normal contact stays well under.
    vc, va = params.obj_vel_cap, params.obj_ang_vel_cap           # linear, angular (rad/s)
    v_obj = torch.cat([v_obj[..., :2].clamp(-vc, vc), v_obj[..., 2:3].clamp(-va, va)], dim=-1)
    new_state = state + v_vine * params.dt
    new_pose = obj_pose + v_obj * params.dt                       # integrate all 3 DOFs (incl. theta)
    out = (new_state, v_vine, bodies, new_pose, v_obj)
    return (*out, ok) if return_ok else out
