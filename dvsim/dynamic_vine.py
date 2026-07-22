"""Movable-obstacle contact on top of the stable dvsim vine base — ORIENTED boxes.

Reuses DiffVine's vine physics (forward_batched_part) untouched, and adds to the QP:
  - object pose DOFs (cx, cy, theta) with mass + rotational inertia
  - TWO-SIDED vine<->object contact (the vine can translate AND rotate objects)
  - object<->static-wall contact (one-sided) and object<->object contact (two-sided)

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
from functools import partial
from .vine import StateTensor, create_M, forward_batched_part, point_obb_gap, obb_obb_sep, box_inertia
from .solver import init_layers, solve_layers


# ------------------------------------------------------------------ object mass
def create_M_obj(obj_mass, obj_hw, obj_hh):
    """Block-diagonal object mass matrix [m, m, I] per object, with the rotational inertia I
    DERIVED PHYSICALLY from the object's mass and size (uniform rectangle) -- so bigger/heavier
    boxes resist rotation correctly. No hand-set inertia, no magic factor: I/m = (hw^2+hh^2)/3
    is the true physical ratio. Same [m, m] scale as the vine's create_M (units are consistent)."""
    per = torch.stack([obj_mass, obj_mass, box_inertia(obj_mass, obj_hw, obj_hh)], dim=1)
    return torch.diag(per.reshape(-1))


# ------------------------------------------------------------------ measures
def proximity_measure(params, state, obj_pose, bodies):
    """Per (body, object) gap between the vine body circle and the oriented box.
    Phantom bodies (>= bodies) forced separated so they never bind."""
    st = StateTensor(state)
    cx, cy = st.x, st.y

    def gap_one(pose, hw, hh):
        return torch.vmap(lambda px, py: point_obb_gap(px, py, pose, hw, hh, params.radius))(cx, cy)

    gaps = torch.vmap(gap_one)(obj_pose, params.obj_hw, params.obj_hh).transpose(0, 1)   # (max_bodies, n_obj)
    idx = torch.arange(gaps.shape[0]).unsqueeze(-1)
    return torch.where(idx >= bodies, torch.full_like(gaps, 1e3), gaps)


def object_env_constraints(params, obj_pose):
    """Per-batch object<->wall and object<->object separations + Jacobians w.r.t. pose.
    Walls are the same oriented boxes the vine collides with (params.obstacle_*)."""
    wall_pose, wall_hw, wall_hh = params.obstacle_pose, params.obstacle_hw, params.obstacle_hh

    def owall(pose):                                               # (n_obj, n_walls)
        def one(p, hw, hh):
            return torch.vmap(lambda wp, whw, whh: obb_obb_sep(p, hw, hh, wp, whw, whh))(wall_pose, wall_hw, wall_hh)
        return torch.vmap(one)(pose, params.obj_hw, params.obj_hh)

    def oobj(pose):                                                # (n_obj, n_obj)
        def one(pA, hwA, hhA):
            return torch.vmap(lambda pB, hwB, hhB: obb_obb_sep(pA, hwA, hhA, pB, hwB, hhB))(pose, params.obj_hw, params.obj_hh)
        return torch.vmap(one)(pose, params.obj_hw, params.obj_hh)

    ow_now = owall(obj_pose); ow_J = torch.func.jacrev(owall)(obj_pose)   # (n_obj, n_walls, n_obj, 3)
    oo_now = oobj(obj_pose); oo_J = torch.func.jacrev(oobj)(obj_pose)     # (n_obj, n_obj, n_obj, 3)
    return ow_now, ow_J, oo_now, oo_J


def dynamic_forward_part(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose):
    """Per-batch vine measures + vine<->object proximity measures/Jacobians (jacrev w.r.t. pose
    gives the object velocity coupling directly, torque included)."""
    bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd = \
        forward_batched_part(params, init_heading, init_x, init_y, state, dstate, bodies)
    prox_now = proximity_measure(params, state, obj_pose, bodies)
    prox_J_state = torch.func.jacrev(lambda s: proximity_measure(params, s, obj_pose, bodies))(state)
    prox_J_pose = torch.func.jacrev(lambda o: proximity_measure(params, state, o, bodies))(obj_pose)
    return bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd, prox_now, prox_J_state, prox_J_pose


# ------------------------------------------------------------------ solve
def dynamic_solve(params, dstate, obj_dstate, forces, growth, sdf_now, dev_now, L, J, gws, gwd,
                  prox_now, prox_J_state, prox_J_pose, ow_now, ow_J, oo_now, oo_J):
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
    obj_damp = getattr(params, 'obj_damp', 0.2)   # viscous friction (free objects decay to rest)
    p = torch.cat([forces * dt - torch.matmul(dstate, Mv),
                   -obj_damp * torch.matmul(obj_dstate.reshape(B, No), Mo)], dim=1)

    Gs, hs = [], []
    # vine self-collision sdf (object cols zero)
    Gs.append(torch.cat([-L * dt, torch.zeros(B, sdf_now.shape[1], No)], dim=2))
    hs.append(sdf_now)
    # two-sided vine<->object proximity. prox_J_pose reshaped to No is the object-velocity
    # coupling directly (measure(j,o) depends only on pose[o], so only o's 3 slots are nonzero).
    Gp = torch.cat([-dt * prox_J_state, (-dt * prox_J_pose).reshape(B, mb, n_obj, No)], dim=3)
    Gs.append(Gp.reshape(B, mb * n_obj, N))
    hs.append(prox_now.reshape(B, mb * n_obj))
    # object<->wall (one-sided)
    n_walls = ow_now.shape[2]
    if n_walls > 0:
        Gs.append(pad_vine((-dt * ow_J).reshape(B, n_obj * n_walls, No)))
        hs.append(ow_now.reshape(B, n_obj * n_walls))
    # object<->object (two-sided); mask self-pairs (i==j) to slack
    Goo = (-dt * oo_J).reshape(B, n_obj, n_obj, No)
    self_mask = torch.eye(n_obj, dtype=torch.bool)[None, :, :, None]
    Goo = torch.where(self_mask, torch.zeros_like(Goo), Goo).reshape(B, n_obj * n_obj, No)
    hoo = torch.where(torch.eye(n_obj, dtype=torch.bool)[None], torch.full_like(oo_now, 1e3), oo_now)
    Gs.append(pad_vine(Goo))
    hs.append(hoo.reshape(B, n_obj * n_obj))

    G = torch.cat(Gs, dim=1)
    h = torch.cat(hs, dim=1)

    # equalities A v == b (vine only; object cols zero)
    A_joint = torch.cat([J * dt, torch.zeros(B, J.shape[1], No)], dim=2)
    g_con = (growth.squeeze(1) - 1000 * params.grow_rate -
             torch.bmm(gwd, dstate.unsqueeze(2)).squeeze(2).squeeze(1))
    A_growth = torch.cat([gws * dt + gwd, torch.zeros(B, 1, No)], dim=2)
    A = torch.cat([A_joint, A_growth], dim=1)
    b = torch.cat([-dev_now, -g_con.unsqueeze(1)], dim=1)

    init_layers(N, M.shape, p.shape[1:], G.shape[1:], h.shape[1:], A.shape[1:], b.shape[1:])
    return solve_layers(M, p, G, h, A, b)


def dynamic_step(params, init_heading, init_x, init_y, state, dstate, bodies, obj_pose, obj_dstate):
    """One full step. Returns (new_state, new_dstate, bodies, new_obj_pose, new_obj_dstate).
    obj_pose is (B, n_obj, 3) = (cx, cy, theta); obj_dstate is (B, n_obj, 3) = (vx, vy, vtheta)."""
    fwd = torch.func.vmap(partial(dynamic_forward_part, params), in_dims=(0, 0, 0, 0, 0, 0, 0))
    bodies, forces, growth, sdf_now, dev_now, L, J, gws, gwd, prox_now, pjs, pjp = \
        fwd(init_heading, init_x, init_y, state, dstate, bodies, obj_pose)
    ow_now, ow_J, oo_now, oo_J = torch.func.vmap(partial(object_env_constraints, params))(obj_pose)
    try:
        sol = dynamic_solve(params, dstate, obj_dstate, forces, growth, sdf_now, dev_now,
                            L, J, gws, gwd, prox_now, pjs, pjp, ow_now, ow_J, oo_now, oo_J)
    except Exception:
        # QP infeasible (vine fully blocked, hard growth equality unmeetable): freeze this step
        # rather than crash, so a planner rollout survives. (A softer growth model would buckle.)
        return state, torch.zeros_like(dstate), bodies, obj_pose, torch.zeros_like(obj_dstate)

    B = sol.shape[0]
    Nv = params.max_bodies * 3
    # Cap vine velocities: fast FREE growth (no contact to regularize) drives the sliding tip
    # joint unstable above ~1.5x the growth target; clamping to a growth-relative bound catches
    # that spike without limiting normal growth (same technique as DiffVine's solve_layers_new).
    vvc = getattr(params, 'vine_vel_cap', 1.5 * 1000.0 * float(params.grow_rate))
    v_vine = sol[:, :Nv].clamp(-vvc, vvc)
    v_obj = sol[:, Nv:].reshape(B, -1, 3)                         # (B, n_obj, 3) = (vx, vy, vtheta)
    # Clamp linear (vx,vy) and angular (vtheta) velocities SEPARATELY -- they have different
    # units/scales, and the promotion spike must be capped for both (a shared cap of ~600 would
    # allow ~600 rad/s of spin). These caps only bite on spikes; normal contact stays well under.
    vc = getattr(params, 'obj_vel_cap', 2.0 * 1000.0 * float(params.grow_rate))   # linear
    va = getattr(params, 'obj_ang_vel_cap', 5.0)                                  # angular (rad/s)
    v_obj = torch.cat([v_obj[..., :2].clamp(-vc, vc), v_obj[..., 2:3].clamp(-va, va)], dim=-1)
    new_state = state + v_vine * params.dt
    new_pose = obj_pose + v_obj * params.dt                       # integrate all 3 DOFs (incl. theta)
    return new_state, v_vine, bodies, new_pose, v_obj
