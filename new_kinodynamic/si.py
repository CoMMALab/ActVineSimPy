"""SI-unit interface for dvsim, with internal non-dimensionalization.

WHY: conic QP solvers (SCS) are best-conditioned when the data is O(1). In true SI, rotational
inertia ~ mass*length^2 becomes tiny for cm-scale bodies (a ~1e4x spread in the mass matrix),
which the solver handles poorly. So internally the sim runs in a NON-DIMENSIONAL unit system
defined by characteristic scales (L0, M0, T0); this module lets you specify parameters, geometry
and state in SI (m, kg, s, N, Pa) -- what you want for system identification and for the sPAM
force model -- and converts to/from the internal units.

It also ABSORBS the DiffVine base's ad-hoc scaling constants (grow_rate*1000, I*100,
stiffness*100000, damping*100) into the stored non-dim parameter values, so the proven internal
solver is used UNCHANGED. Defaults reproduce the DiffVine base's proven-stable regime, expressed
in SI -- system identification will replace the placeholder values with fitted ones.
"""
import torch
from .vine import VineParams, GROW_FACTOR, I_FACTOR, DAMP_FACTOR

# ---- characteristic scales (SI). internal (non-dim) value = SI value / scale ----
L0 = 1.0e-3     # length : 1 mm   (keeps internal lengths O(1..1e3): the DiffVine proven range)
M0 = 1.0        # mass   : 1 kg
T0 = 1.0        # time   : 1 s

V0 = L0 / T0                # velocity  (m/s)
F0 = M0 * L0 / T0 ** 2      # force     (N)
TAU0 = M0 * L0 ** 2 / T0 ** 2   # torque (N*m)
J0 = M0 * L0 ** 2           # moment of inertia (kg*m^2)

# DiffVine base's baked-in ad-hoc unit constants (GROW_FACTOR, I_FACTOR, DAMP_FACTOR) are defined
# ONCE in vine.py and imported here, so the SI<->non-dim conversions below can never drift out of
# sync with the values the solver actually uses.


# ---- SI <-> non-dimensional converters ----
def len_to_nd(x):   return x / L0
def len_to_si(x):   return x * L0
def vel_to_nd(v):   return v / V0
def vel_to_si(v):   return v * V0
def force_to_nd(f): return f / F0
def torque_to_nd(t):return t / TAU0


def vine_params_si(max_bodies=40, obstacles_m=None,
                   radius_m=0.0125, seg_len_m=0.018, seg_mass_kg=0.02, seg_inertia_kgm2=1.0e-5,
                   grow_rate_mps=0.3, ang_damp=5.0e-5, lin_damp=0.10, dt_s=1.0 / 90,
                   stiffness_mode='linear'):
    """Build a VineParams from SI quantities. The returned params carry the internal non-dim
    values used by the (unchanged) solver; `.si` records the scales for downstream conversion.

    Defaults are the SI equivalent of the DiffVine proven-stable config (radius 12.5mm, 18mm
    segment, 20g, 0.3 m/s growth) -- placeholders until system identification.
    """
    # default "no obstacles": one tiny box far out of reach (10 m) -- kept at a modest internal
    # magnitude so it doesn't blow up the QP's constraint scaling (a 1e6 m dummy -> 1e9 internal did).
    obs_nd = [_obstacle_to_nd(o) for o in obstacles_m]
    p = VineParams(max_bodies=max_bodies, obstacles=obs_nd,
                   grow_rate=grow_rate_mps * T0 / (L0 * GROW_FACTOR),
                   stiffness_mode=stiffness_mode)
    # override the DiffVine-hardcoded geometry/inertia with the SI-derived non-dim values
    p.dt = dt_s / T0
    p.radius = radius_m / L0
    p.half_len = torch.tensor(seg_len_m / (2.0 * L0), dtype=torch.float32)
    p.m = torch.tensor([seg_mass_kg / M0], dtype=torch.float32)
    p.I = torch.tensor([seg_inertia_kgm2 / (J0 * I_FACTOR)], dtype=torch.float32)
    p.damping = torch.tensor(ang_damp * T0 / (J0 * DAMP_FACTOR), dtype=torch.float32)
    p.vel_damping = torch.tensor(lin_damp * T0 / M0, dtype=torch.float32)
    p.si = dict(L0=L0, M0=M0, T0=T0, V0=V0, F0=F0, TAU0=TAU0, J0=J0)
    return p


def len_to_mm(x):   return x * L0 * 1000.0   # internal length -> millimeters (for display)


def _obstacle_to_nd(o):
    """Convert one SI obstacle spec to internal non-dim units. Accepts an axis-aligned box
    [x1,y1,x2,y2] (meters) or an oriented box [cx,cy,theta,hw,hh] (meters, radians for theta --
    the angle is dimensionless and is NOT scaled by L0)."""
    if len(o) == 5:
        cx, cy, th, hw, hh = o
        return [cx / L0, cy / L0, th, hw / L0, hh / L0]

    return [c / L0 for c in o]   # AABB: every coordinate is a length


def obstacles_si_to_nd(obstacles_m):
    """Convert a list of SI obstacles (AABB [x1,y1,x2,y2] or OBB [cx,cy,theta,hw,hh]) to
    internal non-dim units."""
    return [_obstacle_to_nd(o) for o in obstacles_m]


def set_objects_si(params, aabbs_m, masses_kg):
    """Configure movable objects from SI. `aabbs_m`: list of [x1,y1,x2,y2] in meters;
    `masses_kg`: list of masses in kg. Sets params.obj_hw / obj_hh / obj_mass (internal non-dim;
    inertia is derived physically downstream) and returns the initial obj_pose
    (internal, shape (1, n_obj, 3) = (cx, cy, theta=0))."""
    aabb = torch.tensor(aabbs_m, dtype=torch.float32) / L0          # meters -> internal
    n = aabb.shape[0]
    params.obj_hw = (aabb[:, 2] - aabb[:, 0]) / 2
    params.obj_hh = (aabb[:, 3] - aabb[:, 1]) / 2
    params.obj_mass = torch.tensor([m / M0 for m in masses_kg], dtype=torch.float32)
    pose = torch.stack([(aabb[:, 0] + aabb[:, 2]) / 2,
                        (aabb[:, 1] + aabb[:, 3]) / 2,
                        torch.zeros(n)], dim=1)[None]
    return pose


def spam_moment_to_nd():
    """Units scale to feed the sPAM force model (which is SI) into the non-dim bending:
        params.bend_length_scale = segment_length_m   (sPAM's bend-radius input is in METERS)
        params.spam_moment_scale = spam_moment_to_nd() (SI N*m -> internal non-dim torque)
    This is the correct DIMENSIONAL conversion, so sPAM 'drops in' with no fudge. HOWEVER, whether
    the resulting bend MAGNITUDE is physical depends on the vine's stiffness/inertia being on the
    same SI footing -- which only holds once they are fit by system identification. Pre-sysid the
    vine still carries DiffVine's fitted (non-SI-consistent) inertia, so the raw sPAM force will
    over/under-power it; use an empirical spam_moment_scale for visualization until sysid provides
    matched SI parameters, then switch to this."""
    return 1.0 / TAU0
