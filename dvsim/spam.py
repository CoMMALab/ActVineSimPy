"""sPAM actuation for the dvsim vine.

The actuator's moment on a joint is CONSTANT for a given design: it is evaluated once, at the
contraction the actuator has at its design bend radius, and does not change with the joint's
current angle. What changes with the angle is the vine's own bending moment (the linear spring
in vine.bending_energy). The joint settles where the two cancel.

    eps      = (2 R_vine + R_act) / (design_radius + R_vine)     contraction at the design radius
    phi, m   = MLP(eps, |l0|)                                     trained surrogate of the sPAM model
    F_t      = p * pi * R_c^2 * (1 - 2m) / (2m cos^2(phi))        actuator tensile force
    M_act    = sign(l0) * F_t * (2 R_vine + R_act)                moment about the far wall of the vine

F_t and eps are the same expressions master uses (sPAM/nns_usage.py). The difference from master's
solve_fwd is WHERE they are evaluated: master re-evaluates them at the joint's current radius every
step and subtracts a constant vine force; here the actuator is constant and the vine does the varying.
"""
import math
import torch

_predict = None
_act_params = None

# rest lengths master's inverse model searches over [m]
L0_CANDIDATES = [0.020 + 0.001 * i for i in range(23)]
P_ACT_MAX = 35e3   # Pa, the pressure limit master's inverse model uses


def _load():
    global _predict, _act_params
    if _predict is None:
        from sPAM.torch_nns import get_or_train_model, get_prediction_function
        from sPAM.torch_spam import params as act_params
        scaling_info, model = get_or_train_model(act_params)
        _predict = get_prediction_function(scaling_info, model)
        _act_params = act_params
    return _predict, _act_params


def moment_per_pressure(l0, design_radius):
    """Actuator moment per unit pressure [N*m / Pa] for rest length l0 [m] at a design bend radius [m]
    (both taken as magnitudes). Negative means the surrogate is outside its valid range (m > 0.5)."""
    predict, ap = _load()
    eps = (2 * ap.R_beam + ap.R_act_max) / (abs(float(design_radius)) + ap.R_beam)
    phi, m = predict(torch.tensor([eps, abs(float(l0))], dtype=torch.float32))
    force_per_pressure = math.pi * ap.R_c ** 2 * (1 - 2 * m) / (2 * m * torch.cos(phi) ** 2)
    return float(force_per_pressure) * (2 * ap.R_beam + ap.R_act_max)


def actuation_moment(p, l0, design_radius):
    """Constant moment [N*m] of an actuator with pressure p [Pa] and rest length l0 [m] (its sign is
    the curl direction), designed for |design_radius| [m]."""
    return math.copysign(1.0, l0) * float(p) * moment_per_pressure(l0, design_radius)


def design_actuator(moment, design_radius):
    """Inverse: (p [Pa], l0 [m]) of the actuator that gives `moment` [N*m] (signed) at a design bend
    radius. Picks the rest length that needs the least pressure. Raises if no rest length works."""
    per_pressure, l0 = max((moment_per_pressure(c, design_radius), c) for c in L0_CANDIDATES)
    if per_pressure <= 0:
        raise ValueError(f"no valid sPAM rest length for a {design_radius} m bend radius")
    return abs(moment) / per_pressure, math.copysign(l0, moment)
