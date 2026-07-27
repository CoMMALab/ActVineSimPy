"""sPAM elastic + actuation bending moment for the dvsim vine (the ActVine design model).

Wraps the trained torch sPAM surrogate. `make_actuation_fn()` returns the paper's ~constant sPAM
actuation moment F_t*(2R_vine+R_act) (Gao et al. 2025, Eq. 1d); the curvature-dependent vine
wrinkling restoring moment (Eq. 2) lives in dvsim.vine.vine_wrinkle_moment, and bending_energy
combines them as Eq. 3b.
"""
import math
import torch

_predict = None
_act_params = None


def _load():
    global _predict, _act_params
    if _predict is None:
        from sPAM.torch_nns import get_or_train_model, get_prediction_function
        from sPAM.torch_spam import params as act_params
        scaling_info, model = get_or_train_model(act_params)
        _predict = get_prediction_function(scaling_info, model)
        _act_params = act_params
    return _predict, _act_params


def make_actuation_fn(eps_design=0.25):
    """Joint-vectorized sPAM ACTUATION moment M_act = F_t*(2R_vine+R_act) (paper Eq. 1d), signed by the
    actuator's curl direction (sign of l0). F_t = pi*P_act*Rc^2*(1-2m)/(2m*cos^2(phi_Rc)); m (contraction)
    comes from the trained surrogate. Crucially it is evaluated ONCE at a valid design strain
    `eps_design` (the surrogate is only valid for m<0.5, i.e. small strain) and is otherwise ~constant
    for a given pressure -- so it ignores the current joint curvature. The curvature dependence that
    makes the curl self-limiting is the vine wrinkling restoring moment (Eq. 2), applied separately in
    bending_energy (Eq. 3b: M_tot = M_act - M_vine(theta)). Returns a callable (turning_radius, p, l0)
    -> M_act; the turning_radius arg is accepted (bending_energy's moment-fn interface) but unused."""
    predict, ap = _load()
    Rc, Ract, Rvine = float(ap.R_c), float(ap.R_act_max), float(ap.R_beam)
    phi_rc = math.acos(Rc / Ract)                       # phi_Rc = acos(Rc/Ract), constant (Eq. 1c)
    arm = 2.0 * Rvine + Ract
    cos2 = math.cos(phi_rc) ** 2
    eps_t = torch.tensor(float(eps_design))

    def one(r, p, l0):                                  # r (turning_radius) ignored: F_t ~ constant
        m = predict(torch.stack([eps_t, l0.abs()]))[1]  # contraction from the surrogate at the design strain
        F_t = math.pi * p * Rc ** 2 * (1.0 - 2.0 * m) / (2.0 * m * cos2)
        return -torch.sign(l0) * F_t * arm              # actuation moment, +drives the l0<0 curl direction

    return torch.vmap(one, in_dims=(0, 0, 0))
