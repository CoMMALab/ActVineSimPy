"""The wrinkling bending model of an inflated vine (Sicheng's, ported from DiffVine's 'real' mode,
~/vine/from_diffvine/vine.py predict_moment), in physical units.

A pressurized fabric tube bends like a linear beam only up to a small transition angle phiTrans(P);
past it the compressed side wrinkles and the moment saturates toward the Comer-Levy constant
moment pi P R^3. The empirical fits (cubics in the pressure P [Pa]) give phiTrans and the critical
wrinkling strain eps_crit; the moment is

    M(phi) = scale * alp(phi) * pi P R^3,   alp linear up to phiTrans, then the wrinkling formula

alp -> 1 for large bends, so a joint loaded past ~pi P R^3 has almost nothing left to resist with
and kinks: this is what makes the real vine buckle. DiffVine used P = R = 1 and a fitted scale;
here P is a pressure, R the vine radius, and `scale` takes up how one joint's angle relates to the
bend angle of the experiments the fits came from.
"""
import torch


def eps_crit(P):
    """Critical strain that starts wrinkling at pressure P [Pa]."""
    s = (P - 6167) / 5836
    return 0.002077863400343 * s ** 3 + 0.009091543113141 * s ** 2 + 0.014512785114617 * s + 0.007656015122415


def phi_trans(P):
    """Bend angle [rad] where the linear model hands over to the wrinkling one, at pressure P [Pa]."""
    s = (P - 8957) / 5149
    return 0.003180574067535 * s ** 3 + 0.020924128997619 * s ** 2 + 0.048366932757916 * s + 0.037544481890778


def _alp_wrinkled(phi, eps):
    the0 = torch.arccos((2 * eps / torch.sin(phi / 2) - 1).clamp(-1.0, 1.0))
    return (torch.sin(2 * the0) + 2 * torch.pi - 2 * the0) / \
        (4 * (torch.sin(the0) + torch.cos(the0) * (torch.pi - the0)))


def moment_fraction(phi, P):
    """alp: the bending moment at |bend angle| phi [rad] as a fraction of pi P R^3."""
    if not torch.is_tensor(P):          # (a fill, not a host-to-device copy: CUDA-graph safe)
        P = torch.full_like(phi, float(P))
    eps, pt = eps_crit(P), phi_trans(P)
    wrinkled = _alp_wrinkled(torch.maximum(phi, pt), eps)       # (evaluated at >= phiTrans only)
    linear = _alp_wrinkled(pt, eps) / pt * phi
    return torch.where(phi > pt, wrinkled, linear)


def wrinkle_moment(params, angle):
    """Restoring moment [internal torque] per joint for a signed bend `angle` [rad] away from rest.
    Uses params.pressure [Pa], params.wrinkle_scale and the vine radius."""
    L0, TAU0 = params.si["L0"], params.si["TAU0"]
    R = params.radius * L0                                       # [m]
    full = torch.pi * params.pressure * R ** 3                   # [N*m]
    return -angle.sign() * params.wrinkle_scale * full * moment_fraction(angle.abs(), params.pressure) / TAU0
