from sPAM.torch_ellip import F, E
from collections import namedtuple

import torch
from torchmin import minimize
from torchimize.functions import lsq_lma
import functools

vine_tube_width = 4.125 # in
vine_perimeter = vine_tube_width * 2 * 25.4e-3 # 2 * pi * r
vine_radius = vine_perimeter / (2 * 3.14159)

actuator_tube_width = 2.125 # in
actuator_perimeter = actuator_tube_width * 2 * 25.4e-3 # 2 * pi * r
r_act_max = actuator_perimeter / (2 * 3.14159)

paramstype = namedtuple('params', ['R_beam', 'R_act_max', 'R_c', 'P_beam', 'a', 'min_l_0', 'max_l_0'])
params = paramstype(R_beam=vine_radius, P_beam=6894.76 * 1.5, R_act_max=r_act_max, R_c=0.005, a=0.0001, min_l_0=0.02, max_l_0=0.080)
# P_beam=6894.76 * 1.5

def stack(*args):
    assert [arg.ndim == 0 for arg in args]
    return torch.stack(args, dim=-1)

def relu(x):
    return torch.maximum(torch.tensor(0), x)

def objective_solve_for_m_crit(m, phi_sat, l_0, R_c, a):
    """
    Objective function for solving for m_crit.
    Returns a scalar value to be minimized by an optimizer like BFGS.
    """
    # To avoid NaNs, add small epsilon to m in denominators
    m_safe = m + 1e-9

    # Residual from phi_m_l0_relation
    res = F(phi_sat, m) / (torch.sqrt(m_safe) * torch.cos(phi_sat)) - \
          l_0 / R_c * (1 + a / (2 * m_safe * torch.cos(phi_sat)**2))

    # Penalties for out-of-range values to guide the solver
    m_range_penalty = relu(m - (0.5 - 1e-3)) + relu(1e-3 - m)

    penalty_weight = 100.0

    return res**2 + penalty_weight * m_range_penalty


############### SECOND ONE ################

def objective_solve_for_phi_eps(vals, l_0, m, R, a):
    """
    Objective for unsaturated case: solve for phi and eps given l_0, m.
    """
    phi, eps = vals[0], vals[1]
    m_safe = m + 1e-6

    res1 = (E(phi, m) - 0.5 * F(phi, m)) / (torch.sqrt(m_safe) * torch.cos(phi)) - l_0 * (1 - eps) / (2 * R)
    res2 = F(phi, m) - l_0 / R * (torch.sqrt(m_safe) * torch.cos(phi) + a / (2 * torch.sqrt(m_safe) * torch.cos(phi)))

    phi_range_penalty = relu(phi - (torch.pi/2 - 1e-3)) + relu(1e-3 - phi)
    eps_range_penalty = relu(eps - (1.0 - 1e-3)) + relu(1e-6 - eps)

    penalty_weight = 100.0

    return torch.stack([res1, res2, penalty_weight * phi_range_penalty, penalty_weight * eps_range_penalty])

def objective_solve_for_eps_l_a(vals, l_0, m, phi_sat, R_c, a):
    """
    Objective for saturated case: solve for eps and l_a given l_0, m, phi_sat.
    """
    eps, l_a = vals[0], vals[1]
    m_safe = m + 1e-6
    l_a_safe = l_a + 1e-6

    eps_prime = l_0 / l_a_safe * eps

    res1 = (E(phi_sat, m) - 0.5 * F(phi_sat, m)) / (torch.sqrt(m_safe) * torch.cos(phi_sat)) - l_a / (2 * R_c) * (1 - eps_prime)
    res2 = F(phi_sat, m) - l_a / R_c * (torch.sqrt(m_safe) * torch.cos(phi_sat) + a / (2 * torch.sqrt(m_safe) * torch.cos(phi_sat)))

    l_a_range_penalty = relu(l_a - l_0) + relu(1e-3 - l_a)
    eps_range_penalty = relu(eps - (1.0 - 1e-3)) + relu(1e-6 - eps)

    penalty_weight = 100.0

    return torch.stack([res1, res2, penalty_weight * l_a_range_penalty, penalty_weight * eps_range_penalty])

def l_m_to_phi_eps(base_seed, l_0, m, params):
    """
    Given l_0 and m, solve for phi and eps (unsaturated) or eps and l_a (saturated).
    Returns: phi, eps, is_sat, info
    """

    # 1. Compute phi_sat and m_crit
    phi_sat = torch.arccos(torch.tensor(params.R_c / params.R_act_max))

    def objective_m_crit(m_val, phi_sat, l_0):
        return objective_solve_for_m_crit(m_val, phi_sat, l_0, params.R_c, params.a)

    loss_closure = functools.partial(objective_m_crit, phi_sat=phi_sat, l_0=l_0)
    opt_mcrit = minimize(loss_closure, torch.tensor(0.25), method='bfgs',
                         max_iter=100, tol=1e-4)
    m_crit = opt_mcrit.x

    def torch_split(seed):
        g_master = torch.Generator().manual_seed(seed)
        seed1 = torch.randint(0, 2**31, (1,), generator=g_master).item()
        seed2 = torch.randint(0, 2**31, (1,), generator=g_master).item()

        g1 = torch.Generator().manual_seed(seed1)
        g2 = torch.Generator().manual_seed(seed2)

        return g1, g2


    # 2. Branch on m < m_crit (unsaturated) or m >= m_crit (saturated)
    def unsat_branch(args):
        key, l_0, m = args

        # Initial guess: phi in (1e-3, phi_sat), eps in (1e-6, 0.5)
        num_samples = 500

        g_phi, g_eps = torch_split(base_seed)
        phi_samples = torch.empty(num_samples).uniform_(1e-3, phi_sat, generator=g_phi)
        eps_samples = torch.empty(num_samples).uniform_(1e-6, 0.5, generator=g_eps)

        def single_error(phi, eps):
            vals = stack(phi, eps)
            res = objective_solve_for_phi_eps(vals, l_0, m, params.R_c, params.a)
            return torch.sum(res[:2]**2)

        errors = []
        for eps_entry, phi_entry in zip(eps_samples, phi_samples):
            errors.append(single_error(eps_entry, phi_entry))

        errors = torch.stack(errors)

        mask = torch.isnan(errors)
        errors_filled = errors.masked_fill(mask, float('inf'))
        best_idx = torch.argmin(errors_filled)

        guess = stack(phi_samples[best_idx], eps_samples[best_idx])

        residual_fn = functools.partial(
            objective_solve_for_phi_eps, l_0=l_0, m=m, R=params.R_c, a=params.a
        )
        result = lsq_lma(guess, residual_fn, max_iter=100, gtol=1e-4, ptol=1e-4, ftol=1e-4)
        solution = result[-1]

        phi, eps = solution[0], solution[1]

        #NOTE: added with torch edition, as info has disappeared and can't inform error field
        final_residual = residual_fn(solution)
        final_error = torch.sum(final_residual[:2] ** 2)

        return phi, eps, False, {'m_crit': m_crit, 'error': final_error}

    def sat_branch(args):
        key, l_0, m = args
        # phi = phi_sat, solve for eps and l_a
        num_samples = 500

        g_eps, g_la = torch_split(base_seed)
        eps_samples = torch.empty(num_samples).uniform_(1e-6, 0.5, generator=g_eps)
        l_a_samples = torch.empty(num_samples).uniform_(1e-3, l_0, generator=g_la)


        def single_error(eps, l_a):
            vals = stack(eps, l_a)
            res = objective_solve_for_eps_l_a(vals, l_0, m, phi_sat, params.R_c, params.a)
            return torch.sum(res[:2]**2)

        errors = []
        for eps_entry, l_a_entry in zip(eps_samples, l_a_samples):
            errors.append(single_error(eps_entry, l_a_entry))

        errors = torch.stack(errors)

        mask = torch.isnan(errors)
        errors_filled = errors.masked_fill(mask, float('inf'))
        best_idx = torch.argmin(errors_filled)

        guess = stack(eps_samples[best_idx], l_a_samples[best_idx])

        residual_fn = functools.partial(objective_solve_for_eps_l_a,
                                        l_0=l_0, m=m, phi_sat = phi_sat, R_c=params.R_c, a=params.a)
        result = lsq_lma(guess, residual_fn, max_iter=100, ftol=1e-4, ptol=1e-4, gtol=1e-4,)
        solution = result[-1]
        eps, l_a = solution[0], solution[1]

        # actual eps = l_a / l_0
        eps_actual = l_a / l_0 * eps

        #NOTE: added with torch edition, as info has disappeared and can't inform error field
        final_residual = residual_fn(solution)
        final_error = torch.sum(final_residual[:2] ** 2)

        return phi_sat, eps_actual, True, {'m_crit': m_crit, 'error': final_error}

    key = "ignore" # no longer needed in torch version
    if m < m_crit:
        phi, eps, is_sat, info = unsat_branch(args = (key, l_0, m))
    else:
        phi, eps, is_sat, info = sat_branch(args = (key, l_0, m))

    return phi, eps, is_sat, info
