import jax
import jax.numpy as jnp
from sPAM.ellip import F, E
import jaxopt
from collections import namedtuple

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
    assert [jnp.ndim(arg) == 0 for arg in args]
    return jnp.stack(args, axis=-1)

def relu(x):
    return jnp.maximum(0, x)

def objective_solve_for_phi_m(vals, eps, R, a, l):
    """
    Objective function for solving the system of equations for phi_r and m.
    Returns residuals for a least-squares solver.
    """
    # jax.debug.print("Solving for phi_r and m with eps: {}, R: {}, a: {}, l: {}", eps, R, a, l)
    
    phi_r, m = vals[0], vals[1]

    # To avoid NaNs, add small epsilon to m in denominators
    m_safe = m + 1e-6

    # Residual from the strain-based equation (implicit_1 in ppam)
    res1 = (E(phi_r, m) - 0.5 * F(phi_r, m)) / (jnp.sqrt(m_safe) * jnp.cos(phi_r)) - \
           l * (1 - eps) / (2 * R)

    # Residual from the force-balance equation (implicit_3 in ppam)
    res2 = F(phi_r, m) - l / R * (jnp.sqrt(m_safe) * jnp.cos(phi_r) + a / (2 * jnp.sqrt(m_safe) * jnp.cos(phi_r)))

    # Penalties for out-of-range values to guide the solver
    phi_range_penalty = relu(phi_r - (jnp.pi/2 - 1e-3)) + relu(1e-3 - phi_r)
    m_range_penalty = relu(m - (0.5 - 1e-3)) + relu(1e-3 - m)
    
    penalty_weight = 100.0

    return jnp.stack([res1, res2, penalty_weight * phi_range_penalty, penalty_weight * m_range_penalty])

def objective_solve_for_l_a_m_sat(vals, l_0, eps, phi_sat, R_c, a):
    """
    Objective function for solving the system of equations for l_a and m_sat in the saturated case.
    Returns residuals for a least-squares solver.
    """
    l_a, m_sat = vals[0], vals[1]

    # To avoid NaNs, add small epsilon to denominators
    m_safe = m_sat + 1e-6
    l_a_safe = l_a + 1e-6

    eps_prime = l_0 / l_a_safe * eps

    # Residual from the strain-based equation
    res1 = (E(phi_sat, m_sat) - 0.5 * F(phi_sat, m_sat)) / (jnp.sqrt(m_safe) * jnp.cos(phi_sat)) - \
           l_a / (2 * R_c) * (1 - eps_prime)

    # Residual from the force-balance equation (stable form)
    res2 = F(phi_sat, m_sat) - l_a / R_c * (jnp.sqrt(m_safe) * jnp.cos(phi_sat) + a / (2 * jnp.sqrt(m_safe) * jnp.cos(phi_sat)))

    # Penalties for out-of-range values to guide the solver
    # l_a must be <= l_0 and > 0
    l_a_range_penalty = relu(l_a - l_0) + relu(1e-3 - l_a)
    m_range_penalty = relu(m_sat - (0.5 - 1e-3)) + relu(1e-3 - m_sat)
    
    penalty_weight = 100.0

    return jnp.stack([res1, res2, penalty_weight * l_a_range_penalty, penalty_weight * m_range_penalty])


def objective_solve_for_m_crit(m, phi_sat, l_0, R_c, a):
    """
    Objective function for solving for m_crit.
    Returns a scalar value to be minimized by an optimizer like BFGS.
    """
    # To avoid NaNs, add small epsilon to m in denominators
    m_safe = m + 1e-9

    # Residual from phi_m_l0_relation
    res = F(phi_sat, m) / (jnp.sqrt(m_safe) * jnp.cos(phi_sat)) - \
          l_0 / R_c * (1 + a / (2 * m_safe * jnp.cos(phi_sat)**2))

    # Penalties for out-of-range values to guide the solver
    m_range_penalty = relu(m - (0.5 - 1e-3)) + relu(1e-3 - m)
    
    penalty_weight = 100.0

    return res**2 + penalty_weight * m_range_penalty

    
def solve_inner(key, eps, l_0, params):
    '''
    radius is the radius we want the beam to bend by
    '''
    
    # Solve for actuator saturation point 
    phi_sat = jnp.arccos(params.R_c / params.R_act_max)
    
    # jax.debug.print("eps: {} phi_sat: {}", eps, phi_sat)
    
    # --- Find m_crit ---
    def objective_m_crit(m, phi_sat, l_0):
        return objective_solve_for_m_crit(m, phi_sat, l_0, params.R_c, params.a)
    
    opt_mcrit = jaxopt.BFGS(objective_m_crit, maxiter=100, tol=1e-4)
    
    def solve_m_crit_for_l0(l_0, phi_sat):
        m_crit, info = opt_mcrit.run(jnp.array(0.25), phi_sat=phi_sat, l_0=l_0)
        return m_crit

    m_crit = solve_m_crit_for_l0(l_0, phi_sat)

    # --- Solve for unsaturated case ---
    def solve_unsat(l_0, eps, key):
        # 1. Random Sampling for initial guesses
        num_samples = 1000
        key_phi, key_m = jax.random.split(key)
        
        phi_r_samples = jax.random.uniform(key_phi, (num_samples,), minval=1e-3, maxval=phi_sat)
        m_samples = jax.random.uniform(key_m, (num_samples,), minval=1e-3, maxval=0.5 - 1e-3)
        
        # 2. Error Calculation to find best initial guess
        def single_error(phi_r, m):
            vals = stack(phi_r, m)
            res = objective_solve_for_phi_m(vals, eps, params.R_c, params.a, l_0)
            return jnp.sum(res[:2]**2)
        
        errors = jax.vmap(single_error)(phi_r_samples, m_samples)
        best_idx = jnp.nanargmin(errors)
        guess = stack(phi_r_samples[best_idx], m_samples[best_idx])
        
        # 3. Optimization
        opt = jaxopt.LevenbergMarquardt(objective_solve_for_phi_m, maxiter=100, tol=1e-4)
        
        solution, info = opt.run(guess, l=l_0, eps=eps, R=params.R_c, a=params.a)
        error = jnp.sum(jnp.abs(objective_solve_for_phi_m(solution, eps, params.R_c, params.a, l_0)[:2]))
        return solution[0], solution[1], error

    # --- Solve for saturated case ---
    def objective_sat(vals, l_0, eps, phi_sat):
        return objective_solve_for_l_a_m_sat(vals, l_0, eps, phi_sat, params.R_c, params.a)

    def solve_sat(l_0, eps, phi_sat, key):
        # 1. Random Sampling for initial guesses
        num_samples = 1000
        key_la, key_m = jax.random.split(key)

        l_a_samples = jax.random.uniform(key_la, (num_samples,), minval=1e-3, maxval=l_0)
        m_sat_samples = jax.random.uniform(key_m, (num_samples,), minval=1e-3, maxval=0.5 - 1e-3)

        # 2. Error Calculation to find best initial guess
        def single_error(l_a, m_sat):
            vals = stack(l_a, m_sat)
            res = objective_sat(vals, l_0, eps, phi_sat)
            return jnp.sum(res[:2]**2)

        errors = jax.vmap(single_error)(l_a_samples, m_sat_samples)
        best_idx = jnp.nanargmin(errors)
        guess = stack(l_a_samples[best_idx], m_sat_samples[best_idx])

        # 3. Optimization
        opt = jaxopt.LevenbergMarquardt(objective_sat, maxiter=100, tol=1e-4)
        solution, info = opt.run(guess, l_0=l_0, eps=eps, phi_sat=phi_sat)
        error = jnp.sum(jnp.abs(objective_sat(solution, l_0, eps, phi_sat)[:2]))
        return solution[0], solution[1], error

    # --- Vectorize solvers over l_0 candidates ---
    key1, key2 = jax.random.split(key, 2)
    phi_unsat_cand, m_unsat_cand, err_unsat = solve_unsat(l_0, eps, key1)
    l_a_sat_cand, m_sat_cand, err_sat = solve_sat(l_0, eps, phi_sat, key2)
    
    # m_unsat greater than m_crit
    m_unsat_error = m_unsat_cand > m_crit
    # m_sat less than m_crit
    m_sat_error = m_sat_cand < m_crit
    
    is_sat = False # m_sat_cand > m_crit # FIXME
    phi = jnp.where(is_sat, phi_sat, phi_unsat_cand)
    m = jnp.where(is_sat, m_sat_cand, m_unsat_cand)
    err = jnp.where(is_sat, err_sat, err_unsat)
    
    # TODO verify these numbers are sane
    # jax.debug.print("unsat phi {} \n m {}", phi_unsat_cand, m_unsat_cand)
    # jax.debug.print("sat phi {} \n m {}", phi_sat, m_sat_cand)
                            
    return err, is_sat, phi, m, {
        'm_unsat_error': m_unsat_error,
        'm_sat_error': m_sat_error,
    }                       
    
solve_inner_vmap = jax.vmap(solve_inner, in_axes=(None, None, 0, None))

def solve(key, params: paramstype, radius, l0_candidates):
    
    # The ratio shortened
    eps = (2 * params.R_beam + params.R_act_max) / (radius + params.R_beam)
    # The force of each actuator at the desired contraction eps
    force = (jnp.pi * params.P_beam * params.R_beam**3) / (2 * params.R_beam + params.R_act_max)
    
    err, is_sat, phi, m, info = solve_inner_vmap(key, eps, l0_candidates, params)
    
    # --- Solve for pressure ---
    p_act = force / (jnp.pi * params.R_c**2) * (2 * m * jnp.cos(phi)**2) / (1 - 2 * m)

    return l0_candidates, p_act, err, is_sat, phi, m

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir") 
# jax.config.update("jax_enable_x64", False)

if __name__ == '__main__':
    
    key = jax.random.PRNGKey(0)
    
    l_0_candidates = jnp.linspace(0.05, 0.05, 1) # jnp.linspace(params.min_l_0, params.max_l_0, 100)

    solve_jit = jax.jit(solve, static_argnames=('params'))
    
    l_0, p_act, err, is_sat, phi, m = solve_jit(key, params, radius=1.0, l0_candidates=l_0_candidates)
    
    # Print all solutions in rows
    for l, p, e, s in zip(l_0, p_act, err, is_sat):
        print(f'l_0: {l:.4f}, P_act: {p:.4f}, Error: {e:.4f}, Saturated: {s}')



############### SECOND ONE ################

def objective_solve_for_phi_eps(vals, l_0, m, R, a):
    """
    Objective for unsaturated case: solve for phi and eps given l_0, m.
    """
    phi, eps = vals[0], vals[1]
    m_safe = m + 1e-6

    res1 = (E(phi, m) - 0.5 * F(phi, m)) / (jnp.sqrt(m_safe) * jnp.cos(phi)) - l_0 * (1 - eps) / (2 * R)
    res2 = F(phi, m) - l_0 / R * (jnp.sqrt(m_safe) * jnp.cos(phi) + a / (2 * jnp.sqrt(m_safe) * jnp.cos(phi)))

    phi_range_penalty = relu(phi - (jnp.pi/2 - 1e-3)) + relu(1e-3 - phi)
    eps_range_penalty = relu(eps - (1.0 - 1e-3)) + relu(1e-6 - eps)

    penalty_weight = 100.0

    return jnp.stack([res1, res2, penalty_weight * phi_range_penalty, penalty_weight * eps_range_penalty])

def objective_solve_for_eps_l_a(vals, l_0, m, phi_sat, R_c, a):
    """
    Objective for saturated case: solve for eps and l_a given l_0, m, phi_sat.
    """
    eps, l_a = vals[0], vals[1]
    m_safe = m + 1e-6
    l_a_safe = l_a + 1e-6

    eps_prime = l_0 / l_a_safe * eps

    res1 = (E(phi_sat, m) - 0.5 * F(phi_sat, m)) / (jnp.sqrt(m_safe) * jnp.cos(phi_sat)) - l_a / (2 * R_c) * (1 - eps_prime)
    res2 = F(phi_sat, m) - l_a / R_c * (jnp.sqrt(m_safe) * jnp.cos(phi_sat) + a / (2 * jnp.sqrt(m_safe) * jnp.cos(phi_sat)))

    l_a_range_penalty = relu(l_a - l_0) + relu(1e-3 - l_a)
    eps_range_penalty = relu(eps - (1.0 - 1e-3)) + relu(1e-6 - eps)

    penalty_weight = 100.0

    return jnp.stack([res1, res2, penalty_weight * l_a_range_penalty, penalty_weight * eps_range_penalty])

def l_m_to_phi_eps(key, l_0, m, params):
    """
    Given l_0 and m, solve for phi and eps (unsaturated) or eps and l_a (saturated).
    Returns: phi, eps, is_sat, info
    """
    # 1. Compute phi_sat and m_crit
    phi_sat = jnp.arccos(params.R_c / params.R_act_max)

    def objective_m_crit(m_val, phi_sat, l_0):
        return objective_solve_for_m_crit(m_val, phi_sat, l_0, params.R_c, params.a)

    opt_mcrit = jaxopt.BFGS(objective_m_crit, maxiter=100, tol=1e-4)
    m_crit, info_crit = opt_mcrit.run(jnp.array(0.25), phi_sat=phi_sat, l_0=l_0)
    
    # jax.debug.print("m_crit: {} phi_sat: {}", m_crit, phi_sat)
    # jax.debug.print("m_val: {} phi_sat: {} l_0: {} R_c: {} a: {}", m, phi_sat, l_0, params.R_c, params.a)
    
    # Check error
    # error = jnp.abs(objective_solve_for_m_crit(m_crit, phi_sat, l_0, params.R_c, params.a))
    # jax.debug.print("m_crit error: {}", error)
    
    # 2. Branch on m < m_crit (unsaturated) or m >= m_crit (saturated)
    def unsat_branch(args):
        key, l_0, m = args
        # Initial guess: phi in (1e-3, phi_sat), eps in (1e-6, 0.5)
        num_samples = 500
        key_phi, key_eps = jax.random.split(key)
        phi_samples = jax.random.uniform(key_phi, (num_samples,), minval=1e-3, maxval=phi_sat)
        eps_samples = jax.random.uniform(key_eps, (num_samples,), minval=1e-6, maxval=0.5)
        def single_error(phi, eps):
            vals = stack(phi, eps)
            res = objective_solve_for_phi_eps(vals, l_0, m, params.R_c, params.a)
            return jnp.sum(res[:2]**2)
        errors = jax.vmap(single_error)(phi_samples, eps_samples)
        best_idx = jnp.nanargmin(errors)
        guess = stack(phi_samples[best_idx], eps_samples[best_idx])
        opt = jaxopt.LevenbergMarquardt(objective_solve_for_phi_eps, maxiter=100, tol=1e-4)
        solution, info = opt.run(guess, l_0=l_0, m=m, R=params.R_c, a=params.a)
        phi, eps = solution[0], solution[1]
        return phi, eps, False, {'m_crit': m_crit, 'error': info.error}

    def sat_branch(args):
        key, l_0, m = args
        # phi = phi_sat, solve for eps and l_a
        num_samples = 500
        key_eps, key_la = jax.random.split(key)
        eps_samples = jax.random.uniform(key_eps, (num_samples,), minval=1e-6, maxval=0.5)
        l_a_samples = jax.random.uniform(key_la, (num_samples,), minval=1e-3, maxval=l_0)
        def single_error(eps, l_a):
            vals = stack(eps, l_a)
            res = objective_solve_for_eps_l_a(vals, l_0, m, phi_sat, params.R_c, params.a)
            return jnp.sum(res[:2]**2)
        errors = jax.vmap(single_error)(eps_samples, l_a_samples)
        best_idx = jnp.nanargmin(errors)
        guess = stack(eps_samples[best_idx], l_a_samples[best_idx])
        opt = jaxopt.LevenbergMarquardt(objective_solve_for_eps_l_a, maxiter=100, tol=1e-4)
        solution, info = opt.run(guess, l_0=l_0, m=m, phi_sat=phi_sat, R_c=params.R_c, a=params.a)
        eps, l_a = solution[0], solution[1]
        # actual eps = l_a / l_0
        eps_actual = l_a / l_0 * eps
        return phi_sat, eps_actual, True, {'m_crit': m_crit, 'error': info.error}

    phi, eps, is_sat, info = jax.lax.cond(m < m_crit,
                                          unsat_branch,
                                          sat_branch,
                                          (key, l_0, m))
    
    return phi, eps, is_sat, info