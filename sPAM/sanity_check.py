import io
import csv
import jax
import jax.numpy as jnp
import numpy as np
from sPAM.ellip import F, E

from sPAM.nns import get_or_train_model, get_prediction_function
# from sPAM.nns_usage import solve
from sPAM.spam import paramstype, params as default_params

# Hardcoded CSV data
csv_data = """Rc,l0,R_tube,P_act,m,phi_rc,R_act,num_actuators,fab_buffer_length,film_elasticity_a,P_act_max
0.005,0.035,0.0171807761,6714.1872239907,0.1610759501,1.1406766723,0.0119909955,2.0,0.005,0.0001,34473.8
0.005,0.03,0.0171807761,11113.2564912162,0.1901281255,1.1153188579,0.0113664497,2.0,0.005,0.0001,34473.8
0.005,0.035,0.0171807761,10550.6682773926,0.2001324147,1.1723556507,0.01288721,2.0,0.005,0.0001,34473.8
0.005,0.025,0.0171807761,20880.2904851354,0.2323297723,1.084513174,0.0106987777,2.0,0.005,0.0001,34473.8
0.005,0.03,0.0171807761,18185.7876543166,0.2352800301,1.1472288648,0.0121649996,2.0,0.005,0.0001,34473.8
0.005,0.03,0.0171807761,29126.0719174334,0.2784416909,1.1706017086,0.0128337558,2.0,0.005,0.0001,34473.8
0.005,0.035,0.0171807761,15744.1666175773,0.2377579925,1.195674957,0.0136468312,2.0,0.005,0.0001,34473.8
0.005,0.035,0.0171807761,23002.3865550793,0.2740908946,1.2136610721,0.0143024009,2.0,0.005,0.0001,34473.8
"""

def solve_eps_unsat(phi_rc, m, l0, R_c):
    eps_unsat = 1 - ( (E(phi_rc, m) - 0.5 * F(phi_rc, m)) * 2 * R_c) / (jnp.sqrt(m) * jnp.cos(phi_rc) * l0)
    return eps_unsat

def solve(predict, params: paramstype, eps):
    # radius_sign = jnp.sign(radius)
    # radius = jnp.abs(radius)
    
    # # The ratio shortened
    # eps = (2 * params.R_beam + params.R_act_max) / (radius + params.R_beam)
    
    # The force of the vine
    force = (jnp.pi * params.P_beam * params.R_beam**3) / (2 * params.R_beam + params.R_act_max)
            
    # l_0 = jnp.arange(params.min_l_0 * eps, params.max_l_0 + 1e-3, step=params.min_l_0 * eps)
    l_0 = jnp.array([params.min_l_0 * (1 - eps) * 1, 
                     params.min_l_0 * (1 - eps) * 1.5, 
                     params.min_l_0 * (1 - eps) * 2,
                     params.min_l_0 * (1 - eps) * 2.5,
                     params.min_l_0 * (1 - eps) * 3,
                     params.min_l_0 * (1 - eps) * 3.5,
                     params.min_l_0 * (1 - eps) * 4,
                     params.min_l_0 * (1 - eps) * 5]) # TODO cover the whole range?
    inputs = jnp.stack((eps.repeat(len(l_0)), l_0), axis=-1)
    ouputs = predict(inputs)
    
    # Inputs are (eps, l_0) --> (phi, m)
    phi, m = ouputs[:, 0], ouputs[:, 1]
    
    jax.debug.print("   eps: {}", eps)
    jax.debug.print("   l_0: {}", l_0)
    jax.debug.print("   phi: {}", phi)
    jax.debug.print("   m: {}", m)
    
    # --- Solve for pressure ---
    p_act = force / (jnp.pi * params.R_c**2) * (2 * m * jnp.cos(phi)**2) / (1 - 2 * m)
    
    p_act /= 2  # TODO

    # Valid is where pressure is positive and pressure is less than 28kPa
    valid_mask = (p_act > 1e-3) & (p_act < 35e3) & (l_0 > params.min_l_0 * eps) & (l_0 < params.max_l_0 * eps)

    # Pick the largest l_0 with err < 1e-3
    best_idx = jnp.argmax(jnp.where(valid_mask, l_0, -9999))
    
    # Print eps, phi, m in rows
    # jax.debug.print("Actuator design eps {} \n l_0, phi, m, p, \n {}", eps, jnp.column_stack((l_0, phi, m, p_act)))

    return p_act[best_idx], l_0[best_idx] * 2.0 # TODO

def run_sanity_check():
    """
    Compares the output of the solve function with hardcoded data.
    """
    # Load the model
    trained_state, scaling_info, model = get_or_train_model(default_params)
    predict = get_prediction_function(trained_state, scaling_info, model)
    solve_jit = jax.jit(solve, static_argnames=('predict', 'params'))

    # Parse the CSV data
    reader = csv.DictReader(io.StringIO(csv_data))
    
    print("Running Sanity Check...")
    print("-" * 80)
    print(f"{'Test Case':<10}{'Predicted P_act':<20}{'Ground Truth P_act':<22}{'P_act Error (%)':<20}")
    print(f"{'':<10}{'Predicted l0':<20}{'Ground Truth l0':<22}{'l0 Error (%)':<20}")
    print("-" * 80)

    for i, row in enumerate(reader):
        # Extract parameters from the row
        R_c = float(row['Rc'])
        l0_gt = float(row['l0'])
        # R_beam = float(row['R_tube'])
        P_act_gt = float(row['P_act'])
        R_act_max = float(row['R_tube'])
        a = float(row['film_elasticity_a'])
        phi_rc = float(row['phi_rc'])
        m = float(row['m'])
        
        # Create params object for the current test case
        # Using some default values from spam.py for params not in CSV
        current_params = default_params._replace(
            R_c=R_c,
            R_act_max=R_act_max,
            a=a
        )
        # Assert that the params are the same as the default params
        assert np.isclose(R_c, default_params.R_c), "R_c does not match default params"
        assert np.isclose(R_act_max, default_params.R_act_max), "R_act_max does not match default params"
        assert np.isclose(a, default_params.a), "film_elasticity_a does not match default params"

        # Run the solver
        eps_unsat = solve_eps_unsat(phi_rc, m, l0_gt, R_c)
        
        P_act_pred, l0_pred = solve_jit(predict, current_params, eps_unsat)

        # Compare results
        p_act_error = jnp.abs((P_act_pred - P_act_gt) / P_act_gt) * 100
        l0_error = jnp.abs((l0_pred - l0_gt) / l0_gt) * 100

        print(f"#{i+1:<9}{P_act_pred:<20.4f}{P_act_gt:<22.4f}{p_act_error:<20.2f}")
        print(f"{'':<10}{l0_pred:<20.4f}{l0_gt:<22.4f}{l0_error:<20.2f}")
        print("-" * 80)

if __name__ == "__main__":
    run_sanity_check()
