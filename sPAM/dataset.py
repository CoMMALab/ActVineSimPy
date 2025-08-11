import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from sPAM.spam import l_m_to_phi_eps, params
from sPAM.ellip import F, E
import os

# It's better to have this defined where it is used, to avoid circular dependencies
# if other modules need to use generate_data_from_csv.
solve_inner_vmap = jax.vmap(l_m_to_phi_eps, in_axes=(None, 0, 0, None))
solve_inner_vmap = jax.jit(solve_inner_vmap)

def generate_data_from_csv(csv_path="sPAM/actuator_design_df_0.csv", sanity_check=True):
    """
    Generates a dataset from a CSV file.

    The CSV file should contain columns for 'l0', 'm', 'phi_rc', 'Rc', and 'film_elasticity_a'.
    It calculates 'eps' based on 'phi_rc' and performs a sanity check against
    the physics solver if requested.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    print(f"Loaded phi/l0/m dataset from {csv_path}")

    # Parameter consistency check
    # Check if Rc and film_elasticity_a are constant and match params
    if not np.allclose(df['Rc'].iloc[0], params.R_c):
        raise ValueError(f"Rc in CSV ({df['Rc'].iloc[0]}) does not match params.R_c ({params.R_c})")
    if not df['Rc'].nunique() == 1:
        raise ValueError("Rc is not constant in the CSV file.")

    if not np.allclose(df['film_elasticity_a'].iloc[0], params.a):
        raise ValueError(f"film_elasticity_a in CSV ({df['film_elasticity_a'].iloc[0]}) does not match params.a ({params.a})")
    if not df['film_elasticity_a'].nunique() == 1:
        raise ValueError("film_elasticity_a is not constant in the CSV file.")

    l0 = jnp.array(df['l0'].values)
    m = jnp.array(df['m'].values)
    phi_rc = jnp.array(df['phi_rc'].values)

    # Calculate eps
    phi_sat_val = 1.2755004409
    
    # Unsaturated case
    def solve_eps_unsat(phi_rc, m, l0):
        eps_unsat = 1 - ( (E(phi_rc, m) - 0.5 * F(phi_rc, m)) * 2 * params.R_c) / (jnp.sqrt(m) * jnp.cos(phi_rc) * l0)
        return eps_unsat

    print('Computing unsaturated eps...')
    solve_eps_unsat = jax.jit(jax.vmap(solve_eps_unsat, in_axes=(0, 0, 0)))
    eps_unsat = solve_eps_unsat(phi_rc, m, l0)
    
    # Saturated case (needs solving)
    key = jax.random.PRNGKey(0) # Use a fixed key for reproducibility
    
    # We only need to solve for the saturated cases.
    sat_mask_np = np.isclose(df['phi_rc'].values, phi_sat_val)
    
    eps_calc = np.zeros(len(df))

    if np.any(sat_mask_np):
        l0_sat = jnp.array(df.loc[sat_mask_np, 'l0'].values)
        m_sat = jnp.array(df.loc[sat_mask_np, 'm'].values)
        
        # We need to provide a key for each sample.
        keys_sat, _ = jax.random.split(key, 2)
        
        # solve_inner_vmap is defined on l_m_to_phi_eps, which returns (phi, eps, is_sat, info)
        # We need to call it with the saturated inputs.
        # Note: l_m_to_phi_eps is defined in spam.py and expects (key, l_0, m, params)
        _, eps_res, _, _ = solve_inner_vmap(keys_sat, l0_sat, m_sat, params)
        
        eps_calc[sat_mask_np] = eps_res

    # Combine results
    eps = jnp.where(jnp.isclose(phi_rc, phi_sat_val), eps_calc, eps_unsat)

    is_sat = jnp.isclose(phi_rc, phi_sat_val)

    if sanity_check:
        print("Performing sanity check...")
        # We need to provide a key for each sample.
        keys, _ = jax.random.split(key, 2)
        phi_pred, eps_pred, _, info = solve_inner_vmap(keys, l0, m, params)
        
        phi_error = jnp.abs(phi_pred - phi_rc)
        phi_error_unsat = phi_error[~is_sat]
        phi_error_sat = phi_error[is_sat]
        
        print(f"Sanity check for phi for unsaturated min={jnp.min(phi_error_unsat):.4f}, "
              f"max={jnp.max(phi_error_unsat):.4f}, avg={jnp.mean(phi_error_unsat):.4f}")
        print(f"Sanity check for phi for saturated min={jnp.min(phi_error_sat):.4f}, "
              f"max={jnp.max(phi_error_sat):.4f}, avg={jnp.mean(phi_error_sat):.4f}") 

    # expect inputs=(eps, l0) and outputs=(phi, m)
    inputs = jnp.stack([eps, l0], axis=1)
    outputs = jnp.stack([phi_rc, m], axis=1)    

    return inputs, outputs, is_sat
