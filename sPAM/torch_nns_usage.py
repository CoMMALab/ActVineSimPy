from sPAM.nns import get_or_train_model, get_prediction_function
from sPAM.spam import paramstype, params
import jax.numpy as jnp
import numpy as np
import jax

import torch

def solve(predict, params: paramstype, radius):
    radius_sign = jnp.sign(radius)
    radius = jnp.abs(radius)
    
    # The ratio shortened
    eps = (2 * params.R_beam + params.R_act_max) / (radius + params.R_beam)
    
    # The force of each actuator at the desired contraction eps
    force = (jnp.pi * params.P_beam * params.R_beam**3) / (2 * params.R_beam + params.R_act_max)
            
    # l_0 = jnp.arange(params.min_l_0 * eps, params.max_l_0 + 1e-3, step=params.min_l_0 * eps)
    l_0 = jnp.array([0.020, 
                     0.021,
                     0.022,
                    0.023,  
                    0.024,
                    0.025,
                    0.026,
                    0.027,
                    0.028,
                    0.029,
                    0.030,
                    0.031,
                    0.032,
                    0.033,
                    0.034,
                    0.035,
                    0.036,
                    0.037,
                    0.038,
                    0.039,
                    0.040,  
                    0.041,
                    0.042,
                     ]) 
    # l_0 = jnp.array([0.02, 
    #                  params.min_l_0 * (1 - eps) * 1.5, 
    #                  params.min_l_0 * (1 - eps) * 2,
    #                  params.min_l_0 * (1 - eps) * 2.5,
    #                  params.min_l_0 * (1 - eps) * 3,
    #                  params.min_l_0 * (1 - eps) * 3.5,
    #                  params.min_l_0 * (1 - eps) * 4,
    #                  params.min_l_0 * (1 - eps) * 5]) # TODO cover the whole range?
    inputs = jnp.stack((eps.repeat(len(l_0)), l_0), axis=-1)
    ouputs = predict(inputs)
    
    # Inputs are (eps, l_0) --> (phi, m)
    phi, m = ouputs[:, 0], ouputs[:, 1]
    
    # --- Solve for pressure ---
    p_act = force / (jnp.pi * params.R_c**2) * (2 * m * jnp.cos(phi)**2) / (1 - 2 * m)
    
    # p_act /= 2 
    
    # Valid is where pressure is positive and pressure is less than 28kPa
    # valid_mask = (p_act > 1e-3) & (p_act < 35e3) & 
    valid_mask = (p_act > 1e-3) & (p_act < 35e3) & (l_0 > params.min_l_0) & (l_0 < params.max_l_0)

    # Pick the largest l_0 with err < 1e-3
    best_idx = jnp.argmax(jnp.where(valid_mask, p_act, -9999))
    
    # Print eps, phi, m in rows
    # jax.debug.print("Actuator design eps {} \n l_0, p, \n {}", eps, jnp.column_stack((l_0, p_act)))

    return p_act[best_idx], radius_sign * l_0[best_idx] * 2.0 


def torch_solve(predict, params: paramstype, radius):
    '''
    Torch-compatible for of solve()
    '''
    radius_sign = torch.sign(radius)
    radius = torch.abs(radius)

    # The ratio shortened
    eps = (2 * params.R_beam + params.R_act_max) / (radius + params.R_beam)
    
    # The force of each actuator at the desired contraction eps
    force = (torch.pi * params.P_beam * params.R_beam**3) / (2 * params.R_beam + params.R_act_max)
            
    l_0 = torch.tensor([0.020, 
                     0.021,
                     0.022,
                    0.023,  
                    0.024,
                    0.025,
                    0.026,
                    0.027,
                    0.028,
                    0.029,
                    0.030,
                    0.031,
                    0.032,
                    0.033,
                    0.034,
                    0.035,
                    0.036,
                    0.037,
                    0.038,
                    0.039,
                    0.040,  
                    0.041,
                    0.042,
                     ]) 
    
    inputs = torch.stack((eps.repeat(len(l_0)), l_0), dim=-1)
    ouputs = predict(inputs)
    
    # Inputs are (eps, l_0) --> (phi, m)
    phi, m = ouputs[:, 0], ouputs[:, 1]
    
    # --- Solve for pressure ---
    p_act = force / (torch.pi * params.R_c**2) * (2 * m * torch.cos(phi)**2) / (1 - 2 * m)
        
    # Valid is where pressure is positive and pressure is less than 28kPa
    valid_mask = (p_act > 1e-3) & (p_act < 35e3) & (l_0 > params.min_l_0) & (l_0 < params.max_l_0)

    # Pick the largest l_0 with err < 1e-3
    best_idx = torch.argmax(torch.where(valid_mask, p_act, -9999))

    return p_act[best_idx], radius_sign * l_0[best_idx] * 2.0 


def solve_fwd(predict, params: paramstype, radius, p_act, l_0):
    l0_sign = torch.sign(l_0) # The direction the actuator is meant to curl
    l_0 = torch.abs(l_0)
    
    # So now radius is positive if the actuator curls in the direction its meant for
    # Otherwise, it is negative
    radius = l0_sign * radius 
    
    # The ratio shortened
    eps = (2 * params.R_beam + params.R_act_max) / (radius + params.R_beam)
    
    # The force of the vine
    force_vine = (torch.pi * params.P_beam * params.R_beam**3) / (2 * params.R_beam + params.R_act_max)
            
    inputs = torch.stack([eps, l_0])
    ouputs = predict(inputs)
    
    # Inputs are (eps, l_0) --> (phi, m)
    phi, m = ouputs[0], ouputs[1]
    
    # --- Solve for force ---
    force_act = p_act * torch.pi * params.R_c**2 * (1 - 2 * m) / (2 * m * torch.cos(phi)**2) 

    # jax.debug.print("force_act: {}, force_vine: {}", force_act, force_vine)
    
    # Force_act tries to curl more
    # Force_vine tries to curl less
    # The return is the total moment direction of curl
    # Need l0_sign to convert from one-direction to both
    return l0_sign * torch.where(radius > 0, 
                      force_act - force_vine,
                      force_vine * 30)

vine_tube_width = 4.125 # in
vine_perimeter = vine_tube_width * 2 * 25.4e-3 # 2 * pi * r
vine_radius = vine_perimeter / (2 * torch.pi)

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir") 
# jax.config.update("jax_enable_x64", False)

if __name__ == "__main__":

    trained_state, scaling_info, model = get_or_train_model(params)

    predict = get_prediction_function(trained_state, scaling_info, model)
    solve_jit = jax.jit(solve, static_argnames=('predict', 'params'))
    solve_fwd_jit = jax.jit(solve_fwd, static_argnames=('predict', 'params'))
    
    p, l = solve_jit(predict, params, 200.0)
    print(f'predicted p 0.9: {p} l: {l}')
    
    p, l = solve_jit(predict, params, 0.8)
    print(f'predicted p 0.8: {p} l: {l}')   
    
    p, l = solve_jit(predict, params, 0.7)
    print(f'predicted p 0.7: {p} l: {l}')
    
    p, l = solve_jit(predict, params, 0.6)
    print(f'predicted p 0.6: {p} l: {l}')
    
    p5, l5 = solve_jit(predict, params, 0.5)
    print(f'predicted p 0.5: {p5} l: {l5}')
    
    p, l = solve_jit(predict, params, 0.4)
    print(f'predicted p 0.4: {p} l: {l}')
    
    p, l = solve_jit(predict, params, 0.3)
    print(f'predicted p 0.3: {p} l: {l}')
    
    p, l = solve_jit(predict, params, 0.275)
    print(f'predicted p 0.275: {p} l: {l}') 
    
    p, l = solve_jit(predict, params, 0.25)
    print(f'predicted p 0.25: {p} l: {l}')  
    
    p, l = solve_jit(predict, params, 0.2)
    print(f'predicted p 0.2: {p} l: {l}')   
    
    p, l = solve_jit(predict, params, 0.1)
    print(f'predicted p 0.1: {p} l: {l}')
    
    force = solve_fwd_jit(predict, params, 50, p5, l5)
    print(f'predicted force from (design 0.5), actual 1000 {force}')
    force = solve_fwd_jit(predict, params, 1.0, p5, l5)
    print(f'predicted force from (design 0.5), actual 1.0 {force}')
    force = solve_fwd_jit(predict, params, 0.9, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.9 {force}')
    force = solve_fwd_jit(predict, params, 0.8, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.8 {force}')
    force = solve_fwd_jit(predict, params, 0.7, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.7 {force}')
    force = solve_fwd_jit(predict, params, 0.6, p5, l5)       
    print(f'predicted force from (design 0.5), actual 0.6 {force}')
    force = solve_fwd_jit(predict, params, 0.5, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.5 {force}')
    force = solve_fwd_jit(predict, params, 0.4, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.4 {force}')
    force = solve_fwd_jit(predict, params, 0.3, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.3 {force}')
    force = solve_fwd_jit(predict, params, 0.2, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.2 {force}')
    force = solve_fwd_jit(predict, params, 0.1, p5, l5)
    print(f'predicted force from (design 0.5), actual 0.1 {force}')
    
    p5, l5 = solve_jit(predict, params, -0.5)
    
    force = solve_fwd_jit(predict, params, -4.5, p5, l5)
    print(f'predicted force from (design -0.5), actual -4.5 {force}')
    force = solve_fwd_jit(predict, params, -0.5, p5, l5)
    print(f'predicted force from (design -0.5), actual -0.5 {force}')
    force = solve_fwd_jit(predict, params, -0.1, p5, l5)
    print(f'predicted force from (design -0.5), actual -0.1 {force}')
    force = solve_fwd_jit(predict, params, 0.5, p5, l5)
    print(f'predicted force from (design -0.5), actual 0.5 {force}')
    force = solve_fwd_jit(predict, params, 0.2, p5, l5)
    print(f'predicted force from (design -0.5), actual 0.2 {force}')
    

    # # Create some sample inputs (eps, l_0)
    # # Ensure they are within the training range for best results
    # sample_inputs = np.array([
    #     [0.1, 0.05],  # eps, l_0
    #     [0.2, 0.08],
    #     [0.3, 0.03],
    # ])
    
    # # Get predictions
    # predictions = predict(sample_inputs)
    # predictions = predict(sample_inputs)
