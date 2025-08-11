import numpy as np

def sample_noise(shape, scale=1.0):
    return np.random.normal(loc=0.0, scale=scale, size=shape)

def perturb_obstacle(obstacle, pos_sigma=0.01, size_sigma=0.05, noise_fn=sample_noise, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x1, y1, x2, y2 = obstacle
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    # Sample noise for center shift
    dx, dy = noise_fn(shape=2, scale=pos_sigma)
    cx_perturbed = cx + dx
    cy_perturbed = cy + dy

    # Sample noise for size scaling
    dw_scale, dh_scale = noise_fn(shape=2, scale=size_sigma)
    w_perturbed = max(1e-6, w * (1 + dw_scale))
    h_perturbed = max(1e-6, h * (1 + dh_scale))

    # Recompute rectangle from center
    x1_new = cx_perturbed - w_perturbed / 2
    x2_new = cx_perturbed + w_perturbed / 2
    y1_new = cy_perturbed - h_perturbed / 2
    y2_new = cy_perturbed + h_perturbed / 2

    return [x1_new, y1_new, x2_new, y2_new]

def new_obs(obs):
    new_ob = []
    for i, ob in enumerate(obs):
        if i < 4:
            continue
        if ob is not None:
            # pos_sigma is absolute units, size_sigma is relative to the size of the obstacle
            new_ob.append(perturb_obstacle(ob, pos_sigma=20, size_sigma=.10, noise_fn=sample_noise))
    return obs, new_ob
