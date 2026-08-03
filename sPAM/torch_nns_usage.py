from sPAM.torch_spam import paramstype, params

import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')

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
    outputs = predict(inputs)

    # Inputs are (eps, l_0) --> (phi, m)
    phi, m = outputs[:, 0], outputs[:, 1]

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
