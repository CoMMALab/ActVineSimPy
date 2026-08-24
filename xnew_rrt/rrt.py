'''
Simplified, unbatched RRT algorithm for the new sim.

Obstacle and other scene metrics are given in SI units (meters, grams, etc.)
The solver operates with non-dim units.

Millimeters are used for display.
'''

# TODO:
# 1. Get the sim to run first 
# 2. Surround it with simple RRT (establish goal states)

import torch
import matplotlib as plt

# Sim modules:
from dvsim import si
import dvsim.solver as solver

from dvsim.dynamic_vine import step 
from dvsim.vine import create_state_batched, init_state_batched


#-------------------------- Helper functions

def init_params(max_bodies, grow_rate_mps,
                bend_length_scale, spam_moment_scale, p, l0):
    params = si.vine_params_si(max_bodies=max_bodies, grow_rate_mps=grow_rate_mps)
    params.stiffness_mode = 'spam'
    params.bend_length_scale = torch.tensor(bend_length_scale)
    params.spam_moment_scale = torch.tensor(spam_moment_scale)
    params.spam_p = torch.full((max_bodies,), float(p))
    params.spam_l0 = torch.full((max_bodies,), float(l0))

    return params


def draw():




#------------------------------------- Main

B = 1 # unbatched => batch size of 1


if __name__ == "__main__":

    # All obstacles:
    
    walls = [(0.0, 0.0, 1.525, 0.1),
             (0.0, 0.9, 1.525, 1.0),
             (0.0, 0.0, 0.1, 1.0),
             (1.425, 0.0, 1.525, 1.0)]

    static_objs = [(0.7125, 0.3, 0.8125, 0.2)]

    dynamic_objs = [(0.7125, 0.5000, 0.8125, 0.4000, 0.2)]


    # Define params:

    solver.cvxpylayer = None
    vine_params = init_params(max_bodies=40,
                              grow_rate_mps=0.3,
                              bend_length_scale=0.018,
                              spam_moment_scale=22000.0,
                              p=8000.0, l0=-0.04)
    
    # Initialize state info

    max_bodies = 40 # same as vine params

    init_heading = torch.zeros(1, 1)
    init_x = torch.zeros(B, 1)
    init_y = torch.zeros(B, 1)

    state, dstate = create_state_batched(B, max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(vine_params, state, bodies, init_heading)


    # Init stuff for drawing:

    wall_xs = []; wall_ys = []

    for wall in walls:
        wall_xs.append(wall[0]); wall_xs.append(wall[2])
        wall_ys.append(wall[1]); wall_ys.append(wall[3])

    xlim_mm = (min(wall_xs) * 1000, max(wall_xs) * 1000)
    ylim_mm = (min(wall_ys) * 1000, min(wall_ys) * 1000)

    