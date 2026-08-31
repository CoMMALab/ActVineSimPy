import torch, math, matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation

# Sim modules:
from dvsim import si
import dvsim.solver as solver
from dvsim.dynamic_vine import step 
from dvsim.vine import create_state_batched, init_state_batched




# RRT utility:
from rrt_util import StateInfo, Node, RRTTree


#---------------------------------------------------------------------- Sim/Animation Helper Defs

B = 1 # batch size
MM = lambda x: si.len_to_mm(x)  # internal length -> mm (for display)


def init_params(max_bodies=40, grow_rate_mps=0.3,
                bend_length_scale=None, spam_moment_scale=None, 
                p=None, l0=None, radius_m=0.0125,
                static_objects=None,
                dynamic_obj_coords=None, dynamic_obj_masses=None):

    params = si.vine_params_si(max_bodies=max_bodies, obstacles_m=static_objects, grow_rate_mps=grow_rate_mps,
                               radius_m=radius_m)

    # params.stiffness_mode = 'spam'
    if bend_length_scale is not None: params.bend_length_scale = torch.tensor(bend_length_scale)
    if spam_moment_scale is not None: params.spam_moment_scale = torch.tensor(spam_moment_scale)
    if p is not None: params.spam_p = torch.full((max_bodies,), float(p))
    if l0 is not None: params.spam_l0 = torch.full((max_bodies,), float(l0))

    if dynamic_obj_coords is not None:
        dynamic_obj_pose = si.set_objects_si(params, dynamic_obj_coords, 
                                                    dynamic_obj_masses)
        num_dynamic_objs = len(dynamic_obj_masses)
        dynamic_obj_dstate = torch.zeros(B, num_dynamic_objs, 3)
    else:
        dynamic_obj_pose = None
        num_dynamic_objs = 0
        dynamic_obj_dstate = None

    return params, num_dynamic_objs, dynamic_obj_pose, dynamic_obj_dstate


def _obb_corners_mm(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c * lx - s * ly), MM(cy + s * lx + c * ly))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def find_frame_dims(walls, max_dim):

    '''
    Given the wall dimensions, return the appropriate width/height
    of the animation frame in inches.
    The neither of the returned dims will go past the given max_dim.
    '''
    
    def find_borders(walls):
        '''
        Just find the outer edges of the walls to define as the mins/maxes for the frame
        '''
        wall_xs = []; wall_ys = []
        
        for wall in walls:

            wall_xs.append(wall[0]); wall_xs.append(wall[2])
            wall_ys.append(wall[1]); wall_ys.append(wall[3])

        xlim_mm = (min(wall_xs) * 1000, max(wall_xs) * 1000)
        ylim_mm = (min(wall_ys) * 1000, max(wall_ys) * 1000)

        return xlim_mm, ylim_mm

    xlim_mm, ylim_mm = find_borders(walls)

    frame_width_mm = xlim_mm[1] - xlim_mm[0]
    frame_height_mm = ylim_mm[1] - ylim_mm[0]

    aspect_ratio = frame_height_mm / frame_width_mm
    if aspect_ratio <= 1:
        width_in = max_dim
        height_in = max_dim * aspect_ratio
    else:
        height_in = max_dim
        width_in = max_dim / aspect_ratio

    return width_in, height_in


#----------------------------------------------------------------------- For RRT

def sampleRandomState


#---------------------------------------------------------------------- Main

'''
Basically, run RRT to try to reach a goal state.
Then animate the sim's steps using the path that is returned,
which saves the information for each state from start to goal.
'''

if __name__ == "__main__":

    solver.cvxpylayer = None

    # Initialize vine parameters:

    walls = [(0.0, 0.0, 1.525, 0.1),
            (0.0, 0.9, 1.525, 1.0),
            (0.0, 0.0, 0.1, 1.0),
            (1.425, 0.0, 1.525, 1.0)]
    static_objs = [(0.7125, 0.3, 0.8125, 0.2)]

    dynamic_obj_coords = [(0.7125, 0.5000, 0.8125, 0.4000)]
    dynamic_obj_masses = [0.2]

    max_bodies = 40
    grow_rate_mps = 0.5
    radius_m = 0.0125 * 4

    initial_vine_pose = (400, 400, 0) # x:mm, y:mm, theta:degrees

    vine_params, num_moveable_objs, \
    moveable_obj_pose, moveable_obj_dstate = init_params(max_bodies=max_bodies, grow_rate_mps=grow_rate_mps,
                                                         static_objects=(walls + static_objs), radius_m=radius_m,
                                                         dynamic_obj_coords=dynamic_obj_coords,
                                                         dynamic_obj_masses=dynamic_obj_masses)

    static_obj_poses = [([float(v) for v in vine_params.obstacle_pose[k]],
                          float(vine_params.obstacle_hw[k]), float(vine_params.obstacle_hh[k]))
                          for k in range(vine_params.obstacle_pose.shape[0])]    
    
    init_x = torch.zeros(B, 1); init_x[:, 0] = initial_vine_pose[0]
    init_y = torch.zeros(B, 1); init_y[:, 0] = initial_vine_pose[1]
    init_heading = torch.zeros(B, 1); init_heading[:, 0] = initial_vine_pose[2]

    # State initialization steps

    vine_state, vine_dstate = create_state_batched(B, max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(vine_params, vine_state, bodies, init_heading, init_x, init_y)


    # Run RRT to try to find a path towards the goal region
    
    max_iters = 100