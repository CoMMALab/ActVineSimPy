import torch, math, matplotlib, random, os

import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
# from matplotlib.animation import FuncAnimation

# Sim modules:
from dvsim import si
import dvsim.solver as solver
from dvsim.dynamic_vine import step 
from dvsim.vine import create_state_batched, init_state_batched

# RRT utility:
from rrt_util import StateInfo, Node, RRTTree

# For predicting p, l0 given a bending angle:
from sPAM.torch_nns import get_or_train_model, get_prediction_function
from sPAM.spam import params as act_params
from sPAM.torch_nns_usage import torch_solve as find_actuator_params

scaling_info, model = get_or_train_model(act_params)
predict = get_prediction_function(scaling_info, model)

find_actuator_params = torch.vmap(find_actuator_params, in_dims=(None, None, 0))

#---------------------------------------------------------------------- Sim/Animation Helper Defs

B = 1 # batch size
MM = lambda x: si.len_to_mm(x)    # internal length -> mm (for display)
ND = lambda x: x / (si.L0 * 1000) # mm -> internal length (used by sim)      


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


def find_frame_dims(walls, max_dim):

    '''
    Given the wall dimensions, return the appropriate width/height
    of the animation frame in inches.
    The neither of the returned dims will go past the given max_dim.
    '''    

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


def draw_frame(frame_idx, sim_state: StateInfo, walls,
               static_obj_poses, vine_params, vine_radius_mm,
               goal_coords_m, goal_radius_m, 
               gif_path):
    '''
    Draws what the current scene looks like given the StateInfo
    '''

    width_in, height_in = find_frame_dims(walls, max_dim=12)
    xlim_mm, ylim_mm = find_borders(walls)

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    vine_state = sim_state["state"]
    moveable_obj_poses = sim_state["moveable_obj_pose"]
    num_bodies = sim_state["bodies"]

    # Draw static objs:
    for (pose, hw, hh) in static_obj_poses:
        corners = _obb_corners_mm(pose[0], pose[1], pose[2], hw, hh)
        ax.add_patch(Polygon(corners, closed=True, facecolor=(1, .89, .71), 
                                edgecolor="k", zorder=1))

    # Draw goal radius:
    goal_x = goal_coords_m[0] * 1000; goal_y = goal_coords_m[1] * 1000
    goal_radius_mm = goal_radius_m * 1000
    ax.add_patch(Circle(goal_x, goal_y), goal_radius_mm,
                 facecolor="red", alpha=0.4, edgecolor=(.1, .2, .6), zorder=4)

    # Draw moveable objs:
    moveable_obj_poses = sim_state["moveable_obj_pose"]

    for k in range(num_bodies):
        obj_x, obj_y, obj_theta = [float(v) for v in moveable_obj_poses[0, k]]
        corners = _obb_corners_mm(obj_x, obj_y, obj_theta, float(vine_params.obj_hw[k]),
                                    float(vine_params.obj_hh[k]))
        ax.add_patch(Polygon(corners, closed=True, facecolor=(.55, .8, .95), 
                                edgecolor="b", lw=2, zorder=3))

    # Draw the vine's bodies:
    n = int(num_bodies[0])
    xs = [MM(float(vine_state[0, 3 * j])) for j in range(n)] 
    ys = [MM(float(vine_state[0, 3 * j + 1])) for j in range(n)]

    ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), vine_radius_mm, 
        facecolor=(.3, .5, .95, .4), edgecolor=(.1, .2, .6), zorder=4))

    ax.plot([init_x[:, 0].item()], [init_y[:, 0].item()], "g^", ms=9, zorder=6)
    ax.set_xlim(*xlim_mm) 
    ax.set_ylim(*ylim_mm) 
    ax.set_aspect("equal")
    ax.set_xlabel("mm"); ax.set_title(f"Frame {frame_idx}")

    fig.savefig(os.path.join(gif_path, f"f{frame_idx:04d}.png"), dpi=70, bbox_inches="tight")
    plt.close(fig)



#NOTE: should probably just move RRT stuff to rrt_util.py later on ...
#----------------------------------------------------------------------- RRT Distance Funcs


def euclidean_distance(state_obj1: StateInfo, state_obj2: StateInfo, 
                 max_bodies, num_moveable_objs,
                 theta_weight = 1, 
                 weight_list = (1, 1, 1, 1, 1)):
    '''
    Default distance function for comparing states returned by sim
    On args:
        - theta_weight: how much to weight diff in theta compared to diff in position
        - weight_list: how much to multiply each measure by (arbitrarily decided)

    NOTE: everything stays in non-dim units, but I don't think that really matters... :p
    '''

    def vmapped_distance(pose1, pose2, theta_weight):
        # Tentative measure for angle diff: 
        # dtheta = torch.atan2(torch.sin(theta1 - theta2), torch.cos(theta1 - theta2))

        # Euclidean distance is used otherwise

        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        euclid_dist = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dtheta = torch.atan2(torch.sin(theta2 - theta1), torch.cos(theta2 - theta1))

        return euclid_dist + theta_weight * dtheta

    #NOTE: reshapes are to make things easier for vmap

    vine_state1 = state_obj1["state"].reshape(B * max_bodies, 3)
    vine_state2 = state_obj2["state"].reshape(B * max_bodies, 3)

    vine_dstate1 = state_obj1["dstate"].reshape(B * max_bodies, 3)
    vine_dstate2 = state_obj2["dstate"].reshape(B * max_bodies, 3)

    bodies1 = state_obj1["bodies"].reshape(B * 1)
    bodies2 = state_obj2["bodies"].reshape(B * 1)

    obj_state1 = state_obj1["moveable_obj_pose"].reshape(B * num_moveable_objs, 3)
    obj_state2 = state_obj2["moveable_obj_pose"].reshape(B * num_moveable_objs, 3)

    obj_dstate1 = state_obj1["moveable_obj_dstate"].reshape(B * num_moveable_objs, 3)
    obj_dstate2 = state_obj2["moveable_obj_dstate"].reshape(B * num_moveable_objs, 3)

    # Diff for vine states:
    vine_state_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(vine_state1, vine_state2, theta_weight)
    vine_state_diff = torch.sum(vine_state_diff)

    # Diff for vine dstates:
    vine_dstate_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(vine_dstate1, vine_dstate2, theta_weight)
    vine_dstate_diff = torch.sum(vine_dstate_diff)

    # Diff for bodies:
    bodies_diff = torch.abs(bodies2 - bodies1)

    # Diff for moveable object states:
    obj_state_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(obj_state1, obj_state2, theta_weight)
    obj_state_diff = torch.sum(obj_state_diff)

    # Diff for moveable object dstates:
    obj_dstate_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(obj_dstate1, obj_dstate2, theta_weight)
    obj_dstate_diff = torch.sum(obj_dstate_diff)

    # Return one weighted measure:

    state_w, dstate_w, bodies_w, obj_state_w, obj_dstate_w = weight_list

    return (vine_state_diff * state_w) + \
           (vine_dstate_diff * dstate_w) + \
           (bodies_diff * bodies_w) + \
           (obj_state_diff * obj_state_w) + \
           (obj_dstate_diff * obj_dstate_w)


#----------------------------------------------------------------------- RRT Goal Tests


def last_body_goal_test(state_obj: StateInfo, 
                        max_bodies,
                        vine_radius, # in m, for consistency
                        goal_coords, # (x, y) given in m
                        goal_radius  # given in m
                        ):
    '''
    Simple goal test: if the last body is anywhere within the 
    goal region, then returns True (otherwise returns False).

    NOTE: assumes both the vine body's and the goal region's geometries
          are simple circles.
    '''

    num_bodies = state_obj["bodies"].reshape(B * 1)

    # print(state_obj["state"].reshape(B * max_bodies, 3).shape)
    # print(state_obj["state"].reshape(B * max_bodies, 3)[num_bodies - 1].shape)

    last_body_x, last_body_y, last_body_theta = state_obj["state"].reshape(B * max_bodies, 3)[num_bodies - 1].squeeze()
    min_dist_before_collision = goal_radius + vine_radius

    distance = torch.sqrt((goal_coords[0] - last_body_x)**2 + (goal_coords[1] - last_body_y)**2)

    return distance < min_dist_before_collision


#----------------------------------------------------------------------- Everything else for RRT

def getRandomState(walls, max_bodies, num_moveable_objs,
                    velocity_cap, curr_bodies):
    '''
    Generates completely random state for kinodynamic RRT. 
    Does not actually have to be (and isn't likely to be)
    an achievable state for the sim. Just being used to help the
    algorithm explore the state space.

    NOTE: everything is randomly sampled in mm, then converted to non-dim units
          (which is actually what the sim uses)

          state/dstate: (B, max_bodies * 3)
          bodies: (B, 1)
          moveable_obj_state/dstate: (B, num_moveable_objs, 3)

    NOTE's on args:
        - velocity_cap: an arbitrary constraint (change if you wish)
        - curr_bodies: each step should grow exactly 1 more body, so this is used
                       as a constraint
    '''

    xlim_mm, ylim_mm = find_borders(walls)
    MAX_DEG = 360
    MIN_X = xlim_mm[0]; MAX_X = xlim_mm[1] + 1
    MIN_Y = ylim_mm[0]; MAX_Y = ylim_mm[1] + 1

    # Randomly sample vine's state (within walls' borders):    
    rand_state = torch.zeros((B, max_bodies*3))
    rand_state[:, 0::3] = MIN_X + torch.rand(B, max_bodies) * (MAX_X - MIN_X)
    rand_state[:, 1::3] = MIN_Y + torch.rand(B, max_bodies) * (MAX_Y - MIN_Y)
    rand_state[:, 2::3] = torch.rand(B, max_bodies) * MAX_DEG

    # Randomly sample vine's dstate:
    rand_dstate = torch.zeros((B, max_bodies*3))
    rand_dstate[:, 0::3] = torch.rand(B, max_bodies) * velocity_cap
    rand_dstate[:, 1::3] = torch.rand(B, max_bodies) * velocity_cap
    rand_dstate[:, 2::3] = torch.rand(B, max_bodies) * MAX_DEG

    # Grow body:
    rand_bodies = torch.zeros((B, 1)); rand_bodies[:, 0] = curr_bodies + 1

    # Randomly sample dynamic objects' state:
    rand_obj_state = torch.zeros((B, num_moveable_objs, 3))
    rand_obj_state[:, :, 0::3] = MIN_X + torch.rand(B, num_moveable_objs, 1) * (MAX_X - MIN_X)
    rand_obj_state[:, :, 1::3] = MIN_Y + torch.rand(B, num_moveable_objs, 1) * (MAX_Y - MIN_Y)
    rand_obj_state[:, :, 2::3] = torch.rand(B, num_moveable_objs, 1) * MAX_DEG

    # Randomly sample dynamic objects' dstate:
    rand_obj_dstate = torch.zeros((B, num_moveable_objs, 3))
    rand_obj_dstate[:, :, 0::3] = torch.rand(B, num_moveable_objs, 1) * velocity_cap
    rand_obj_dstate[:, :, 1::3] = torch.rand(B, num_moveable_objs, 1) * velocity_cap
    rand_obj_dstate[:, :, 2::3] = torch.rand(B, num_moveable_objs, 1) * MAX_DEG

    # Convert everything to non-dim units for the sim:
    rand_state = ND(rand_state)
    rand_dstate = ND(rand_dstate)
    rand_obj_state = ND(rand_obj_state)
    rand_obj_dstate = ND(rand_obj_dstate)

    return StateInfo(rand_state, rand_dstate, rand_bodies, rand_obj_state, rand_obj_dstate)


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

    '''
    The algorithm:
    1. Randomly sample a state (randomize state, dstate, bodies, moveable_obj_state, moveable_obj_dstate) => x_rand
    2. Find x_nearest => closest state to x_rand
    3. Find x_new by propagating from x_nearest using some randomized controls (p, l0, etc.)
        - x_rand is disposable after this (likely not even reachable)
    4. Add x_new to the tree (returns path if it's a goal state)
    5. Repeat
    '''

    # Initialization for RRT:

    GOAL_COORDS_M = (1.225, 0.3)
    GOAL_RADIUS_M = 0.2
    
    start_state = StateInfo(vine_state, vine_dstate, bodies,
                            moveable_obj_pose, moveable_obj_dstate)

    rrt_tree = RRTTree(start_state, distance_function=euclidean_distance,
                       goal_test=last_body_goal_test,
                       vine_radius=radius_m, 
                       goal_coords=GOAL_COORDS_M, goal_radius=GOAL_RADIUS_M,
                       max_bodies=max_bodies, num_moveable_objs=num_moveable_objs)
    path_to_goal = None

    #NOTE: assume each step grows by one body (updated each iter)
    curr_bodies = start_state["bodies"] 

    RRT_ITERS = 1000
    VEL_CAP = 1000 # for random sampling dstates

    BEND_ANGLE_BOUND = 3.33 #NOTE: ripped from sst(); could change?
    bend_length_scale = torch.tensor(0.0018) #FIXME: should vary and depend on actuators, but not sure how to yet
    spam_moment_scale = 1.0                  #FIXME: may be same problem as above

    print("\nStarting RRT!")

    for iter in range(1, RRT_ITERS+1):
        rand_state = getRandomState(walls, max_bodies, num_moveable_objs,
                                   velocity_cap=VEL_CAP, curr_bodies=curr_bodies)

        nearest_node = rrt_tree.find_nearest_to(rand_state) # Returns Node in graph
        nearest_state = nearest_node.info

        # Propagate from nearest_state by giving random controls to the sim:
        new_bend_angle = np.random.uniform(BEND_ANGLE_BOUND*-1, BEND_ANGLE_BOUND, B)
        new_bend_angle = 1.0 / new_bend_angle

        # Each shaped (1,)
        p, l0 = find_actuator_params(predict, act_params, torch.tensor(new_bend_angle))

        # Make shape (max_bodies,), as documented in VineParams
        p = p.repeat(max_bodies)
        l0 = l0.repeat(max_bodies)

        vine_params.spam_p = p
        vine_params.spam_l0 = l0
        vine_params.bend_length_scale = bend_length_scale
        vine_params.spam_moment_scale = spam_moment_scale

        # Find next state to add to the tree (run the sim)
        try:
            new_state, new_dstate, new_bodies, new_obj_state, new_obj_dstate = step(
                vine_params, init_heading, init_x, init_y,
                nearest_state["state"], nearest_state["dstate"],
                nearest_state["bodies"],
                nearest_state["moveable_obj_pose"], nearest_state["moveable_obj_dstate"]
            )

            new_tree_state = StateInfo(new_state, new_dstate, new_bodies, new_obj_state, new_obj_dstate,
                                       p, l0)

            path_to_goal = rrt_tree.add_edge(nearest_node, new_tree_state)
            if path_to_goal is not None:
                print("RRT has found a path")
                break

        except Exception as e:
            print(f"Error encountered on RRT iter {iter}:\t{e}")
            raise e # just for debugging

        print(f"\r Iteration #{iter:<40}", end="", flush=True)

    # Use the path to create the animation (saved as a gif)

    print("\nRRT finished...")
    

    GIF_PATH="xnew_rrt/sim_animation"

    if path_to_goal is not None: 

        print("RRT found a path!")
        print("Producing gif...")

        for frame_idx in len(path_to_goal):
            draw_frame(frame_idx, path_to_goal[frame_idx],
                    walls, static_obj_poses, vine_params,
                    vine_radius_mm=radius_m*1000,
                    gif_path=GIF_PATH)

        print("Gif generation complete!")

    else:
        print("RRT did not find a path!")