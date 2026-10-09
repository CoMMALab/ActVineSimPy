import torch, math, matplotlib, random, os, re

import numpy as np

from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import colorsys

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

# For creating GIFs:
from PIL import Image

scaling_info, model = get_or_train_model(act_params)
predict = get_prediction_function(scaling_info, model)

find_actuator_params = torch.vmap(find_actuator_params, in_dims=(None, None, 0))

#---------------------------------------------------------------------- Sim/Animation Helper Defs

B = 1 # this RRT implementation is unbatched
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


def draw_frame(frame_idx, sim_states: list[StateInfo], walls,
               static_obj_poses, vine_params, vine_radius_mm,
               goal_coords_m, goal_radius_m, num_moveable_objs,
               gif_path, vine_alpha):
    '''
    Given a list of sim_states (all states at the timestep = frame_idx),
    draw what all the explored scenes look like, all in one gif (similar to ActVine)
    '''

    width_in, height_in = find_frame_dims(walls, max_dim=12)
    xlim_mm, ylim_mm = find_borders(walls)

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Draw static objs:
    for (pose, hw, hh) in static_obj_poses:
        corners = _obb_corners_mm(pose[0], pose[1], pose[2], hw, hh)
        ax.add_patch(Polygon(corners, closed=True, facecolor=(1, .89, .71), 
                                edgecolor="k", zorder=1))

    # Draw goal radius:
    goal_x = goal_coords_m[0] * 1000; goal_y = goal_coords_m[1] * 1000
    goal_radius_mm = goal_radius_m * 1000
    ax.add_patch(Circle((goal_x, goal_y), goal_radius_mm,
                 facecolor="red", alpha=0.4, edgecolor=(.1, .2, .6), zorder=4))


    # Draw what actually changes per frame for this timestep:

    for path_idx, sim_state in enumerate(sim_states):

        if sim_state is None: continue

        # NOTE: for debugging, make each vine a diff color

        cmap = plt.get_cmap("tab20"); 
        VINE_COLOR = cmap(path_idx % 20)

        # default blue: facecolor = (.3, .5, .95)

        vine_state = sim_state.info["state"]
        moveable_obj_poses = sim_state.info["moveable_obj_pose"]
        num_bodies = sim_state.info["bodies"]    

        # Draw moveable objs:
        moveable_obj_poses = sim_state.info["moveable_obj_pose"]

        for k in range(num_moveable_objs):

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
            facecolor=VINE_COLOR, edgecolor=(.1, .2, .6), zorder=4,
            alpha=vine_alpha))

    # Set axes properties:
    ax.plot([init_x[:, 0].item()], [init_y[:, 0].item()], "g^", ms=9, zorder=6)
    ax.set_xlim(*xlim_mm) 
    ax.set_ylim(*ylim_mm) 
    ax.set_aspect("equal")
    ax.set_xlabel("mm"); ax.set_title(f"Frame {frame_idx}")

    # Create the png for this timestep (will be used to compile into gif later)
    fig.savefig(os.path.join(gif_path, f"f{frame_idx:04d}.png"), dpi=70, bbox_inches="tight")

    plt.close(fig)


def draw_frame_multi_gif(frame_idx, sim_states: list[StateInfo], walls,
                        static_obj_poses, vine_params, vine_radius_mm,
                        goal_coords_m, goal_radius_m, num_moveable_objs,
                        vine_alpha):
    '''
    Similar to draw frame: takes in a sim state from each path,
    but now each path gets its own gif.

    NOTE: if the goal is found, then it will have idx of 0 in sim_states (so just check that gif)
    '''

    width_in, height_in = find_frame_dims(walls, max_dim=12)
    xlim_mm, ylim_mm = find_borders(walls)

    PARENT_FOLDER = "xnew_rrt/sim_animation"
    VINE_COLOR = (.3, .5, .95) # default to blue

    # Save pngs to separate folders for each path found (all folders under "xnew_rrt/sim_animation")
    for path_idx, sim_state in enumerate(sim_states):

        if sim_state is None: continue

        fig, ax = plt.subplots(figsize=(width_in, height_in))

        # Draw static objs:
        for (pose, hw, hh) in static_obj_poses:
            corners = _obb_corners_mm(pose[0], pose[1], pose[2], hw, hh)
            ax.add_patch(Polygon(corners, closed=True, facecolor=(1, .89, .71), 
                                    edgecolor="k", zorder=1))

        # Draw goal radius:
        goal_x = goal_coords_m[0] * 1000; goal_y = goal_coords_m[1] * 1000
        goal_radius_mm = goal_radius_m * 1000
        ax.add_patch(Circle((goal_x, goal_y), goal_radius_mm,
                        facecolor="red", alpha=0.4, edgecolor=(.1, .2, .6), zorder=4))

        vine_state = sim_state.info["state"]
        moveable_obj_poses = sim_state.info["moveable_obj_pose"]
        num_bodies = sim_state.info["bodies"]


        # Draw moveable objs:
        moveable_obj_poses = sim_state.info["moveable_obj_pose"]

        for k in range(num_moveable_objs):

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
            facecolor=VINE_COLOR, edgecolor=(.1, .2, .6), zorder=4,
            alpha=vine_alpha))

        # Set axes properties:
        ax.plot([init_x[:, 0].item()], [init_y[:, 0].item()], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim_mm) 
        ax.set_ylim(*ylim_mm) 
        ax.set_aspect("equal")
        ax.set_xlabel("mm"); ax.set_title(f"Frame {frame_idx}")

        # Save this gif frame (png) to the rrt path's proper folder:

        rrt_path_folder = os.path.join(PARENT_FOLDER, f"path_{path_idx}")

        if not os.path.exists(rrt_path_folder):
            os.mkdir(rrt_path_folder)

        if os.path.exists(rrt_path_folder) and frame_idx == 0:
            # clear out the folder before throwing pngs in there:
            for root, dirs, files in os.walk(rrt_path_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))

        new_frame_path = os.path.join(rrt_path_folder, f"path_frame{frame_idx}.png")
        fig.savefig(new_frame_path, dpi=70, bbox_inches="tight")

        plt.close()


#----------------------------------------------------------------------- RRT Goal Tests

def last_body_goal_test(state_obj: StateInfo, 
                        max_bodies,
                        body_radius, # in m, for consistency
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

    last_body_x, last_body_y, last_body_theta = state_obj["state"].reshape(B * max_bodies, 3)[num_bodies - 1].squeeze()

    # Convert internal non-dims to meters:
    last_body_x = MM(last_body_x) / 1000; last_body_y = MM(last_body_y) / 1000

    min_dist_before_collision = goal_radius + body_radius

    distance = torch.sqrt((goal_coords[0] - last_body_x)**2 + (goal_coords[1] - last_body_y)**2)

    return distance < min_dist_before_collision


#------------------------------------------------------------------------ RRT Quality Measures

def euclidean_quality(node: Node, max_bodies, goal_coords_m):
    '''
    Just sees how far away the goal is from the given state's last body
    (Except for returned value, it's very similar to last_body_goal_test)
    '''

    num_bodies = node.info["bodies"].reshape(B * 1)
    
    last_body_x, last_body_y, last_body_theta = node.info["state"].reshape(B * max_bodies, 3)[num_bodies - 1].squeeze()

    # Convert internal non-dims to meters:
    last_body_x = MM(last_body_x) / 1000; last_body_y = MM(last_body_y) / 1000

    distance = torch.sqrt((goal_coords_m[0] - last_body_x)**2 + (goal_coords_m[1] - last_body_y)**2)
    return distance.item()


#----------------------------------------------------------------------- New Distance Func/Random Sampler (based on last body)

def get_random_body_position(walls):
    '''
    Samples random last body position/orientation within the wall boundaries.
    
    NOTE: pose is kept in mm
    '''

    xlim_mm, ylim_mm = find_borders(walls)
    MAX_DEG = 360
    MIN_X = xlim_mm[0]; MAX_X = xlim_mm[1] + 1
    MIN_Y = ylim_mm[0]; MAX_Y = ylim_mm[1] + 1

    random_last_body = torch.zeros((B, 3))

    random_last_body[:, 0] = MIN_X + torch.rand(B, 1) * (MAX_X - MIN_X)
    random_last_body[:, 1] = MIN_Y + torch.rand(B, 1) * (MAX_Y - MIN_Y)
    random_last_body[:, 2] = torch.rand(B, 1) * MAX_DEG

    return random_last_body


def get_last_body_distance(state_obj1: StateInfo, random_last_body: torch.tensor):
    '''
    Gets the distance of the given vine's last body and the randomly generated
    body (2nd arg)
    '''

    x2, y2, theta2 = random_last_body.squeeze()
    

    num_bodies1 = state_obj1["bodies"]
    x1, y1, theta1 = state_obj1["state"].reshape(B * max_bodies, 3)[num_bodies1 - 1].squeeze()

    # Convert vine last body pose to mm:
    x1 = MM(x1)
    y1 = MM(y1)
    theta1 = MM(theta1)

    # Find diff btw the two args:
    euclid_dist = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    theta_diff = torch.atan2(torch.sin(theta1 - theta2),
                             torch.cos(theta1 - theta2))

    return euclid_dist + theta_diff


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

    aux_goal_test_args = (max_bodies, radius_m, GOAL_COORDS_M, GOAL_RADIUS_M)

    rrt_tree = RRTTree(start_state,
                       goal_test=last_body_goal_test,
                       aux_goal_test_args=aux_goal_test_args)
    path_to_goal = None

    RRT_ITERS = 300

    BEND_ANGLE_BOUND = 3.33                  #NOTE: ripped from sst(); could change?
    bend_length_scale = torch.tensor(0.0018) #FIXME: should vary and depend on actuators, but not sure how to yet
    spam_moment_scale = 1.0                  #FIXME: may be same problem as above

    # Run RRT:

    print("\nStarting RRT!")

    for iter in range(1, RRT_ITERS+1):

        random_last_body_pose = get_random_body_position(walls)

        nearest_node = rrt_tree.find_nearest_to(random_last_body_pose,
                                                get_last_body_distance)
        nearest_state = nearest_node.info

        # Try random sampling again if nearest state exceeds max bodies:
        if nearest_state["bodies"] >= max_bodies:
            continue

        # Propagate from nearest_state by giving random controls to the sim:
        new_bend_angle = np.random.uniform(BEND_ANGLE_BOUND*-1, BEND_ANGLE_BOUND, B)
        new_bend_angle = 1.0 / new_bend_angle

        p, l0 = find_actuator_params(predict, act_params, torch.tensor(new_bend_angle)) # each (1,)

        p = p.repeat(max_bodies)   # Make shape (max_bodies,), as documented in VineParams
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

    # GIF_DIR = "xnew_rrt/sim_animation" <= previously used for gif with vine overlay
    # GIF_NAME = "test_gif.gif"

    ANI_FOLDER = "xnew_rrt/sim_animation"
    NUM_CLOSEST_PATHS = rrt_tree.get_num_leaves()
    # NUM_CLOSEST_PATHS = 10
    

    if path_to_goal is not None: 
        print("RRT found a path!")

    else:
        print("RRT did not find a path!")

    print(f"Animating the closest {NUM_CLOSEST_PATHS} paths...")

    aux_args = (max_bodies, GOAL_COORDS_M)
    all_paths = rrt_tree.get_closest_paths(NUM_CLOSEST_PATHS, euclidean_quality,
                                            aux_args)

    rrt_tree.print_diagnostics(euclidean_quality)

    print("Starting to draw frames...")

    # Find max timesteps you need to draw:

    saved_indices = []

    if path_to_goal is not None:
        all_paths.insert(0, path_to_goal)

    len_to_use = max([len(path) for path in all_paths])
        
    # Actually draw the gif:

    VINE_ALPHA = 0.6

    for frame_idx in range(len_to_use):

        # Creates a buncha images that are used to create the gif:
        # saved_indices.append(frame_idx)

        states_to_draw = []

        for path in all_paths:
            try:
                states_to_draw.append(path[frame_idx])
            except IndexError:
                # path is shorter than others, so just throw this in:
                states_to_draw.append(None)


        # draw_frame(frame_idx, sim_states=states_to_draw,
        #         walls=walls, static_obj_poses=static_obj_poses, vine_params=vine_params,
        #         vine_radius_mm=radius_m*1000,
        #         goal_coords_m=GOAL_COORDS_M, goal_radius_m=GOAL_RADIUS_M,
        #         num_moveable_objs=num_moveable_objs,
        #         gif_path=GIF_DIR,
        #         vine_alpha=VINE_ALPHA)

        draw_frame_multi_gif(frame_idx, states_to_draw, 
                            walls=walls, static_obj_poses=static_obj_poses, vine_params=vine_params,
                            vine_radius_mm=radius_m*1000,
                            goal_coords_m=GOAL_COORDS_M, goal_radius_m=GOAL_RADIUS_M,
                            num_moveable_objs=num_moveable_objs,
                            vine_alpha=VINE_ALPHA)

    print("Compiling gifs...")

    # For sorting files by name:
    def num_key(name):
        return int(re.search(r"\d+", name).group())

    for path_idx in range(len(all_paths)):

        path_folder = os.path.join(ANI_FOLDER, f"path_{path_idx}")
        names = sorted(os.listdir(path_folder), key=num_key)

        imgs = [Image.open(os.path.join(path_folder, name)) for name in names]
        out = os.path.join(ANI_FOLDER, f"path_{path_idx}.gif")
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=110, loop=0)

    print("Cleaning up...")

    # Delete all the path folders and their pngs
    for root, dirs, files in os.walk(ANI_FOLDER, topdown=False):
        for name in files:
            if name.endswith(".png"):
                os.remove(os.path.join(root, name))
        for name in dirs:
            if name.startswith("path"):
                full = os.path.join(root, name)
                os.rmdir(full)

    print("GIF generation complete!")
