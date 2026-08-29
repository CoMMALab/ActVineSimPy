'''
This file is just for animating the sim (RRT is not implemented here)
It can also generate gifs
'''

# TODO:
# 1. Get the sim to run first 
# 2. Surround it with simple RRT (establish goal states)

import torch, math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation

# Sim modules:
from dvsim import si
import dvsim.solver as solver

from dvsim.dynamic_vine import step 
from dvsim.vine import create_state_batched, init_state_batched


#--------------------------- Global Variables (for update())

fig, ax = None, None
ani = None
recorded_frames = [] # for saving gifs

vine_params = None
init_heading = None
init_x = None
init_y = None
sim_state = None

num_moveable_obj = 0
body_radius_mm = 0.0

#-------------------------- Helper functions

MM = lambda x: si.len_to_mm(x)  # internal length -> mm (for display)

def _obb_corners_mm(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c * lx - s * ly), MM(cy + s * lx + c * ly))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def init_params(max_bodies=40, grow_rate_mps=0.3,
                bend_length_scale=None, spam_moment_scale=None, 
                p=None, l0=None, radius_m=0.0125,
                static_objects=None):

    params = si.vine_params_si(max_bodies=max_bodies, obstacles_m=static_objects, grow_rate_mps=grow_rate_mps,
                               radius_m=radius_m)

    # params.stiffness_mode = 'spam'
    if bend_length_scale is not None: params.bend_length_scale = torch.tensor(bend_length_scale)
    if spam_moment_scale is not None: params.spam_moment_scale = torch.tensor(spam_moment_scale)
    if p is not None: params.spam_p = torch.full((max_bodies,), float(p))
    if l0 is not None: params.spam_l0 = torch.full((max_bodies,), float(l0))

    return params


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


def render(i, curr_sim_state=None):
        '''
        Draw the current scene completely based on the current sim_state
        '''
        ax.clear()

        # Use global record if no alternative is provided
        if curr_sim_state == None:
            curr_sim_state = sim_state

        # Draw static objs:
        for (pose, hw, hh) in static_obj_poses:
            corners = _obb_corners_mm(pose[0], pose[1], pose[2], hw, hh)
            ax.add_patch(Polygon(corners, closed=True, facecolor=(1, .89, .71), 
                                 edgecolor="k", zorder=1))

        # Draw dynamic objs:
        dyn_obj_poses = curr_sim_state["moveable_obj_pose"]

        for k in range(num_moveable_obj):
            obj_x, obj_y, obj_theta = [float(v) for v in dyn_obj_poses[0, k]]
            corners = _obb_corners_mm(obj_x, obj_y, obj_theta, float(vine_params.obj_hw[k]),
                                      float(vine_params.obj_hh[k]))
            ax.add_patch(Polygon(corners, closed=True, facecolor=(.55, .8, .95), 
                                 edgecolor="b", lw=2, zorder=3))

        # Draw the vine's bodies:
        curr_state = curr_sim_state["state"]
        curr_bodies = curr_sim_state["bodies"]

        n = int(curr_bodies[0])
        xs = [MM(float(curr_state[0, 3 * j])) for j in range(n)] 
        ys = [MM(float(curr_state[0, 3 * j + 1])) for j in range(n)]


        if curr_sim_state==sim_state:
            obj_x, obj_y, obj_theta = [float(v) for v in dyn_obj_poses[0, k]]
            last_body_x = xs[-1]; last_body_y = ys[-1]

            # dist = math.sqrt(pow(obj_x - last_body_x, 2) + pow(obj_y - last_body_y, 2))

            print(f"Dist of last body to center of blue obj: {abs(obj_x - last_body_x)}, {abs(obj_y - last_body_y)}")



        ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)

        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), body_radius_mm, 
                                facecolor=(.3, .5, .95, .4), edgecolor=(.1, .2, .6), zorder=4))

        ax.plot([init_x[:, 0].item()], [init_y[:, 0].item()], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim_mm) 
        ax.set_ylim(*ylim_mm) 
        ax.set_aspect("equal")
        ax.set_xlabel("mm"); ax.set_title(f"Frame {i}")


def update(i):
    '''
    Basically does two things:
    1. Does one step of the simulator.
    2. Updates what gsimlobal vars need to be updated for the next step (specifically sim_state),
       then draws whatever the output of the sim was

    This is done indefinitely until interrupted.
    '''

    def close_later():
        timer = fig.canvas.new_timer(interval=100)
        timer.single_shot = True
        timer.add_callback(plt.close, fig)
        sim_state["close_timer"] = timer
        timer.start()

    if sim_state["halted"]:
        print("Stopping sim...")
        close_later()
        return

    try: 
        state, dstate, bodies, dyn_obj_pose, dyn_obj_dstate = step(
            vine_params, init_heading, init_x, init_y,
            sim_state["state"], sim_state["dstate"],
            sim_state["bodies"],
            sim_state["moveable_obj_pose"], sim_state["moveable_obj_dstate"]
        )

        sim_state["state"] = state
        sim_state["dstate"] = dstate
        sim_state["bodies"] = bodies
        sim_state["moveable_obj_pose"] = dyn_obj_pose
        sim_state["moveable_obj_dstate"] = dyn_obj_dstate

        # Also save the data for gif creation later:
        recorded_frames.append(
            {"state": sim_state["state"].clone(),
             "bodies": sim_state["bodies"].clone(),
             "dstate": sim_state["dstate"].clone(),
             "moveable_obj_pose": sim_state["moveable_obj_pose"].clone(),
             "moveable_obj_dstate": sim_state["moveable_obj_dstate"].clone()
             }
        )

        render(i)

    except IndexError:
        print(f"IndexError: vine likely ran out of bodies at frame {i}, stopping sim...")
        sim_state["halted"] = True
        close_later()
        return
    
    except Exception as e:
        print(f"Stopped at frame {i} ({type(e).__name__})")
        raise e
    
        close_later()
        return

    if not torch.isfinite(state).all():
        print("Some element in state is not finite, stopping sim")
        sim_state["halted"] = True
        close_later()
        return


def render_saved(i):
    '''
    Uses the info for each timestep from the sim to render and save a gif
    '''
    snap = recorded_frames[i]
    render(i, snap)


#------------------------------------- Main

'''
Progress Notes:

1. Vine does not react when making contact with moveable obstacles

2. Vine coils upon itself in several situations:
    - when it bumps into static obstacles => dramatic reaction
    - when the initial heading is changed to anything != 0 => massive coily mess
      that grows in on itself

'''


if __name__ == "__main__":

    B = 1 # unbatched => batch size of 1

    # All obstacles:
    
    walls = [(0.0, 0.0, 1.525, 0.1),
             (0.0, 0.9, 1.525, 1.0),
             (0.0, 0.0, 0.1, 1.0),
             (1.425, 0.0, 1.525, 1.0)]

    static_objs = [(0.7125, 0.3, 0.8125, 0.2)]

    dynamic_obj_coords = [(0.7125, 0.5000, 0.8125, 0.4000)]
    dynamic_obj_masses = [0.2]

    # Define params and dynamic object info:
    # NOTE: vine_params args are in SI units

    solver.cvxpylayer = None

    max_bodies = 40
    vine_params = init_params(max_bodies=max_bodies,
                              grow_rate_mps=0.5,
                            #   bend_length_scale=0.018,
                            #   spam_moment_scale=22000.0,
                            #   p=8000.0, 
                            #   l0=-0.04, 
                              static_objects = (walls + static_objs),
                              radius_m= 0.0125*4)

    if dynamic_obj_coords is not None:
        moveable_obj_pose = si.set_objects_si(vine_params, dynamic_obj_coords, 
                                            dynamic_obj_masses)
        num_moveable_obj = len(dynamic_obj_masses)
        moveable_obj_dstate = torch.zeros(B, num_moveable_obj, 3)
    else:
        moveable_obj_pose = None
        moveable_obj_dstate = None
        num_moveable_obj = 0

    # Initialize state:

    # In mm:
    init_x = torch.zeros(B, 1); init_x[:, 0] = 400
    init_y = torch.zeros(B, 1); init_y[:, 0] = 400

    init_heading = torch.zeros(1, 1)
    init_heading[:, 0] = 0 

    state, dstate = create_state_batched(B, max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(vine_params, state, bodies, init_heading, init_x, init_y)

    body_radius_mm = MM(float(vine_params.radius))

    # Define static obj poses:

    static_obj_poses = [([float(v) for v in vine_params.obstacle_pose[k]],
                          float(vine_params.obstacle_hw[k]), float(vine_params.obstacle_hh[k]))
                          for k in range(vine_params.obstacle_pose.shape[0])]

    # Find dims for the display
    # NOTE: the physical display is in inches, but objs in frame should be scaled in mm

    xlim_mm, ylim_mm = find_borders(walls)

    MAX_DIM = 12 # inches
    frame_width_mm = xlim_mm[1] - xlim_mm[0]
    frame_height_mm = ylim_mm[1] - ylim_mm[0]

    aspect_ratio = frame_height_mm / frame_width_mm
    if aspect_ratio <= 1:
        width_in = MAX_DIM
        height_in = MAX_DIM * aspect_ratio
    else:
        height_in = MAX_DIM
        width_in = MAX_DIM / aspect_ratio


    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # Run the sim:

    sim_state = {"state": state, "dstate": dstate,
                 "bodies": bodies,
                 "moveable_obj_pose": moveable_obj_pose,
                 "moveable_obj_dstate": moveable_obj_dstate,
                 "halted": False,
                 "close_timer": None
                 } 


    # For stopping when sim halts
    def frame_gen():
        i = 0
        while not sim_state["halted"]:
            yield i
            i += 1

    live_ani = FuncAnimation(fig, update, frames=frame_gen(), interval=50, blit=False, 
                        save_count=500, # for gif generation
                        repeat=False, cache_frame_data=False) 
    plt.show()

    # Save gif to specified dir:

    print("Will now create gif, please be patient...")
    save_ani = FuncAnimation(fig, render_saved, frames=len(recorded_frames), interval=50, blit=False)
    save_ani.save("xnew_rrt/gifs/dynamic_contact.gif", writer="pillow", fps=20)
    print("Done, now ending process...")

    sim_state.pop("close_timer", None)
    plt.close("all")
    del live_ani, save_ani
    