import pygame
import math
import numpy as np
import os
import torch

from .si import len_to_mm
from .vine import VineParams
from .sst import get_last_body_length

regular_font = None
bold_font = None

# Global references to surfaces so we can do fast overlay
_display_surf = None       # Main display

_text_surf = None          # Text surface (for stats)

_tree_surf = None          # Accumulated states in the tree
_live_surf = None          # Current "next" state
_sst_surf = None           # For persistent SST stuff
_sst_active_surf = None    # For non-persistent SST stuff

_obstacles = None          # The obstacle list (x1,y1,x2,y2)

_dynamic_obstacles = set()  # All unique dynamic object positions

_screen_width = 1200
_screen_height = 800

# Transform parameters: scale mm->pixels, and offset so we see everything nicely
_scale = 1.5
_offset_x = 100.0
_offset_y = 50.0

save_pygame_folder_path = None  # Folder to save pygame frames, if needed

#---------------------------------- Helpers (non-draw functions)


MM = lambda x: len_to_mm(x)          # internal length -> mm (for display)
def _obb_corners_mm(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c * lx - s * ly), MM(cy + s * lx + c * ly))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def _to_screen(x, y):
    """
    Transform world (vine) coordinates into screen coordinates.
    By default, we shift x by +_offset_x, 
    and invert y about _offset_y (so that larger y is drawn lower).
    """

    sx = _offset_x + x * _scale
    sy = _offset_y + y * _scale
    return int(sx), int(sy)


def _to_screen_array(x_array, y_array):
    assert x_array.ndim == 1 or x_array.ndim == 2, f"Expected 2D or 1D array, got {x_array.ndim}"
    assert x_array.shape == y_array.shape
    
    # Vectorized version of _to_screen
    sx = _offset_x + x_array * _scale
    sy = _offset_y + y_array * _scale
    return sx, sy


def _get_blue_red_color_scale(t, max_val=50.0):
    """
    Linearly blend from blue at time=0 to red at time >= max_time.
    Feel free to tweak the color scale to your liking.
    """
    ratio = min(t / max_val, 1.0)
    r = int(255 * ratio)
    g = 120
    b = int(255 * (1.0 - ratio))
    return (r, g, b)


def _compute_vine_points(params, cspace, bodies, x0, y0, heading0):
    
    batch_size = cspace.shape[0]
    
    assert cspace.ndim == 2
    assert cspace.shape[1] == params.max_bodies * 3
    assert bodies.shape == (batch_size,)
    
    heading0 = np.asarray(heading0, dtype=np.float32)
    x0 = np.asarray(x0, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.float32)
    bodies = np.asarray(bodies, dtype=np.int32)
    cspace = np.asarray(cspace, dtype=np.float32)
        
    angles = cspace[:, 2::3]

    # last_len = cspace[:, params.max_bodies, -1]
    last_len = torch.vmap(get_last_body_length, in_dims=(0, 0))(torch.tensor(cspace), torch.tensor(bodies))
    last_len = last_len.cpu().numpy()

    n_bodies = bodies

    global_angle_full = heading0 + np.cumsum(angles, axis=1)
    c_ = np.cos(global_angle_full)
    s_ = np.sin(global_angle_full)
    
    full_lengths = np.full((batch_size, params.max_bodies,), fill_value=params.body_length.cpu().numpy())
    arange = np.arange(batch_size)
    full_lengths[arange, n_bodies-1] = last_len
    
    tip_x = x0 + np.cumsum(full_lengths * c_, axis=1)
    tip_y = y0 + np.cumsum(full_lengths * s_, axis=1)
    
    antitip_x = tip_x - 1.0 * full_lengths * c_
    antitip_y = tip_y - 1.0 * full_lengths * s_
    center_x = tip_x - 0.5 * full_lengths * c_
    center_y = tip_y - 0.5 * full_lengths * s_
    
    # Exception: the last link's center is the tip
    center_x[:, n_bodies-1] = tip_x[:, n_bodies-1]
    center_y[:, n_bodies-1] = tip_y[:, n_bodies-1]
    
    return antitip_x, antitip_y, tip_x, tip_y, center_x, center_y, n_bodies


#---------------------------------- Draw Helpers

def _draw_obstacles():
    """
    Draw the obstacles (rectangles) onto the tree surface,
    so they don't need to be redrawn each frame.
    Obstacles is a list of [x1, y1, x2, y2] rectangles (in non-dim units).
    """

    if _obstacles is not None:
        for obs in _obstacles:

            # Convert non-dim internal units to mm
            x, y, theta, hw, hh = obs
            corners = _obb_corners_mm(x, y, theta, hw, hh)
            
            # # Convert mm to screen coordinates
            # left, top = _to_screen(x1, y1)

            # width = int((x2 - x1) * _scale)
            # height = int((y2 - y1) * _scale)

            corners = [(_to_screen(x, y)) for x, y in corners]
            
            # Draw a brown rectangle for the obstacle
            # pygame.draw.rect(_tree_surf, (255, 228, 181), (left, top, width, height))
            pygame.draw.polygon(_tree_surf, (255, 228, 181), corners)
            
            # Draw a black border around the obstacle
            # pygame.draw.rect(_tree_surf, (0,0,0), (left, top, width, height), 4)
            pygame.draw.polygon(_tree_surf, (0, 0, 0), corners, 4)


def _draw_dynamic_obstacles(new_dynamic_obstacles=None, sim_params=None):
    '''
    Similar to draw_obstacles, but for dynamic obstacles,
    whose position may be changed over time (thus requiring them to be re-drawn).
    '''


    # For redrawing dynamic objects when called by clear_surfaces
    # (In this casse, just redraw everything cached in _dynamic_obstacles set)
    if new_dynamic_obstacles is None or new_dynamic_obstacles.size == 0 or sim_params is None:
        for x, y, theta, hh, hw in _dynamic_obstacles:
            corners = _obb_corners_mm(x, y, theta, hw, hh)
            pygame.draw.polygon(_tree_surf, (173, 216, 230), corners)
            pygame.draw.polygon(_tree_surf, (0, 0, 0), corners, width=2)

    else:
        for i in range(new_dynamic_obstacles.shape[0]): # skip batch or steps*batch dim
            for obj_idx, obj in enumerate(new_dynamic_obstacles[i]):

                x, y, theta = tuple(obj)

                hw = float(sim_params.obj_hw[obj_idx].item())
                hh = float(sim_params.obj_hh[obj_idx].item())

                _dynamic_obstacles.add((x.item(), y.item(), theta.item(), hw, hh))

                corners = _obb_corners_mm(x.item(), y.item(), theta.item(), hw, hh)

                # Scale to screen:
                for i in range(len(corners)):
                    scaled_x = corners[i][0] / 1000
                    scaled_y = corners[i][1] / 1000

                    scaled_x, scaled_y = _to_screen(scaled_x, scaled_y)
                    corners[i] = (scaled_x, scaled_y)

                pygame.draw.polygon(_tree_surf, (173, 216, 230), corners)
                pygame.draw.polygon(_tree_surf, (0, 0, 0), corners, width=2)


def clear_all_surfaces():
    """
    Clear all surfaces to transparent.
    This is useful to clear the tree surface before drawing a new state.
    """
    global _tree_surf, _live_surf, _sst_surf, _sst_active_surf
    _tree_surf.fill((0,0,0,0))
    _live_surf.fill((0,0,0,0))
    _sst_surf.fill((0, 0, 0, 0))
    _sst_active_surf.fill((0, 0, 0, 0))
    _display_surf.fill((255, 250, 240)) 


    _draw_obstacles()

    _draw_dynamic_obstacles()   


def _draw_vine(surface, params, points, circle_col=None, alpha=255, draw_circles=False, circle_thickness=5):
    '''
    Circle col can be None (blue), a single color (RGBA), or 
    an array of colors of shape (batch_size, n_bodies, 4).
    
    '''
    antitip_x, antitip_y, tip_x, tip_y, center_x, center_y, n_bodies = points

    line_thickness  = max(1, int(5 * _scale))
    circle_thickness= 6 # max(1, int(circle_thickness * _scale))
    circle_radius   = max(1, int(params.radius * _scale))
    
    line_col = _get_blue_red_color_scale(0.0) + (alpha,)
    line_col_darker = (line_col[0] * 0.2, line_col[1] * 0.2, line_col[2] * 0.2, alpha)
    
    if circle_col is None:
        circle_col = (line_col[0] * 0.5, line_col[1] * 0.5, line_col[2] * 0.5, alpha)
            
    atx, aty = _to_screen_array(antitip_x, antitip_y)
    tx, ty = _to_screen_array(tip_x, tip_y)
    cx, cy = _to_screen_array(center_x, center_y)
    
    batch_size = atx.shape[0]
    
    # Draw outline
    for i in range(batch_size):
        for j in range(n_bodies[i]):
            if draw_circles:
                pygame.draw.circle(surface, (0, 0, 0, alpha), (cx[i, j], cy[i, j]), circle_radius + 3, 0)

            
    for i in range(batch_size):
        for j in range(n_bodies[i]):

            atx_x = atx[i, j].item(0)
            aty_y = aty[i, j].item(0)
            tx_x = tx[i, j].item(0)
            ty_y = ty[i, j].item(0)

            pygame.draw.line(surface, line_col_darker, (atx_x, aty_y), (tx_x, ty_y), line_thickness + 4)
            pygame.draw.line(surface, line_col, (atx_x, aty_y), (tx_x, ty_y), line_thickness)
            
            this_circle_col = circle_col[i, j] if isinstance(circle_col, np.ndarray) else circle_col
            this_circle_col_dark = (this_circle_col[0] * 0.5, this_circle_col[1] * 0.5, this_circle_col[2] * 0.5, alpha)
                        
            if draw_circles:
                pygame.draw.circle(surface, this_circle_col, (cx[i, j], cy[i, j]), circle_radius, 0)
                # Draw slightly darker circle outline
                pygame.draw.circle(surface, this_circle_col_dark, (cx[i, j], cy[i, j]), circle_radius, 3)


#---------------------- init_vis(): For initializing stuff needed for render

def init_vis(figsize=(12, 8), cfg_obstacles = None, dynamic_obstacles = None, start=None, goal=None, save_pygame_folder=None,
             sim_params=None):
    """
    Initialize pygame, create a main display, create the 
    surfaces for tree and live states, draw obstacles, etc.
    The 'figsize' from old code is used to set the window size in 'inches',
    so we just multiply to get pixel dimensions. (12,8) -> (1200,800).

    NOTE: cfg_obstacles := static obstacles stored as [x1, y1, x2, y2] in whatever units env_loader uses (meters)
          sim_params stores hw, hh so we can actually convert nd => pixels (which is why we store both representations)
    """

    # In case there are no dynamic obstacles in the scene:
    if dynamic_obstacles.size == 0: 
        dynamic_obstacles = None
    
    global _display_surf, _text_surf, _tree_surf, _live_surf, _sst_surf, _sst_active_surf
    global _screen_width, _screen_height
    global _obstacles, _dynamic_obstacles
    global save_pygame_folder_path
    
    # Set the offset and scale to fit the obstacles
    # NOTE: not updated for dynamic obstacles (since static walls are more likely to affect scaling)
    global _offset_x, _offset_y, _scale
    margin = 60
    min_x = min(o[0] * 1000 for o in cfg_obstacles)
    min_y = min(o[1] * 1000 for o in cfg_obstacles)
    max_x = max(o[2] * 1000 for o in cfg_obstacles)
    max_y = max(o[3] * 1000 for o in cfg_obstacles)
    
    width = max_x - min_x + margin
    height = max_y - min_y + margin
    _scale = min(_screen_width / width, _screen_height / height).item()
    _offset_x = (-min_x * _scale + margin/2)
    _offset_y = (-min_y * _scale + margin/2)

    pygame.init()
    
    global regular_font, bold_font
    regular_font = pygame.font.SysFont("Monospace", 20)
    bold_font = pygame.font.SysFont("Go Medium", 10, bold=True) # None 
    
    _screen_width = int(figsize[0] * 100)
    _screen_height = int(figsize[1] * 100)

    _display_surf = pygame.display.set_mode((_screen_width, _screen_height))
    pygame.display.set_caption("Vine Visualization")
    
    # Create a surface for text
    _text_surf = pygame.Surface((_screen_width, _screen_height), pygame.SRCALPHA)
    _text_surf.fill((0, 0, 0, 0))  # Fill with transparent background

    # Create offscreen surfaces for the tree and the live state
    _tree_surf = pygame.Surface((_screen_width, _screen_height), pygame.SRCALPHA)
    _live_surf = pygame.Surface((_screen_width, _screen_height), pygame.SRCALPHA)
    _sst_surf = pygame.Surface((_screen_width, _screen_height), pygame.SRCALPHA)
    _sst_active_surf = pygame.Surface((_screen_width, _screen_height), pygame.SRCALPHA)
    
    # Store the obstacles so we can re-draw them in the background each time
    # Read in static obstacles using sim_params; shaped (x, y, theta, hw, hh)
    _obstacles = torch.cat([
            sim_params.obstacle_pose,             # (n_obs, 3)
            sim_params.obstacle_hw.unsqueeze(1),  # (n_obs, 1)
            sim_params.obstacle_hh.unsqueeze(1),  # (n_obs, 1)
        ], dim=1)

    _obstacles = _obstacles.cpu().numpy()

    # Fill the tree surface with a transparent background initially
    _tree_surf.fill((0,0,0,0))
    _live_surf.fill((0,0,0,0))
    _sst_surf.fill((0, 0, 0, 0))
    _sst_active_surf.fill((0, 0, 0, 0))

    # Optionally draw obstacles on the tree surface immediately (so they are behind everything)
    _draw_obstacles()
    _draw_dynamic_obstacles(dynamic_obstacles, sim_params)
    
    save_pygame_folder_path = save_pygame_folder

    # Clear the display
    _display_surf.fill((255, 255, 255))
    pygame.display.flip()

#---------------------- render(): For setting up the sub-menus created in init_vis()

def render():
    """
    Blit (copy) the tree surface first, then overlay the live surface, 
    then flip the display. This effectively layers them for speed.
    """

    def flip(blit):
        '''
        Flip a blit upside down to match matplotlib coordinate system
        '''
        return pygame.transform.flip(blit, False, True)


    # In a typical main loop, you may also want to handle pygame events 
    # so the window can be closed nicely:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # Draw the tree surface first (it has obstacles, old vines, etc.)
    tree_surf_with_alpha = _tree_surf.copy()
    tree_surf_with_alpha.set_alpha(255)  # Set transparency level (0-255)
    _display_surf.blit(flip(tree_surf_with_alpha), (0,0))
    _display_surf.blit(flip(_sst_surf), (0,0))
    
    # Overlay the live surface
    _display_surf.blit(flip(_live_surf), (0,0))
    _display_surf.blit(flip(_sst_active_surf), (0,0))
    
    # Overlay the text surface
    _display_surf.blit(_text_surf, (0,0))
    
    # Finally update the screen
    pygame.display.flip()
    
    # Save to the folder save_pygame_folder_path
    if save_pygame_folder_path is not None:
        # Create the folder if it doesn't exist
        if not os.path.exists(save_pygame_folder_path):
            os.makedirs(save_pygame_folder_path)
        num_items = len(os.listdir(save_pygame_folder_path))
        pygame.image.save(_display_surf, f"{save_pygame_folder_path}/frame_{num_items}.png")
    
    # Clear it (fully transparent)
    _live_surf.fill((0,0,0,0))
    _sst_active_surf.fill((0,0,0,0))
    _text_surf.fill((0, 0, 0, 0))
    
    # Start with a white background
    _display_surf.fill((255, 255, 255))

#----------------------------- Draw functions used outside render

def draw_witness(tree, idx, radius, color=(0, 0, 0, 100)):
    """
    Draw each witness's tip position as a small circle on _sst_surf.
    Alpha: 255 is opaque, 0 is transparent.
    """
    global _sst_surf
    wx, wy, _ = tree._witness_positions[idx]
    sx, sy = _to_screen(wx, wy)
    pygame.draw.circle(_sst_surf, color, (sx, sy), int(radius * _scale), 2)


def draw_tips(tips, color=(0, 0, 0), costs=None):
    """
    Draw the tips as small black lines on the _sst_surf.
    """
    global _sst_surf
    
    if costs is not None: 
        assert costs.shape[0] == tips.shape[0], f"Expected {tips.shape[0]} costs, got {costs.shape}"
        assert costs.ndim == 1
        global bold_font
        
    max_cost = None
    if costs is not None and costs.shape[0] > 0:
        # Normalize costs to [0, 1]
        min_cost = min(0, np.min(costs))
        max_cost = max(6, np.max(costs))
        
        costs_norm = (costs - min_cost) / (max_cost - min_cost)
    
    for idx in range(tips.shape[0]):
        x, y, theta = tips[idx]
        x, y = _to_screen(x, y)
        x2 = x + math.cos(theta) * 15
        y2 = y + math.sin(theta) * 15
        
        # color = np.random.random(3) * 255
        
        
        # cost_surface = bold_font.render(f"{int(costs[idx])}", True, (255, 55, 55))
        # cost_rect = cost_surface.get_rect()
        # cost_rect.topleft = (x - 7, y - 10)
        # _sst_surf.blit(flip(cost_surface), cost_rect)
            
        if max_cost is not None:
            # Set the color to a scale from blue to red from cost [0 - 10]
            # If the cost if out of this bound make it cyan
            if costs[idx] < 0 or costs[idx] > max_cost:
                color = (0, 255, 255)
            else:
                color = _get_blue_red_color_scale(costs_norm[idx], max_val=1)
                
        # small black circle
        pygame.draw.circle(_sst_surf, color, (x, y), 6)
        # small line
        # pygame.draw.line(_sst_surf, color, (x, y), (x2, y2), 2)


def draw_points(points, costs, color=(0, 0, 0)):
    """
    Draw the points (N, 3) (x, y, theta) as small black circles on the _sst_surf,
    and each circle has a line coming out to indicate the heading.
    """
    global _sst_surf
    
    
    max_cost = np.max(costs)
    range = max_cost - np.min(costs)
    if range < 1e-6:
        # If all costs are the same, just set max_cost to 1
        range = 1.0
    cost_norm = (costs - np.min(costs)) / range
    
    for point, cost in zip(points, cost_norm):
        x, y, theta = point
        x, y = _to_screen(x, y)
        x2 = x + math.cos(theta) * 10
        y2 = y + math.sin(theta) * 10
        
        color = _get_blue_red_color_scale(cost, max_val=1)
        
        # small circle 
        pygame.draw.circle(_sst_surf, color, (x, y), int(3 * _scale))
        # small line
        pygame.draw.line(_sst_surf, color, (x, y), (x2, y2), 3)


def draw_stats(sst_params, sst_iter, total_iters, num_active_nodes, num_total_nodes, num_reps, num_witnesses, costs):
    """
    Write some stats on the right top corner of the screen, line by line.
    """
    global _text_surf
    global _display_surf
    global _screen_width, _screen_height
    global _offset_x, _offset_y
    global _scale
    
    # Clear the text surface
    _text_surf.fill((0, 0, 0, 0))
    
    # Set the font and size
    global regular_font
    
    min_in_queue = -1 if sst_params.solutions.empty() else \
                     sst_params.solutions.queue[0].first
    
    # Prepare lines
    lines = [
        f"Cell radius:        {sst_params.δs:.2f}",
        f"δBN (sampling):     {sst_params.δBN:.2f}",
        f""
        f"Iteration:          {sst_iter} / {int(total_iters)}",
        f"Active/total nodes: {num_active_nodes} / {num_total_nodes}",
        f"Reps/Witnesses:     {num_reps} / {num_witnesses}",
        f"Min/Max sol cost:   {np.min(costs):.1f} / {np.max(costs):.1f}",
        f"Solutions:          {sst_params.solutions.qsize()} (min: {min_in_queue:.2f})",
    ]
    
    max_width = max(regular_font.size(line)[0] for line in lines)
    
    # Render each line separately
    y_offset = 10
    for text_line in lines:
        line_surface = regular_font.render(text_line, True, (0, 0, 0))
        line_rect = line_surface.get_rect()
        line_rect.topleft = (_screen_width - max_width - 10, y_offset)
        _text_surf.blit(line_surface, line_rect)
        y_offset += line_surface.get_height() + 5


def draw_goal(xy, radius):
    '''
    Draw a filled circle at the position
    '''
    global _live_surf
    sx, sy = _to_screen(xy[0], xy[1])
    pygame.draw.circle(_live_surf, (220, 20, 20, 40), (sx, sy), int(radius * _scale), 0)
    pygame.draw.circle(_live_surf, (170, 0, 0), (sx, sy), int(radius * _scale), 4)


def draw_dead_state(params, state, dynamic_positions, bodies, x0, y0, heading0):
    points = _compute_vine_points(params, state, bodies, x0, y0, heading0)
    _draw_vine(_tree_surf, params, points, alpha=120)
    _draw_dynamic_obstacles(dynamic_positions, sim_params=params)
