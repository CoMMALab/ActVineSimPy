import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_box_config(filename: str):
    """
    Reads a config file with fields:
      bound: <float>
      start: <float> <float> <float>
      goal:  <float> <float> <float>
      ob_type: box
      obstacles:
        x1 y1 x2 y2
        ...
    Returns a dict with bound, start, goal, ob_type, obstacles (Nx4).
    """

    cfg = {
        'bound_x': None,
        'bound_y': None,
        'start': None,
        'goal': None,
        'scale': 1.0,
        'ob_type': None,
        'obstacles': []
    }
    reading_obstacles = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if reading_obstacles:
                parts = line.split()
                if len(parts) == 4:
                    cfg['obstacles'].append([float(p) for p in parts])
                continue

            if line.startswith("bound:"):
                cfg['bound_x'] = float(line.split(' ')[1].strip())
                cfg['bound_y'] = float(line.split(' ')[2].strip())
            elif line.startswith("start:"):
                values = line.split(':')[1].split()
                cfg['start'] = [float(v) for v in values]
            elif line.startswith("goal:"):
                values = line.split(':')[1].split()
                cfg['goal'] = [float(v) for v in values]
            elif line.startswith("goal_radius:"):
                cfg['goal_radius'] = float(line.split(':')[1].strip())
            elif line.startswith("ob_type:"):
                cfg['ob_type'] = line.split(':')[1].strip()
            elif line.startswith("scale:"):
                cfg['scale'] = float(line.split(':')[1].strip())
            elif line.startswith("obstacles:"):
                reading_obstacles = True


    # Make sure for obstacles, x1 < x2 and y1 < y2
    for i in range(len(cfg['obstacles'])):
        x1, y1, x2, y2 = cfg['obstacles'][i]
        if x1 > x2:
            cfg['obstacles'][i][0] = x2
            cfg['obstacles'][i][2] = x1
        if y1 > y2:
            cfg['obstacles'][i][1] = y2
            cfg['obstacles'][i][3] = y1
    
    cfg['obstacles'] = np.array(cfg['obstacles'], dtype=np.float32)
    
    # If lack goal radius, set to 100
    if 'goal_radius' not in cfg:
        cfg['goal_radius'] = 100.0 / cfg['scale']

    # Scale the obstacles
    cfg['obstacles'][:, :] *= cfg['scale']
    cfg['start'] = [cfg['start'][0] * cfg['scale'], cfg['start'][1] * cfg['scale'], cfg['start'][2]]
    cfg['goal'] = [cfg['goal'][0] * cfg['scale'], cfg['goal'][1] * cfg['scale'], cfg['goal'][2]]
    cfg['goal_radius'] = cfg['goal_radius'] * cfg['scale']
    cfg['bound_x'] *= cfg['scale']
    cfg['bound_y'] *= cfg['scale']
    
    return cfg

def save_box_config(filename: str, cfg):
    """
    Writes a config dict (with bound, start, goal, ob_type, obstacles)
    into the file in the same format used by load_box_config.
    """
    with open(filename, 'w') as f:
        f.write(f"bound_x: {cfg['bound_x']}\n")
        f.write(f"bound_y: {cfg['bound_y']}\n")
        f.write(f"start: {' '.join(map(str, cfg['start']))}\n")
        f.write(f"goal: {' '.join(map(str, cfg['goal']))}\n")
        f.write(f"ob_type: {cfg['ob_type']}\n")
        f.write(f"goal_radius: {cfg['goal_radius']}\n")
        
        if 'scale' in cfg:
            f.write(f"scale: {cfg['scale']}\n")
        
        f.write("obstacles:\n")
        for obs in cfg['obstacles']:
            f.write(" ".join(map(str, obs)) + "\n")

def generate_box_config():
    """
    Generates a random box configuration.
    """
    
    # Obstacle format: [x, y, x2, y2]
    # Parameters for the field and obstacles
    num_obstacles = 20
    field_width = 400
    field_height = 400
    obstacle_size = 40
    wt = 40

    # Generate random obstacles
    # x1 y1 x2 y2
    obstacles = []
    for _ in range(num_obstacles):
        x = random.randint(0, field_width - obstacle_size)
        y = random.randint(0, field_height - obstacle_size)
        obstacles.append([x, y, x + obstacle_size, y + obstacle_size])

    # Add rectangle wall around the field with thickness
    obstacles.append([-wt, -wt, field_width + wt, 0])  # Top wall
    obstacles.append([-wt, field_height, field_width + wt, field_height + wt])  # Bottom wall
    obstacles.append([-wt, -wt, 0, field_height + wt])  # Left wall
    obstacles.append([field_width, -wt, field_width + wt, field_height + wt])  # Right wall

    obstacles = np.array(obstacles)
    
    cfg = {
        'bound': field_width,
        'start': [1, 1, 0],
        'goal': [field_width - 1, field_height - 1, 0],
        'ob_type': 'box',
        'obstacles': obstacles
    }
    
    return cfg


def generate_divider_config():
    """
    Generates a random box configuration.
    """
    
    def round_to_25(x):
        '''
        in: mm
        round to nearest 25 mm
        out: mm
        '''
        return round(x / 25) * 25
    
    # Obstacle format: [x, y, x2, y2]
    # Parameters for the field and obstacles
    scale = 25.4 # Convert inches to mm
    num_obstacles = 15
    wall_thickness = 4 * 25 # align to 2.5 mm
    field_width = round_to_25(60 * scale) - wall_thickness - wall_thickness
    field_height = round_to_25(39 * scale) - wall_thickness - wall_thickness
    obstacle_size = 4 * scale
    
    start = [wall_thickness * 2, wall_thickness * 2, 1.51]
    goal = [field_width, wall_thickness * 2, 0]
    
    # x1 y1 x2 y2
    obstacles = []
    
    # Add rectangle wall around the field with thickness
    obstacles.append([-wall_thickness, -wall_thickness, field_width + wall_thickness, 0])  # Top wall
    obstacles.append([-wall_thickness, field_height, field_width + wall_thickness, field_height + wall_thickness])  # Bottom wall
    obstacles.append([-wall_thickness, -wall_thickness, 0, field_height + wall_thickness])  # Left wall
    obstacles.append([field_width, -wall_thickness, field_width + wall_thickness, field_height + wall_thickness])  # Right wall
    
    # Add a vertical divider 
    divider_x = field_width // 2
    obstacles.append([divider_x - wall_thickness/2, -wall_thickness, divider_x + wall_thickness/2, field_height/3 + wall_thickness])  # Vertical divider
    
    # Generate random obstacles
    while len(obstacles) < num_obstacles + 5:
        x = random.uniform(0, field_width - obstacle_size)
        y = random.uniform(0, field_height - obstacle_size)
        
        # Round to nearest 25 mm
        x = round_to_25(x)
        y = round_to_25(y)
                
        collides_goal = np.hypot(x + obstacle_size/2 - goal[0], y + obstacle_size/2 - goal[1]) < 200
        if collides_goal:
            continue
        
        collides_start = np.hypot(x + obstacle_size/2 - start[0], y + obstacle_size/2 - start[1]) < 200
        if collides_start:
            continue
        
        # check collision with any of the obstacles
        margin = 0
        collides = np.any([
            (x - margin < obs[2] and x + obstacle_size + margin > obs[0] and \
             y - margin < obs[3] and y + obstacle_size + margin > obs[1])
            for obs in obstacles
        ])
        if collides:
            continue
        
        obstacles.append([x, y, x + obstacle_size, y + obstacle_size])
        
    obstacles = np.array(obstacles)
    
    # Add wall_thickness to all coordinates
    obstacles[:, :] += wall_thickness
    
    cfg = {
        'bound': field_width,
        'start': start,
        'goal': goal,
        'ob_type': 'box',
        'goal_radius': 4 * scale,
        'scale': 1.0,
        'obstacles': obstacles,
    }
        
    return cfg

def render_matplotlib(cfg):
    """
    Renders the environment configuration using Matplotlib.
    - Draws obstacles with specified colors and borders.
    - Adds a dotted grid.
    """
    obstacles = cfg['obstacles']
    start = cfg['start']
    goal = cfg['goal']

    ax = plt.gca()

    # Obstacle colors from fast_render
    obstacle_fill_color = (255/255, 228/255, 181/255)  # Brownish
    obstacle_border_color = (50/255, 30/255, 30/255)      # Black
    border_linewidth = 3.5 # Corresponds to thickness 2 in pygame at typical scales

    # Draw obstacles
    for obs in obstacles:
        x1, y1, x2, y2 = obs
        width = x2 - x1
        height = y2 - y1
        # Filled rectangle
        rect_fill = patches.Rectangle((x1, y1), width, height, linewidth=0, edgecolor='none', facecolor=obstacle_fill_color, zorder=1)
        ax.add_patch(rect_fill)
        # Border
        rect_border = patches.Rectangle((x1, y1), width, height, linewidth=border_linewidth, edgecolor=obstacle_border_color, facecolor='none', zorder=2)
        ax.add_patch(rect_border)
    
    # Draw green transparent circle with solid green borders for start and goal
    if start:
        start_circle = patches.Circle((start[0], start[1]), radius=10, color='green', alpha=0.3, zorder=3)
        ax.add_patch(start_circle)
        start_border = patches.Circle((start[0], start[1]), radius=10, linewidth=border_linewidth * 2, edgecolor='green', facecolor='none', zorder=4)
        ax.add_patch(start_border)
    if goal:
        goal_circle = patches.Circle((goal[0], goal[1]), radius=cfg['goal_radius'], color='green', alpha=0.3, zorder=3)
        ax.add_patch(goal_circle)
        goal_border = patches.Circle((goal[0], goal[1]), radius=cfg['goal_radius'], linewidth=border_linewidth * 2, edgecolor='green', facecolor='none', zorder=4)
        ax.add_patch(goal_border)
    
    # Determine plot limits
    all_x = []
    all_y = []
    if obstacles.size > 0:
        all_x.extend(obstacles[:, 0])
        all_x.extend(obstacles[:, 2])
        all_y.extend(obstacles[:, 1])
        all_y.extend(obstacles[:, 3])
    
    if start:
        all_x.append(start[0])
        all_y.append(start[1])
    if goal:
        all_x.append(goal[0])
        all_y.append(goal[1])

    if not all_x or not all_y: # Handle empty case
        min_x, max_x, min_y, max_y = -10, 10, -10, 10
    else:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

    margin = (max_x - min_x) * 0.05  # 5% margin
    if margin == 0: margin = 10 # Default margin if no extent

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # Add dotted gridlines
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    grid_step = 25.0 # Default to 25mm
    if x_span < 250 and y_span < 250: # If field is small, use smaller grid
        grid_step = 10.0
    if x_span < 100 and y_span < 100:
        grid_step = 2.5

    ax.set_xticks(np.arange(np.floor(ax.get_xlim()[0]/grid_step)*grid_step, np.ceil(ax.get_xlim()[1]/grid_step)*grid_step + grid_step, grid_step))
    ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0]/grid_step)*grid_step, np.ceil(ax.get_ylim()[1]/grid_step)*grid_step + grid_step, grid_step))

    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='x', rotation=-90)
    
    # No -- lines
    plt.grid(False, which='both', axis='both')  # Disable major and minor grid lines
    
    plt.title("Environment Obstacles (Matplotlib)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.pause(0.001)  # Ensure the plot updates immediately


if __name__ == "__main__":
    # Repeatedy read from an env config file, load its obstacles
    # and display them with fast render

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from render import init_vis, render, draw_goal, clear_all_surfaces
    import pygame
    import time
    
    import argparse
    parser = argparse.ArgumentParser(description="Visualize environment configuration.")
    parser.add_argument('--env', type=str, default='envs/env_live.txt', help='Path to the environment configuration file.')
    args = parser.parse_args()
    
    # save_box_config('envs/divider.txt', generate_divider_config())
    
    # cfg = load_box_config('envs/divider.txt')
    
    # Initialize visualization
    # init_vis(figsize=(12, 8), 
    #         obstacles=cfg['obstacles'], 
    #         start=cfg['start'], 
    #         goal=cfg['goal'])
    
    plt.figure(figsize=(16, 12))
    
    while True:
        
        # save_box_config(args.env, generate_divider_config())

        cfg = load_box_config(args.env)
        
        # ----- Fast Render -----
        # import fast_render
        
        # clear_all_surfaces()
        
        # fast_render._obstacles = cfg['obstacles']
        # fast_render._draw_obstacles()

        # # Extract start and goal
        # start = cfg['start']
        # goal = cfg['goal']
        # radius = cfg['goal_radius']
        
        # # Draw the goal as a green circle
        # draw_goal(goal[:2], radius)
        
        # # Render frame
        # render()     
        
        
        # ----- Matplotlib ----- 
        plt.clf()  # Clear the current figure  
        render_matplotlib(cfg)


        time.sleep(0.1)


    pygame.quit()