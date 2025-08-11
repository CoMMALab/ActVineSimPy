import numpy as np
import os
import pygame
import argparse
import render
from pbd_vine import VineParams
from kinodynamic.env_loader import load_box_config
import jax

def view_all_solutions(solutions, sim_params, cfg, frames_to_display=None, no_obs=False, output_path='all_solutions.png'):
    """
    Visualizes multiple final states of vine rollouts overlaid.
    """
    init_x = cfg['start'][0]
    init_y = cfg['start'][1]
    init_heading = cfg['start'][2]

    if no_obs:
        obstacles = np.array(cfg['obstacles'])
        # convert from (x, y, w, h) to (x1, y1, x2, y2)
        obstacles[:, 2:] += obstacles[:, :2]
        min_coords = np.min(obstacles[:, :2], axis=0)
        max_coords = np.max(obstacles[:, 2:], axis=0)
        
        top_left = (min_coords[0], min_coords[1], 10, 10)
        bottom_right = (max_coords[0] - 10, max_coords[1] - 10, 10, 10)
        
        cfg['obstacles'] = [top_left, bottom_right]
        sim_params.obstacle_rects = np.asarray(cfg['obstacles'])
        print("Replaced obstacles with corner markers.")

    render.init_vis(figsize=(12, 9), obstacles=cfg['obstacles'], start=cfg['start'], goal=cfg['goal'],
                         save_pygame_folder='pics/view_solutions_all/')
    render.draw_goal(cfg['goal'], cfg['goal_radius'])

    if frames_to_display is None:
        frames_to_display = range(len(solutions))

    for i in frames_to_display:
        if i >= len(solutions):
            print(f"Warning: Frame index {i} is out of bounds. Skipping.")
            continue
            
        state = solutions[i]
        state_dict = state.second
        
        final_cspace = state_dict['cspace']
        final_bodies = state_dict['bodies']
        bending_control = state_dict['bending_control']
        
        print(f"Drawing solution {i}")
        
        actuator_bendings = np.zeros((1, bending_control.shape[0]+1, 2))
        actuator_bendings[:, 1:, :] = bending_control

        colors = np.sign(actuator_bendings[..., 1]) * actuator_bendings[..., 0] / 10_000_00 * 3
        
        # Use draw_live_state to get more detailed rendering options
        render.draw_live_state(sim_params, final_cspace[None, ...], final_bodies[None, ...], 
                                    init_x, init_y, init_heading, draw_circles=True, actuator_colors=colors)

    render.render()

    print(f"Displaying all specified solutions. Press 'S' to save to '{output_path}', 'Q' to quit.")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_s:
                    pygame.image.save(render._display_surf, output_path)
                    print(f"Saved image to {output_path}")

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multiple vine solutions.")
    parser.add_argument('--frames', nargs='+', type=int, help='List of frame indices to display.')
    parser.add_argument('--no_obs', action='store_true', help='Replace obstacles with top-left and bottom-right corner markers.')
    parser.add_argument('--output', type=str, default='all_solutions.png', help='Output file path for the saved image.')
    parser.add_argument('--load_dir', type=str, default='cache', help='Directory to load solution files from.')
    args = parser.parse_args()

    jax.config.update('jax_platform_name', 'cpu')

    sim_params = np.load(os.path.join(args.load_dir, 'winning_sim_params.npy'), allow_pickle=True).item()
    info = np.load(os.path.join(args.load_dir, 'winning_info.npy'), allow_pickle=True).item()
    
    cfg = load_box_config(info['env_path'])
    
    from kinodynamic.sst import DontCompareSecond
    solutions = np.load(os.path.join(args.load_dir, 'solutions.npy'), allow_pickle=True)
    solutions = list(solutions)
    solutions.sort(key=lambda x: x.first)
    
    print(f"Loaded {len(solutions)} winning states.")

    view_all_solutions(solutions, sim_params, cfg, frames_to_display=args.frames, no_obs=args.no_obs, output_path=args.output)
