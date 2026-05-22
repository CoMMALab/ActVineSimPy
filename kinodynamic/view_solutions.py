import numpy as np
import os
import pygame
import argparse
import render
from pbd_vine import VineParams, step_vine_batched
from kinodynamic.env_loader import load_box_config
import jax
from kinodynamic.sst import actuator_params_fwd

# Jax-compiled function for rolling out vine motion
forward = jax.jit(step_vine_batched, static_argnames=['params', 'x0_list', 'y0_list', 'heading0_list', 'bend_energy_func'])

def view_all_solutions(solutions, sim_params, cfg, frames_to_display=None, no_obs=False, output_path='all_solutions.png'):
    """
    Visualizes multiple final states of vine rollouts overlaid.
    """
    init_x = cfg['start'][0]
    init_y = cfg['start'][1]
    init_heading = cfg['start'][2] # maze - 1.1

    if no_obs:
        obstacles = np.array(cfg['obstacles'])
        min_coords = np.min(obstacles[:, :2], axis=0)
        max_coords = np.max(obstacles[:, 2:], axis=0)
        
        top_left = (min_coords[0], min_coords[1], 10, 10)
        bottom_right = (max_coords[0] - 10, max_coords[1] - 10, 10, 10)
        
        cfg['obstacles'] = [top_left, bottom_right]
        
        #convert x1 y1 w h to x1 y1 x2 y2
        cfg['obstacles'] = [
            [top_left[0], top_left[1], top_left[0] + top_left[2], top_left[1] + top_left[3]],
            [bottom_right[0], bottom_right[1], bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]]
        ]
        sim_params.obstacle_rects = np.asarray(cfg['obstacles'])
        print("Replaced obstacles with corner markers.")

    render.init_vis(figsize=(12, 9), obstacles=cfg['obstacles'], dynamic_obstacles=cfg['dynamic_obstacles'], start=cfg['start'], goal=cfg['goal'],
                         save_pygame_folder='pics/view_solutions_free/')
    # fast_render.draw_goal(cfg['goal'], cfg['goal_radius'])

    if frames_to_display is None:
        frames_to_display = range(len(solutions))

    for i in frames_to_display:
        if i >= len(solutions):
            print(f"Warning: Frame index {i} is out of bounds. Skipping.")
            continue
            
        state = solutions[i]
        state_dict = state.second
        
        play_vine_rollout_all(i, sim_params, state_dict, init_x, init_y, init_heading, cfg)

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

def play_vine_rollout_all(rollout_num, sim_params, state_dict, init_x, init_y, init_heading, cfg):
    """
    Renders the growth of a vine from start to its final state.
    """
    final_cspace = state_dict['cspace']
    final_bodies = state_dict['bodies']
    bending_control = state_dict['bending_control']
    
    current_bodies = np.array([1])
    current_cspace = np.zeros((1, sim_params.max_bodies + 1))
    current_cspace[0, -1] = sim_params.body_length

    steps = 1500
    for i in range(steps + 999999):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print("Skipping to next vine...")
                    return

        if i % 20 == 0:
            # fast_render.draw_goal(cfg['goal'], cfg['goal_radius'])
            
            actuator_bendings = np.zeros((1, bending_control.shape[0]+1, 2))
            actuator_bendings[:, 1:, :] = bending_control

            colors = np.sign(actuator_bendings[..., 1]) * actuator_bendings[..., 0] / 10_000_00 * 3

            render.draw_live_state(sim_params, current_cspace, current_bodies, init_x, init_y, init_heading,
                                        draw_circles=True, actuator_colors=colors)
            
            regular_font =  pygame.font.SysFont("Go Medium", 40, bold=False) 
            lines = [f'Rollout num: {rollout_num}']
            max_width = max(regular_font.size(line)[0] for line in lines)
    
            y_offset = 700
            for text_line in lines:
                line_surface = regular_font.render(text_line, True, (0, 0, 0))
                line_surface = render.flip(line_surface)
                line_rect = line_surface.get_rect()
                line_rect.topleft = (600 - max_width - 10, y_offset)
                # fast_render._live_surf.blit(line_surface, line_rect)
                y_offset += line_surface.get_height() + 5
        
            render.render()

        current_cspace, current_bodies = forward(
            sim_params, current_cspace, current_bodies, bending_control[None, ...],
            init_x, init_y, init_heading, actuator_params_fwd
        )
                
        if current_bodies[0] >= final_bodies:
            break
        
    print('Press Enter in the window to see the next solution (you can do this during the animation too)')
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multiple vine solutions.")
    parser.add_argument('--frames', nargs='+', type=int, help='List of frame indices to display.')
    parser.add_argument('--no_obs', action='store_true', help='Replace obstacles with top-left and bottom-right corner markers.')
    parser.add_argument('--output', type=str, default='all_solutions.png', help='Output file path for the saved image.')
    parser.add_argument('--load_dir', type=str, default='cache', help='Directory to load solution files from.')
    args = parser.parse_args()

    jax.config.update('jax_platform_name', 'cpu')
    
    from kinodynamic.sst import DontCompareSecond

    sim_params = np.load(os.path.join(args.load_dir, 'winning_sim_params.npy'), allow_pickle=True).item()
    
    sim_params.radius = 50
    info = np.load(os.path.join(args.load_dir, 'winning_info.npy'), allow_pickle=True).item()
    
    cfg = load_box_config(info['env_path'])
    
    solutions = np.load(os.path.join(args.load_dir, 'solutions.npy'), allow_pickle=True)
    solutions = list(solutions)
    solutions.sort(key=lambda x: x.first)
    
    print(f"Loaded {len(solutions)} winning states.")

    view_all_solutions(solutions, sim_params, cfg, frames_to_display=args.frames, no_obs=args.no_obs, output_path=args.output)
