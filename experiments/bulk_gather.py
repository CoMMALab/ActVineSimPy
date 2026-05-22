'''
This programmatically launches a ton of experiments and records their result to fill in the ablations table.

Ablations: Default, No Maximal, No Set Cover, No Geo Curve Limits, No Geo
Envs (in env/ folder) env_long, env_maze, env_needle, env_plus, env_tube'


On env_tube, the time_to_evolve should be 200
'''

# Tries to average over per cell in the table. (So for a unique combo of settings)
import pickle
import time
from render import init_vis
from kinodynamic.env_loader import load_box_config
from geometric.biarc_rrtstar import main as geometric_plan
from kinodynamic.sst import SSTparams, sst_star
import numpy as np
from render import *
from pbd_vine import VineParams
import io
import sys
from contextlib import contextmanager
import os

trials_per_cell = 5

# ablations = ['default', 'no_maximal', 'no_set_cover', 'geo_curve_limits', 'no_geo']
# envs = ['maze', 'needle', 'plus', 'tube', 'long']
ablations = ['default', 'no_geo']
# ablations = ['default', 'geo_curve_limits']
# envs = ['maze', 'needle', 'plus', 'tube', 'long']
# envs = ['maze', 'long', 'needle', 'plus']
envs = ['tube']

def print_green(*args, **kwargs):
    print('\033[92m', *args, '\033[0m', **kwargs)
    
def combos(a, b, c):
    set = []
    for thing in a:
        for otherthing in b:
            for thirdthing in c:
                set.append((thing, otherthing, thirdthing))
    return set

# Create a context manager that collects all prints (std, err) and adds it to a list (passed as arg)
@contextmanager
def capture_output(output_file):
    """A context manager to capture stdout and stderr and 
    write them to a file as the output comes in"""
    with open(output_file, 'w', buffering=1) as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    

def go(args):
    
    reps_per_expy = 1
    
    repeats = list(range(reps_per_expy))
    
    experiments = combos(ablations, envs, repeats)
    
    for trial_idx, (ablation, env, _) in enumerate(experiments):
        if trial_idx < args.start_index:
            continue
        
        if env == 'long' and ablation == 'geo_curve_limits':
            continue  
        
        print('\n')
        print_green(f'Running experiment with ablation: {ablation}, env: {env}')
        print_green(f'This is {trial_idx} out of {len(experiments)}')
        
        env_path = f'envs/env_{env}.txt'
        
        # Check if there are other data files at f'cache/ablation_{ablation}_{env}_{rep}_data.npy'
        all_files = os.listdir('cache')
        all_files_starting_with = [name for name in all_files if name.startswith(f'ablation_{ablation}_{env}_') and name.endswith('.pkl')]
        all_reps = [int(name.split('_')[-2]) for name in all_files_starting_with]
        greatest_rep = max(all_reps) if all_reps else -1
        rep = greatest_rep + 1
        
        print_green('Starting rep:', rep)
        
        save_pygame_folder = f'cache/ablation_{ablation}_{env}_{rep}_images'
        

        cfg = load_box_config(env_path)
        
        # delete save_pygame_folder
        if os.path.exists(save_pygame_folder):
            for file in os.listdir(save_pygame_folder):
                file_path = os.path.join(save_pygame_folder, file)
                os.remove(file_path)
            os.rmdir(save_pygame_folder)
    
        init_vis(figsize=(12,9), obstacles=cfg['obstacles'], dynamic_obstacles=cfg['dynamic_obstacles'], start=cfg['start'], goal=cfg['goal'], 
                save_pygame_folder=save_pygame_folder)
        
        output_std = []
        
        max_bodies = 70
        # if env == 'long':
        #     max_bodies = 110
        
        load_points = True
        if load_points:
            min_r = 0.001
            geo_time = 60.0
            
            if ablation == 'geo_curve_limits':
                min_r = 200
                geo_time = 60.0 * 5
                            
            save_points_path = f'cache/ablation_costs_env_{env}_minr_{min_r}.npy'
            has_saved_points = os.path.exists(save_points_path)
            
            if not has_saved_points:
                print('Did not find a saved points file at', save_points_path, 'planning (60s)...')
                geometric_plan(env=env_path, time=geo_time, thresh=0.1,
                                save_points_path=save_points_path,
                                plan_goal_to_start=True, min_r=min_r,)
            else:
                print('Found saved points file at', save_points_path)
            
            points = np.load(save_points_path)
            point_costs = np.load(save_points_path.replace('.npy', '_costs.npy')) # We'll just assume if points exists, then points_costs is up to date too
                                
            # Draw the points
            draw_points(points, point_costs)
        else:
            # We didn't load points
            points = None
            point_costs = None
            
        render()
        
        
        sim_params = VineParams(
            max_bodies=max_bodies,
            body_length=68.0, # 25.0 mm
            radius=50, # 16.0,
            dt=1/10,
            grow_rate=20.0,
            grow_force=15.0 if env != 'tube' else 15.0, 
            stiffness=20.0 if env != 'tube' else 5.0, 
            damping=50.0,
            # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
            substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
            alpha=1e-2,
            obstacle_rects=cfg['obstacles'],
            use_tube_obstacle=(env=='tube'),
        )
        
        # SST params
        sst_params = SSTparams(
            batch_size=100,
            δBN=60.0,
            δs=45.0, # 20
            min_x=0.0,
            max_x=cfg['bound_x'],
            min_y=0.0,
            max_y=cfg['bound_y'],
            start=cfg['start'],
            goal=cfg['goal'],
            goal_radius=cfg['goal_radius'],
            points=points,
            point_costs=point_costs,
            info={'env_path': env_path}, # Not used, except when we serialize this object for view_solutions
            do_cost_to_go=not ablation == 'no_geo', 
            do_maximal=not ablation == 'no_maximal',
            do_set_cover=not ablation == 'no_set_cover',
            time_to_evolve=100.0 if env != 'tube' else 200.0,
            record_every_multiplier=4 if env == 'tube' else 1
        )
        
        min_iters = 15
        # if env == 'tube':
        #     min_iters = 10
        
        # Iteration count, wall time, min cost, num_solutions, and DontCompareSecond
        # DontCompareSecond has (total_cost, {cspace bodies bending_control cost_to_come cost_total tip}}
        data = []
        iters = 0
        min_cost_has_not_changed_for = 0
        last_min_cost = None
        
        start_time = None
    
        def callback(solutions_ranked):
            # global start_time, iters, min_cost_has_not_changed_for, last_min_cost
            nonlocal start_time, iters, min_cost_has_not_changed_for, last_min_cost
             
            if start_time is None:
                start_time = time.time()
                            
            if solutions_ranked is None or solutions_ranked.__len__() == 0:
                min_cost = None
            else:
                min_cost = solutions_ranked[0].first
                                
            if min_cost == last_min_cost:
                min_cost_has_not_changed_for += 1
                # So this is incremented even when there is no solution
                # In fact its purpose is to kill plans that are not making any progress
                # and will likely not find a solution
                # the thing keeping this from triggering in early stages is iter > 10
            else:
                min_cost_has_not_changed_for = 0
                
            last_min_cost = min_cost
            
            elapsed_time = time.time() - start_time
            
            print('Elapsed time:', elapsed_time, 'seconds')
            
            data.append((iters, 
                         elapsed_time,
                         min_cost,
                         solutions_ranked.__len__() if solutions_ranked is not None else 0,
                         solutions_ranked
                         ))
            
            print(f'   iter: {iters}, elapsed_time: {elapsed_time:.2f}, min_cost_has_not_changed_for: {min_cost_has_not_changed_for}')
            
            iters += 1
            
            # End condition; the min cost has not changed for 10 iterations and 
            # at least 30 sec have passed,
            # and at least 20 iterations have been done.
            if min_cost_has_not_changed_for > 5 and \
               elapsed_time > 90 and \
               iters > min_iters:
                
                print(f'Experiment Done, killing min_cost_has_not_changed_for: {min_cost_has_not_changed_for}, iters: {iters}, elapsed_time: {elapsed_time}')
                return True
                
            if elapsed_time > 60 * 5:
                return True
            
            return False
        
        # Delete the output_file
        output_file = f'cache/ablation_{ablation}_{env}_{rep}_output.txt'
        if os.path.exists(output_file):
            os.remove(output_file)
        
        print('output txt at', output_file)
        
        with capture_output(output_file): 
            sst_star(sst_params, sim_params, callback)
        
        # Save
        with open(f'cache/ablation_{ablation}_{env}_{rep}_data.pkl', "wb") as f:
            pickle.dump(data, f)
        print(f'Saved data cache/ablation_{ablation}_{env}_{rep}_data.pkl')

if __name__ == '__main__':
    
    # Argparse for start_index
    import argparse
    parser = argparse.ArgumentParser(description='Run bulk ablation experiments.')
    parser.add_argument('--start_index', type=int, default=0, help='The index to start from in the experiments list.')
    args = parser.parse_args()

    while True:
        go(args)