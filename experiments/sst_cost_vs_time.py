import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Assuming sst.py is in the same directory or accessible via PYTHONPATH
# and contains sst_star, SSTparams, VineParams.
# Also assuming kinodynamic.env_loader and fast_render are accessible.
from kinodynamic.sst import sst_star, SSTparams, VineParams
from kinodynamic.env_loader import load_box_config
from render import init_vis # For initializing visualization context if sst's draw functions need it

class SabotageException(Exception):
    '''Kill SST from the inside.'''
    pass

def main():
    # --- Configuration ---
    num_trials = 10
    max_iters_per_trial = 15
    env_file = 'envs/divider.txt' 
    points_file = 'cache/points.npy'
    point_costs_file = 'cache/point_costs.npy'

    # --- Ensure cache and plot directories exist ---
    os.makedirs('cache', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # --- Load shared configuration (points, costs, env config) ---
    if not (os.path.exists(points_file) and os.path.exists(point_costs_file)):
        print(f"Error: Required cache files {points_file} or {point_costs_file} not found.")
        print("Please ensure these files are generated, possibly by running the geometric planner")
        print("from sst.py (e.g., by running sst.py without --noplan argument).")
        return

    cfg = load_box_config(env_file)
    points = np.load(points_file)
    point_costs = np.load(point_costs_file)

    # --- Initialize visualization (mimicking sst.py's setup) ---
    # This is called to ensure that if sst_star or sst internally use drawing functions
    # that depend on this initialization, they do not fail.
    init_vis(figsize=(12,9), obstacles=cfg['obstacles'], start=cfg['start'], goal=cfg['goal'])

    # --- Base Simulation Parameters (VineParams) ---
    # Copied from sst.py main for consistency
    base_sim_params = VineParams(
        max_bodies=110, body_length=25.0, radius=30, dt=1/10.0,
        grow_rate=20.0, grow_force=15.0, stiffness=20.0, damping=50.0,
        substeps=15, alpha=1e-2, obstacle_rects=cfg['obstacles'],
    )

    # --- Data storage ---

    # For saving raw (iter, time, cost) series per trial to NPY file
    # This will be a list of lists; each inner list contains (iter, time, cost) tuples for a trial
    raw_data_for_npy = []

    # --- Benchmarking Loop ---
    for trial_idx in range(num_trials):
        print(f"Running Trial {trial_idx + 1}/{num_trials}...")

        # Re-initialize SSTparams for a fresh trial state. This is crucial because
        # sst_star modifies sst_params.δs, sst_params.δBN, and uses sst_params.solutions.
        trial_sst_params = SSTparams(
            batch_size=80,  # Default from sst.py
            δBN=60.0,       # Default from sst.py
            δs=45.0,        # Default from sst.py
            min_x=0.0, max_x=cfg['bound_x'], min_y=0.0, max_y=cfg['bound_y'],
            start=cfg['start'], goal=cfg['goal'], goal_radius=cfg['goal_radius'],
            points=points, point_costs=point_costs,
            geo_cost_to_go_weight=0.2 # Default in SSTparams class, explicit here
        )

        trial_start_time = time.time()
        invocation_idx = [0] # Needs to be 1D list to be mutable in the callback
        current_trial_raw_points = [] # Stores (iter, time, cost) for the current trial

        def trial_specific_callback(cost_from_sst):
            elapsed_time = time.time() - trial_start_time
            current_trial_raw_points.append((invocation_idx[0], elapsed_time, cost_from_sst))
            invocation_idx[0] += 1
            
            if invocation_idx[0] >= max_iters_per_trial:
                raise SabotageException("Stopping SST* early due to max iterations reached.")
        
        try:
            sst_star(sst_params=trial_sst_params, 
                        sim_params=base_sim_params, 
                        callback=trial_specific_callback)
        except SabotageException:
            # This is the intended way we stop the trial early.
            # It beats adding in early stop logic to the sst* which is already complicated enough
            print(f"Trial {trial_idx + 1} completed with {invocation_idx[0]} iterations.")

        raw_data_for_npy.append(current_trial_raw_points)

    # --- Save raw data ---
    # The NPY file will store a list of lists. Each inner list is the time series
    # [(iter_0, time_0, cost_0), (iter_1, time_1, cost_1), ...] for one trial.
    np.save('cache/sst_cost_vs_time.npy', np.array(raw_data_for_npy, dtype=object))
    print("Raw benchmark data saved to cache/sst_cost_vs_time.npy")
    

def plot_cost_vs_time():
    # Load the raw data from the NPY file 
    raw_data = np.load('cache/sst_cost_vs_time.npy', allow_pickle=True)
    
    # List of trials, where each entry is a np array of (iters, 2) [time, cost]
    costs = []
    for trial_data in raw_data:
        trial_costs = np.array([ (point[1], point[2]) for point in trial_data])
        trial_costs = np.where(np.isfinite(trial_costs) & (trial_costs != 0), trial_costs, np.nan)  # Replace zero with nan
        costs.append(trial_costs)
        
    min_end_time = min([trial[-1][0] for trial in costs])
    num_trials = len(costs)
    
    # Now we have an annoying situation. We have serveral time series of (t, y),
    # but the times don't align. So we will interpolate each series to linspace(0, min_end_time, 100)
    costs_interpolated = np.zeros((num_trials, 200)) # Shape: (num_trials, 100) for (time, cost)
    times = np.linspace(0, min_end_time, 200)
    
    for i, trial_costs in enumerate(costs):
        for j, t in enumerate(times):
            # Get the cost at the time right before (or equal) t
            idx = np.searchsorted(trial_costs[:, 0], t, side='left') - 1
            if idx < 0:
                costs_interpolated[i, j] = np.nan
            else:
                left_t = trial_costs[idx, 0]
                right_t = trial_costs[idx + 1, 0]
                left_cost = trial_costs[idx, 1]
                right_cost = trial_costs[idx + 1, 1]
                
                interped = np.interp(t, [left_t, right_t], [left_cost, right_cost])
                costs_interpolated[i, j] = interped
                        
    mean_costs = np.nanmean(costs_interpolated[:, :], axis=0)  # Mean across trials
    std_costs = np.nanstd(costs_interpolated[:, :], axis=0)  # Std deviation across trials
    
    # --- Plot results ---
    plt.figure(figsize=(10, 6))
    
    # Regular: Plot with std ranges
    # plt.plot(times, mean_costs, label='Mean Min Cost', linewidth=2.5) # Thick line
    # plt.fill_between(times, 
    #                  mean_costs - std_costs, 
    #                  mean_costs + std_costs, 
    #                  alpha=0.3, label='Std Dev Min Cost', color='skyblue') # Lighter filled color

    # Test: Plot each trial separately
    for i in range(num_trials):
        plt.plot(times, costs_interpolated[i], label=f'Trial {i + 1}', alpha=0.5)

    plt.xlabel('Time (s)')
    plt.xticks(rotation=-90)
    plt.ylabel('Minimum Solution Cost')
    plt.title(f'SST* Performance: Cost vs. Time ({num_trials} Trials)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(times[::3]) # Show ticks for each callback invocation
        
    
    plot_filename = 'plots/sst_cost_vs_time.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Uncomment to display plot interactively if running in a GUI environment

if __name__ == '__main__':
    # main()
    plot_cost_vs_time()