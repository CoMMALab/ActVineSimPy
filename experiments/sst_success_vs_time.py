import numpy as np
import matplotlib.pyplot as plt
import os

def plot_success_vs_time():
    """
    Loads trial data and plots the number of successful trials over time.
    A trial is considered successful at time t if a solution (finite, non-zero cost)
    has been found by or at time t.
    """
    data_file = 'cache/sst_cost_vs_time.npy'
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        print("Please run a script that generates this file (e.g., sst_cost_vs_time.py's main function).")
        return

    raw_data = np.load(data_file, allow_pickle=True) # List of trials

    if raw_data.size == 0:
        print("No data found in cache/sst_cost_vs_time.npy. Cannot generate plot.")
        return
        
    num_trials = len(raw_data)
    if num_trials == 0:
        print("No trials found in the data. Cannot generate plot.")
        return

    first_success_times_per_trial = []
    trial_actual_end_times = []

    for trial_data_points in raw_data:
        # trial_data_points is a list of (iter, time, cost) tuples


        # Find the first success time for this trial
        # Yeah this code is pretty bad since gemini wrote it, but it works
        current_trial_first_success_time = float('inf') 
        
        for point in trial_data_points:
            time_val, cost_val = point[1], point[2]
            if np.isfinite(cost_val) and cost_val != 0:
                current_trial_first_success_time = min(current_trial_first_success_time, time_val)
                break 
        
        first_success_times_per_trial.append(current_trial_first_success_time)
        trial_actual_end_times.append(trial_data_points[-1][1]) # Time of the last data point for this trial
            
    num_interpolation_points = 400 
    max_end_time = max(trial_actual_end_times)

    common_times = np.linspace(0, max_end_time, num_interpolation_points)

    success_counts_over_time = np.zeros(num_interpolation_points)

    for i in range(num_interpolation_points):
        count = 0
        for t_succ in first_success_times_per_trial:
            if common_times[i] >= t_succ:
                count += 1
        success_counts_over_time[i] = count

    # --- Plot results ---
    plt.figure(figsize=(10, 6))
    plt.plot(common_times, success_counts_over_time / num_trials, label='Number of Successful Trials', linewidth=2)
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Fraction of Successful Trials (out of {num_trials})')
    plt.title(f'SST* Performance: Success Count vs. Time ({num_trials} Trials)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)  # Set y-axis limits to [0, 1.05] for better visibility

    # Configure x-axis ticks
    num_xticks = 10
    step = max(1, len(common_times) // num_xticks)
    selected_ticks = common_times[::step]
    
    # Ensure the last time point is included if not already by the step
    if common_times[-1] not in selected_ticks and len(common_times) > 1 :
        selected_ticks = np.append(selected_ticks[:-1], common_times[-1]) # Replace last auto-tick with actual end

    tick_labels = [f"{t:.2f}" for t in selected_ticks]
    plt.xticks(ticks=selected_ticks, labels=tick_labels, rotation=-45, ha="left")
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    plot_filename = 'plots/sst_success_vs_time.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Uncomment to display plot interactively

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True) # Ensure plots directory exists
    plot_success_vs_time()
