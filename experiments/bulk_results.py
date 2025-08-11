'''

Read data from bulk_ablations.py and compute stats.

Do the following;

Get a graph of the ratio of envs that have found a solution per expy vs time. So get 
the first time of the solution (or non, then ignore) for each rep, then sort the times.
Then save to a dict {{ablation}_{env}: times}. And save the whole dict to a file.

Then do the same for min cost vs time. So here we will first get the time range. So [0, max(all reps)].
Then for each rep, get the min cost at all time points (nan if min cost in None at some time). Then sample it
over the aforementioned time range. Make the linspace precision high (like 10,000). Specifically, if at t=1 the min_cost was 4 and at t=2 it becamse 3,
then from t=1 to t=2 all the values will be 4, then at t=2 it will become 3, and so on.

Now that we have some common x values, get the average and 25, and 75th percentiles of the min costs at each time point across all reps. Save this to a dict

Save as {{ablation}_{env}: 25, avg, 75 percentile series as a 3-tuple}. And save the whole dict to a file.

Average itertime vs iter count. Save as {{ablation}_{env}: 25 percentile, avg, 75 percentile}
'''

import os
import numpy as np

# from experiments.bulk_ablations import ablations, envs, combos, trials_per_cell

ablations = ['default', 'geo_curve_limits', 'no_geo']
envs = ['plus', 'tube', 'maze', 'long', 'needle', 'pickone']

def combos(a, b, c):
    set = []
    for thing in a:
        for otherthing in b:
            for thirdthing in c:
                set.append((thing, otherthing, thirdthing))
    return set

if __name__ == '__main__':
    # Figure out reps    
    experiments = combos(ablations, envs, (1,))
    
    all_times_to_first_solution = {}
    all_min_cost_vs_time = {}
    all_avg_itertime = {}
    
    for ablation, env, _ in experiments:
        # Check if there are other data files at f'cache/ablation_{ablation}_{env}_{rep}_data.npy'
        all_files = os.listdir('cache')
        all_files_starting_with = [name for name in all_files if name.startswith(f'ablation_{ablation}_{env}_') and name.endswith('.pkl')]
        all_reps = [int(name.split('_')[-2]) for name in all_files_starting_with]
        greatest_rep = max(all_reps) if all_reps else -1
        reps = greatest_rep + 1
        
        if reps == 0:
            print(f"No data for {ablation} on {env}")
            continue

        print(f"Processing {ablation} on {env} with {reps} reps")

        # --- Success Rate vs. Time ---
        times_to_first_solution = []
        for rep in range(reps):
            data = np.load(f'cache/ablation_{ablation}_{env}_{rep}_data.pkl', allow_pickle=True)
            for row in data:
                _, elapsed_time, min_cost, _, _ = row
                if min_cost is not None:
                    times_to_first_solution.append(elapsed_time)
                    break # Found first solution for this rep
        
        all_times_to_first_solution[f'{ablation}_{env}'] = np.array(sorted(times_to_first_solution))

        # --- Min Cost vs. Time ---
        max_time = 0
        all_rep_data_cost = []
        
        # Figure out the max time over all reps
        for rep in range(reps):
            data = np.load(f'cache/ablation_{ablation}_{env}_{rep}_data.pkl', allow_pickle=True)
            if len(data) > 0:
                max_time = max(max_time, data[-1][1])
                all_rep_data_cost.append(data)

        if max_time == 0:
            print(f"No time data for {ablation} on {env}")
            continue
        
        time_points = np.linspace(0, max_time, 1001)
        rep_time_cost_arrays = []
        for data in all_rep_data_cost:
            costs = []
            times = []
            current_cost = np.nan
            data_idx = 0
            for t0, t in zip(time_points[:-1], time_points[1:]):
                while data_idx < len(data) and t0 < data[data_idx][1] and data[data_idx][1] <= t:
                    current_cost = data[data_idx][2]
                    data_idx += 1
                times.append(t0)
                costs.append(current_cost)
            # Add extra point at t = max_time, y = second last cost
            if len(costs) > 1:
                times.append(max_time)
                costs.append(costs[-2])
            else:
                times.append(max_time)
                costs.append(costs[-1])
            arr = np.column_stack((np.array(times), np.array(costs)))
            rep_time_cost_arrays.append(arr)
            
        all_min_cost_vs_time[f'{ablation}_{env}'] = rep_time_cost_arrays
        
        # --- Average Iteration Time vs. Iteration Count ---
        max_iters = 0
        all_rep_data = []
        
        # Figure out the max iters over all reps
        for rep in range(reps):
            data = np.load(f'cache/ablation_{ablation}_{env}_{rep}_data.pkl', allow_pickle=True)
            if len(data) > 1:
                max_iters = max(max_iters, len(data) -1)
                all_rep_data.append(data)
        
        if max_iters > 0:
            itertimes_matrix = np.full((reps, max_iters), np.nan)

            for i, data in enumerate(all_rep_data):
                if len(data) > 1:
                    data_item_1 = [a[1] for a in data]
                    itertimes = np.diff(data_item_1)
                    itertimes_matrix[i, :len(itertimes)] = itertimes
            
            iter_counts = np.arange(max_iters)
            avg_itertimes = np.nanmean(itertimes_matrix, axis=0)
            p25_itertimes = np.nanpercentile(itertimes_matrix, 25, axis=0)
            p75_itertimes = np.nanpercentile(itertimes_matrix, 75, axis=0)

            all_avg_itertime[f'{ablation}_{env}'] = (iter_counts, p25_itertimes, avg_itertimes, p75_itertimes)


    np.savez('cache/bulk_results_success_rate.npz', **all_times_to_first_solution)
    np.savez('cache/bulk_results_min_cost.npz', **all_min_cost_vs_time)
    np.savez('cache/bulk_results_itertime.npz', **all_avg_itertime)

    print("Finished processing. Results saved to cache directory.")


