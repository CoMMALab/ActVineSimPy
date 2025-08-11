'''
Load the results of bulk_results.py and plot them

First, create a 2D table (jsut with text, maybe use a library to format it). This has columns of envs
and rows of ablations. Each cell contains the average time to first solution, average time per iter, and 
average final min_cost.

Then, in figures/ create three plots for each ablation/env combo:
1. Time to first solution vs. env
2. Min cost vs. time
3. Average time per iter vs. env

Save them as PNG files in figures/ with names like
# figures/ablation_{ablation}_{env}_time_to_first_solution.png
# figures/ablation_{ablation}_{env}_min_cost_vs_time.png
# figures/ablation_{ablation}_{env}_avg_time_per_iter.png

The figures should be academic and suitable for publication, with appropriate labels, titles, and legends.
Use seaborn for styling, use bold text for titles and axes names, use black text for all numbers,
use colored title (dark blue).

Plot the percentiles range as a shaded area around the average line, with the 25th and 75th percentiles.
the average line should be thick

Also smooth out the percentile bounds.

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tabulate import tabulate
from scipy.ndimage import gaussian_filter1d
from experiments.bulk_results import envs

def smooth(y, sigma=2):
    return gaussian_filter1d(y, sigma=sigma)

def plot_data(data, title, xlabel, ylabel, save_path, x_data=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if x_data is None:
        x_data = data[0]
    
    p25, avg, p75 = data[1], data[2], data[3]

    # Remove NaNs for plotting
    valid_indices = ~np.isnan(avg)
    x_data = x_data[valid_indices]
    p25 = p25[valid_indices]
    avg = avg[valid_indices]
    p75 = p75[valid_indices]

    if len(x_data) < 2:
        print(f"Not enough data to plot for {title}. Skipping.")
        plt.close()
        return

    # Smooth the percentile bounds
    # p25 = smooth(p25)
    # p75 = smooth(p75)

    ax.plot(x_data, avg, linewidth=3, label='Average')
    ax.fill_between(x_data, p25, p75, alpha=0.2, label='25th-75th Percentile')

    title_font = {'family': 'sans-serif', 'color':  'darkblue', 'weight': 'bold', 'size': 20}
    label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}

    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel(xlabel, fontdict=label_font)
    ax.set_ylabel(ylabel, fontdict=label_font)
    
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def plot_min_costs(data, title, xlabel, ylabel, save_path, x_data=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # data is a list of (times, costs)
    for rep in data:
        times, costs = rep.T
        if len(costs) < 2:
                continue
            
        ax.plot(times, costs, linewidth=3, label='Average')

    title_font = {'family': 'sans-serif', 'color':  'darkblue', 'weight': 'bold', 'size': 20}
    label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}

    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel(xlabel, fontdict=label_font)
    ax.set_ylabel(ylabel, fontdict=label_font)
    
    ax.set_ylim(ymin=0)
    
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_success_rate(times, title, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(times) > 0:
        # Calculate cumulative success rate
        sorted_times = np.sort(times)
        success_rate = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
        ax.plot(sorted_times, success_rate, marker='o', linestyle='-', label='Success Rate')

    title_font = {'family': 'sans-serif', 'color': 'darkblue', 'weight': 'bold', 'size': 20}
    label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 16}

    ax.set_title(title, fontdict=title_font)
    ax.set_xlabel("Time (s)", fontdict=label_font)
    ax.set_ylabel("Success Rate (%)", fontdict=label_font)
    ax.set_ylim(0, 105)
    
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    if not os.path.exists('figures'):
        os.makedirs('figures')

    success_data = np.load('cache/bulk_results_success_rate.npz', allow_pickle=True)
    cost_data = np.load('cache/bulk_results_min_cost.npz', allow_pickle=True)
    itertime_data = np.load('cache/bulk_results_itertime.npz', allow_pickle=True)

    # --- Create Table ---
    header = ['Ablation'] + [env.capitalize() for env in envs]
    table = []
    
    ablations = ['default', 'no_geo'] # , 'geo_curve_limits']
    
    def last_nonnull(collection):
        """Returns the last non-null value in a collection."""
        for item in reversed(collection):
            if item is not None and not np.isnan(item):
                return item
        return None
        
    for ablation in ablations:
        row = [ablation.replace('_', ' ').capitalize()]
        for env in envs:
            key = f'{ablation}_{env}'
            
            # Get average solution time
            times = success_data.get(key)
            avg_time_sol = f"{np.mean(times):.2f}s" if times is not None and times.size > 0 else "N/A"
            
            if times is None or len(times) == 0:
                print("WARNING: No avg_time_sol for key:", key)
            
            # Get average min cost
            costs = cost_data.get(key)
            final_cost = "N/A"
                                    
            # costs is a list of (times, costs)
            # Get avg final cost
            cost_sum = 0
            cost_count = 0
            num_trials = 0
            if costs is not None:
                num_trials = len(costs)
                
                # For each rep
                for cost_entry in costs:
                    if len(cost_entry[1]) == 0:
                        continue
                    
                    # Get last non none cost
                    last_cost = cost_entry[-1, 1]
                    if last_cost is None:
                        continue
                                        
                    cost_sum += last_cost
                    cost_count += 1
                    
                if cost_count > 0:
                    final_cost = f"{cost_sum/cost_count:.2f}"
            
            # Get iteration times
            itertimes = itertime_data.get(key)
            avg_iter_t = "N/A"
            if itertimes is not None and len(itertimes[2]) > 0:
                 avg_iter_t = f"{np.nanmean(itertimes[2]):.2f}s"
            else:
                print("WARNING: No avg_iter_time for key:", key)

            # row.append(f"{avg_time_sol}\n{final_cost}\n{avg_iter_t}\n{num_trials} trials")
            row.append(f"{avg_time_sol}\n{final_cost}\n{avg_iter_t}")

        table.append(row)
    
    print("--- Summary Table ---")
    print("   Each cell lists:\n   1. Average time to first solution\n   2. Final minimum cost\n   3. Average time per iteration")
    print(tabulate(table, headers=header, tablefmt="grid"))
    print("\n" * 2)

    print("--- LaTeX Table for Copy/Paste ---")
    print(" & ".join(header) + " \\\\")
    print("\\hline")
    
    for row in table:
        row1 = []
        for cell in row:
            row1.extend(cell.split('\n'))
        row1 = np.array(row1)
        row1 = row1[1:]
        # print row[0::3] separated by & 
        print(" & ".join(row1[0::3]).replace('\n', ' ') + " \\\\")
        print(" & ".join(row1[1::3]).replace('\n', ' ') + " \\\\")
        print(" & ".join(row1[2::3]).replace('\n', ' ') + " \\\\")

    # --- Create Second Table: Mean, Stddev, Median ---
    stat_header = ['Ablation'] + [env.capitalize() for env in envs]
    stat_table = []
    for ablation in ablations:
        stat_row = [ablation.replace('_', ' ').capitalize()]
        for env in envs:
            key = f'{ablation}_{env}'
            # Time to first solution
            times = success_data.get(key)
            if times is not None and times.size > 0:
                mean_time = np.nanmean(times)
                std_time = np.nanstd(times)
                median_time = np.nanmedian(times)
                time_str = f"Time: {mean_time:.2f}s ± {std_time:.2f}s, med {median_time:.2f}s"
            else:
                time_str = "Time: N/A"

            # Final min cost
            costs = cost_data.get(key)
            final_costs = []
            if costs is not None:
                for cost_entry in costs:
                    if len(cost_entry[1]) == 0:
                        continue
                    last_cost = cost_entry[-1, 1]
                    if last_cost is not None and not np.isnan(last_cost):
                        final_costs.append(last_cost)
            if final_costs:
                mean_cost = np.nanmean(final_costs)
                std_cost = np.nanstd(final_costs)
                median_cost = np.nanmedian(final_costs)
                cost_str = f"Cost: {mean_cost:.2f} ± {std_cost:.2f}, med {median_cost:.2f}"
            else:
                cost_str = "Cost: N/A"

            # Avg time per iteration
            itertimes = itertime_data.get(key)
            if itertimes is not None and len(itertimes[2]) > 0:
                mean_iter = np.nanmean(itertimes[2])
                std_iter = np.nanstd(itertimes[2])
                median_iter = np.nanmedian(itertimes[2])
                iter_str = f"Iter: {mean_iter:.2f}s ± {std_iter:.2f}s, med {median_iter:.2f}s"
            else:
                iter_str = "Iter: N/A"

            stat_row.append(f"{time_str}\n{cost_str}\n{iter_str}")
        stat_table.append(stat_row)
        
    print("--- Statistics Table (Mean, Stddev, Median) ---")
    print("   Each cell lists:\n   1. Time to first solution (mean ± stddev, median)\n   2. Final minimum cost (mean ± stddev, median)\n   3. Avg time per iteration (mean ± stddev, median)")
    print(tabulate(stat_table, headers=stat_header, tablefmt="grid"))

    # --- Generate Plots ---
    for ablation in ablations:
        for env in envs:
            key = f'{ablation}_{env}'
            print(f"Generating plots for {key}...")

            # 1. Time to first solution (Success Rate Plot)
            if key in success_data:
                plot_success_rate(success_data[key], 
                                  title=f'Success Rate vs. Time ({ablation.capitalize()}, {env.capitalize()})',
                                  save_path=f'figures/ablation_{ablation}_{env}_success_rate.png')

            # 2. Min cost vs. time
            if key in cost_data:
                plot_min_costs(cost_data[key],
                          title=f'Min Cost vs. Time ({ablation.capitalize()}, {env.capitalize()})',
                          xlabel='Time (s)',
                          ylabel='Minimum Cost',
                          save_path=f'figures/ablation_{ablation}_{env}_min_cost_vs_time.png')

            # 3. Average time per iter vs. iter count
            if key in itertime_data:
                plot_data(itertime_data[key],
                          title=f'Avg. Time per Iteration vs. Iteration Count ({ablation.capitalize()}, {env.capitalize()})',
                          xlabel='Iteration Count',
                          ylabel='Time per Iteration (s)',
                          save_path=f'figures/ablation_{ablation}_{env}_avg_time_per_iter.png')
    
    print("All figures saved in the 'figures/' directory.")
    
    