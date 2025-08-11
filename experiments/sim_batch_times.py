import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import os
import time
import jax
import jax.numpy as jnp
from pbd_vine import VineParams, step_vine_batched 
from kinodynamic.env_loader import load_box_config # Import for loading obstacles

slow_avg_time_per_iter_s = 0.10206422591460368

def run_single_batch_benchmark(vine_params: VineParams, 
                               forward_jitted_fn, 
                               batch_size: int, 
                               num_trials: int, 
                               num_steps_per_trial: int):
    """
    Runs the benchmark for a single batch size.
    """
    print(f"Benchmarking batch_size: {batch_size}")
    
    # Initialize inputs for the current batch size
    current_cspaces = jnp.zeros((batch_size, vine_params.max_bodies + 1))
    current_cspaces = current_cspaces.at[:, vine_params.max_bodies].set(vine_params.body_length)
    current_n_bodies = jnp.full((batch_size,), 1, dtype=jnp.int32)
    current_target_angles = jnp.zeros((batch_size, vine_params.max_bodies))
    
    x0_scalar = 0.0
    y0_scalar = 0.0
    heading0_scalar = 0.0

    times_for_current_batch = []

    for trial_idx in range(num_trials):
        cspaces_trial = jnp.copy(current_cspaces)
        n_bodies_trial = jnp.copy(current_n_bodies)

        # JIT Warm-up run
        try:
            warmup_cspaces, warmup_n_bodies = forward_jitted_fn(
                vine_params,
                cspaces_trial,
                n_bodies_trial,
                current_target_angles,
                x0_scalar,
                y0_scalar,
                heading0_scalar
            )
            warmup_cspaces.block_until_ready() # type: ignore
        except Exception as e:
            print(f"Error during warm-up for batch_size {batch_size}, trial {trial_idx+1}: {e}")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()
            times_for_current_batch.append(np.nan) # Record NaN for this trial
            continue # Move to next trial if warm-up fails

        start_time = time.perf_counter()
        for _ in range(num_steps_per_trial):
            cspaces_trial, n_bodies_trial = forward_jitted_fn(
                vine_params,
                cspaces_trial,
                n_bodies_trial,
                current_target_angles,
                x0_scalar,
                y0_scalar,
                heading0_scalar
            )
        cspaces_trial.block_until_ready() # type: ignore
        end_time = time.perf_counter()

        total_time_for_trial = end_time - start_time
        avg_time_per_step = total_time_for_trial / num_steps_per_trial
        times_for_current_batch.append(avg_time_per_step)
        print(f"  Trial {trial_idx+1}/{num_trials}: {avg_time_per_step:.6f} s/step")
    
    if not times_for_current_batch: # If all warm-ups failed
        return [np.nan] * num_trials
        
    return times_for_current_batch

def run_all_benchmarks(output_path='cache/sim_batch_times.npy'):
    """
    Manages the overall benchmarking process across different batch sizes.
    """
    if not os.path.exists('cache'):
        os.makedirs('cache')

    # Load obstacle configuration from file, similar to sst.py
    cfg = load_box_config('envs/divider.txt')
    cfg_obstacles = cfg['obstacles']
    
    assert cfg_obstacles.ndim == 2 and cfg_obstacles.shape[1] == 4, \
        f"Expected obstacles to be a 2D array with shape (N, 4), got {cfg_obstacles.shape}"

    default_params = VineParams(
        max_bodies=70,
        body_length=68.0, # 25.0 mm
        radius=50, # 16.0,
        dt=1/10,
        grow_rate=20.0,
        grow_force=15.0,
        stiffness=20.0,
        damping=50.0,
        # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
        substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
        alpha=1e-2,
        obstacle_rects=cfg['obstacles'],
    )
    default_params.hash = int(time.time())

    batch_sizes_to_test = [1, 5, 10, 50, 100, 500, 1000, 5_000, 10_000, 50_000]
    # batch_sizes_to_test = [1, 10, 100] # For quicker testing

    results_all_batches = {}
    num_trials = 10
    num_steps_per_trial = 10

    # JIT compile the function
    forward_jitted = jax.jit(step_vine_batched, static_argnames=['params'])

    for batch_size in batch_sizes_to_test:
        trial_times = run_single_batch_benchmark(default_params, 
                                                 forward_jitted, 
                                                 batch_size, 
                                                 num_trials, 
                                                 num_steps_per_trial)
        results_all_batches[batch_size] = trial_times
         # If all trials for this batch size resulted in NaN (e.g. OOM)
        assert not np.all(np.isnan(trial_times))
            
    np.save(output_path, results_all_batches)
    print(f"Benchmarking complete. Results saved to {output_path}")


def plot_benchmark_results(data_path='cache/sim_batch_times.npy'):
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run the benchmarking script in pbd_vine.py first.")
        return

    # Load the results
    # The results are saved as a dictionary, allow_pickle=True and .item() are needed.
    results_all_batches = np.load(data_path, allow_pickle=True).item()

    batch_sizes = sorted(results_all_batches.keys())
    mean_times_per_step = []

    for bs in batch_sizes:
        times = np.array(results_all_batches[bs])
        # Filter out NaNs if any trial failed (e.g., OOM)
        times = times[~np.isnan(times)]
        if len(times) > 0:
            mean_times_per_step.append(np.mean((slow_avg_time_per_iter_s * bs) / times))
        else:
            # If all trials for a batch size failed, append NaN
            mean_times_per_step.append(np.nan)


    # Convert to numpy arrays for easier plotting
    batch_sizes_plot = np.array(batch_sizes, dtype=float)
    mean_times_plot = np.array(mean_times_per_step, dtype=float)

    # Filter out batch sizes where mean_time is NaN (all trials failed)
    valid_mask = ~np.isnan(mean_times_plot)
    batch_sizes_plot = batch_sizes_plot[valid_mask]
    mean_times_plot = mean_times_plot[valid_mask]

    if len(batch_sizes_plot) == 0:
        print("No valid data to plot. All benchmark trials might have failed.")
        return

    # plt.rcParams['font.family'] = 'Helvetica'
    #list all fonts
    # print("Available fonts:", plt.rcParams['font.family'])
    plt.figure(figsize=(5, 4.5))
    plt.subplots_adjust(left=0.14, bottom=0.14)  # Increase left and bottom margins
    fontsize = plt.rcParams['font.size'] * 1.4
    # Plot mean line
    plt.plot(batch_sizes_plot, mean_times_plot, marker='o', linestyle='-', color='blue', linewidth=2, label='Mean Time per Step')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    # set min at 0
    # plt.ylim(bottom=0)
    plt.xlabel('Surrogate Batch Size', fontsize=fontsize, labelpad=5)
    plt.ylabel('Surrogate / Baseline Throughput', fontsize=fontsize, labelpad=5)
    plt.title('Neural Surrogate Throughput vs. Baseline', fontsize=fontsize, pad=15)
    # plt.legend()
    plt.grid(True, which="major", ls="--")
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Ensure plot directory exists
    plot_dir = 'figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_filename = os.path.join(plot_dir, 'sim_batch_times_performance.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    # run_all_benchmarks()
    plot_benchmark_results()
