import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from sPAM.nns import get_or_train_model, generate_data, get_prediction_function
from sPAM.spam import params

def generate_and_save_errors(filepath='cache/squared_errors.npy'):
    """
    Loads the NN model, generates data, computes MSE loss, and saves the squared errors.
    """
    # 1. Load the pre-trained model
    print("Loading model...")
    trained_state, scaling_info, model = get_or_train_model(params)
    predict = get_prediction_function(trained_state, scaling_info, model)

    # 2. Generate the dataset
    print("Generating data...")
    inputs, outputs, is_sat = generate_data(params)
    
    # Filter out failed solver runs (NaNs)
    valid_mask = ~jnp.isnan(outputs).any(axis=1)
    print(f"Generated {len(inputs)} total samples, {jnp.sum(valid_mask)} are valid.")
    
    inputs = inputs[valid_mask]
    outputs = outputs[valid_mask]

    # 3. Get predictions from the model
    print("Making predictions on the entire dataset...")
    predictions = predict(inputs)

    # 4. Compute MSE loss for each sample
    squared_errors = jnp.sum((predictions - outputs)**2, axis=1)
    
    # The overall MSE is the mean of these squared errors.
    mse = jnp.mean(squared_errors)
    print(f"Overall MSE on the dataset: {mse:.6f}")

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save the squared errors
    np.save(filepath, np.array(squared_errors))
    print(f"Saved squared errors to {filepath}")
    
    return filepath

def plot_error_distribution(filepath='cache/squared_errors.npy'):
    """
    Loads squared errors from a file and plots their distribution.
    """
    print(f"Loading squared errors from {filepath}...")
    squared_errors = np.load(filepath)

    # Filter out large errors for better visualization
    max_error = 0.013
    original_count = len(squared_errors)
    squared_errors = squared_errors[squared_errors < max_error]
    filtered_count = len(squared_errors)
    print(f"Filtered out {original_count - filtered_count} samples with squared error > {max_error}")

    # 5. Plot the distribution of the loss
    print("Plotting loss distribution...")
    plt.figure(figsize=(5, 4.5))
    plt.subplots_adjust(left=0.18, bottom=0.18)  # Increase left and bottom margins
    fontsize = plt.rcParams['font.size'] * 1.4
    # Use density=True to plot the ratio of samples
    plt.hist(squared_errors, bins=50, color='blue', edgecolor='black', weights=np.ones_like(squared_errors) / len(squared_errors))
    plt.xlabel('Squared Error', fontsize=fontsize, labelpad=5)
    plt.ylabel('Fraction of Samples', fontsize=fontsize, labelpad=5)
    plt.title('Squared Error Distribution of Neural Surrogate', fontsize=fontsize, pad=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=-25, ha='left')
    
    # A log scale on the y-axis helps to see the distribution of less frequent, larger errors.
    # plt.yscale('log')
    
    # Adding some statistics to the plot
    mean_error = np.mean(squared_errors)
    median_error = np.median(squared_errors)
    
    plt.axvline(mean_error, color='red', linestyle='dotted', linewidth=2, label=f'Mean: {mean_error:.4f}')
    plt.axvline(median_error, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_error:.4f}')
    
    plt.legend(fontsize=fontsize)
    
    # Save the plot to a file
    plot_filename = 'figures/mse_error_distribution.png'
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    print(f"Saved plot to '{plot_filename}'")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Standard JAX configuration
    jax.config.update("jax_enable_x64", False)
    # error_filepath = generate_and_save_errors()
    error_filepath = 'cache/squared_errors.npy'
    plot_error_distribution(error_filepath)
