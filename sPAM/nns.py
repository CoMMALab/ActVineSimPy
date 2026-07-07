import os
import time
import functools
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sPAM.dataset import generate_data_from_csv
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, orbax_utils, checkpoints
from flax import struct
from flax.linen import initializers
import optax
import jaxopt
from sPAM.ellip import F, E
from sPAM.spam import l_m_to_phi_eps, params
import pandas as pd
# --------------------------
# 1. Data Generation (from thesis_fig2.py)
# --------------------------

solve_inner_vmap = jax.vmap(l_m_to_phi_eps, in_axes=(None, 0, 0, None))    
solve_inner_vmap = jax.jit(solve_inner_vmap)

def generate_data(params):
    """Generates a dataset by solving for phi and m over a grid of eps and l_0 values."""
    jax.debug.print("Generating training data...")

    m_vals = jnp.linspace(0, 0.5, 200)
    l0_vals = jnp.linspace(params.min_l_0, params.max_l_0, 200)
    
    # Create a grid of inputs
    m_grid, l0_grid = jnp.meshgrid(m_vals, l0_vals)
    
    # Vectorize the solver over the grid of inputs
    jax.debug.print("Solving for {} samples (batched)...", m_grid.size)

    # Create a JAX PRNG key for reproducibility
    key = jax.random.PRNGKey(42)

    # Vectorize the solver over keys, eps, and l0.
    phi, eps, is_sat, info = solve_inner_vmap(key, l0_grid.ravel(), m_grid.ravel(), params)
        
    jax.debug.print('err min {}', jnp.min(info['error']))
    jax.debug.print('err 25th percentile {}', jnp.percentile(info['error'], 25))
    jax.debug.print('err 50th percentile {}', jnp.percentile(info['error'], 50))
    jax.debug.print('err 75th percentile {}', jnp.percentile(info['error'], 75))
    jax.debug.print('err max {}', jnp.max(info['error']))
    
    inputs_grid = jnp.stack([eps, l0_grid.ravel()], axis=1)
    outputs_grid = jnp.stack([phi, m_grid.ravel()], axis=1)
    
    # Filter by error < 3
    valid_mask = info['error'] < 100
    inputs_grid = inputs_grid[valid_mask]
    outputs_grid = outputs_grid[valid_mask]
    is_sat = is_sat[valid_mask]

    return inputs_grid, outputs_grid, is_sat

# generate_data = jax.jit(generate_data, static_argnames=('params'))

# --------------------------
# 2. Dataset Scaling
# --------------------------

def create_dataset_and_scale(inputs, outputs):
    """Min-max scaling for inputs and outputs."""
    
    # Scale inputs (eps, l0)
    in_min = inputs.min(axis=0)
    in_max = inputs.max(axis=0)
    in_range = in_max - in_min
    scaled_inputs = (inputs - in_min) / in_range
    
    # Scale outputs (phi, m)
    out_min = outputs.min(axis=0)
    out_max = outputs.max(axis=0)
    out_range = out_max - out_min
    scaled_outputs = (outputs - out_min) / out_range
    
    scaling_info = {
        'in_min': in_min, 'in_range': in_range,
        'out_min': out_min, 'out_range': out_range,
    }
    
    return scaled_inputs, scaled_outputs, scaling_info

def unscale_outputs(output_scaled, scaling_info):
    """Un-scale predicted outputs back to their original range."""
    out_min, out_rng = scaling_info['out_min'], scaling_info['out_range']
    return output_scaled * out_rng + out_min

# --------------------------
# 3. Flax MLP Model
# --------------------------

class MLP(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, 2)
        # x = x.astype(jnp.float16)
        x = nn.Dense(features=32, kernel_init=initializers.he_normal())(x)
        x = nn.relu(x)
        # x = nn.Dense(features=16, kernel_init=initializers.he_normal())(x)
        # x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs, kernel_init=initializers.he_normal())(x)
        
        return x

# -------------------------------
# 4. TrainState and Metrics
# -------------------------------

@struct.dataclass
class Metrics:
    mse: float
    count: int

    @staticmethod
    def empty():
        return Metrics(mse=0.0, count=0)

    def update(self, preds, targets):
        loss = jnp.mean((preds - targets) ** 2)
        return Metrics(mse=self.mse + loss * preds.shape[0], count=self.count + preds.shape[0])

    def compute(self):
        return {'mse': (self.mse / self.count) if self.count > 0 else 0.0}

class TrainState(train_state.TrainState):
    metrics: Metrics = Metrics.empty()

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, metrics=Metrics.empty())

# ----------------------
# 5. Training and Evaluation Steps
# ----------------------

@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, x_batch)
        return jnp.mean((preds - y_batch)**2)
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    preds = new_state.apply_fn({'params': new_state.params}, x_batch)
    new_metrics = new_state.metrics.update(preds, y_batch)
    return new_state.replace(metrics=new_metrics)

@jax.jit
def eval_step(state, x_batch, y_batch):
    preds = state.apply_fn({'params': state.params}, x_batch)
    return state.metrics.update(preds, y_batch)

# ------------------------
# 6. Main Orchestration
# ------------------------

def get_or_train_model(params, epochs=100, learning_rate=5e-2, batch_size=256):
    """
    Main function to load a pre-trained model or train a new one.
    - Checkpoint name is derived from `params`.
    - If no checkpoint, it generates data, trains, and saves plots/model.
    """
    
    # Define checkpoint directory and name
    ckpt_dir = './sPAM'
    ckpt_name = f"model_a_{params.a}_Rc_{params.R_c}_R_act_max_{params.R_act_max}_l0_{params.min_l_0}-{params.max_l_0}"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_path = os.path.abspath(ckpt_path)
    
    model = MLP(num_outputs=2)
    
    # Try to load from checkpoint and only if scaling info exists
    if os.path.exists(ckpt_path) and os.path.exists(f'{ckpt_path}/scaling_info.npy'):
        print(f"Loading trained model from {ckpt_path}...")
        # Create a dummy state to restore into
        key = jax.random.PRNGKey(0)
        dummy_state = create_train_state(key, model, learning_rate, input_shape=(1, 2))
        state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=dummy_state)
        
        scaling_info = np.load(f'{ckpt_path}/scaling_info.npy', allow_pickle=True).item()
        print("Neural surrogate model loaded successfully.")
        return state, scaling_info, model

    print(f"No checkpoint found {ckpt_path}. Starting new training run.")
    os.makedirs(ckpt_path, exist_ok=True)
    
    # 1. Generate and scale data

    #NOTE NOTE NOTE: HAD TO GO INTO THIS FUNCTION TO CHANGE
    inputs, outputs, is_sat = generate_data(params)
    
    # Save inputs and outputs as one pandas csv file
    data_dict = {
        'm': outputs[:, 1],
        'l0': inputs[:, 1],
        'phi': outputs[:, 0],
        'eps': inputs[:, 0],
        'is_sat': is_sat,
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(f'{ckpt_path}/data.csv', index=False)
    print(f'Saved generated data to {ckpt_path}/data.csv')
    
    # Filter out failed solver runs (NaNs)
    valid_mask = ~jnp.isnan(outputs).any(axis=1)
    print(f"Generated {len(inputs)} total samples, {jnp.sum(valid_mask)} are valid.")
    
    inputs = inputs[valid_mask]
    outputs = outputs[valid_mask]
    
    x_scaled, y_scaled, scaling_info = create_dataset_and_scale(inputs, outputs)
    
    
    np.save(f'{ckpt_path}/scaling_info.npy', scaling_info)
    
    # 2. Train/Val split
    total_size = x_scaled.shape[0]
    rng = np.random.default_rng(seed=42)
    indices = np.arange(total_size)
    rng.shuffle(indices)
    
    train_count = int(0.8 * total_size)
    train_idx, test_idx = indices[:train_count], indices[train_count:]
    x_train, y_train = x_scaled[train_idx], y_scaled[train_idx]
    x_test, y_test = x_scaled[test_idx], y_scaled[test_idx]
    
    print(f"Train size: {x_train.shape}, Test size: {x_test.shape}")

    # 3. Create model and state
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, model, learning_rate, input_shape=(batch_size, 2))
    
    # 4. Training loop
    train_mses, val_mses = [], []
    
    def get_batches(x, y, size):
        for start in range(0, x.shape[0], size):
            yield x[start:start+size], y[start:start+size]

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        state = state.replace(metrics=Metrics.empty())
        perm = rng.permutation(train_count)
        for x_b, y_b in get_batches(x_train[perm], y_train[perm], batch_size):
            state = train_step(state, x_b, y_b)
        train_metrics = state.metrics.compute()

        # Validation
        val_metrics_agg = Metrics.empty()
        for x_b, y_b in get_batches(x_test, y_test, batch_size):
            val_metrics_agg = eval_step(state, x_b, y_b)
        val_metrics = val_metrics_agg.compute()
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_metrics['mse']:.6f} | Val MSE: {val_metrics['mse']:.6f} | Time: {time.time() - epoch_start:.2f}s")
        train_mses.append(train_metrics['mse'])
        val_mses.append(val_metrics['mse'])

    # 5. Save model checkpoint
    # Reset metrics to avoid saving JAX arrays in the state, which can cause
    # issues with some jax/orbax version combinations.
    state_to_save = state.replace(metrics=Metrics.empty())
    checkpoints.save_checkpoint(ckpt_dir=ckpt_path, target=state_to_save, step=epochs, overwrite=True)
    print(f"\nSaved final model checkpoint to {ckpt_path}")

    # 6. Save MSE plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_mses, label='Train MSE')
    plt.plot(val_mses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training vs. Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    mse_plot_path = os.path.join(ckpt_path, 'mse_plot.png')
    plt.savefig(mse_plot_path)
    print(f"Saved MSE plot to {mse_plot_path}")
    plt.close()

    # 7. Save PCA comparison plot
    plot_pca_comparison(state, x_test, y_test, scaling_info, ckpt_path)
        
    return state, scaling_info, model

def plot_pca_comparison(state, x_test_scaled, y_test_scaled, scaling_info, save_dir):
    """Generates and saves a plot comparing predictions vs true values over a PCA of the input."""
    print("Generating PCA comparison plot...")
    
    # Use a random sample for cleaner plotting
    sample_size = min(1000, len(x_test_scaled))
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(len(x_test_scaled), sample_size, replace=False)

    x_samp = x_test_scaled[sample_indices]
    y_samp_true_scaled = y_test_scaled[sample_indices]

    # PCA from 2D -> 1D on scaled input
    pca = PCA(n_components=1)
    x_samp_1d = pca.fit_transform(x_samp)

    # Get predicted outputs
    pred_scaled_samp = state.apply_fn({'params': state.params}, x_samp)

    # Unscale for comparison
    pred_unscaled_samp = unscale_outputs(np.array(pred_scaled_samp), scaling_info)
    y_unscaled_samp_true = unscale_outputs(np.array(y_samp_true_scaled), scaling_info)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sort_indices = np.argsort(x_samp_1d.ravel())

    # Plot phi
    axs[0].scatter(x_samp_1d[sort_indices], y_unscaled_samp_true[sort_indices, 0], label='True phi', s=10, alpha=0.7)
    axs[0].scatter(x_samp_1d[sort_indices], pred_unscaled_samp[sort_indices, 0], label='Predicted phi', s=10, alpha=0.7)
    axs[0].set_xlabel('PCA(scaled eps, scaled l0)')
    axs[0].set_ylabel('phi (rad)')
    axs[0].legend()
    axs[0].set_title('phi: True vs. Predicted')
    axs[0].grid(True)

    # Plot m
    axs[1].scatter(x_samp_1d[sort_indices], y_unscaled_samp_true[sort_indices, 1], label='True m', s=10, alpha=0.7)
    axs[1].scatter(x_samp_1d[sort_indices], pred_unscaled_samp[sort_indices, 1], label='Predicted m', s=10, alpha=0.7)
    axs[1].set_xlabel('PCA(scaled eps, scaled l0)')
    axs[1].set_ylabel('m')
    axs[1].legend()
    axs[1].set_title('m: True vs. Predicted')
    axs[1].grid(True)

    plt.tight_layout()
    pca_plot_path = os.path.join(save_dir, 'pca_comparison.png')
    plt.savefig(pca_plot_path)
    print(f"Saved PCA plot to {pca_plot_path}")
    plt.close()


# ------------------------
# 7. Prediction Function
# ------------------------
def get_prediction_function(state, scaling_info, model):
    """Returns a jitted function for making predictions."""
    
    def predict_fn(params, inputs_unscaled, model):
        # Scale inputs
        print("")
        print("TYPES:", type(scaling_info['in_min']), type(scaling_info['in_min']))
        print("")

        in_min, in_rng = scaling_info['in_min'], scaling_info['in_range']
        scaled_inputs = (inputs_unscaled - in_min) / in_rng
        
        # Predict
        preds_scaled = model.apply({'params': params}, scaled_inputs)
        
        # Unscale outputs
        return unscale_outputs(preds_scaled, scaling_info)

    return lambda inputs: predict_fn(state.params, inputs, model)


if __name__ == '__main__':
    # Configure JAX
    # jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_enable_x64", False)
    
    # This function will either load the checkpoint or run the full training process
    trained_state, scaling_info, model = get_or_train_model(params)

    # --- Example Usage ---
    print("\n--- Running Example Prediction ---")
    
    # Get a callable prediction function
    predict = get_prediction_function(trained_state, scaling_info, model)
    
    # Create some sample inputs (eps, l_0)
    # Ensure they are within the training range for best results
    sample_inputs = np.array([
        [0.1, 0.05],  # eps, l_0
        [0.2, 0.08],
        [0.3, 0.03],
    ])
    
    # Get predictions
    predictions = predict(sample_inputs)
    
    print("Sample Inputs (eps, l_0):")
    print(sample_inputs)
    print("\nPredicted Outputs (phi, m):")
    print(predictions)