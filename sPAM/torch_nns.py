'''
Supposed to be nns.py, but compatible with torch
'''

import os
import time
import functools
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# from flax.training import train_state, orbax_utils, checkpoints
# from flax import struct
# from flax.linen import initializers
# import optax
# import jaxopt
from sPAM.torch_ellip import F, E
from sPAM.torch_spam import l_m_to_phi_eps, params
import pandas as pd

import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print("PREDICTION MODEL: USING CUDA")
else:
    print("PREDICTION MODEL: USING CPU")
    

# NOTE: deliberately do NOT call torch.set_default_device('cuda') here. This module is imported as a
# library (dvsim/spam.py -> the vine sim), and flipping the *global* default device forces the whole
# cpu-native dvsim sim onto CUDA, where its growth path stalls. The sPAM MLP is tiny; it runs on the
# caller's device (get_or_train_model(device=...), default cpu to match the sim).

import torch.nn as nn
from torch import optim
import csv

# --------------------------
# 1. Data Generation (from thesis_fig2.py)
# --------------------------

# solve_inner_vmap = torch.vmap(l_m_to_phi_eps, in_dims=(None, 0, 0, None))    
# solve_inner_vmap = torch.compile(solve_inner_vmap)

def generate_data(params):
    """Generates a dataset by solving for phi and m over a grid of eps and l_0 values."""
    print("Generating training data...")

    m_vals = torch.linspace(0, 0.5, 200)
    l0_vals = torch.linspace(params.min_l_0, params.max_l_0, 200)
    
    # Create a grid of inputs
    m_grid, l0_grid = torch.meshgrid(m_vals, l0_vals)
    
    # Vectorize the solver over the grid of inputs
    print("Solving for {} samples (batched)...", m_grid.size)

    # Create torch seed for reproducibility
    # key = jax.random.PRNGKey(42)
    base_seed = 42
    torch.manual_seed(base_seed)

    # Vectorize the solver over keys, eps, and l0.
    
    # phi, eps, is_sat, info = solve_inner_vmap(base_seed, l0_grid.ravel(), m_grid.ravel(), params)

    l0_flat = l0_grid.ravel()
    m_flat = m_grid.ravel()
    phi_list, eps_list, is_sat_list, info_list = [], [], [], []

    for l_0_i, m_i in zip(l0_flat, m_flat):
        phi_i, eps_i, is_sat_i, info_i = l_m_to_phi_eps(base_seed, l_0_i, m_i, params)
        phi_list.append(phi_i)
        eps_list.append(eps_i)
        is_sat_list.append(is_sat_i)
        info_list.append(info_i)

    phi = torch.stack(phi_list)
    eps = torch.stack(eps_list)
    is_sat = torch.stack(is_sat_list)
    errors = torch.stack([info_i.error for info_i in info_list])

        
    # print('err min {}', torch.min(info['error']))
    # print('err 25th percentile {}', torch.quantile(info['error'], 25/100))
    # print('err 50th percentile {}', torch.quantile(info['error'], 50/100))
    # print('err 75th percentile {}', torch.quantile(info['error'], 75/100))
    # print('err max {}', torch.quantile.max(info['error']))
    
    inputs_grid = torch.stack([eps, l0_grid.ravel()], axis=1)
    outputs_grid = torch.stack([phi, m_grid.ravel()], axis=1)
    
    # Filter by error < 3
    # valid_mask = info['error'] < 100

    valid_mask = errors < 100
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
    in_range = in_max[0] - in_min[0]
    scaled_inputs = (inputs - in_min[0]) / in_range
    
    # Scale outputs (phi, m)
    out_min = outputs.min(axis=0)
    out_max = outputs.max(axis=0)
    out_range = out_max[0] - out_min[0]
    scaled_outputs = (outputs - out_min[0]) / out_range
    
    scaling_info = {
        'in_min': in_min[0], 'in_range': in_range,
        'out_min': out_min[0], 'out_range': out_range,
    }
    
    return scaled_inputs, scaled_outputs, scaling_info

def unscale_outputs(output_scaled, scaling_info):
    """Un-scale predicted outputs back to their original range. Follow the model output's device
    (the saved scaling_info tensors may be on a different device than the loaded model)."""
    out_min, out_rng = scaling_info['out_min'], scaling_info['out_range']
    out_min = out_min.to(output_scaled.device)
    out_rng = out_rng.to(output_scaled.device)
    return output_scaled * out_rng + out_min

# --------------------------
# 3. MLP Model
# --------------------------

class MLP(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, num_outputs)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    


# -------------------------------
# 4. TrainState and Metrics
# -------------------------------

class Metrics:
    def __init__(self):
        self.mse = 0.0
        self.count = 0

    def update(self, preds, targets):
        loss = torch.mean((preds - targets) ** 2)
        n = preds.shape[0]
        self.mse += loss.item() * n
        self.count += n
    
    def compute(self):
        return {'mse': (self.mse / self.count) if self.count > 0 else 0.0}



# ----------------------
# 5. Training and Evaluation Steps
# ----------------------

def train_step(model, optimizer, metrics, x_batch, y_batch, device):
    
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    model.train()
    
    optimizer.zero_grad()
    preds = model(x_batch)
    loss = torch.mean((preds - y_batch) ** 2)
    loss.backward()
    optimizer.step()

    metrics.update(preds.detach(), y_batch)
    return metrics

@torch.no_grad()
def eval_step(model, metrics, x_batch, y_batch, device):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    model.eval()

    preds = model(x_batch)
    metrics.update(preds, y_batch)
    return metrics

# ------------------------
# 6. Main Orchestration
# ------------------------

def get_or_train_model(params, epochs=100, learning_rate=5e-2, batch_size=256, device='cpu'):
    """
    Main function to load a pre-trained model or train a new one.
    - Checkpoint name is derived from `params`.
    - If no checkpoint, it generates data, trains, and saves plots/model.
    - device: where to place the model. Defaults to 'cpu' so the surrogate matches dvsim's
      cpu-native sim (the caller can pass 'cuda' for standalone GPU training/inference).
    """

    # Define checkpoint directory and name
    ckpt_dir = './sPAM'
    ckpt_name = f"model_a_{params.a}_Rc_{params.R_c}_R_act_max_{params.R_act_max}_l0_{params.min_l_0}-{params.max_l_0}"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_path = os.path.abspath(ckpt_path)

    # Define model, optimizer, and device
    device = torch.device(device)

    model = MLP(num_outputs=2)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Try to load from checkpoint and only if scaling info exists
    if os.path.exists(ckpt_path) and os.path.exists(f'{ckpt_path}/scaling_info.npy') \
        and os.path.exists(f'{ckpt_path}/checkpoint.pt'):
    
        print(f"Loading trained model from {ckpt_path}...")
        # Create a dummy state to restore into
        # key = jax.random.PRNGKey(0)
        # dummy_state = create_train_state(key, model, learning_rate, input_shape=(1, 2))
        # state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=dummy_state)
        
        checkpoint = torch.load(f'{ckpt_path}/checkpoint.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scaling_info = np.load(f'{ckpt_path}/scaling_info.npy', allow_pickle=True).item()
        print("Neural surrogate model loaded successfully.")
        return scaling_info, model

    print(f"No checkpoint found {ckpt_path}. Starting new training run.")
    os.makedirs(ckpt_path, exist_ok=True)
    
    # NOTE: don't do this, it's way too slow
    # 1. Generate and scale data
    # inputs, outputs, is_sat = generate_data(params) <= IGNORE THIS LINE
    
    # # Save inputs and outputs as one pandas csv file
    # data_dict = {
    #     'm': outputs[:, 1],
    #     'l0': inputs[:, 1],
    #     'phi': outputs[:, 0],
    #     'eps': inputs[:, 0],
    #     'is_sat': is_sat,
    # }
    # df = pd.DataFrame(data_dict)
    # df.to_csv(f'{ckpt_path}/data.csv', index=False)
    # print(f'Saved generated data to {ckpt_path}/data.csv')
    

    # 1. Load in saved data used for training
    csv_path = f'{ckpt_path}/data.csv'

    ms, phis, eps, l0s = [], [], [], []
    with open(csv_path, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for record in reader:
            ms.append(float(record['m']))
            phis.append(float(record['phi']))
            eps.append(float(record['eps']))
            l0s.append(float(record['l0']))
    
    ms = torch.tensor(ms)
    phis = torch.tensor(phis)
    eps = torch.tensor(eps)
    l0s = torch.tensor(l0s)

    inputs = torch.stack((eps, l0s))
    inputs = inputs.T

    outputs = torch.stack((phis, ms))
    outputs = outputs.T

    print(inputs.shape, outputs.shape)

    # Filter out failed solver runs (NaNs)
    valid_mask = ~torch.isnan(outputs).any(dim=1)

    print("Successfully loaded data...")
    # print(f"Generated {len(inputs)} total samples, {torch.sum(valid_mask)} are valid.")
    
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
    # key = jax.random.PRNGKey(0)
    # state = create_train_state(key, model, learning_rate, input_shape=(batch_size, 2))
    
    # 4. Training loop
    train_mses, val_mses = [], []
    
    def get_batches(x, y, size):
        for start in range(0, x.shape[0], size):
            yield x[start:start+size], y[start:start+size]

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        # state = state.replace(metrics=Metrics.empty())
        # perm = rng.permutation(train_count)
        # for x_b, y_b in get_batches(x_train[perm], y_train[perm], batch_size):
        #     state = train_step(state, x_b, y_b)
        # train_metrics = state.metrics.compute()

        model.train()
        train_metrics_agg = Metrics()
        perm = rng.permutation(train_count)
        for x_b, y_b in get_batches(x_train[perm], y_train[perm], batch_size):
            # print(x_b.shape, y_b.shape)

            train_step(model, optimizer, train_metrics_agg, x_b, y_b, device)
        train_metrics = train_metrics_agg.compute()

        # Validation
        # val_metrics_agg = Metrics.empty()
        # for x_b, y_b in get_batches(x_test, y_test, batch_size):
        #     val_metrics_agg = eval_step(state, x_b, y_b)
        # val_metrics = val_metrics_agg.compute()

        model.eval()
        val_metrics_agg = Metrics()
        for x_b, y_b in get_batches(x_test, y_test, batch_size):
            eval_step(model, val_metrics_agg, x_b, y_b, device)
        val_metrics = val_metrics_agg.compute()
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_metrics['mse']:.6f} | Val MSE: {val_metrics['mse']:.6f} | Time: {time.time() - epoch_start:.2f}s")
        train_mses.append(train_metrics['mse'])
        val_mses.append(val_metrics['mse'])

    # 5. Save model checkpoint
    # Reset metrics to avoid saving JAX arrays in the state, which can cause
    # issues with some jax/orbax version combinations.
    
    # state_to_save = state.replace(metrics=Metrics.empty())
    # checkpoints.save_checkpoint(ckpt_dir=ckpt_path, target=state_to_save, step=epochs, overwrite=True)
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                f'{ckpt_path}/checkpoint.pt')
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
    plot_pca_comparison(model, x_test, y_test, scaling_info, ckpt_path)
        
    return scaling_info, model

def plot_pca_comparison(model, x_test_scaled, y_test_scaled, scaling_info, save_dir):
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

    # pred_scaled_samp = state.apply_fn({'params': state.params}, x_samp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():        
        x_samp_tensor = torch.as_tensor(x_samp, dtype=torch.float32)
        x_samp_tensor = x_samp_tensor.to(device)

        pred_scaled_samp = model(x_samp_tensor).numpy()

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
def get_prediction_function(scaling_info, model):
    """Returns a jitted function for making predictions."""
    
    def predict_fn(params, inputs_unscaled, model):
        # Scale inputs

        in_min, in_rng = scaling_info['in_min'], scaling_info['in_range']

        if in_min.device != inputs_unscaled.device:
            in_min = in_min.to(inputs_unscaled.device)
        if in_rng.device != inputs_unscaled.device:
            in_rng = in_rng.to(inputs_unscaled.device)

        scaled_inputs = (inputs_unscaled - in_min) / in_rng
        
        # Predict
        model.eval()
        model = model.to(scaled_inputs.device)

        with torch.no_grad():
            scaled_inputs = scaled_inputs.to(torch.float32)
            preds_scaled = model(scaled_inputs)

        # Unscale outputs
        return unscale_outputs(preds_scaled, scaling_info)

    return lambda inputs: predict_fn(model.parameters(), inputs, model)
