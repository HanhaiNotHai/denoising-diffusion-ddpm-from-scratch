"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F
from torch import Tensor


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    '''return a linear beta schedule of length T'''

    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

# Step 2 - alphas_from_betas
def alphas_from_betas(betas: Tensor):
    '''return 1 - betas'''

    return 1.0 - betas

# Step 3 - cumprod_alphas
def cumprod_alphas(alphas: Tensor):
    '''cumulative product of alphas'''

    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
def extract_into_batch(a: Tensor, t: Tensor, x: Tensor):
    '''gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x'''

    return a.gather(0, t.long()).reshape(-1, 1, 1, 1)

# Step 5 - q_sample (not yet solved)
# TODO: implement

# Step 6 - build_diffusion_schedule (not yet solved)
# TODO: implement

# Step 7 - noise_prediction_loss (not yet solved)
# TODO: implement

# Step 8 - diffusion_training_loss (not yet solved)
# TODO: implement

# Step 9 - timestep_embedding (not yet solved)
# TODO: implement

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

