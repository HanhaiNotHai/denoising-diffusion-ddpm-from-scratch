"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
from functools import partial
from typing import Callable

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

# Step 5 - q_sample
def q_sample(x0: Tensor, t: Tensor, noise: Tensor, alphas_cumprod: Tensor):
    '''x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise'''

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return (
        extract_into_batch(sqrt_alphas_cumprod, t, x0) * x0
        + extract_into_batch(sqrt_one_minus_alphas_cumprod, t, x0) * noise
    )

# Step 6 - build_diffusion_schedule
def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
    '''build betas, alphas, alphas_cumprod and useful sqrts'''

    betas = linear_beta_schedule(T, beta_start, beta_end)
    alphas = alphas_from_betas(betas)
    alphas_cumprod = cumprod_alphas(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'T': T,
    }

# Step 7 - noise_prediction_loss
def noise_prediction_loss(noise_pred: Tensor, noise: Tensor):
    '''MSE between predicted and true noise'''

    return ((noise_pred - noise) ** 2).mean()

# Step 8 - diffusion_training_loss
def diffusion_training_loss(
    model: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t: Tensor,
    noise: Tensor,
    alphas_cumprod: Tensor,
):
    '''q_sample -> model -> MSE(noise_pred, noise)'''

    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
def timestep_embedding(t: Tensor, dim: int):
    '''sinusoidal timestep embedding of shape (B, dim)'''

    half = dim // 2

    if half == 1:
        exponent = torch.zeros(1)
    else:
        exponent = torch.arange(half) / (half - 1)

    freqs = 10000**exponent

    args = t[:, None] / freqs[None]

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return emb

# Step 10 - init_tiny_unet
def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0):
    '''initialize tiny residual denoiser parameters'''

    torch.manual_seed(seed)

    normal = partial(torch.normal, mean=0, std=0.02, requires_grad=True)
    zeros = partial(torch.zeros, requires_grad=True)

    return {
        'conv_in_w': normal(size=(hidden, in_ch, 3, 3)),
        'conv_in_b': zeros(size=(hidden,)),
        'time_mlp_w': normal(size=(time_dim, hidden)),
        'time_mlp_b': zeros(size=(time_dim,)),
        'conv_mid_w': normal(size=(hidden, time_dim, 3, 3)),
        'conv_mid_b': zeros(size=(hidden,)),
        'conv_out_w': normal(size=(in_ch, hidden, 3, 3)),
        'conv_out_b': zeros(size=(in_ch,)),
    }

# Step 11 - tiny_unet_forward
def tiny_unet_forward(x: Tensor, t: Tensor, params: dict[str, Tensor]):
    '''time-conditioned tiny CNN predicting noise'''

    conv_in_w = params['conv_in_w']
    conv_in_b = params['conv_in_b']
    time_mlp_w = params['time_mlp_w']
    time_mlp_b = params['time_mlp_b']
    conv_mid_w = params['conv_mid_w']
    conv_mid_b = params['conv_mid_b']
    conv_out_w = params['conv_out_w']
    conv_out_b = params['conv_out_b']

    h = F.conv2d(x, conv_in_w, conv_in_b, padding=1)
    temb = timestep_embedding(t, time_mlp_w.shape[1])
    temb = F.relu(F.linear(temb, time_mlp_w, time_mlp_b))
    h += temb[..., None, None]
    h = F.relu(h)
    h = F.relu(F.conv2d(h, conv_mid_w, conv_mid_b, padding=1))
    return F.conv2d(h, conv_out_w, conv_out_b, padding=1)

# Step 12 - make_blob_dataset
def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    '''n images with a random bright disk on a black background'''

    torch.manual_seed(seed)

    radius = size // 4
    images = torch.zeros(n, 1, size, size)

    # coordinate grid
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')

    for i in range(n):
        # random integer center
        cy, cx = torch.randint(radius, size - radius, (2,))

        # filled disk mask
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

        images[i, 0][mask] = 1.0

    return images

# Step 13 - ddpm_train_step
def ddpm_train_step(
    params: dict[str, Tensor],
    x0: Tensor,
    schedule: dict[str, Tensor | int],
    lr: float = 1e-2,
    seed: int = 0,
):
    '''sample t,noise -> loss -> SGD on params'''

    T: int = schedule['T']
    alphas_cumprod: Tensor = schedule['alphas_cumprod']
    B = x0.shape[0]

    torch.manual_seed(seed)

    t = torch.randint(0, T, size=(B,))
    noise = torch.rand(size=x0.shape)
    loss = diffusion_training_loss(
        (lambda x, t: tiny_unet_forward(x, t, params)), x0, t, noise, alphas_cumprod
    )
    loss.backward()

    new_params: dict[str, Tensor] = dict()
    for m, p in params.items():
        if p.grad is None:
            new_params[m] = p.clone()
        else:
            new_params[m] = (p - lr * p.grad).detach().requires_grad_()

    return new_params, float(loss)

# Step 14 - train_ddpm
def train_ddpm(
    dataset: Tensor,
    params: dict[str, Tensor],
    schedule: dict[str, Tensor | int],
    num_steps: int = 50,
    batch_size: int = 16,
    lr: float = 1e-2,
    seed: int = 0,
):
    '''minibatch SGD training loop'''

    N = dataset.shape[0]

    history: list[float] = []
    for step in range(num_steps):
        torch.manual_seed(seed + step)

        idx = torch.randint(0, N, size=(batch_size,))
        batch = dataset[idx]
        params, loss = ddpm_train_step(params, batch, schedule, lr, seed)
        history.append(loss)

    return params, history

# Step 15 - predict_x0_from_eps
def predict_x0_from_eps(x_t: Tensor, t: Tensor, eps: Tensor, alphas_cumprod: Tensor):
    '''invert the q_sample equation for x0'''

    alphas_cumprod_t = extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat = (x_t - torch.sqrt(1 - alphas_cumprod_t) * eps) / torch.sqrt(alphas_cumprod_t)
    return x0_hat

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

