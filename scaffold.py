"""
Denoising Diffusion (DDPM) from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: train a tiny DDPM on synthetic blob images and sample new ones.

Story: pure Gaussian noise is unstructured (high nearest-neighbor MSE to the data).
After a short training run the reverse process produces images much closer to the
bright-disk manifold — visible both as a drop in training loss and as a lower
sample_quality_mse than the noise baseline.
"""
# Imports live here too: /assemble concatenates solutions FIRST, then this
# scaffolding. Names like F are resolved at call time inside main(), so these
# imports cover user solutions that used F.conv2d / torch.* without importing.
import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    result = ddpm_experiment(
        n_data=64,
        size=8,
        T=20,
        hidden=16,
        num_steps=60,
        batch_size=16,
        lr=5e-2,
        n_samples=8,
        seed=0,
    )
    print("steps:", len(result["train_losses"]))
    print(f"loss: {result['train_losses'][0]:.4f} -> {result['final_loss']:.4f}")
    print(f"noise baseline MSE:  {result['noise_mse']:.4f}")
    print(f"trained sample MSE:  {result['sample_mse']:.4f}")
    print(f"improvement (noise - sample): {result['improvement']:.4f}")


if __name__ == "__main__":
    main()

