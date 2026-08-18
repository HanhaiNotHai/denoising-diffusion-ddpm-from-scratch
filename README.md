# Denoising Diffusion (DDPM) from Scratch

Implement the Denoising Diffusion Probabilistic Model (Ho et al., 2020) in pure PyTorch: linear noise schedules, closed-form forward sampling, the simplified noise-prediction loss, a tiny time-conditioned denoiser, ancestral DDPM sampling, and an end-to-end experiment on synthetic blob images that beats a pure-noise baseline.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** linear_beta_schedule
- [x] **2.** alphas_from_betas
- [x] **3.** cumprod_alphas
- [x] **4.** extract_into_batch
- [x] **5.** q_sample
- [x] **6.** build_diffusion_schedule
- [x] **7.** noise_prediction_loss
- [x] **8.** diffusion_training_loss
- [x] **9.** timestep_embedding
- [x] **10.** init_tiny_unet
- [x] **11.** tiny_unet_forward
- [x] **12.** make_blob_dataset
- [x] **13.** ddpm_train_step
- [x] **14.** train_ddpm
- [x] **15.** predict_x0_from_eps
- [ ] **16.** ddpm_p_mean_variance
- [ ] **17.** ddpm_p_sample
- [ ] **18.** ddpm_sample_loop
- [ ] **19.** sample_quality_mse
- [ ] **20.** ddpm_experiment

---

Built on Deep-ML.
