#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 20251021
# @Author  : lby
# @File    : sampling.py
# @Description: 从噪声中反向采样

import torch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


from noise import NoiseScheduler
from forward_process import DiffusionModel

@torch.no_grad()
def sample(model, scheduler, num_samples=5000):
    """
    按照Algorithm 2 (Sampling)从噪声中反向采样。
    """
    model.eval()
    device = next(model.parameters()).device
    xt = torch.randn(num_samples, 2).to(device)

    for t_int in reversed(range(scheduler.betas.size(0))):
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)
        eps_pred = model(xt, t)
        beta_t = scheduler.betas[t].view(-1, 1)
        alpha_t = scheduler.alphas[t].view(-1, 1)
        alpha_bar_t = scheduler.alphas_cumprod[t].view(-1, 1)
        mean = (1. / torch.sqrt(alpha_t)) * (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
        noise = torch.zeros_like(xt)
        if t_int > 0:
            noise = torch.sqrt(beta_t) * torch.randn_like(xt)
        xt = mean + noise
    return xt.cpu()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = NoiseScheduler(1000, device=device)
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))

    samples = sample(model, scheduler, num_samples=5000)
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.6)
    plt.axis('equal'); plt.axis('off')
    plt.title("Generated Samples")
    plt.savefig("generated_samples.png")    
    print("✅ 采样完成，已保存到 generated_samples.png")
