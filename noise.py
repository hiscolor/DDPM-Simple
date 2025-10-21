import torch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class NoiseScheduler:
    """
    负责扩散过程的噪声调度与采样。
    """
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        # 计算累积乘积 α̅_t = Π_{i=1}^t α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 计算 sqrt(α̅_t) 和 sqrt(1-ᾱ_t)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x0, t):
        """
        根据 q(x_t|x_0) = sqrt(ā_t)x_0 + sqrt(1-ā_t)ε
        向数据添加噪声。
        """
        eps = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        xt = sqrt_ac * x0 + sqrt_om * eps
        return xt, eps

if __name__ == "__main__":
    from make_data import make_swiss_roll
    data = make_swiss_roll(5000)
    scheduler = NoiseScheduler(num_timesteps=1000)
    selected_t = [0, 29, 99, 299, 999]

    plt.figure(figsize=(15, 3))
    for i, t in enumerate(selected_t):
        xt, _ = scheduler.add_noise(data, torch.full((data.size(0),), t))
        plt.subplot(1, len(selected_t), i + 1)
        plt.scatter(xt[:, 0], xt[:, 1], s=1, alpha=0.6)
        plt.title(f"t={t}")
        plt.axis('equal'); plt.axis('off')
    plt.savefig("noise_progression.png")
    print("✅ 噪声可视化完成，已保存到 noise_progression.png")
