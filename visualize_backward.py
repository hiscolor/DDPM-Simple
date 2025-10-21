import torch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from forward_process import DiffusionModel  
from noise import NoiseScheduler

@torch.no_grad()
def reverse_sampling_visualization(model, scheduler, num_samples=2000, selected_t=None, save_path="reverse_progression.png"):
    """
    可视化反向采样过程，将不同时间步的样本画在一张图上。
    Args:
        model: 训练好的噪声预测模型
        scheduler: 噪声调度器
        num_samples: 采样点数量
        selected_t: 选取的可视化时间步列表，例如 [999, 299, 99, 29, 0]
        save_path: 图片保存路径
    """
    device = next(model.parameters()).device
    T = scheduler.betas.size(0)
    if selected_t is None:
        selected_t = [999, 299, 99, 29, 0]

    # 初始化 x_T ~ N(0, I)
    xt = torch.randn(num_samples, 2, device=device)

    # 存储不同时间步的采样结果
    saved_samples = {T - 1: xt.clone().cpu()}

    # 开始反向去噪
    print("🚀 开始反向采样过程 ...")
    for t_int in reversed(range(T)):
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)
        eps_pred = model(xt, t)

        beta_t = scheduler.betas[t].view(-1, 1)
        alpha_t = scheduler.alphas[t].view(-1, 1)
        alpha_bar_t = scheduler.alphas_cumprod[t].view(-1, 1)

        # DDPM均值公式
        mean = (1. / torch.sqrt(alpha_t)) * (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred)

        # 采样噪声项（最后一步不再加噪声）
        noise = torch.zeros_like(xt)
        if t_int > 0:
            noise = torch.sqrt(beta_t) * torch.randn_like(xt)

        xt = mean + noise

        # 保存选定时间步的样本
        if t_int in selected_t:
            saved_samples[t_int] = xt.clone().cpu()

    # 绘制结果
    plt.figure(figsize=(15, 3))
    for i, t in enumerate(sorted(saved_samples.keys(), reverse=True)):
        data = saved_samples[t].numpy()
        plt.subplot(1, len(saved_samples), i + 1)
        plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.6)
        plt.title(f"t={t}")
        plt.axis('equal'); plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 反向采样可视化完成，已保存到 {save_path}")


if __name__ == "__main__":
    # ========== 初始化 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载噪声调度器
    scheduler = NoiseScheduler(num_timesteps=1000, device=device)

    # 加载模型
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
    model.eval()

    # 执行可视化
    reverse_sampling_visualization(
        model=model,
        scheduler=scheduler,
        num_samples=2000,
        selected_t=[999, 299, 99, 29, 0],
        save_path="reverse_progression.png"
    )
