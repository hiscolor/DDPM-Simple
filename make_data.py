import numpy as np
import torch
import matplotlib.pyplot as plt

def make_swiss_roll(n_samples=5000, noise_std=0.1):
    """
    生成二维瑞士卷（Swiss Roll）数据。
    每个样本是一个二维坐标点 (x, y)，用于模拟真实分布 q(x_0)。

    参数：
        n_samples: 样本数
        noise_std: 噪声标准差，控制数据分布的模糊程度
    返回：
        data: torch.FloatTensor，形状 [n_samples, 2]
    """
    # 生成角度参数 t，控制卷曲程度
    # t 分布在 [0, 2π] 之间，控制数据点的分布范围
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))
    # 生成 x, y 坐标
    # x = t * cos(t) 控制数据点的水平分布
    # y = t * sin(t) 控制数据点的垂直分布
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1)
    data = data + noise_std * np.random.randn(n_samples, 2)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)

if __name__ == "__main__":
    data = make_swiss_roll(5000)
    plt.figure(figsize=(4, 4))
    plt.scatter(data[:, 0], data[:, 1], s=2, alpha=0.7,
                c=np.linspace(0, 1, len(data)), cmap='viridis')
    plt.title("Swiss Roll Data")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("00_swiss_roll_data.png")
    plt.close()
    print("✅ 数据图已保存为 00_swiss_roll_data.png")
