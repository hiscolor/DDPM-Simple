import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """
    正弦-余弦时间嵌入模块，用于编码时间步信息。
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        t = t.float().unsqueeze(1)
        sinusoid = t * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
        return emb

class DiffusionModel(nn.Module):
    """
    简单MLP结构的Diffusion噪声预测模型。
    输入：x_t, t
    输出：预测噪声 ε̂
    """
    def __init__(self, input_dim=2, hidden_dim=256, time_dim=64):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.time_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU()
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        x_emb = self.input_net(x)
        t_emb = self.time_net(self.time_emb(t))
        h = torch.cat([x_emb, t_emb], dim=-1)
        return self.output_net(h)
