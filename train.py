import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from make_data import make_swiss_roll
from noise import NoiseScheduler
from forward_process import DiffusionModel

# ========== 初始化 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_timesteps = 1000
epochs = 500
batch_size = 128
lr = 1e-4

data = make_swiss_roll(5000).to(device)
loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
scheduler = NoiseScheduler(num_timesteps, device=device)
model = DiffusionModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

loss_history = []

# ========== 训练循环 ==========
for epoch in tqdm(range(epochs), desc="Training"):
    for (x0,) in loader:
        x0 = x0.to(device)
        t = torch.randint(1, num_timesteps, (x0.size(0),), device=device)
        xt, eps = scheduler.add_noise(x0, t)
        eps_pred = model(xt, t)
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())

# 保存Loss曲线
plt.figure(figsize=(8, 3))
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.savefig("training_loss.png")
torch.save(model.state_dict(), "diffusion_model.pth")
print("✅ 模型训练完成，权重已保存到 diffusion_model.pth")
