import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cgru_error import CRVAE, VRAE4E, train_phase1, train_phase2
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
df = pd.read_csv('csv/filtered_data.csv', header=None,low_memory=False)

# 移除前一行和前一列
df = df.iloc[1:,1:]

# 移除所有列值全为0的列
df = df.loc[:, (df != 0).any(axis=0)]

# 应用均值归一化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.values)

# 转换为Tensor
data_tensor = torch.tensor(df_scaled, dtype=torch.float32)

# 创建一个简单的Dataset类
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建Dataset和DataLoader
dataset = SimpleDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 选择使用整个数据集作为输入
X = data_tensor

# 确保数据在正确的设备上
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
X = X.to(device)

# 数据维度
dim = X.shape[1]  # 特征数
GC = torch.eye(dim, device=device)  # 初始化因果关系矩阵
for i in range(1, dim):
    GC[i, i-1] = 1

# 模型初始化
full_connect = torch.ones_like(GC)
# cgru = CRVAE(dim, full_connect.numpy(), hidden=64).to(device)
cgru = CRVAE(dim, full_connect, hidden=64).to(device)
vrae = VRAE4E(dim, hidden=64).to(device)

# 因为您的数据已经是Tensor了，需要调整shape以符合模型的输入要求
X = X.unsqueeze(0)  # 添加一个批次维度

# 模型训练
train_loss_list = train_phase1(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000,
    check_every=50)

# 计算并显示因果关系矩阵
GC_est = cgru.GC().cpu().numpy()
print("因果矩阵 (Causal-effect Matrix):\n", GC_est)
np.savetxt("GC_est_matrix.txt", GC_est, fmt="%s")

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
GC_cpu = GC.cpu().numpy()
axarr[0].imshow(GC_cpu, cmap='Blues')
axarr[0].set_title('Causal-effect matrix')
axarr[0].set_ylabel('Effect series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_ylabel('Effect series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.savefig('lala.png')
plt.close(fig)

np.save('GC_trans.npy', GC_est)
full_connect = np.load('GC_trans.npy')
#full_connect = torch.ones_like(GC)


#%%
cgru = CRVAE(X.shape[-1], full_connect, hidden=64).cuda(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)


train_loss_list = train_phase2(
    cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
    check_every=50)

