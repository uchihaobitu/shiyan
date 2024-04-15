# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 20:00:04 2022

@author: 61995
"""

import numpy as np
import torch

from models.cgru_error import CRVAE, VRAE4E, train_phase1,train_phase3,train_phase4

# device = torch.device('cuda')
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:1")

# X_np = np.load("henon.npy").T
X_np = np.load("train_use.npy")

X_np = X_np.astype(np.float32)

print(X_np.shape)
print(X_np)

dim = X_np.shape[-1]
GC = np.zeros([dim, dim])
for i in range(dim):
    GC[i, i] = 1
    if i != 0:
        GC[i, i - 1] = 1
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)


# full_connect = np.ones(GC.shape)
full_connect = np.load('W_est.npy')
# full_connect = torch.tensor(full_connect,device=device,dtype=torch.float64)
cgru = CRVAE(X.shape[-1], full_connect, hidden=64).to(device=device)
vrae = VRAE4E(X.shape[-1], hidden=64).to(device=device)

# %%


# train_loss_list = train_phase1(
#     cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000, check_every=50, batch_size=32
# )  # 0.1
train_loss_list = train_phase4(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000, check_every=50, batch_size=128
)  # 0.1


# %%no
GC_est = cgru.GC().cpu().data.numpy()
np.save('GC_henon_my.npy', GC_est)

# # Make figures
# fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
# axarr[0].imshow(GC, cmap='Blues')
# axarr[0].set_title('Causal-effect matrix')
# axarr[0].set_ylabel('Effect series')
# axarr[0].set_xlabel('Causal series')
# axarr[0].set_xticks([])
# axarr[0].set_yticks([])

# axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
# axarr[1].set_ylabel('Effect series')
# axarr[1].set_xlabel('Causal series')
# axarr[1].set_xticks([])
# axarr[1].set_yticks([])

# # Mark disagreements
# for i in range(len(GC_est)):
#     for j in range(len(GC_est)):
#         if GC[i, j] != GC_est[i, j]:
#             rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
#             axarr[1].add_patch(rect)

# # plt.show()
# plt.savefig('GC_henon.png')
#
# #np.save('GC_henon.npy', GC_est)
# full_connect = np.load('GC_henon.npy')
#
#
# #%%
# cgru = CRVAE(X.shape[-1], full_connect, hidden=64).cuda(device=device)
# vrae = VRAE4E(X.shape[-1], hidden=64).cuda(device=device)
#
#
# train_loss_list = train_phase2(
#     cgru, vrae, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=10000,
#     check_every=50)
