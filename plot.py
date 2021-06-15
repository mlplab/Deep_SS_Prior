# coding: UTF-8


import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


sns.set()


ckpt_dir = '../SCI_ckpt/CAVE_0614/all_trained'
model_names = ['Mix', 'Vanilla']
losses = ['mse', 'mse_sam']
mix_list = glob(os.path.join(ckpt_dir, 'Mix_*'))
print(mix_list)
train_loss = []
val_loss = []
output_names = []
for name in mix_list:
    output_name = name.split('/')[-1].split('.')[0]
    ckpt = torch.load(name)
    train_loss.append(ckpt['train_loss'])
    val_loss.append(ckpt['val_loss'])
    output_names.append(output_name)
train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
title_list = ['RMSE', 'PSNR', 'SSIM', 'SAM']
for i, title in enumerate(title_list):
    plt.figure(figsize=(16, 9))
    for j, name in enumerate(output_names):
        plt.plot(val_loss[j, :, i].T, label=name)
    plt.title(title)
    plt.legend()
    plt.show()
# plt.plot(train_loss)
# plt.show()
