# coding: utf-8


import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from data_loader import PatchEvalDataset
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.PixelwiseFusionReconst import FusionReconstHSI
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM


# parser = argparse.ArgumentParser(description='Evaluate Model')
# parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
# parser.add_argument('--block_num', '-b', default=9, type=int, help='Model Block Number')
# parser.add_argument('--learned_time', '-lt', default='0000', type=str, help='learned model day')
# args = parser.parse_args()


device = 'cpu'

datasets = ['CAVE', 'Harvard']
block_num = 9
dt_now = '0513'
for data_name in datasets:

    ckpt_dir = f'../SCI_ckpt/{data_name}_{dt_now}/all_trained'
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort()

    output_path = os.path.join('../SCI_result/', f'{data_name}_{dt_now}')
    output_learned_path = os.path.join(output_path, 'validation_process')
    os.makedirs(output_learned_path, exist_ok=True)

    titles = ['RMSE', 'PSNR', 'SSIM', 'SAM']
    all_evaluaters = dict([(title, []) for title in titles])
    show_legend = []
    for ckpt_name in ckpt_list:
        show_legend.append(ckpt_name)
        output_path = os.path.join('../SCI_result/', f'{data_name}_{dt_now}', ckpt_name)
        ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=torch.device(device))
        val_loss = np.array(ckpt['val_loss'])
        if val_loss.shape == ():
            continue
        for i, title in enumerate(titles):
            all_evaluaters[title].append(val_loss[:, i])
    for i, title in enumerate(titles):
        plt.figure(figsize=(16, 9))
        plt.plot(np.array(all_evaluaters[title]).T)
        plt.legend(show_legend, loc='best')
        plt.title(title)
        plt.savefig(os.path.join(output_learned_path, f'{data_name}_{title}.png'), bbox_inches='tight')
