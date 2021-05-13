# coding: UTF-8


import os
import h5py
from tqdm import tqdm
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize


data_dir = '../SCI_dataset/ICVL_2021_comp'
data_list = os.listdir(data_dir)
data_list.sort()
img_save_dir = '../SCI_dataset/ICVL_2021_img'
os.makedirs(img_save_dir, exist_ok=True)

for i, name in enumerate(data_list):
    new_data = {}
    with h5py.File(os.path.join(data_dir, name), 'r') as data:
        print(f'{i:05d}, {name:40s}, {data["data"].shape}')
        '''
        img = np.array(data['data'])
        img = normalize(img)
        plt.imshow(img[:, :, (26, 16, 9)])
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(img_save_dir, name[:-4] + '.png'), bbox_inches='tight')
        '''
