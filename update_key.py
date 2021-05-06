# coding: UTF-8


import os
import h5py
from tqdm import tqdm
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize


data_dir = '../SCI_dataset/ICVL_2021/'
data_list = os.listdir(data_dir)
move_dir = '../SCI_dataset/ICVL_2021_comp'
if os.path.exists(move_dir):
    shutil.rmtree(move_dir)
os.makedirs(move_dir, exist_ok=True)

# data = h5py.File(os.path.join(data_dir, data_list[0]))
# data_reshape = np.array(data['rad']).transpose(1, 2, 0)[::-1]
# plt.imshow(normalize(data_reshape[:, :, :3]))
# plt.show()


for name in tqdm(data_list):
    new_data = {}
    with h5py.File(os.path.join(data_dir, name), 'r') as data:
        new_data['bands'] = np.array(data['bands'])
        new_data['data'] = np.array(data['rad']).transpose(1, 2, 0)[::-1]
        new_data['rgb'] = np.array(data['rgb'])
    with h5py.File(os.path.join(move_dir, name), 'a') as f:
        f.create_dataset('bands', data=new_data['bands'])
        f.create_dataset('data', data=new_data['data'])
        f.create_dataset('rgb', data=new_data['rgb'])
