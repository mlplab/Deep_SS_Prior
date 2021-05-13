# coding: UTF-8


import os
import h5py
from tqdm import tqdm
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize
import torch


data_dir = '../SCI_dataset/ICVL_2021/'
data_list = os.listdir(data_dir)
data_list.sort()
move_dir = '../SCI_dataset/ICVL_2021_h5'
# if os.path.exists(move_dir):
#     shutil.rmtree(move_dir)
# os.makedirs(move_dir, exist_ok=True)

# data = h5py.File(os.path.join(data_dir, data_list[0]))
# data_reshape = np.array(data['rad']).transpose(1, 2, 0)[::-1]
# plt.imshow(normalize(data_reshape[:, :, :3]))
# plt.show()


# for name in tqdm(data_list):
name = data_list[0]
print(name)
# new_data = {}
with h5py.File(os.path.join(data_dir, name), 'r') as data:
    # new_data['bands'] = listray(data['bands'])
    print(data['rad'])
    # new_data['data'] = np.array(data['rad'], dtype=np.float32).transpose(1, 2, 0)[::-1]
    # new_data['rgb'] = np.array(data['rgb'])
with h5py.File(os.path.join(move_dir, name), 'r') as data:
    # f.create_dataset('bands', data=new_data['bands'])
    print(data['data'])
    my_data = np.array(data['data'])
    # f.create_dataset('rgb', data=new_data['rgb'])

print(my_data.shape)
my_data = normalize(my_data)
my_data = np.expand_dims(my_data.transpose(1, 2, 0), axis=0)
my_data = torch.as_tensor(my_data)
