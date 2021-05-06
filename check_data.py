# coding: UTF-8


import os
import h5py
import tqdm
import shutil
import pickle
import numpy as np


data_dir = '../SCI_dataset/Download_ICVL/'
data_list = os.listdir(data_dir)
# data_list.sort()
can_list = []
cannot_list = []


for name in data_list:
    # with h5py.File(os.path.join(data_dir, name), mode='r') as f:
    #     print(f.shape)
    try:
        data = h5py.File(os.path.join(data_dir, name))
        can_list.append(name)
    except OSError:
        cannot_list.append(name)

with open('can_open_list.pkl', 'wb') as f:
    pickle.dump(can_list, f)
with open('cannot_open_list.pkl', 'wb') as f:
    pickle.dump(cannot_list, f)
