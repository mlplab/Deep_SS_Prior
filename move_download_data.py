# coding: UTF-8


import os
import h5py
from tqdm import tqdm
import shutil
import pickle
import numpy as np


data_dir = '../SCI_dataset/Download_ICVL/'
data_list = os.listdir(data_dir)
move_dir = '../SCI_dataset/ICVL_2021'
os.makedirs(move_dir, exist_ok=True)


with open('can_open_list.pkl', 'rb') as f:
    can_list = pickle.load(f)

for name in tqdm(can_list):
    shutil.copy(os.path.join(data_dir, name), os.path.join(move_dir, name))
# with open('cannot_open_list.pkl', 'wb') as f:
#     cannot_list = pickle.load(f)
