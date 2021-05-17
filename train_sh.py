# coding: utf-8


import os
import sys
import shutil
import argparse
import datetime
import torch
import torchvision
from torchsummary import summary
from trainer import Trainer
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.PixelwiseFusionReconst import FusionReconstHSI
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output
from utils import plot_progress


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=9, type=int, help='Model Block Number')
parser.add_argument('--feature_block', '-fb', default=2, type=int, help='fusion feature num')
parser.add_argument('--start_time', '-st', default='0000', type=str, help='start learning time')
args = parser.parse_args()


batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 31
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num
feature_block = args.feature_block
dt_now = args.start_time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


img_path = f'../SCI_dataset/My_{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('../SCI_result', f'{data_name}_{dt_now}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
filter_path = os.path.join('../SCI_dataset', 'D700_CSF.mat')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now}')
os.makedirs(ckpt_path, exist_ok=True)
all_ckpt_path = os.path.join(ckpt_path, 'all_trained')
os.makedirs(all_ckpt_path, exist_ok=True)


model_obj = {'HSCNN': HSCNN, 'HyperReconNet': HyperReconNet, 'DeepSSPrior': DeepSSPrior, 'FusionReconst': FusionReconstHSI}
activations = {'HSCNN': 'leaky', 'HyperReconNet': 'relu', 'DeepSSPrior': 'relu', 'FusionReconst': 'none'}


train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, transform=train_transform, concat=concat_flag)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, transform=test_transform, concat=concat_flag)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if model_name not in model_obj.keys():
    print('Enter Model Name')
    sys.exit(0)


activation = activations[model_name]
model = model_obj[model_name](input_ch, 31, block_num=block_num,
                              activation=activation, feature_block=feature_block)


if model_name == 'FusionReconst':
    save_model_name = f'{model_name}_{activation}_{block_num:02d}_{feature_block:02d}'
else:
    save_model_name = f'{model_name}_{activation}_{block_num:02d}'
finish_ckpt = os.path.join(all_ckpt_path, f'{save_model_name}_{dt_now}.tar')
if os.path.exists(os.path.join(finish_ckpt)):
    print('already trained')
    sys.exit(0)
if os.path.exists(f'../SCI_ckpt/{data_name}_SOTA/{model_name}_{block_num:02d}.tar'):
    shutil.copy(f'../SCI_ckpt/{data_name}_SOTA/{model_name}_{block_num:02d}.tar',
        f'../SCI_ckpt/{data_name}_{dt_now}/all_trained/{model_name}_{block_num:02d}.tar')
    print('copy model')
    sys.exit(0)


model.to(device)
criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, .5)


summary(model, (input_ch, 64, 64))
print(model_name)


ckpt_cb = ModelCheckPoint(ckpt_path, save_model_name,
                          mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler,
                  callbacks=[ckpt_cb],
                  output_progress_path=os.path.join(ckpt_path, save_model_name))
train_loss, val_loss = trainer.train(epochs, train_dataloader, test_dataloader)
if model_name in ['HSCNN', 'DeepSSPrior', 'HyperReconNet']:
    torch.save({'model_state_dict': model.state_dict(),
                'optim': optim.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'epoch': epochs},
               f'../SCI_ckpt/{data_name}_SOTA/{model_Name}_{block_num:02d}.tar')
torch.save({'model_state_dict': model.state_dict(),
            'optim': optim.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'epoch': epochs},
           finish_ckpt)
plot_progress(ckpt_path, mode='train')
plot_progress(ckpt_path, mode='val')
