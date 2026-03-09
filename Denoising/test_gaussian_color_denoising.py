## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='/root/Restormer/demo/input/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/gaussian_color_denoising', type=str, help='Path to weights')
parser.add_argument('--model_type', default='non_blind', choices=['non_blind','blind'], type=str, help='non_blind: fixed sigma, blind: variable sigma')
parser.add_argument('--sigma', default='50', type=str, help='Sigma values')

args = parser.parse_args()

####### Load yaml #######
if args.model_type == 'blind':
    yaml_file = 'Options/GaussianColorDenoising_Restormer.yml'
else:
    yaml_file = f'Options/GaussianColorDenoising_RestormerSigma{args.sigma}.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = [int(args.sigma)]

factor = 8

# 自定义数据集路径
datasets = [
    '/root/autodl-tmp/datasets/valid/DIV2K_valid_HR',
    '/root/autodl-tmp/datasets/valid/LSDIR_valid_HR'
]

for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration = Restormer(**x['network_g'])
    if args.model_type == 'blind':
        weights = args.weights+'_blind.pth'
    else:
        weights = args.weights + '_sigma' + str(sigma_test) +'.pth'
    # 强制使用正确的模型路径
    weights = '/root/autodl-tmp/Restormer/pretrained_models/gaussian_color_denoising_sigma50.pth'
    print(f'加载模型: {weights}')
    checkpoint = torch.load(weights)
    model_restoration.load_state_dict(checkpoint['params'])

    print("===>Testing using weights: ",weights)
    print("------------------------------------------------")
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        result_dir_tmp = os.path.join(args.result_dir, args.model_type, dataset, str(sigma_test))
        os.makedirs(result_dir_tmp, exist_ok=True)

    #添加开始（本人加的）
    if args.model_type == 'non_blind' and args.sigma == '50':
        print("使用 sigma50 专用模型")
    # 如果脚本支持 --weights 参数，可以在这里重新指定
    if hasattr(args, 'weights') and args.weights:
        weights_path = args.weights
    else:
        weights_path = '/root/Restormer/pretrained_models/gaussian_color_denoising_sigma50.pth'
        print(f"强制使用模型: {weights_path}")
        # 重新加载模型（如果需要）
        checkpoint = torch.load(weights_path)
        model_restoration.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    #添加结束
        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img = np.float32(utils.load_img(file_))/255.

                np.random.seed(seed=0)  # for reproducibility
                img += np.random.normal(0, sigma_test/255., img.shape)

                img = torch.from_numpy(img).permute(2,0,1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 8
                h,w = input_.shape[2], input_.shape[3]
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:,:,:h,:w]

                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                utils.save_img(save_file, img_as_ubyte(restored))
