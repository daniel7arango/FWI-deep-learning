import os
from typing import Any
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm import trange

from torch import optim
from utils import *
from modules import UNet
from guided_diffusion.script_util import create_model
import logging
from torch.utils.tensorboard import SummaryWriter
from ddpm import Diffusion
from guided_diffusion.NetworkPaul import AttU_Net
import utils_image as util
from guided_diffusion.solverDIP import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

device = torch.device("cpu")
model = create_model(image_size=128,num_channels=64,num_res_blocks=3)
model = model.to(device)
PSNR = PeakSignalNoiseRatio().to(device)
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

ckpt = torch.load("models/DDPM_cosine_vels/ckpt.pt", map_location='cpu')
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=128, device=device)

#LOAD DATA EXAMPLE
pathimg = 'testvmodels/curvevel_a.jpg'

img_H = util.imread_uint(pathimg, n_channels=1)/255.
img_H = img_H.transpose(2,0,1)[np.newaxis,:,:,:]
print(img_H.shape)

img_H: int | Any=img_H*2-1
img_H=torch.as_tensor(img_H.copy())

##########################
noise_steps=1000 # no modificar

seq = list(range(noise_steps))
seq = seq[::-1]
progress_seq = seq[::(len(seq) // 20)]

###############################
#noise schedule from diffusion
beta = diffusion.prepare_noise_schedule(schedule_name='cosine').to(device)
alpha = 1-beta
alpha_hat = torch.cumprod(alpha, dim=0)
##################################

#### definir el steps desde donde se inserta x_ti
ti=200
seq_new=[indx for indx in seq if indx <= ti]
x = diffusion.noise_images(img_H.to(device),(torch.ones(1) * (ti)).long().to(device))[0]

### preparar la difusion iniciando desde t_i y x_ti
progress_img = []
progress_xdip= []
pbar =  trange(len(seq_new))

for i in tqdm(range(1, ti), position=0):
    #logging.info(f"Resampling t={seq_new[i]} ")
    #print("resampling at t= ",seq_new[i])
    
    t = (torch.ones(1) * i).long().to(device)
    x = diffusion.resample_single(model, seq_new[i], x.to(torch.float32))

    if seq_new[i] in progress_seq:
        progress_img.append(x)


progress_img.append(x)

result = torch.cat(progress_img, dim=0)
result = (result.clamp(-1, 1) + 1) / 2
result = (result * 255).type(torch.uint8)

# guardar resultado
save_images(result, f"results/diffusion_ti{ti}.jpg")
psnr0=PSNR(x.clamp(-1, 1) ,img_H.to(device, dtype=torch.float32))
ssim0=SSIM(x.clamp(-1, 1) ,img_H.to(device, dtype=torch.float32))

print('PSNR:', psnr0)
print('SSIM:', ssim0)

