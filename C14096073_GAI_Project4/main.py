from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
from models import *
import torch
import torch.optim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from skimage.color import rgb2gray
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
import sys
sys.path.insert(0, './utils.py')
from utils2 import ExponentialMovingAverage
import os
import math
import argparse
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
imsize =-1
PLOT = False
sigma = 25
sigma_ = sigma/255.
import torch
from tqdm import tqdm
import time
import math
from torchvision.utils import save_image
# deJPEG
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/denoising/3.png'

"""# Load image"""

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)

    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np

    if PLOT:
        plot_image_grid([img_np], 4, 5);

elif fname == 'data/denoising/3.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=4)# make pic size :nxn
    img_np = pil_to_np(img_pil)
    print("Initial shape of img_np:", img_np.shape)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)# Add Gaussian noise to an image.(≒Ground Truth)

    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6);
else:
    assert False

"""# Setup"""

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99

if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3
    figsize = 5

    net = skip(
                input_depth, 3,
                num_channels_down = [8, 16, 32, 64, 128],
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

elif fname == 'data/denoising/3.png':

    num_iter =100
    input_depth = 3
    figsize = 4

    net = skip(
                input_depth, 3,
                num_channels_down = [8, 16, 32],
                num_channels_up   = [8, 16, 32],
                num_channels_skip = [0, 0, 0],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)
else:
    assert False

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0]),noise_type='n', var=1.0).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

"""# Optimize"""

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0
out_np_list = []
i = 0
def closure():

    global i, out_avg, psrn_noisy_last, last_net, net_input,out_np_list

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()


    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         img_np], factor=figsize, nrow=1)

    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        out_np_list.append(np.clip(out_np, 0, 1))

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=64)    
    parser.add_argument('--epochs',type = int,default=50)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='./results/steps_00046900.pt')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=16)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args

def main(args):
    device="cuda"

    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=64,
                in_channels=3,
                base_dim=args.model_base_dim,
                dim_mults=[1,2,4,8]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])
        model_ema.eval()
        start_time = time.time()
        out_np = torch_to_np(net(net_input))
        #q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);
        samples=model_ema.module.sampling2(i,out_np,args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        elapsed_time = time.time() - start_time
        print(f"Sampling with DIP (Elapsed Time: {elapsed_time:.2f} seconds)")
        save_image(samples,"ddpm_dip.png",nrow=int(math.sqrt(args.n_samples)))
        start_time = time.time()
        samples=model_ema.module.sampling(i,args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        elapsed_time = time.time() - start_time
        print(f"Sampling without DIP (Elapsed Time: {elapsed_time:.2f} seconds)")
        save_image(samples,"ddpm_nodip.png",nrow=int(math.sqrt(args.n_samples)))


if __name__=="__main__":
    args=parse_args()
    main(args)







