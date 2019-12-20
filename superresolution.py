#Most of the code is taken from to https://github.com/xinntao/ESRGAN , full credits to their authors


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import time
import ImagePipeline_utils as IP
from ImagePipeline_utils import timing
import subprocess
import shutil

from os import path

import glob
import cv2
import numpy as np
import torch


import functools
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

	
def superres(args):
	
	#os.chdir(args.git_dir)
	
	device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
	# device = torch.device('cpu')	
	
	
	test_img_folder = path.join(args.input_dir, '*')
	sr_output_dir = args.output_dir

	use_16bit = args.half
	
	model = RRDBNet(3, 3, 64, 23, gc=32)
	
	if use_16bit:
		model.half()
		
	if args.arch == 'PSNR':
		model_path = path.join(args.model_dir, 'RRDB_PSNR_x4.pth')
	else:
		model_path = path.join(args.model_dir, 'RRDB_ESRGAN_x4.pth')
		
	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()
	model = model.to(device)

	print('Model path {:s}. \nTesting...'.format(model_path))

	with timing('super-resolution'):

		idx = 0
		for im_path in glob.glob(test_img_folder):
			idx += 1
			base, ext = path.splitext(path.basename(im_path))
			print(idx, base)
			# read images
			img = cv2.imread(im_path, cv2.IMREAD_COLOR)
			img = img * 1.0 / 255
			if use_16bit:
				img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).half()
			else:
				img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

			img_LR = img.unsqueeze(0)
			img_LR = img_LR.to(device)

			with torch.no_grad():
				output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
			output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
			output = (output * 255.0).round()
			cv2.imwrite(path.join(sr_output_dir, base + ext), output)


				
if __name__ == '__main__':	
	
	parser = argparse.ArgumentParser()

	parser.add_argument( '-i', '--input-dir', help='location to load input images', required=True)
	parser.add_argument('-o', '--output-dir', help='location to load output images', required=True)	
	parser.add_argument('-m','--model-dir', help='location to the model to load', default='./ESRGAN/models', type=str)
	parser.add_argument('-a','--arch', help='Architecture to use (GAN or PSNR)', default='PSNR', type=str)
	parser.add_argument('-b','--half', help='Use 16bit (if True) or 32bit model (if False)', default=True, type=bool)

	args = parser.parse_args()
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	#Clone git if not present
	#if not path.exists(args.git_dir):
	#	print('Cloning ESRGAN repository...')
	#	command = "git clone https://github.com/xinntao/ESRGAN.git " + args.git_dir
	#	sub = subprocess.call(command, shell=True)
	
	#Create model directory if not present
	if not path.exists(args.model_dir):	
		os.makedirs(args.model_dir)
		
	#Download model if not present
	if args.arch == 'PSNR':
		model_path = path.join(args.model_dir, 'RRDB_PSNR_x4.pth')
		command = "wget -O " + model_path + " --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN'"
	else:
		model_path = path.join(args.model_dir, 'RRDB_ESRGAN_x4.pth')
		command = "wget -O " + model_path + " --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene'"
	
	if not path.isfile(model_path):
		print("Model mising. Let's download it...")
		#command = "wget wget --no-check-certificate " + model_url + " -O " + model_path
		sub = subprocess.call(command, shell=True)
	
	superres(args)

	#Uncomment this if you have issues with gpu memory release
	#IP.reset_gpu(0)