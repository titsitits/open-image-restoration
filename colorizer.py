#The colorization model comes from https://github.com/jantic/DeOldify , full credit to their authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from PIL import Image
import time
import ImagePipeline_utils as IP
import glob
from ImagePipeline_utils import timing, suppress_stdout
import subprocess
import shutil


from os import path
import torch



def colorize(args):
	
	os.chdir(args.git_dir)
	try:
		colorizer = get_image_colorizer(artistic=True)
	except Exception as e:
		print("Model could not be loaded. The file may be missing or corrupted. Let's try to download it again...")
			model_dir = path.join(args.git_dir, 'models')

			if not path.exists(model_dir):
				os.makedirs(model_dir)

			model_path = path.join(args.git_dir, 'models/ColorizeArtistic_gen.pth')

			if path.isfile(model_path):
				os.remove(model_path)
			
			command = "wget wget https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeArtistic_gen.pth?dl=0 -O " + model_path
			sub = subprocess.call(command, shell=True)
	
	
	count = 0
	
	from os import path
	
	col_output_dir = args.output_dir
	col_input_dir = args.input_dir
	
	if not path.exists(col_output_dir):
		os.mkdir(col_output_dir)

	imname = '*'
	orignames = glob.glob(col_input_dir + '/' + imname)

	with timing('Colorization'):

		for source_path in orignames:
			count = count+1
			if source_path.endswith(".jpg") or source_path.endswith(".bmp") or source_path.endswith(".png"):
				
				filepath, file = path.split(source_path)
				filename, ext = path.splitext(file)	
				
				with timing(str(count) + '. ' + file):
					
					for i in args.render_factor:


						
						if len(args.render_factor) > 1:
							newfile = path.join(col_output_dir, filename + '_r_' + str(i) + ext)						
						else:
							newfile = path.join(col_output_dir, file)
						
						#check if file was not already processed
						#if not path.isfile(newfile):
						try:
							result = colorizer.get_transformed_image(source_path, render_factor=i)
							result.save(newfile)
						except Exception as e:
							print(e)

				
if __name__ == '__main__':	
	
	parser = argparse.ArgumentParser()

	parser.add_argument( '-i', '--input-dir', help='location to load input images', required=True)
	parser.add_argument('-o', '--output-dir', help='location to load output images', required=True)	
	parser.add_argument('-g','--git-dir', help='location to download DeOldify repository', default='./DeOldify', type=str)
	parser.add_argument('-r','--render-factor', help='rendering factor (DeOldify parameter)', nargs='+', default=[20], type=int)
	
	args = parser.parse_args()
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	if not path.exists(args.git_dir):
		print('Cloning DeOldify repository...')
		command = "git clone https://github.com/jantic/DeOldify.git " + args.git_dir
		sub = subprocess.call(command, shell=True)
	
	model_dir = path.join(args.git_dir, 'models')
	
	if not path.exists(model_dir):	
		os.makedirs(model_dir)

	model_path = path.join(args.git_dir, 'models/ColorizeArtistic_gen.pth')
	
	if not path.isfile(model_path):
		print("Model mising. Let's download it...")
		command = "wget wget https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeArtistic_gen.pth?dl=0 -O " + model_path
		sub = subprocess.call(command, shell=True)

	if not path.exists('./fasterai'):
		shutil.copytree(path.join(args.git_dir, 'fasterai'), './fasterai' )	
	
	from fasterai.visualize import *
	plt.style.use('dark_background')
	torch.backends.cudnn.benchmark=True
	
	colorize(args)

	#Uncomment this if you have issues with gpu memory release
	#IP.reset_gpu(0)