#The code is mostly taken from https://github.com/Ding-Liu/NLRN , full credit to their authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from hashlib import sha256
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import sys, os
from contextlib import contextmanager
import time
import ImagePipeline_utils as IP
#import glob
from ImagePipeline_utils import timing, suppress_stdout
import subprocess
import shutil

#avoid GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def NLRN(args):
	
	tf.logging.set_verbosity(tf.logging.ERROR)
	
	config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
	#config.gpu_options.allow_growth = True #allocate dynamically
	#sess = tf.Session(config = config)
	
	#mysess = 
	
	with tf.Session(graph=tf.Graph(), config = config) as sess:

		metagraph_def = tf.saved_model.loader.load(
			sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
		signature_def = metagraph_def.signature_def[
			tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
		input_tensor = sess.graph.get_tensor_by_name(
			signature_def.inputs['inputs'].name)
		output_tensor = sess.graph.get_tensor_by_name(
			signature_def.outputs['output'].name)
		if not os.path.isdir(args.output_dir):
			os.makedirs(args.output_dir)

		print("NLRN Model loaded...")

		with timing('Denoising - NLRN'):

			for input_file in os.listdir(args.input_dir):

				if input_file.endswith(".jpg") or input_file.endswith(".bmp") or input_file.endswith(".png"):

					with timing(input_file):

						sha = sha256(input_file.encode('utf-8'))
						seed = np.frombuffer(sha.digest(), dtype='uint32')
						rstate = np.random.RandomState(seed)

						output_file = os.path.join(args.output_dir, input_file)
						input_file = os.path.join(args.input_dir, input_file)
						input_image = np.asarray(Image.open(input_file))
						input_image = input_image.astype(np.float32) / 255.0

						def forward_images(images):
							images = output_tensor.eval(feed_dict={input_tensor: images})
							return images

						stride = args.stride
						h_idx_list = list(
							range(0, input_image.shape[0] - args.patch_size,
								stride)) + [input_image.shape[0] - args.patch_size]
						w_idx_list = list(
							range(0, input_image.shape[1] - args.patch_size,
								stride)) + [input_image.shape[1] - args.patch_size]
						output_image = np.zeros(input_image.shape)
						overlap = np.zeros(input_image.shape)

						noise_image = input_image + rstate.normal(0, args.noise_sigma / 255.0, input_image.shape)

						for h_idx in h_idx_list:
							for w_idx in w_idx_list:
								# print(h_idx, w_idx)
								input_patch = noise_image[h_idx:h_idx + args.patch_size, w_idx:
														w_idx + args.patch_size]
								input_patch = np.expand_dims(input_patch, axis=-1)
								input_patch = np.expand_dims(input_patch, axis=0)
								output_patch = forward_images(input_patch)
								output_patch = output_patch[0, :, :, 0]
								output_image[h_idx:h_idx + args.patch_size, w_idx:
											 w_idx + args.patch_size] += output_patch
								overlap[h_idx:h_idx + args.patch_size, w_idx:
										w_idx + args.patch_size] += 1
						output_image /= overlap

						output_image = np.around(output_image * 255.0).astype(np.uint8)
						output_image = Image.fromarray(output_image)
						output_image.save(output_file)
						
	#tf.reset_default_graph()

def download_model(args):
	
	#if os.path.exists("./sigma15"):
	#	shutil.rmtree('./sigma15')
	
	#if os.path.exists("sigma15_12states.zip"):
	#	os.remove('sigma15_12states.zip')
	
	
	command = "rm sigma15_12states.zip; rm -r sigma15; wget -O 'sigma15_12states.zip' --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cbaLYjPn_H6TRbLPfA6B3j45TmKdUrfa' && unzip sigma15_12states.zip"
	
	sub = subprocess.call(command, shell=True)
	#(output, err) = sub.communicate() 
	#status = sub.wait()
	#print(output)
		
	shutil.move('sigma15', args.model_dir)
	
	os.remove('sigma15_12states.zip')
	
				
if __name__ == '__main__':	
	
	parser = argparse.ArgumentParser()

	parser.add_argument( '-i', '--input-dir', help='location to load input images', required=True)
	parser.add_argument('-o', '--output-dir', help='location to load output images', required=True)	
	
	#larger patches allow to use more context, and it allows to make an average of more individual predictions (averaging ovarlapping patches is basically ensemble learning) but need more GPU resources (and the process is slower)
    #Larger strides (hop size between windows i suppose) make the process faster, but lead to lower quality (if stride is close to patch_size, squares appear in the processed image)
    #actually add noise to image before process limits the oversmoothing of the algorithm, making it more real
	
	parser.add_argument('-n',
		'--noise-sigma',
		help='Input noise',
		default=15,
		type=float)

	parser.add_argument('-p',
		'--patch-size',
		help='Number of pixels in height or width of patches (only for NLRN)',
		default=25,
		type=int)
	parser.add_argument('-s',
		'--stride',
		help='Overlapping between patches (only for NLRN)',
		default=12,
		type=int)
	parser.add_argument('-m','--model-dir', help='location to load exported model', default=os.path.join('./NLRN/models','sigma15'), type=str)
	
	args = parser.parse_args()
	
	#Verify if model is present, otherwise download it:
	if not os.path.exists(args.model_dir):
		print("Model files missing, let's download them...")
		download_model(args)

	
	NLRN(args)

	#Uncomment this if you have issues with gpu memory release
	#IP.reset_gpu(0)
	
	#input("Press Enter to continue...")
	