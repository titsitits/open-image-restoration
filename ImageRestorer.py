import time
from PIL import Image
import glob
import numpy as np
import os, sys
from contextlib import contextmanager
from numba import cuda as ncuda
from PIL import ImageFilter
import cv2

fast_denoising = False
import time
import ImagePipeline_utils as IP
import os, shutil
import warnings

import subprocess
from contextlib import contextmanager
import time, sys


#imports needed for stripe removal (see below)
from keras import backend as K
import tensorflow as tf
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam,SGD
from skimage.measure import compare_psnr, compare_ssim
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,Multiply,Add,Concatenate
from keras import regularizers
from keras.utils import plot_model
from keras import initializers
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from pywt import dwt2,idwt2 
import pywt
import scipy.io as sio


class ImageRestorer:

	def __init__(self):
		
		self._history = []
		#record current interpreter path (to ensure calling the right interpreter when calling another process)
		self._python_dir = sys.executable
		
		
	def preprocess(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		preprocess images: convert them to RGB format, resize them if too large (optional, True by default), convert them to grayscale (optional, False by default)
		Defaults options (you can override any option with a keyword argument): 
		options = {'gray':False, 'resize':True, 'size':(1000,1000), 'quiet':True, 'raising':True}
		"""
		
		#default parameters
		options = {'gray':False, 'resize':True, 'size':(1000,1000), 'quiet':True, 'raising':True}
		#default parameters are overriden by keywords arguments (e.g. gray = True) (passed by **kwargs) and are unpacked as class attributes (self.gray = True, ...)
		inputdir, outputdir = self._init_process(inputdir, outputdir, "preprocess", options, **kwargs)		
		
		with IP.quiet_and_timeit("Image preprocessing", raising = self.raising, quiet = self.quiet):
			imname = '*'
			orignames = glob.glob(os.path.join(inputdir, imname))

			for orig in orignames:

                
				try:
					im = Image.open(orig)
					#print(orig)

					#remove alpha component
					#print("torgb")
					im = IP.to_RGB(im)

					#convert to grayscale
					if self.gray:
						#print("togray")
						im = IP.to_grayscale(im)

					#resize
					if self.resize:

						width, height = im.size

						#resize only if larger than limit
						if width > self.size[0] or height > self.size[1]:
							im.thumbnail(self.size,Image.ANTIALIAS)

					#save as png (and remove previous version if inputdir = outputdir)
					path, file = os.path.split(orig)
					f, e = os.path.splitext(file)
					if inputdir == outputdir:
						os.remove(orig)
					output_name = os.path.join(outputdir, f+".png")
					im.save(output_name)
					print(output_name)
					
				except Exception as e:
					IP.force_print(e)

					
	def filter(self, inputdir = None, outputdir = None, **kwargs):		
		
		"""
		Perform basic filtering (median, gaussian and/or mean filtering) on images
		Defaults options (you can override any option with a keyword argument): 
		options = {'median':True, 'median_winsize':5, 'gaussian':True, 'gaussian_x':5, 'gaussian_y':5, 'gaussian_std':0, 'mean':True, 'mean_winsize':3, 'raising':True, 'quiet':True}
		"""
		
		options = {'median':True, 'median_winsize':5, 'gaussian':True, 'gaussian_x':5, 'gaussian_y':5, 'gaussian_std':0, 'mean':True, 'mean_winsize':3, 'raising':True, 'quiet':True}
		
		inputdir, outputdir = self._init_process(inputdir, outputdir, "filter", options, **kwargs)
		
		with IP.quiet_and_timeit("Image filtering", self.raising, self.quiet):
			imname = '*'
			orignames = glob.glob(os.path.join(inputdir, imname))

			for orig in orignames:
				print(orig)
				try:
					im = cv2.imread(orig, cv2.IMREAD_COLOR)


					#median blur
					if self.median:
						im = cv2.medianBlur(im,self.median_winsize)     

					if self.gaussian:
						im = cv2.GaussianBlur(im,(self.gaussian_x,self.gaussian_y),self.gaussian_std)

					#mean blur
					if self.mean:
						im = cv2.blur(im,(self.mean_winsize,self.mean_winsize))

					#save as png (and remove previous version if inputdir = outputdir)
					path, file = os.path.split(orig)
					f, e = os.path.splitext(file)
					if inputdir == outputdir:
						os.remove(orig)
					output_name = os.path.join(outputdir, f+".png")
					cv2.imwrite(output_name, im)
					print(output_name)
					
				except Exception as e:
					IP.force_print(e)

					
	def remove_stripes(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Remove vertical and horizontal stripes from images
		Defaults options (you can override any option with a keyword argument): 
		
		"""
		
		options = {'working_dir':'./WDNN', 'raising':True, 'quiet':True}
		inputdir, outputdir = self._init_process(inputdir, outputdir, "remove_stripes", options, **kwargs)

		stripe_remover = StripeRemover(self.working_dir)
		
		tf.logging.set_verbosity(tf.logging.ERROR)
		
		#Remove vertical stripes
		with IP.quiet_and_timeit("Removing vertical stripes", self.raising, self.quiet):
			stripe_remover.stripes_removal(inputdir, outputdir)
		#Rotate images
		IP.rotate_images(outputdir)
		#Remove horizontal stripes
		with IP.quiet_and_timeit("Removing horizontal stripes", self.raising, self.quiet):
			stripe_remover.stripes_removal(outputdir, outputdir)
		#Unrotate images
		IP.unrotate_images(outputdir)
		
	
	def remove_gaussian_noise(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Remove gaussian noise using NLRN.
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.		
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		#Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python
		
		inputdir, outputdir = self._init_process(inputdir, outputdir, "remove_gaussian_noise", options, **kwargs)
		
		tf.logging.set_verbosity(tf.logging.ERROR)

		command = self.python_dir + ' -u denoiser.py -i ' + inputdir + ' -o ' + outputdir + self.process_args + self.command_suffix
		#print("raising", self.raising)
		with IP.quiet_and_timeit("Removing gaussian noise", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)

		
	def colorize(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Colorize images using deoldify.
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		inputdir, outputdir = self._init_process(inputdir, outputdir, "colorize", options, **kwargs)		
		
		tf.logging.set_verbosity(tf.logging.ERROR)

		command = self.python_dir + ' -u colorizer.py -i ' + inputdir + ' -o ' + outputdir + self.process_args + self.command_suffix
		#print("raising", self.raising)
		with IP.quiet_and_timeit("Colorizing", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)

	
	def super_resolution(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Upsample images using ESRGAN.
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':'python', 'process_args':'', 'command_suffix':" >> log.out 2>&1"}
		inputdir, outputdir = self._init_process(inputdir, outputdir, "colorize", options, **kwargs)		
		
		tf.logging.set_verbosity(tf.logging.ERROR)

		command = self.python_dir + ' -u superresolution.py -i ' + inputdir + ' -o ' + outputdir + self.process_args + self.command_suffix
		#print("raising", self.raising)
		with IP.quiet_and_timeit("Super-resolving", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)
		
	def _init_process(self, inputdir, outputdir, process, default_options, **kwargs):
		
		"""
		This method can (should) be used at the beginning of any other method, to manage process options and history log
		"""
		
		if inputdir == None:
			if len(self._history) == 0:
				raise("Please set the inputdir at least for first processing step")
			else:
				#If no inputdir is provided, take last outputdir as new inputdir
				inputdir = self._history[-1]['output']
		
		#if outputdir is not provided, override inputdir with process result
		if outputdir == None:
			outputdir = inputdir
		elif outputdir != inputdir:
			IP.initdir(outputdir)
		
		
		options = default_options
		
		#override default parameters for the process, and unpack all arguments as class attributes
		options.update(kwargs)
		#create/update (unpack) class attributes with options
		for key in options: 
			self.__setattr__(key, options[key])
		
		
		self._history.append({"input":inputdir,"output":outputdir,"process":process,"options" : options})		
		
		IP.createdir_ifnotexists(outputdir)
		
		return inputdir, outputdir
	
	
	
	
#Stripes removal: adapted from https://github.com/jtguan/Wavelet-Deep-Neural-Network-for-Stripe-Noise-Removal (CC-BY license)
#################################
		
class StripeRemover:
	
	def __init__(self, working_dir):
		self.working_dir = working_dir
		
	#copy-paste from main.py (defined as a function) (from https://github.com/jtguan/Wavelet-Deep-Neural-Network-for-Stripe-Noise-Removal CC-BY license)
	def stripes_removal(self, input_dir, output_dir):

		working_dir = self.working_dir
		# -*- coding: utf-8 -*-
		"""
		Created on Sun Nov 25 19:08:40 2018

		@author: jtguan@stu.xidian.edu.cn
		"""




		#-----------------------------------------------------------------------#
		#-----------------------------------------------------------------------#
		#-----------------------------------------------------------------------#
		L2 =None
		init = 'he_normal'

		def SNRDWNN():

			inpt = Input(shape=(None,None,4))
			x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init,name='Conv-1')(inpt)
			x = Activation('relu')(x)
			for i in range(8):
				x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init)(x)
				x = Activation('relu')(x)
			residual = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), padding='same' ,kernel_initializer=init, name = 'residual')(x)
			res = Add(name = 'res')([inpt,residual])
			model = Model(inputs=inpt, 
						  outputs=[res,residual],
						  name = 'DWSRN'
						  )

			return model



		checkpoint_file = 'weights'

		WEIGHT_PATH = working_dir+'/'+checkpoint_file+'/weight.hdf5'
		save_dir = output_dir
		multi_GPU = 0
		#----------------------------------------------------------------------#
		test_dir = input_dir+'/'


		model =  SNRDWNN()
		model.load_weights(WEIGHT_PATH)
		print('Start to test on {}'.format(test_dir))
		out_dir = output_dir + '/'
		if not os.path.exists(out_dir):
				os.mkdir(out_dir)

		name = []
		
		#file_list = os.listdir(test_dir)
		#process only files
		#file_list = [f for f in file_list if os.path.isfile(os.path.join(test_dir,f))]
		
		orignames = glob.glob(os.path.join(test_dir, '*'))
		
		for orig in orignames:
			
			fold,file = os.path.split(orig)
			# read image
			img_clean = np.array(Image.open(test_dir + file), dtype='float32') / 255.0
			img_test = img_clean.astype('float32')
			if(len(img_test.shape)>2):
				img_test = img_test[:,:,0]
			# predict
			x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
			LLY,(LHY,HLY,HHY) = pywt.dwt2(img_test, 'haar')
			Y = np.stack((LLY,LHY,HLY,HHY),axis=2)
			# predict
			x_test = np.expand_dims(Y,axis=0)
			y_pred,noise = model.predict(x_test)
			# calculate numeric metrics
			pred = np.stack((y_pred[0,:,:,0],y_pred[0,:,:,1],y_pred[0,:,:,2],y_pred[0,:,:,3]),axis=2)
			coeffs_pred = y_pred[0,:,:,0],(y_pred[0,:,:,1],y_pred[0,:,:,2],y_pred[0,:,:,3])
			img_out = pywt.idwt2(coeffs_pred, 'haar')
			# calculate numeric metrics
			img_out = np.clip(img_out, 0, 1)
			filename = file    # get the name of image file
			name.append(filename)
			img_out = Image.fromarray((img_out*255).astype('uint8')) 
			img_out.save(out_dir + filename)

			print('Saving result to '+out_dir + filename)

		print('Test Over')
