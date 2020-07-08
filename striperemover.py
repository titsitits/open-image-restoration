#Stripes removal: adapted from https://github.com/jtguan/Wavelet-Deep-Neural-Network-for-Stripe-Noise-Removal (Apache-2.0 license)
#Original author: jtguan@stu.xidian.edu.cn Nov 25 2018
#Edit: mickael.tits@cetic.be Dec 12 2019

#################################

#imports needed for stripe removal (see below)

#reduce tensorflow verbosity
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

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
import argparse
import ImagePipeline_utils as IP



class StripeRemover:

	def __init__(self, model_path):

		self.model = self.SNRDWNN()
		self.model.load_weights(model_path)

	def SNRDWNN(self):

		#-----------------------------------------------------------------------#
		#-----------------------------------------------------------------------#
		#-----------------------------------------------------------------------#
		init = 'he_normal'
		#----------------------------------------------------------------------#

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

	def stripes_removal(self, input_dir, output_dir):

		test_dir = input_dir+'/'

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
			y_pred,noise = self.model.predict(x_test)
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


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#IP.reset_gpu(0)

	parser = argparse.ArgumentParser()

	parser.add_argument( '-i', '--input-dir', help='location to load input images', required=True)
	parser.add_argument('-o', '--output-dir', help='location to load output images', required=True)
	parser.add_argument('-n','--ntimes', help='number of iterations to pass', default=1, type=int)
	parser.add_argument('-m','--model_path', help='path to model weights', default='./WDNN/weights/weight.hdf5', type=str)

	args = parser.parse_args()

	stripe_remover = StripeRemover(args.model_path)

	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	#removed for tf2 compatibility:
	#if type(tf.contrib) != type(tf):
	#	tf.contrib._warning = None

	if args.input_dir != args.output_dir:
		IP.duplicatedir(args.input_dir, args.output_dir)

	for i in range(args.ntimes):
		#Remove vertical stripes
		stripe_remover.stripes_removal(args.output_dir, args.output_dir)
		#Rotate images
		IP.rotate_images(args.output_dir)
		#Remove horizontal stripes
		stripe_remover.stripes_removal(args.output_dir, args.output_dir)
		#Unrotate images
		IP.unrotate_images(args.output_dir)
