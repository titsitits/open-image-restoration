import time
from PIL import Image
from glob import glob
import numpy as np
import os, sys
from contextlib import contextmanager
from numba import cuda as ncuda
from PIL import ImageFilter
import cv2

fast_denoising = False
import time
import ImagePipeline_utils as IP
from ImagePipeline_utils import Quiet
import os, shutil
import warnings
from os import path as osp
import subprocess
from contextlib import contextmanager
import time, sys


class ImageRestorer:

	def __init__(self, resetgpu = True):
		
		self._history = []
		self._resetgpu = resetgpu
		#record current interpreter path (to ensure calling the right interpreter when calling another process)
		self._python_dir = sys.executable
		self.denoise = self.remove_gaussian_noise
		self.Q = Quiet()
		
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
		
		with self.Q.quiet_and_timeit("Image preprocessing", raising = self.raising, quiet = self.quiet):
			imname = '*'
			orignames = glob(os.path.join(inputdir, imname))

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
					self.Q.force_print(e)

					
	def filter(self, inputdir = None, outputdir = None, **kwargs):		
		
		"""
		Perform basic filtering (median, gaussian and/or mean filtering) on images
		Defaults options (you can override any option with a keyword argument): 
		options = {'median':True, 'median_winsize':5, 'gaussian':True, 'gaussian_x':5, 'gaussian_y':5, 'gaussian_std':1, 'mean':True, 'mean_winsize':3, 'raising':True, 'quiet':True}
		"""
		
		options = {'median':True, 'median_winsize':5, 'gaussian':True, 'gaussian_x':5, 'gaussian_y':5, 'gaussian_std':1, 'mean':True, 'mean_winsize':3, 'raising':True, 'quiet':True}
		
		inputdir, outputdir = self._init_process(inputdir, outputdir, "filter", options, **kwargs)
		
		with self.Q.quiet_and_timeit("Image filtering", self.raising, self.quiet):
			imname = '*'
			orignames = glob(os.path.join(inputdir, imname))

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
					self.Q.force_print(e)

					
	def remove_stripes(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Remove vertical and horizontal stripes from images
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./WDNN', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		"""
		
		options = {'working_dir':'./WDNN', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		
		inputdir, outputdir = self._init_process(inputdir, outputdir, "remove_stripes", options, **kwargs)
		
		command = "%s -W ignore -u striperemover.py -i %s -o %s %s %s" % (self.python_dir, inputdir, outputdir, self.process_args, self.command_suffix)	
		
		#Remove vertical stripes
		with self.Q.quiet_and_timeit("Removing stripes", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)
			
		if self.raising:
			self.logerr()		
						
		if not self.quiet:
			for l in self.log():
				print(l)			

		
		
	def remove_gaussian_noise(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Remove gaussian noise using NLRN (or DNCNN if fast is True).
		Defaults options (you can override any option with a keyword argument): 
		options = {'fast':False, 'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.		
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'fast':False, 'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		#Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python
		
		inputdir, outputdir = self._init_process(inputdir, outputdir, "remove_gaussian_noise", options, **kwargs)
		
		command = "%s -W ignore -u denoiser.py -i %s -o %s %s %s" % (self.python_dir, inputdir, outputdir, self.process_args, self.command_suffix)

		if self.fast:
			#IP.reset_gpu(0)
			command = '%s -W ignore -u denoiser_NLRN_DNCNN.py -i %s -o %s --method DNCNN %s %s' % (self.python_dir, inputdir, outputdir, self.process_args, self.command_suffix)
		
		#print("raising", self.raising)
		with self.Q.quiet_and_timeit("Removing gaussian noise", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)
			
		if self.raising:
			self.logerr()		
						
		if not self.quiet:
			for l in self.log():
				print(l)
		
		#if self.fast:
			#IP.reset_gpu(0)
			
			
	def colorize(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Colorize images using deoldify.
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		inputdir, outputdir = self._init_process(inputdir, outputdir, "colorize", options, **kwargs)		
				
		command = "%s -W ignore -u colorizer.py -i %s -o %s %s %s" % (self.python_dir, inputdir, outputdir, self.process_args, self.command_suffix)
		
		with self.Q.quiet_and_timeit("Colorizing", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)

		if self.raising:
			self.logerr()
			
		if not self.quiet:
			for l in self.log():
				print(l)


	def super_resolution(self, inputdir = None, outputdir = None, **kwargs):
		
		"""
		Upsample images using ESRGAN.
		Defaults options (you can override any option with a keyword argument): 
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		Note that we use a command suffix to export console outputs to a log file. If you remove this, you could have bugs in jupyter notebooks. It should work in standard python.
		You can monitor log.out output by using the command "tail -f log.out" in a terminal.
		You can launch the process with a specific python environment by providing its path with the keyword argument "python_dir".
		"""
		
		#defaults attributes to instance for method
		options = {'working_dir':'./', 'raising':True, 'quiet':True, 'python_dir':sys.executable, 'process_args':'', 'command_suffix':" 2> log.err 1>> log.out"}
		inputdir, outputdir = self._init_process(inputdir, outputdir, "super_resolution", options, **kwargs)		

		command = "%s -W ignore -u superresolution.py -i %s -o %s %s %s" % (self.python_dir, inputdir, outputdir, self.process_args, self.command_suffix)
		
		with self.Q.quiet_and_timeit("Super-resolving", self.raising, self.quiet):
			IP.createdir_ifnotexists(outputdir)
			subprocess.run(command, shell = True)
		
		if self.raising:
			self.logerr()
			
		if not self.quiet:
			for l in self.log():
				print(l)


				
	def merge(self, inputdirs, outputdir, **kwargs):
		
		"""		
		merge (compute average) of folders parwise images
		inputdirs: list of input directories
		Defaults options (you can override any option with a keyword argument): 
		options = {'weights':[1/len(inputdirs)]*len(inputdirs), 'raising':True, 'quiet':True}
		"""
		
		
		options = {'weights':[1/len(inputdirs)]*len(inputdirs), 'raising':True, 'quiet':True}
		
		inputdirs, outputdir = self._init_process(inputdirs, outputdir, "merge", options, **kwargs)
		
		with self.Q.quiet_and_timeit("Image merging", self.raising, self.quiet):
			
			names = IP.get_filenames(inputdirs[0])
			
			for n in names:
				
				try:
					files = [os.path.join(i,n) for i in inputdirs]

					merged = IP.image_average(files,self.weights)

					merged.save(os.path.join(outputdir,n))
					
				except Exception as e:
					
					self.Q.force_print(e)
	
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
			if not ( (type(inputdir) is list) and (outputdir in inputdir) ):
				#initialize only if diffrent from inputdir
				IP.initdir(outputdir)
		
		
		options = default_options
		
		#override default parameters for the process, and unpack all arguments as class attributes
		options.update(kwargs)
		#create/update (unpack) class attributes with options
		for key in options: 
			self.__setattr__(key, options[key])
		
		
		self._history.append({"input":inputdir,"output":outputdir,"process":process,"options" : options})		
		
		IP.createdir_ifnotexists(outputdir)
		
		if self._resetgpu:
			IP.reset_gpu(0)
		
		return inputdir, outputdir


	def display(self, **kwargs):
		
		if len(self._history) == 0:
			print("You did not perform any process yet. No image folder to display.")
			return
		
		last_folder = self._history[-1]["output"]
		IP.display_folder(last_folder, **kwargs)
	
	def log(self, lines = 10):
		
		logdata = []
		
		if os.path.exists('log.out'):
		
			with open('log.out', 'r') as myfile:
				logdata = myfile.readlines()
				
		if os.path.exists('log.err'):
		
			with open('log.err', 'r') as myfile:
				logdataerr = myfile.readlines()	
		
			logdata = logdata+logdataerr

		if len(logdata) > 0:
			if len(logdata) < lines:
				return logdata

			return logdata[-lines:]
		
		else:
			logdata = 'No log.out file. Using restorer history instead.\n\n %s' % (self._history)
			
			return logdata
	
	
	def history(self):
		
		return self._history
	
	
	def logerr(self, raising = False):
		
		if os.path.exists('log.err'):
			with open('log.err') as f:
				logdata = f.readlines()
				if len(logdata) > 0:						
					print('Error or warning occured during process. Please check output below.')
					for l in logdata:
						if raising:
							raise Exception(l)
						else:
							print(l)
		
		return logdata
	