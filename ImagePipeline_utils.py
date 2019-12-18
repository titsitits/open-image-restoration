import time
import glob
import numpy as np
import os, sys, shutil
from contextlib import contextmanager
from numba import cuda as ncuda
import PIL
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2
import contextlib
from copy import deepcopy
import subprocess

utilspath = os.path.join(os.getcwd(), 'utils/')

@contextmanager
def timing(description: str) -> None:
  
	start = time.time()
	
	yield
	elapsed_time = time.time() - start

	print( description + ': finished in ' + f"{elapsed_time:.4f}" + ' s' )

@contextmanager
def suppress_stdout(raising = False):
	
	with open(os.devnull, "w") as devnull:
		error_raised = False
		error = "there was an error"
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:  
			yield
		except Exception as e:
			error_raised = True  
			error = e
			sys.stdout = old_stdout
			print(e)
		finally:
			finished = True
			sys.stdout = old_stdout
			
	sys.stdout = old_stdout		 
	if error_raised:
		if raising:
			raise(error)
		else:
			print(error)
			
global old_stdout
old_stdout = sys.stdout

#Mute stdout inside this context
@contextmanager
def quiet_and_timeit(description = "Process running", raising = False, quiet = True):
	
	global old_stdout
	old_stdout = sys.stdout
	print(description+"...", end = '')
	start = time.time()
	try:
		
		if quiet:
			#with suppress_stdout(raising):	
			sys.stdout = open(os.devnull, "w")
		yield
		if quiet:
			sys.stdout.close()
			sys.stdout = old_stdout
	except Exception as e:
		if quiet:
			sys.stdout.close()
			sys.stdout = old_stdout
		if raising:
			raise(e)
		else:
			print(e)
			
	elapsed_time = time.time() - start
	print(': finished in ' + f"{elapsed_time:.4f}" + ' s' )

#Force printing in stdout, regardless of the context (such as the one defined above)	
def force_print(value):
	prev_stdout = sys.stdout
	global old_stdout
	sys.stdout = old_stdout
	print(value)
	sys.stdout = prev_stdout



def duplicatedir(src,dst):

	if os.path.exists(dst):
		shutil.rmtree(dst)
		
	shutil.copytree(src=src,dst=dst) 

def createdir_ifnotexists(directory):
	#create directory, recursively if needed, and do nothing if directory already exists
	os.makedirs(directory, exist_ok=True)

def initdir(directory):

	if os.path.exists(directory):
		shutil.rmtree(directory)   
	os.makedirs(directory)
			
def to_RGB(image):
	return image.convert('RGB')

def to_grayscale(image):	
	return image.convert('L')

def split_RGB_images(input_dir):
	
	imname = '*'
	orignames = glob.glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)			

			#remove alpha component
			im = to_RGB(im)
			
			#split channels
			r, g, b = Image.Image.split(im)
			r = to_RGB(r)
			g = to_RGB(g)
			b = to_RGB(b)


			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			
			r.save(f+"_red.png")
			g.save(f+"_green.png")
			b.save(f+"_blue.png")
			
		except Exception as e:
			print(e)	

def unsplit_RGB_images(input_dir):
	
	imname = '*_red.png'
	orignames = glob.glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			substring = orig[:-8]
			r = to_grayscale(Image.open(substring+'_red.png'))
			g = to_grayscale(Image.open(substring+'_green.png'))
			b = to_grayscale(Image.open(substring+'_blue.png'))
			
			im = Image.merge('RGB', (r,g,b) )
			
			#save as png (and remove monochannel images)
			os.remove(substring+'_red.png')
			os.remove(substring+'_green.png')
			os.remove(substring+'_blue.png')
			
			im.save(substring+".png")
			
		except Exception as e:
			print(e)			
			
	
			
def preprocess(input_dir, gray = True, resize = True, size = (1000,1000)):

	imname = '*'
	orignames = glob.glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = to_RGB(im)

			#convert to grayscale
			if gray:
				im = to_grayscale(im)

			#resize
			if resize:

				width, height = im.size

				#resize only if larger than limit
				if width > size[0] or height > size[1]:
					im.thumbnail(size,Image.ANTIALIAS)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)

def filtering(input_dir, median = True, median_winsize = 5, mean = True, mean_winsize = 5):

	with timing("Filtering (median) with PIL (consider using filtering_opencv for faster processing)"):
		imname = '*'
		orignames = glob.glob(os.path.join(input_dir, imname))

		for orig in orignames:

			try:
				im = Image.open(orig)

				
				#median blur
				if median:
					im = im.filter(ImageFilter.MedianFilter(median_winsize))  
					
				#mean blur
				if mean:
					im = im.filter(ImageFilter.Meanfilter(mean_winsize))				 

				#save as png (and remove previous version)
				f, e = os.path.splitext(orig)
				os.remove(orig)
				im.save(f+".png")
			except Exception as e:
				print(e)

def filtering_opencv(input_dir, median = True, median_winsize = 5, gaussian = True, gaussian_x = 5, gaussian_y = 5, gaussian_std = 0, mean = True, mean_winsize = 3):

	with timing("Filtering (median) with opencv"):
		imname = '*'
		orignames = glob.glob(os.path.join(input_dir, imname))

		for orig in orignames:
			print(orig)
			try:
				im = cv2.imread(orig, cv2.IMREAD_COLOR)


				#median blur
				if median:
					im = cv2.medianBlur(im,median_winsize)	 
					
				if gaussian:
					im = cv2.GaussianBlur(im,(gaussian_x,gaussian_y),gaussian_std)

				#mean blur
				if mean:
					im = cv2.blur(im,(mean_winsize,mean_winsize))
					
				

				#save as png (and remove previous version)
				f, e = os.path.splitext(orig)
				os.remove(orig)
				cv2.imwrite(f+".png", im)
			except Exception as e:
				print(e)
	
			
def rotate_images(input_dir):

	imname = '*'
	orignames = glob.glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = im.transpose(Image.ROTATE_90)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)

def unrotate_images(input_dir):

	imname = '*'
	orignames = glob.glob(os.path.join(input_dir, imname))
	
	for orig in orignames:
		
		try:
			im = Image.open(orig)

			#remove alpha component
			im = im.transpose(Image.ROTATE_270)

			#save as png (and remove previous version)
			f, e = os.path.splitext(orig)
			os.remove(orig)
			im.save(f+".png")
		except Exception as e:
			print(e)			
			
def reset_gpu(device = 0):  
	
	ncuda.select_device(device)
	ncuda.close()
	
import os, time, datetime
#import PIL.Image as Image
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave

def to_tensor(img):
	if img.ndim == 2:
		return img[np.newaxis,...,np.newaxis]
	elif img.ndim == 3:
		return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
	return np.squeeze(np.moveaxis(img[...,0],0,-1))

def save_result(result,path):
	path = path if path.find('.') != -1 else path+'.png'
	ext = os.path.splitext(path)[-1]
	if ext in ('.txt','.dlm'):
		np.savetxt(path,result,fmt='%2.4f')
	else:
		imsave(path,np.clip(result,0,1))

fontfile = os.path.join(utilspath,"arial.ttf")

def addnoise(im, sigma = 10, imagetype = 'L', add_label = False):
	
	x = np.array(im)
	y = x + np.random.normal(0, sigma, x.shape)
	y=np.clip(y, 0, 255)
	
	im = PIL.Image.fromarray(y.astype('uint8'), imagetype)

	if add_label:
		d = ImageDraw.Draw(im)
		fnt = ImageFont.truetype(fontfile, 40)
		if imagetype == 'L':
			fill = 240
		elif imagetype == 'RGB':
			fill = (255, 0, 0)
		elif imagetype == 'RGBA':
			fill = (255,0,0,0)
		d.text((10,10), "sigma = %s" % sigma, font = fnt, fill = fill)

	return im


utilspath = os.path.join(os.getcwd(), 'utils/')

fontfile = os.path.join(utilspath,"arial.ttf")

def concat_images(img_list, labels = [], imagetype = None, samesize = True, labelsize = 30, labelpos = (10,10), labelcolor = None):

	images = deepcopy(img_list)
	if imagetype == None:

		imagetype = images[0].mode
		
	widths, heights = zip(*(i.size for i in images))

	#if images have various heights, put height to the smallest one
	if (len(set(heights)) > 1) & samesize:
			for im in images:
				size = (im.width, min(heights))
				im.thumbnail(size,PIL.Image.ANTIALIAS)

	widths, heights = zip(*(i.size for i in images))
	total_width = sum(widths)
	max_height = max(heights)

	new_im = PIL.Image.new(imagetype, (total_width, max_height))

	#add labels to images
	if len(labels) == len(images):

		fnt = ImageFont.truetype(fontfile, labelsize)
		if imagetype == 'L':
			fill = 240
		elif imagetype == 'RGB':
			fill = (176,196,222)
		elif imagetype == 'RGBA':
			fill = (176,196,222,0)

		if labelcolor is not None:
			fill = labelcolor

		for i in range(len(labels)):
			d = ImageDraw.Draw(images[i]).text(labelpos, labels[i], font = fnt, fill = fill)

	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset,0))
		x_offset += im.size[0]
	
	return new_im

def display_images(im_list, labels = [], **kwargs):
	display(concat_images(im_list, labels, **kwargs))

def get_filepaths(directory):
	files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
	return files

def get_filenames(directory):
	files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
	return files

def display_folder(directory, limit = 10):
	
	files = get_filepaths(directory)
	if len(files) > limit:
		files = files[:limit]
	
	display_images([PIL.Image.open(f) for f in files], [os.path.split(f)[1] for f in files])
	
def clone_git(url, dir_name = None, tag = None, reclone = False):

	"""	
	url: url of the git repository to clone
	dir_name: name of the folder to give to the repository. If not given, the git repository name is used
	tag: allows to checkout a specific commit if given
	reclone: overwrite existing repo
	"""
	
	old_dir = os.getcwd()
	
	if dir_name is None:		
		dir_name = os.path.split(url)[1] #use git repo name
		dir_name = os.path.splitext(dir_name)[0] #remove ".git" if present
	
	if reclone and os.path.exists(dir_name):
		shutil.rmtree(dir_name)
		
	if not os.path.exists(dir_name):
		command = "git clone %s %s" % (url, dir_name)
		subprocess.run(command, shell = True)
		
	os.chdir(dir_name)
	
	if tag is not None:
		command = "git checkout %s" % tag
		subprocess.run(command, shell = True)
	
	git_path = os.path.join(os.getcwd())
	
	os.chdir(old_dir)
	
	return git_path

def download_gdrive(file_id):
	
	subprocess.run("wget https://raw.githubusercontent.com/GitHub30/gdrive.sh/master/gdrive.sh", shell = True)
	subprocess.run("curl gdrive.sh | bash -s %s" % file_id, shell = True)
	subprocess.run("rm gdrive.sh", shell = True)