import time
from PIL import Image
import glob
import numpy as np
import os
from contextlib import contextmanager
from numba import cuda as ncuda
import sys

@contextmanager
def timing(description: str) -> None:
  
    start = time.time()
    
    yield
    elapsed_time = time.time() - start

    print( description + ': finished in ' + f"{elapsed_time:.4f}" + ' s' )

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        except Exception as e:
            sys.stdout = old_stdout
            print(e)
        finally:
            sys.stdout = old_stdout 
    
def to_RGB(image):
    return image.convert('RGB')

def to_grayscale(image):    
    return image.convert('L')

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