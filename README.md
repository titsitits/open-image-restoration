# Open-Image-Restoration Toolkit
## A selection of State-ot-the-art, Open-source, Usable, and Pythonic techniques for Image Restoration

<p align="center">
<img src="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/Anne-Franck.png" width="800" />
</p>

<!--p align="center">
<img style="height: 100%; width: 100%; object-fit: contain" src="https://github.com/titsitits/open-image-restoration/blob/master/Anne-Franck.png" />
</p-->

## Description
This project gathers together and packages various image restoration techniques that follow various criteria:
* State-of-the-art (they are all based on Deep Learning; as of today (25/07/2019 at time of writing), NLRN and ESRGAN are leaders in various leaderboards maintained by paperswithcode.com, see [here](https://paperswithcode.com/paper/non-local-recurrent-network-for-image) and [here](https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative) ).
* Open source (the selected implementations are under MIT or Apache licenses)
* Usable (a pretrained model is available, and the code does not need painfull<sup>1</sup> dependencies)
* Python implementation (easier to use together, to share, and especially to use in Google Colab).

## Demo
The project is a work in progress. However, it is already functional and **can be tested on your own images** through this **[demo in Google Colab](https://colab.research.google.com/github/titsitits/open-image-restoration/blob/master/Colab_Notebooks/Demo_Image_Restoration.ipynb)**.

## Technical details
The algorithms currently included in the packages are directly replicated or slightly adapted from external github repositories (see below). These methods were selected based on the above criteria, and after a comparison with other methods (comparison colab notebooks are coming soon). 

### 1. Denoising (grain removal)


[NLRN](https://github.com/Ding-Liu/NLRN) - Liu et al. 2018. Non-Local Recurrent Network for Image Restoration (NeurIPS 2018) - MIT License

[Paperswithcode.com ranking](https://paperswithcode.com/paper/non-local-recurrent-network-for-image)

<hr>
<p style="text-align:center;">
<a href="https://titsitits.github.io/image_restoration/images/denoising_annefranck.png"><img src="https://titsitits.github.io/image_restoration/images/denoising_annefranck.png" style="width:100%; height:auto;"/></a>
<br>
Denoising methods comparison
</p>
<hr>

### 2. Moire removal (stripe noise removal)

[WDNN](https://github.com/jtguan/Wavelet-Deep-Neural-Network-for-Stripe-Noise-Removal) - Guan et al. 2019. Wavelet Deep Neural Network for Stripe Noise Removal. (IEEE Access, 7, 2019) - Apache-2.0 License

(no leaderboard associated... yet).

The original algorithm has been adapted to allow removal of both horizontal and vertical stripes. Note that in some cases, multiple steps of this technique can provide better results (see image below).

<hr>
<p style="text-align:center;">
<a href="https://titsitits.github.io/image_restoration/images/robaeys_stripe_removal.png"><img src="https://titsitits.github.io/image_restoration/images/robaeys_stripe_removal.png" style="width:100%; height:auto;"/></a>
<br>
Multiple steps of stripe removal with adapted WDNN (image credits: Jean Vandendries. Reuse of this image is prohibited).
</p>
<hr>

### 3. Colorization
[DeOldify](https://github.com/jantic/DeOldify) ("NoGAN" algorithm) - Jason Antic, 2019 - MIT License

(no leaderboard associated... yet).

<hr>
<p style="text-align:center;">
<a href="https://titsitits.github.io/image_restoration/images/colorization_goffaux.png"><img src="https://titsitits.github.io/image_restoration/images/colorization_goffaux.png" style="width:100%; height:auto;"/></a>
<br>
Comparison of image colorization methods (image credits: Jean Vandendries. Reuse of this image is prohibited).
</p>
<hr>

### 4. Super-resolution
[ESRGAN](https://github.com/xinntao/ESRGAN) - Wang et al. 2018. ESRGAN: Enhanced super-resolution generative adversarial networks (ECCV 2018) - Apache-2.0 License

[Paperswithcode.com ranking](https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative)

<hr>
<p style="text-align:center;">
<a href="https://titsitits.github.io/image_restoration/images/superres_anne_franck.png"><img src="https://titsitits.github.io/image_restoration/images/superres_anne_franck.png" style="width:100%; height:auto;"/></a>
<br>
Comparison of image super-resolution methods.
</p>
<hr>

Many thanks to the authors of these awesome contributions to image restoration research, and for sharing them as open-source projects. 

## Requirements/Installation

(more detailed info coming soon)

You need a CUDA-compatible GPU.

Tested with:
* Python 3.*
* CUDA
* Tensorflow
* Pytorch >= 1.1
* Torchvision >= 0.3.0


In a Google Colab notebook, you can use this:

```python
import os
from os.path import *
basedir = "/content"

# Import library
repodir = join(basedir,"open-image-restoration")
if not exists(repodir):
  os.chdir(basedir)
  !git clone https://github.com/titsitits/open-image-restoration {repodir}

os.chdir(repodir)

#Todo: create a pip module
#Specific dependencies needed for colorization
!pip install --quiet -r requirements.txt

import ImagePipeline_utils as IP
import ImageRestorer
restorer = ImageRestorer.ImageRestorer()
```

## Usage

(more detailed info to come)

```python
inputdir = "path/to/your/images"
outputdir = "path/to/output" #(can be inputdir)

#Todo: make restorer available from outside repo directory
os.chdir(repodir)

# Restore images
restorer.preprocess(inputdir, outputdir, gray=True) #resize if needed (by default limit to 1000x1000px), convert to 1-channel image
restorer.remove_stripes() #reduce image moire (WDNN)
restorer.denoise() #remove image grain (NLRN)
restorer.remove_stripes(process_args="-n 2") #reduce remaining image moire (n iterations)
restorer.colorize() #colorize image (DeOldify) (first time takes a long time as large models must be downloaded)
restorer.super_resolution() #upsample image (ESRGAN)

#Compare input and output folders
IP.compare_folders([inputdir, outputdir])

```

## Output examples (Christmas Truce, 1914)

These results can be reproduced in this notebook in Google Colab: 
https://github.com/titsitits/open-image-restoration/blob/master/Colab_Notebooks/Christmas_Truce_Restoration.ipynb

<hr>
<p style="text-align:center;">
<a href="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration1.png"><img src="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration1.png" style="width:100%; height:auto;"/></a>
  <a href="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration2.png"><img src="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration2.png" style="width:100%; height:auto;"/></a>
  <a href="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration3.png"><img src="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration3.png" style="width:100%; height:auto;"/></a>
  <a href="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration4.png"><img src="https://raw.githubusercontent.com/titsitits/open-image-restoration/master/blog%20image%20restoration/restoration4.png" style="width:100%; height:auto;"/></a>
<br>
Examples of old image restoration (Christmas Truce, 1914).
</p>
<hr>


## More information

A more detailed comparison of State-of-the-art super-resolution algorithms can be found in this **[Google Colab Notebook](https://colab.research.google.com/drive/1x7wHaiJ-_rPfRqz1DcRJ_Hbq61t6lFKi)**.
A blog post (in french) presents an overview on the subject [here](https://titsitits.github.io/super_resolution/)

---

<sup>1</sup>: In my research of IR algorithms, I subjectively considered Matlab and Caffe as painfull dependencies. Matlab is simply not free (and simpy not usable in Google colab), and Caffe hard to install, especially in Google Colab. Both of these issues make the algorithms hard to share to the community.
