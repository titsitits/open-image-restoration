# An Open-source framework for image restoration
This project gathers together and packages various image restoration techniques that follow various criteria:
* State-of-the-art (they are all based on Deep Learning)
* Open source (MIT or Apache licenses)
* Usable (a pretrained model is available)
* Python implementation (easier to 

## Demo
The project is a work in progress. However, it is already functional and can be tested through this [demo in google Colab](https://colab.research.google.com/github/titsitits/open-image-restoration/blob/master/ImageRestorationColab.ipynb).

## Technical details
The algorithms currently included in the packages are directly replicated or slightly adapted from external github repositories (see below). These methods were selected based on the above criteria, and after a comparison with other methods (comparison colab notebooks are coming soon).

### 1. Denoising
NLRN - Liu et al. 2018. Non-Local Recurrent Network for Image Restoration (NeurIPS 2018) - MIT License
https://github.com/Ding-Liu/NLRN

### 2. Colorization
DeOldify ("NoGAN" algorithm) - Jason Antic, 2019 - MIT License
https://github.com/jantic/DeOldify

### 3. Super-resolution
ESRGAN - Wang et al. 2018. ESRGAN: Enhanced super-resolution generative adversarial networks (ECCV 2018) - Apache-2.0 license
https://github.com/xinntao/ESRGAN

Many thanks to the authors of these awesome works and for sharing them as open-source projects. 

