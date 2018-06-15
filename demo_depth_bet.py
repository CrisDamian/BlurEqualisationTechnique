# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from time import clock

from camera_model import Camera
from depth_bet import depth_bet


#%% Two cameras with are taken with different focus.

fst_cam = Camera(focal_length = 50.0,
                 f_stop = 1.8,
                 pixel_pitch = 4.65e-3,
                 focus = 670) 
snd_cam = Camera(focal_length = 50.0,
                 f_stop = 1.8,
                 pixel_pitch = 4.65e-3,
                 focus = 710)


#%% A synthetic scene is generated.

x = np.linspace(-1,1,300)
x, y = np.meshgrid(x,x)
I1 = np.where((x**2 + y**2) < 0.75**2  ,0.75,0.25)
I1 = I1 +  0.1 * np.random.rand(*I1.shape)

plt.figure('Texture')
plt.imshow(I1)
plt.axis('off')


#%% Simulates acquisition.

diams = np.array([13,15,17,19]); # Diametri PSF  
dists = fst_cam.distance(diams*fst_cam.pitch);                 
dists

def gen_acq(the_cam):
    I = np.empty((I1.shape[0],I1.shape[1]*len(dists)))
    for i, d in enumerate(dists):
        V = ndi.convolve(I1,the_cam.psf(d))
        I[:,I1.shape[1]*i:I1.shape[1]*(i+1)] = V
    return I

fst_im = gen_acq(fst_cam);
snd_im = gen_acq(snd_cam);

plt.figure('Scene')
plt.subplot(2,1,1)
plt.imshow(fst_im)
plt.subplot(2,1,2)
plt.imshow(snd_im)


#%% BET depth technique

fst_psfs = [fst_cam.psf(d) for d in dists]
snd_psfs = [snd_cam.psf(d) for d in dists]
est_time = -clock()
emap = depth_bet(fst_im, snd_im, fst_psfs, snd_psfs, 0)
est_time += clock()
print("Estimation time:", est_time)


rdmap = np.argmin(emap,2)
plt.figure("Noiseless Depth Map")
plt.imshow(rdmap, cmap='rainbow')


#%% Noisy verio

fst_imz = fst_im + np.random.normal(0,1e-3, size=fst_im.shape)
snd_imz = snd_im + np.random.normal(0,1e-3, size=snd_im.shape)


# In[9]:

est_time = -clock()
emapz = depth_bet(fst_imz, snd_imz, fst_psfs, snd_psfs, 0)
est_time += clock()
print("Estimation time:", est_time)


# In[10]:

rdmapz = np.argmin(emapz,2)
plt.figure("Depth Map")
plt.imshow(rdmapz, cmap='rainbow')



