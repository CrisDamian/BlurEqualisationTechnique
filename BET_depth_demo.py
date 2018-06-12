
# coding: utf-8

# Blur Equalisation Technique
# ============================

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy import ndimage as ndi
from time import clock

from camera_model import Camera
from depth_bet import depth_bet


# Two images are taken with different focus settings.

# In[1]:

fst_cam = Camera(focal_length = 50.0,
                 f_stop = 1.8,
                 pixel_pitch = 4.65e-3,
                 focus = 670) 
snd_cam = Camera(focal_length = 50.0,
                 f_stop = 1.8,
                 pixel_pitch = 4.65e-3,
                 focus = 710)


# A synthetic scene is generated.

# In[3]:

x = np.linspace(-1,1,300)
x, y = np.meshgrid(x,x)
I1 = np.where((x**2 + y**2) < 0.75**2  ,0.75,0.25)
I1 = I1 +  0.1 * np.random.rand(*I1.shape)

plt.matshow(I1)
plt.axis('off')


# Simulates acquisition.

# In[4]:

diams = np.array([13,15,17,19]); # Diametri PSF  
dists = fst_cam.distance(diams*fst_cam.pitch);                 
dists


# In[5]:


def gen_acq(the_cam):
    I = np.empty((I1.shape[0],I1.shape[1]*len(dists)))
    for i, d in enumerate(dists):
        V = ndi.convolve(I1,the_cam.psf(d))
        I[:,I1.shape[1]*i:I1.shape[1]*(i+1)] = V
    return I

fst_im = gen_acq(fst_cam);
snd_im = gen_acq(snd_cam);

plt.figure()
plt.subplot(2,1,1)
plt.imshow(fst_im)
plt.subplot(2,1,2)
plt.imshow(snd_im)


# Aplicăm estimarea de adînicime.

# In[6]:

fst_psfs = [fst_cam.psf(d) for d in dists]
snd_psfs = [snd_cam.psf(d) for d in dists]
est_time = -clock()
emap = depth_bet(fst_im, snd_im, fst_psfs, snd_psfs, 0)
est_time += clock()
print("Estimation time:", est_time)


# In[7]:

rdmap = np.argmin(emap,2)
plt.figure("Raw Depth Map")
plt.imshow(rdmap, cmap='rainbow')


# Adăugăm zgomot.

# In[8]:

fst_imz = fst_im + np.random.normal(0,1e-3,size=fst_im.shape)
snd_imz = snd_im + np.random.normal(0,1e-3,size=snd_im.shape)


# In[9]:

est_time = -clock()
emapz = depth_bet(fst_imz, snd_imz, fst_psfs, snd_psfs, 0)
est_time += clock()
print("Estimation time:", est_time)


# In[10]:

rdmapz = np.argmin(emapz,2)
plt.figure("Raw Depth Map")
plt.imshow(rdmapz, cmap='rainbow')


# Real images
# ===========

# In[11]:

fst_cam = Camera(focal_length = 25.0,
                 f_stop = 8.3,
                 pixel_pitch = 0.025,
                 focus = 869) # Focalizarea este diferită.
snd_cam = Camera(focal_length = 25.0,
                 f_stop = 8.3,
                 pixel_pitch = 0.025,
                 focus = 529) # Focalizarea este diferită.


# In[12]:

from imageio import imread
folder = "data/"
fst_im = imread(folder + "breakfast_far.png")[:,:,1]/255
snd_im = imread(folder + "breakfast_near.png")[:,:,1]/255


# In[13]:

plt.figure()
plt.subplot(1,2,1)
plt.imshow(fst_im)
plt.subplot(1,2,2)
plt.imshow(snd_im)


# In[14]:

dists =np.linspace(529, 869, 10)
fst_psfs = [fst_cam.psf(d) for d in dists]
snd_psfs = [snd_cam.psf(d) for d in dists]
est_time = -clock()
emap = depth_bet(fst_im, snd_im, fst_psfs, snd_psfs, 1)
est_time += clock()
print("Estimation time:", est_time)


# In[15]:

rdmap = np.argmin(emap,2)
plt.figure("Raw Depth Map")
plt.imshow(rdmap)
plt.colorbar()


# In[ ]:



