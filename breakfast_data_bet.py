# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from time import clock

from camera_model import Camera
from depth_bet import depth_bet


#%% Cameras

far_cam = Camera(focal_length = 25.0,
                 f_stop = 8.3,
                 pixel_pitch = 0.025,
                 focus = 869) 
near_cam = Camera(focal_length = 25.0,
                 f_stop = 8.3,
                 pixel_pitch = 0.025,
                 focus = 529) 


#%% Input

folder = "data/breakfast/"
far_im = imread(folder + "breakfast_far.png").mean(2)/256
near_im = imread(folder + "breakfast_near.png").mean(2)/256

plt.figure("Input images")
plt.subplot(1,2,1)
plt.title('Far')
plt.imshow(far_im)
plt.subplot(1,2,2)
plt.title('Near')
plt.imshow(near_im)


#%% Estimation

dists =np.linspace(529, 869, 10)
far_psfs = [far_cam.psf(d) for d in dists]
near_psfs = [near_cam.psf(d) for d in dists]
est_time = -clock()
emap = depth_bet(far_im, near_im, far_psfs, near_psfs, 1)
est_time += clock()
print("Estimation time:", est_time)


rdmap = np.argmin(emap,2)
plt.figure("Raw Depth Map")
plt.imshow(rdmap)
plt.colorbar()

