# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from time import clock

from camera_model import Camera
from depth_bet import depth_bet


#%% Parameters

# Patio

near_cam = Camera(focal_length = 2.5,
                 f_stop = 2.5/10,
                 pixel_pitch = 5.56e-3,
                 focus = 584) 
far_cam = Camera(focal_length = 2.5,
                 f_stop = 2.5/10,
                 pixel_pitch = 5.56e-3,
                 focus = 970)

min_depth = 200
max_depth = 1500

folder = r"data/patio/patio_stereo2/"  
near_file = "patio_stereo2_near2.png"
far_file = "patio_stereo2_far2.png"
all_file = "patio_stereo2_all.png"
depth_file = "patio_stereo2.depth.npy"


# Office

#near_cam = Camera(focal_length = 22,
#                 f_stop = 22/0.40,
#                 pixel_pitch = 0.0028121,
#                 focus = 267) 
#far_cam = Camera(focal_length = 22,
#                 f_stop = 22/0.40,
#                 pixel_pitch = 0.0028121,
#                 focus = 534)
#
#min_depth = 150
#max_depth = 850
#
#folder = r"data/office/office_stereo2/"  
#near_file = "office_stereo2_near2.png"
#far_file = "office_stereo2_far2.png"
#all_file = "office_stereo2_all.png"
#depth_file = "office_stereo2.depth.npy"



#%% Input Data

near_im = imread(folder + near_file).mean(2)/256
far_im = imread(folder + far_file).mean(2)/256
all_im = imread(folder + all_file).mean(2)/256
true_depth = np.load(folder + depth_file);

plt.figure("Input Images")
plt.subplot(2,2,1)
plt.title('Near')
plt.imshow(near_im)
plt.subplot(2,2,2)
plt.title('Far')
plt.imshow(far_im)
plt.subplot(2,2,3)
plt.title('All')
plt.imshow(all_im)
plt.subplot(2,2,4)
plt.title('Depth')
plt.imshow(true_depth, 
           vmin=  np.percentile(true_depth, 1), 
           vmax=  np.percentile(true_depth, 99))

#%%  Depth map

n_levels = 16
true_map =  np.rint( n_levels*(true_depth-min_depth)/(max_depth-min_depth)).astype('uint');
np.clip(true_map, 0, n_levels-1, out=true_map)
dists = np.arange(n_levels)/n_levels*(max_depth - min_depth) + min_depth

plt.figure("True labeling").clear()
plt.imshow(true_map)
plt.colorbar()

#%% Performs BET 

near_dim = [near_cam.psf_diameter(d)/near_cam.pitch for d in dists]
far_dim = [far_cam.psf_diameter(d)/near_cam.pitch for d in dists]
near_psfs = [near_cam.psf(d) for d in dists]
far_psfs = [far_cam.psf(d) for d in dists]

est_time = -clock()
emap = depth_bet(near_im, far_im, near_psfs, far_psfs, 1)
est_time += clock()
print("Estimation time:", est_time)


#%% 

rdmap = np.argmin(emap,2)
plt.figure("Raw Depth Map").clear()
plt.imshow(rdmap)
plt.colorbar()


#%% Confusion matrix

h = np.zeros((n_levels, n_levels), dtype=int)
for i, j in zip(true_map.flat, rdmap.flat):
     h[i,j] += 1
    
plt.figure("Confusion for raw map").clear()
plt.imshow(h, vmin=0)
plt.colorbar()

#%%

plt.figure('Map of good estimates').clear()
plt.imshow(true_depth*(true_map == rdmap), vmax=max_depth)
plt.colorbar()
