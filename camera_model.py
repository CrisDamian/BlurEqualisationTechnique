# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:15:30 2017

@author: DamianCristian
"""


import numpy as np
from scipy import ndimage as ndi
from scipy import signal

sigm = lambda x, w: -np.where(abs(x)<w/2, 2*x, np.sign(x))/2 + .5
scale = lambda x: (x-x.min())/np.sum(x-x.min())


def psf_mesh(dim):
    w = 2/abs(dim)
    x = np.arange(w,1+w,w)
    x = np.concatenate((-x[::-1],[0],x[::1]))
    x, y = np.meshgrid(x,x)
    return x, y, w

def pillbox_psf(dim):
    x, y, w = psf_mesh(dim)
    h = sigm((x**2 + y**2)**0.5 - 1, w)
    return h

def box_psf(dim):
    x, y, w = psf_mesh(dim)
    h = sigm(np.maximum(abs(x)+ abs(y)) - 1, w)
    return h


class Coded_psf:
    """ The class that models a coded aperture. It holds a matrix describing
    the aperture and when called returns a PSF of the given size.

    Parameters
    -----------

    prototype: array like
        A matrix holding the transmissivity profile of the aperture.

    """

    def __init__(self, prototype):
        self.mat = scale(np.array(prototype, dtype= float))
        self.s = max(self.mat.shape)

    def __call__(self, d):
        sfact = self.s//d + 1
        #w =  np.bartlett(2*sfact+1)
        w = signal.bspline(np.linspace(-2,2,4*sfact+1),3)
        fh = ndi.convolve1d(self.mat, w, axis=0, mode='constant')
        fh = ndi.convolve1d(fh, w, axis= 1, mode='constant')
        sh = ndi.zoom( fh, d/self.s, order=3, prefilter=False)
        return sh


class Camera:
    """This class holds the characteristics of the camera.
    And performs basic calculations such as computing the PSF according to distance. It uses a thin lens model for computation.

    By convention all lengths are expressed in millimeters.


    Parameters
    -----------

    focal_length: float
        The focal length of the lens.

    f_stop: float
        The ratio of focal length over the diameter of the lens.

    pixel_pitch: float
        The distance between the centers of two adjacent pixels.

    aperture:
        The shape of the aperture of the camera.

    focus:
        The distance at witch the camera is focused.


    Example
    --------
    ::

        testCam = de.Camera(focal_length = 105.0,
                      f_stop = 4.0,
                      pixel_pitch = 0.0082)
        testCam.set_focal_plane(1600)
        print('Psf diameter:', testCam.d_psf(1700))
        print('Psf')
        print(testCam.psf(1700))

    """

    def __init__(self,
                 focal_length,
                 f_stop,
                 pixel_pitch,
                 aperture=None,
                 focus=np.inf):

        self.fL = focal_length
        self.fN = f_stop
        self.pitch = pixel_pitch
        self.focus = focus

        if aperture is None:
            self.kernel = pillbox_psf
        else:
            self.kernel = aperture


    def set_focal_plane(self, distance):
        "Sets the focal plane of the camera."
        self.focus = distance

    def set_f_stop(self, value):
        self.fN = value

    @property
    def aperture_diameter(self):
        "Diameter of the camera's aperture."
        return self.fL/self.fN

    def psf_diameter(self,distance):
        "Returns the diameter of the psf given the distance to the scene."
        fL = self.fL
        focus = self.focus
        V = 1/(1/fL - 1/focus)
        diam = fL / self.fN
        return diam*V*(1/distance-1/focus)


    def distance(self, diam_psf):
        "Returns the distance of the object if given the diameter of the PSF."
        fL = self.fL
        focus = self.focus
        s = 1/(1/fL - 1/focus)
        diam = fL / self.fN
        return s/(s/fL - diam_psf/diam -1)


    def set_aperture(self, fun):
        "Sets the shape of the aperture for the camera."
        self.kernel = fun


    def psf(self, distance= None, diameter=None):
        "Returns the PSF given the distance to the imaged object or the diameter of the PSF."

        if distance is None and diameter is None:
            raise TypeError("At least one parameter must be defined.")

        if not distance is None:
            pitch = self.pitch
            psfd = self.psf_diameter(distance)
            kerd = abs(psfd/pitch)
        else:
            kerd = diameter

        if kerd < 1:
            return np.array(1, ndmin = 2)

        h = self.kernel(kerd)
        return h/np.sum(h)
