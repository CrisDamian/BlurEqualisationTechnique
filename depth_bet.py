"""
depth_bet.py

Generate depth error map from 2 defocused images using
Blur Equalisation Technique (BET).


References
----------
Xian, Tao & Subbarao, M. (2006). Depth-from-defocus: blur equalization technique. Proceedings of SPIE - The International Society for Optical Engineering. 10.1117/12.688615.

"""
import numpy as np
from scipy.ndimage import convolve, gaussian_filter

def depth_bet(image1, image2, psfs1, psfs2, sigma=2.0):
    """Generate depth error map from 2 images using Blur Equalisation Technique (BET).

    Parameters
    -----------
    image1: MxN array like
        First image of the scene.

    image2: MxN array like
        Second image of the scene. It has to be an acquisition of
        the same scene with the camera in the same position but
        with different focus settings.

    psfs1: list of array
        Defocus Point Spread Functions (PSFs) for a set of detphs.

    psfs2: list of array
        Defocus PSFs for the same set of depths as `psfs1`.

    sigma: float,  >= 0
        Parameter for smoothing the error map with a gaussian filter.

    Returns
    -------
    error_map : array
        A MxNxL array of error values where MxN is the size of the
        imput images and L is the number of depths.
        The smaller the value of the error the more likely it is that  the object is at the corresponding depth.

    """

    if not len(psfs1)==len(psfs2):
        raise ValueError(
            "`psf1` and `psf2` must have the same lenghts")

    error_map = np.empty(image1.shape + (len(psfs1),))
    for i, (p1,p2) in enumerate( zip(psfs1,psfs2)):
        evidence1 = convolve(image1, p2)
        evidence2 = convolve(image2, p1)
        error_map[:,:,i] = gaussian_filter((evidence2 - evidence1)**2, sigma)

    return error_map
