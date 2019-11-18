"""
Generate porous pictures
Code taken from PoreSpy with adding param k to blobs() function to make orthogonal anisotropy
"""

import scipy as sp
from scipy import ndimage as spim
from scipy import special
import numpy as np
import skimage as skim
import skimage.filters as filters


def norm_to_uniform(im, scale=None):

    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - sp.mean(im)) / sp.std(im)
    im = 1 / 2 * special.erfc(-im / sp.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]

    return im


def blobs(shape, k, porosity: float = 0.5, blobiness: int = 1):

    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape)/(40*blobiness)
    sigma = sp.array(k) * sigma
    im = sp.random.random(shape)
    im = spim.gaussian_filter(im, sigma=sigma)
    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im < porosity
    return im


def add_noise_to_image(im, high_value=1, low_value=0):
    """
    Naive addition of noise to image
    """
    im = im.astype(np.float)
    im[im > 0] = high_value
    im[im == 0] = low_value
    im = skim.util.random_noise(im, mode='poisson')

    return filters.gaussian(im, sigma=1)


if __name__ == '__main__':
    pass
