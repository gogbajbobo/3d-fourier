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
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import cv2


def time_measurement(func):

    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        finish_time = time.perf_counter()
        elapsed_time = finish_time - start_time

        print(f'{ func.__name__ }() elapsed time: { timedelta(seconds=elapsed_time) }')

        return result

    return wrapper


@time_measurement
def norm_to_uniform(im, scale=None):

    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - sp.mean(im)) / sp.std(im)
    im = 1 / 2 * special.erfc(-im / sp.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]

    return im


def blobs(shape, k, porosity: float = 0.5, blobiness: int = 1, show_figs: bool = False):

    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape)/(40*blobiness)
    sigma = sp.array(k) * sigma
    im = sp.random.random(shape)

    im = spim.gaussian_filter(im, sigma=sigma)

    im = norm_to_uniform(im, scale=[0, 1])
    im_test, _ = image_histogram_equalization(im)
    im_test -= np.min(im_test)
    im_test *= 1.0 / np.max(im_test)

    im_test_opencv = opencv_histogram_equalization(im)
    im_test_opencv -= np.min(im_test_opencv)
    im_test_opencv = im_test_opencv / np.max(im_test_opencv)

    im_test_clahe = opencv_clahe_hist_equal(im)
    im_test_clahe -= np.min(im_test_clahe)
    im_test_clahe = im_test_clahe / np.max(im_test_clahe)

    if porosity:
        im = im < porosity
        im_test = im_test < porosity
        im_test_opencv = im_test_opencv < porosity
        im_test_clahe = im_test_clahe < porosity

    print(np.sum(im)/im.ravel().shape[0])
    print(np.sum(im_test)/im_test.ravel().shape[0])
    print(np.sum(im_test_opencv) / im_test_opencv.ravel().shape[0])
    print(np.sum(im_test_clahe) / im_test_clahe.ravel().shape[0])

    if show_figs:
        show_image_and_histogram(im, 'Porespy norm2uniform')
        show_image_and_histogram(im_test, 'numpy hist eq')
        show_image_and_histogram(im_test_opencv, 'opencv hist eq')
        show_image_and_histogram(im_test_clahe, 'opencv clahe hist eq')

    if show_figs:
        plt.show()

    return im


def show_image_and_histogram(im, title=None):
    plt.figure()
    plt.imshow(im, cmap='gray')
    if title:
        plt.title(title)
    # g_im = np.ravel(im)
    # plt.figure()
    # plt.hist(g_im, 255, [0, 1], density=True)
    # if title:
    #     plt.title(title)


@time_measurement
def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


@time_measurement
def opencv_histogram_equalization(im):
    img = np.uint8(cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX))
    return cv2.equalizeHist(img)


@time_measurement
def opencv_clahe_hist_equal(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = np.uint8(cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX))
    return clahe.apply(img)


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
