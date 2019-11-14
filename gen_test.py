import porespy as ps
import scipy as sp
from scipy import ndimage as spim
import time
from sys import getsizeof
import math
import matplotlib.pyplot as plt


def norm_to_uniform(im, scale=None):

    t_s = time.time()
    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - sp.mean(im)) / sp.std(im)
    # im = (im - sp.mean(im, dtype=sp.float32)) / sp.std(im, dtype=sp.float32)
    im = 1 / 2 * sp.special.erfc(-im / sp.sqrt(2))
    # im = 1 / 2 * math.erfc(-im / sp.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]
    t_f = time.time()
    print('norm_to_uniform:', t_f - t_s, 's')

    return im


def blobs(shape, k, porosity: float = 0.5, blobiness: int = 1):
    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape)/(40*blobiness)
    print(sigma)
    sigma = sp.array(k) * sigma
    print(sigma)
    t_s = time.time()
    im = sp.random.random(shape)
    t_f = time.time()
    print('sp.random.random(shape):', t_f - t_s, 's')

    # print(im)
    print('im size: ', getsizeof(im))

    # t_s = time.time()
    # im = sp.empty(shape, dtype=sp.float32)
    # im[:, :, :] = sp.random.random()
    # t_f = time.time()
    # print('sp.empty + sp.random.random(shape):', t_f - t_s, 's')
    # print(im)
    # print('im f2 size: ', getsizeof(im))

    t_s = time.time()
    im = spim.gaussian_filter(im, sigma=sigma)
    t_f = time.time()
    print('spim.gaussian_filter:', t_f - t_s, 's')

    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im < porosity
    return im


if __name__ == '__main__':

    im_size = 300
    im_shape = (sp.ones(2) * im_size).astype(int)

    # t0 = time.time()
    #
    # ps.generators.blobs(im_shape)
    #
    t1 = time.time()
    # print('ps.generators.blobs:', t1 - t0, 's')

    # result = getsizeof(blobs(im_shape))

    # print('result size: ', result)

    porosity = 0.1

    im = blobs(im_shape, k=[1, 1], porosity=porosity, blobiness=1)

    plt.figure()
    plt.imshow(im)

    im = blobs(im_shape, k=[1, 2], porosity=porosity, blobiness=1)

    plt.figure()
    plt.imshow(im)

    im = blobs(im_shape, k=[1, 0.1], porosity=porosity, blobiness=1)

    plt.figure()
    plt.imshow(im)

    plt.show()

    t2 = time.time()
    print('blobs:', t2 - t1, 's')
