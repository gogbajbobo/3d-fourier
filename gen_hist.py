import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import skimage as skim
import skimage.filters as filters

im_size = 1000
im_shape = [im_size, im_size]
shape_ranges = tuple(int(i * im_size) for i in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
nrows = 3
ncols = 7
figsize = (28, 9)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
plt.tight_layout()


im = ps.generators.blobs(im_shape, blobiness=2)

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[0, i].imshow(im[start:finish, start:finish], cmap='gray')

im = im.astype(np.float)
im[im > 0] = 0.7
im[im == 0] = 0.3
im = skim.util.random_noise(im, mode='poisson')
im = filters.gaussian(im, sigma=1)

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[1, i].imshow(im[start:finish, start:finish], cmap='gray')

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    g_im = im[start:finish, start:finish]
    g_im = np.ravel(g_im)
    axes[2, i].hist(g_im, 255, [0, 1], density=True)


plt.show()
