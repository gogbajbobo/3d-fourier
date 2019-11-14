import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import skimage as skim
import skimage.filters as filters
import time

t0 = time.time()

im_size = 500
im_shape = np.array([1, 1, 1]) * im_size
shape_ranges = tuple(int(i * im_size) for i in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
nrows = 3
ncols = 7
figsize = (28, 9)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
plt.tight_layout()

t1 = time.time()
print('init complete:', t1 - t0, 's')

im = ps.generators.blobs(im_shape, blobiness=2)

t2 = time.time()
print('image generation complete:', t2 - t1, 's')

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[0, i].imshow(im[start:finish, start:finish, center], cmap='gray')

t3 = time.time()
print('imshow 1 complete:', t3 - t2, 's')

im = im.astype(np.float)
im[im > 0] = 0.7
im[im == 0] = 0.3
im = skim.util.random_noise(im, mode='poisson')
im = filters.gaussian(im, sigma=1)

t4 = time.time()
print('noising complete:', t4 - t3, 's')

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[1, i].imshow(im[start:finish, start:finish, center], cmap='gray')

t5 = time.time()
print('imshow 2 complete:', t5 - t4, 's')

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    g_im = im[start:finish, start:finish, start:finish]
    g_im = np.ravel(g_im)
    axes[2, i].hist(g_im, 255, [0, 1], density=True)

t6 = time.time()
print('histogram and imshow 3 complete:', t6 - t5, 's')


plt.show()
