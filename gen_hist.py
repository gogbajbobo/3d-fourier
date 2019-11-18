"""
Generate 2D porous picture and calc gray values histogram
"""

import matplotlib.pyplot as plt
import numpy as np
import generator

im_size = 1000
im_shape = np.ones(2, dtype=int) * im_size
shape_ranges = tuple(int(i * im_size) for i in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
nrows = 3
ncols = 7
figsize = (28, 9)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
plt.tight_layout()

im = generator.blobs(im_shape, k=[1, 1])

for i, r in enumerate(shape_ranges):
    # show generated images
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[0, i].imshow(im[start:finish, start:finish], cmap='gray')

im = generator.add_noise_to_image(im, high_value=0.7, low_value=0.3)

for i, r in enumerate(shape_ranges):
    # show generated images with noise
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[1, i].imshow(im[start:finish, start:finish], cmap='gray')

for i, r in enumerate(shape_ranges):
    # show images gray values histogram
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    g_im = im[start:finish, start:finish]
    g_im = np.ravel(g_im)
    axes[2, i].hist(g_im, 255, [0, 1], density=True)


plt.show()
