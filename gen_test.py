"""
Generate 2D porous picture and calc gray values histogram
"""

import matplotlib.pyplot as plt
import numpy as np
import generator


im_size = 5000
im_shape = np.ones(2, dtype=int) * im_size

image = generator.blobs(im_shape, k=[1, 1], show_figs=True)

# plt.figure()
# plt.imshow(image, cmap='gray')

# for i, r in enumerate(shape_ranges):
#     # show images gray values histogram
#     center = im_size // 2
#     start = center - r//2
#     finish = center + r // 2
#     g_im = im[start:finish, start:finish]
#     g_im = np.ravel(g_im)
#     axes[2, i].hist(g_im, 255, [0, 1], density=True)


# plt.show()
