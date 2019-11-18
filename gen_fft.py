"""
Generate 2D porous picture and fft it's internal parts
"""

import matplotlib.pyplot as plt
import numpy as np
import generator

im_size = 1000
im_shape = np.ones(2, dtype=int) * im_size
shape_ranges = tuple(int(i * im_size) for i in (0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
nrows = 4
ncols = 7
figsize = (28, 12)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
plt.tight_layout()

im = generator.blobs(im_shape, k=[1, 0.25])

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
    # show fft amplitudes
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    f_im = np.fft.fft2(im[start:finish, start:finish])
    f_im_abs = np.fft.fftshift(np.log(1 + np.abs(f_im)))
    # if r > 100:
    #     center = r // 2
    #     f_im_abs = f_im_abs[center-50:center+50, center-50:center+50]
    axes[2, i].imshow(f_im_abs)
    axes[3, i].plot(f_im_abs[(finish - start) // 2, :])

# for i, r in enumerate(shape_ranges):
#     # show fft phase
#     center = im_size // 2
#     start = center - r//2
#     finish = center + r // 2
#     f_im = np.fft.fft2(im[start:finish, start:finish])
#     f_im_angle = np.fft.fftshift(np.angle(f_im))
#     # if r > 100:
#     #     center = r // 2
#     #     f_im_angle = f_im_angle[center-50:center+50, center-50:center+50]
#     axes[3, i].imshow(f_im_angle)

plt.show()
