import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import skimage as skim
import skimage.filters as filters
import gen_test as gt

im_size = 1000
im_shape = [im_size, im_size]
shape_ranges = (20, 40, 80, 150, 300, 600, 1000)
nrows = 4
ncols = 7
figsize = (28, 12)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
plt.tight_layout()


# im = ps.generators.blobs(im_shape, blobiness=2)
im = gt.blobs(im_shape, k=[1, 0.1])

for i, r in enumerate(shape_ranges):
    center = im_size // 2
    start = center - r//2
    finish = center + r // 2
    axes[0, i].imshow(im[start:finish, start:finish], cmap='gray')

im = im.astype(np.float)
im[im > 0] = 0.7
im[im == 0] = 0.3
# im = skim.util.random_noise(im, var=0.125)
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
    f_im = np.fft.fft2(im[start:finish, start:finish])
    f_im_abs = np.fft.fftshift(np.log(1 + np.abs(f_im)))
    # if r > 100:
    #     center = r // 2
    #     f_im_abs = f_im_abs[center-50:center+50, center-50:center+50]
    axes[2, i].imshow(f_im_abs)
    axes[3, i].plot(f_im_abs[(finish - start) // 2, :])

# for i, r in enumerate(shape_ranges):
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
