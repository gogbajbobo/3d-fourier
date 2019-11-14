import porespy as ps
import matplotlib.pyplot as plt
import skimage as skim
import skimage.filters as filters

im = ps.generators.blobs([100, 100])

plt.figure()
plt.imshow(im)

im1 = skim.util.random_noise(im, var=0.125)
im1 = filters.gaussian(im1)

plt.figure()
plt.imshow(im1)

plt.figure()
plt.hist(im1, 100, facecolor='blue', alpha=0.5)

plt.show()
