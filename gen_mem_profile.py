from typing import List
import scipy as sp
import scipy.ndimage as spim
from scipy import special
import matplotlib.pyplot as plt
from memory_profiler import profile
import sys


@profile
def blobs(
        shape: List[int],
        porosity: float = 0.5,
        blobiness: int = 1,
        show_images: bool = False,
        random_seed: int = None
):
    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape) / (40 * blobiness)
    sp.random.seed(random_seed)
    im = sp.random.random(shape).astype(sp.float32)

    im = spim.gaussian_filter(im, sigma=sigma)

    show_image(im, show_images=show_images)

    im = norm_to_uniform(im, scale=[0, 1])

    show_image(im, show_images=show_images)

    if porosity:
        im = im < porosity
    show_image(im, show_images=show_images)

    return im


@profile
def norm_to_uniform(im, scale=None):
    if scale is None:
        scale = [im.min(), im.max()]

    im = (im - sp.mean(im)) / sp.std(im)
    im = 1 / 2 * special.erfc(-im / sp.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]

    return im


def show_image(im, show_images=False):
    if not show_images:
        return
    plt.figure()
    plt.imshow(im, cmap='gray')


@profile
def main():
    size = 1000
    dimensions = 2
    show_images = False
    blobs(
        shape=sp.ones(dimensions, dtype=int) * size,
        blobiness=1,
        show_images=show_images,
        random_seed=1
    )
    if show_images:
        plt.show()


if __name__ == '__main__':
    main()
