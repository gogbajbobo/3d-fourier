import numpy as np
import matplotlib.pyplot as plt


def plot_some(a):

    b = np.fft.fft(a)

    plt.figure(figsize=(14, 3))
    plt.subplot(121)
    plt.plot(a)

    plt.subplot(122)
    _b = np.fft.fftshift(np.log(1 + np.abs(b)))
    center = _b.shape[0]//2
    _b = _b[center-50:center+50]
    plt.plot(_b)

    # plt.subplot(133)
    # __b = np.fft.fftshift(np.angle(b))
    # plt.plot(__b)

    return _b


a = np.random.rand(100)

aa = np.tile(a, 10)
plot_some(aa)
plot_some(aa[0:666])
plot_some(aa[0:333])
plot_some(aa[0:100])

# amp = np.empty((3, 100))
#
# for i in range(0,3):
#
#     amp[i:] = plot_some(np.roll(a, 33 * i))
#     # plot_some(np.flip(a))
#
# print(np.sum(amp[0]-amp[1]))
# print(np.sum(amp[0]-amp[2]))

plt.show()
