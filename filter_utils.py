from os import MFD_HUGE_SHIFT
import numpy as np


def _get_conv_filter(filter_name, filter_size=None):

    if filter_name == 'sobel':
        return np.array([[[[-1,-2,-1],[0,0,0],[1,2,1]]]])
    if filter_name == 'laplace':
        return np.array([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]])
    if filter_name == 'highpass':
        return np.array([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]])
    if filter_name == 'roberts':
        return np.array([[[[1,0],[0,-1]]]])
    if filter_name == 'gauss':
        return np.array([[[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]]])
    if filter_name == 'mean':
        return np.tile(1/9,filter_size)


def _get_fft_filter(filter_name, threshold, img_shape):

    x, y = img_shape[-2]//2, img_shape[-1]//2

    idx = np.indices(img_shape)
    idx_comb = zip(idx[0].flatten(), idx[1].flatten())
    filter = np.array([np.sqrt(np.power(i-x,2)+np.power(j-y,2))
                      for i, j in idx_comb]).reshape(img_shape)

    if filter_name == 'bandpass':

        filter[filter <= threshold] = 1
        filter[filter > threshold] = 0

        return filter

    if filter_name == 'gausslowpass':
        return 1/(1+np.power((filter/threshold),2))

    if filter_name == 'gaussbandpass':
        return np.exp(-np.power((np.power(filter,2) - np.power(threshold,2)) / (5 * filter),2))


def normalize_image(img):

    img += np.abs(np.min(img))
    img /= np.max(img)
    img *= 255

    img = img.astype(int)

    return img