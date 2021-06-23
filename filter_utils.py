import torch
import numpy as np

def _get_conv_filter(filter_name):

    if filter_name == 'sobel1':
        return [[[[1,0,-1],[2,0,-2],[1,0,-1]]]]
    if filter_name == 'sobel2':
        return [[[[1,2,1],[0,0,0],[-1,-2,-1]]]]
    if filter_name == 'roberts1':
        return [[[[1,0],[0,-1]]]]
    if filter_name == 'roberts2':
        return [[[[0,1],[-1,0]]]]
    if filter_name == 'prewitt1':
        return [[[[1,0,-1],[1,0,-1],[1,0,-1]]]]
    if filter_name == 'prewitt2':
        return [[[[1,1,1],[0,0,0],[-1,-1,-1]]]]
    if filter_name == 'laplace':
        return [[[[0,-1,0],[-1,4,-1],[0,-1,0]]]]
    if filter_name == 'highpass':
        return [[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]]
    if filter_name == 'gauss':
        return [[[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]]]
    if filter_name == 'mean':
        return np.tile(1/9,(1,1,3,3))

def _get_morph_filter(filter_name):

    if filter_name == 'erosion':
        return [[[[1,1,1],[1,1,1],[1,1,1]]]]
    if filter_name == 'dilation':
        return [[[[0,1,0],[1,1,1],[0,1,0]]]]
    if filter_name == 'full':
        return [[[[1,1,1],[1,1,1],[1,1,1]]]]

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
        return np.exp(-np.power((np.power(filter,2) - np.power(threshold,2)) / (30 * filter),2))

def normalize_image(img, gray_levels):

    img += torch.abs(torch.min(img))
    img /= torch.max(img)
    img *= gray_levels

    img = img.type(torch.int)

    return img 