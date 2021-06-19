import torch
import numpy as np
import filter_utils as utils
import matplotlib.pyplot as plt
from torch._C import ComplexType
from torch.functional import norm
import torch.nn.functional as func
from torchvision import transforms as T

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def kernel_convolution(img, filter_name):

    if filter_name not in ['sobel', 'prewitt', 'roberts']:
        filter = torch.FloatTensor(utils._get_conv_filter(filter_name)).to(device)
        filtered_img = func.conv2d(img, filter, padding=1)
    else:
        filter_x = torch.FloatTensor(utils._get_conv_filter(filter_name+'1')).to(device)
        filter_y = torch.FloatTensor(utils._get_conv_filter(filter_name+'2')).to(device)

        filtered_img_x = func.conv2d(img, filter_x, padding=1)
        filtered_img_y = func.conv2d(img, filter_y, padding=1)

        filtered_img = torch.sqrt(torch.float_power(filtered_img_x,2) + torch.float_power(filtered_img_y,2))

    filtered_img = filtered_img.reshape(filtered_img.shape[-2:])

    return filtered_img.to('cpu').numpy()

def fourier_transform(img, filter_name, threshold):

    fft_img = torch.fft.fftn(img)
    fft_img_shift = torch.fft.fftshift(fft_img)
    img_shape = fft_img_shift.shape[-2:]

    filter = utils._get_fft_filter(filter_name, threshold, img_shape)

    filter = torch.from_numpy(filter).unsqueeze_(0).to(device)
    fft_img_shift *= filter
    fft_img = torch.fft.fftshift(fft_img_shift)
    ifft_img = torch.fft.ifftn(fft_img)

    npimage = ifft_img.squeeze(0).to('cpu').numpy()
    return npimage.real

def naive_thresolding(img, thresholds):
    
    img[img < thresholds[0]] = 0
    for i in range(1,len(thresholds)-1):
        img[(img > thresholds[i]) & \
            (img < thresholds[i+1])] = thresholds[i]
    img[img > thresholds[-1]] = 1

    return img.squeeze(0).squeeze(0).to('cpu').numpy()

def histogram_equalizer(img, gray_levels):

    keys, counts = torch.unique(img, return_counts=True)
    prob_dict = dict.fromkeys(torch.arange(gray_levels).tolist(), 0)
    
    for key, value in zip(keys.tolist(), counts.tolist()):
        prob_dict[key] = value

    counts = torch.cumsum(torch.FloatTensor(list(prob_dict.values())), dim=0).to(device)
    counts = torch.round(((counts/counts[-1])*(gray_levels))).type(torch.int)

    for key, value in zip(prob_dict.keys(), counts.tolist()):
        img[img == key] = value

    return img