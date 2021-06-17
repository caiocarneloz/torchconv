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

def apply_conv(img, filter_name, filter_size=None):

    filter = torch.from_numpy(utils._get_conv_filter(filter_name, filter_size)) \
                        .type(torch.FloatTensor).to(device)
    filtered_img = func.conv2d(img, filter, padding=1)
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