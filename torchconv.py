import torch
import numpy as np
import filter_utils as utils
import torch.nn.functional as func

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
    #prob_dict = dict.fromkeys(torch.arange(gray_levels).tolist(), 0)
    prob_dict = {}
    
    for key, value in zip(keys.tolist(), counts.tolist()):
        prob_dict[key] = value

    counts = torch.cumsum(torch.FloatTensor(list(prob_dict.values())), dim=0).to(device)
    counts = torch.round(((counts/counts[-1])*(gray_levels))).type(torch.int)

    for key, value in zip(prob_dict.keys(), counts.tolist()):
        img[img == key] = value

    return img

def otsu_segmentation(img):

    values, idx = img.unique().sort()
    hist = torch.histc(img, min=0, max=img.max(), bins=len(img.unique()))
    pixels = hist.sum()
    size = len(hist)
    max_cost = float('inf')

    for i in range(1,size):

        b = torch.arange(i).to(device)
        f = torch.arange(i,size).to(device)

        wb = hist[:i].sum()/pixels
        meanb = (hist[:i]*b).sum()/hist[:i].sum()
        varb = (torch.pow(b - meanb, 2)*hist[:i]).sum()/hist[:i].sum()

        wf = hist[i:].sum()/pixels
        meanf = (hist[i:]*f).sum()/hist[i:].sum()
        varf = (torch.pow(f - meanf, 2)*hist[i:]).sum()/hist[i:].sum()

        cost = wb*varb+wf*varf

        if cost < max_cost:
            max_cost = cost
            best = i
    
    img[img < values[best]] = 0
    img[img > values[best]] = 1

    return img

def morphology(img, mask_name, origin, op):

    mask = torch.FloatTensor(utils._get_morph_filter(mask_name)).to(device)
    new_img = torch.zeros(img.shape).to(device)
    xsize = mask.shape[-2]
    ysize = mask.shape[-1]
    ox, oy = origin[0], origin[1]

    if op == 'dilation':
        for j in range(0,img.shape[-2]-ysize):
            for i in range(0,img.shape[-1]-xsize):
                    if img[:,:,j:j+ysize,i:i+xsize][:,:,ox,oy] == mask[:, :, ox, oy]:
                        new_img[:,:,j:j+ysize,i:i+xsize] = mask

    if op == 'erosion':
        for j in range(0,img.shape[-2]-ysize):
            for i in range(0,img.shape[-1]-xsize):
                if (img[:,:,j:j+ysize,i:i+xsize] == mask).all():
                    new_img[:,:,j+oy,i+ox] = 1

    return new_img.to('cpu')

def opening(img, mask_name, origin):

    img_ = morphology(img, mask_name, origin, 'erosion')
    img_ = morphology(img_.to(device), mask_name, origin, 'dilation')

    return img_.to('cpu')

def closing(img, mask_name, origin):
    
    img_ = morphology(img, mask_name, origin, 'dilation')
    img_ = morphology(img_.to(device), mask_name, origin, 'erosion')

    return img_.to('cpu')

def get_edges(img, mask_name, origin):
    
    img_ = morphology(img, mask_name, origin, 'dilation')
    
    edge = torch.abs(img_.to(device) - img)

    return edge.to('cpu')