import torch
import numpy as np
from skimage import io
from torch._C import ComplexType
from torch.functional import norm
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as func

if torch.cuda.is_available():
    print('using cuda ' + torch.version.cuda)
    device = torch.device('cuda')
else:
    print('using cpu')
    device = torch.device('cpu')

def get_filter(filter, filter_size=None):

    if filter == 'sobel':
        return np.array([[[[-1,-2,-1],[0,0,0],[1,2,1]]]])
    if filter == 'laplace':
        return np.array([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]])
    if filter == 'lowpass':
        return np.array([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]])
    if filter == 'roberts':
        return np.array([[[[1,0],[0,-1]]]])
    if filter == 'gauss':
        return np.array([[[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]])
    if filter == 'mean':
        return np.tile(1/9,filter_size)

def apply_conv(img, filter_name, filter_size=None):

    filter = torch.from_numpy(get_filter(filter_name, filter_size)) \
                        .type(torch.FloatTensor).to(device)
    filtered_img = func.conv2d(img, filter, padding=1)
    filtered_img = filtered_img.reshape(filtered_img.shape[-2:])

    return filtered_img.to('cpu').numpy()

def normalize_image(img):

    img += np.abs(np.min(img))
    img /= np.max(img)
    img *= 255

    img = img.astype(int)

    return img

def fourier_transform(img, filter):


    fft_img = torch.fft.rfftn(img)
    fft_img_shift = torch.fft.fftshift(fft_img)
    img2 = fft_img_shift.imag.unsqueeze(0)
    fft_img_shift.imag = torch.from_numpy(apply_conv(img2, 'lowpass', (1,1,3,3)))
    #fft_img[fft_img.imag < filter] = 0
    fft_img = torch.fft.ifftshift(fft_img_shift)
    ifft_img = torch.fft.irfftn(fft_img)

    npimage = ifft_img.squeeze(0).to('cpu').detach().numpy()
    return npimage.real



#READ IMAGE
filename = 'lena.jpg'
img = io.imread(filename)
plt.imshow(img)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).to(device)


f_img = fourier_transform(gray_img, 10)
plt.imshow(f_img, cmap='gray')


#CONVOLUTE IMAGE USING FILTER
new_img = apply_conv(gray_img.unsqueeze_(0), 'lowpass', (1,1,3,3))
new_img = normalize_image(new_img)
plt.imshow(new_img, cmap='gray')