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

def fourier_transform(img, filter_name):
    
    
    fft_img = torch.fft.fftn(img)
    fft_img_shift = torch.fft.fftshift(fft_img)
    img_shape = fft_img_shift.shape[-2:]

    
    x, y = img_shape[-2]//2, img_shape[-1]//2

    
    idx = np.indices(img_shape)
    idx_comb = zip(idx[0].flatten(), idx[1].flatten())
    values = np.array([np.sqrt(np.power(i-x,2)+np.power(j-y,2))
                                 for i, j in idx_comb]).reshape(img_shape)
    values = 1/(1+np.power((values/1),2))

    filter = torch.from_numpy(values).to(device).type(torch.float32)

    #img2 = fft_img_shift.imag.unsqueeze(0)
    #fft_img_shift.imag = torch.matmul(fft_img_shift.imag,filter)
    fft_img_shift[fft_img_shift.imag > 10] = 0
    fft_img = torch.fft.ifftshift(fft_img_shift)
    ifft_img = torch.fft.ifftn(fft_img)

    npimage = ifft_img.squeeze(0).to('cpu').numpy()
    return npimage.real

f_img = fourier_transform(gray_img, 10)
plt.imshow(f_img, cmap='gray')


#READ IMAGE
filename = 'lena.jpg'
img = io.imread(filename)
plt.imshow(img)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).to(device)




#CONVOLUTE IMAGE USING FILTER
new_img = apply_conv(gray_img.unsqueeze_(0), 'lowpass', (1,1,3,3))
new_img = normalize_image(new_img)
plt.imshow(new_img, cmap='gray')