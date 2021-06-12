import torch
import numpy as np
from skimage import io
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
    filtered_img = func.conv2d(gray_img, filter, padding=1)
    filtered_img = filtered_img.reshape(filtered_img.shape[-2:])
    return filtered_img.to('cpu').numpy()

#READ IMAGE
filename = 'lena.jpg'
img = io.imread(filename)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).unsqueeze(0).to(device)

#CONVOLUTE IMAGE USING FILTER
new_img = apply_conv(img, 'lowpass', (1,1,3,3))
plt.imshow(new_img)
