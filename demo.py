from skimage import io
from torchconv import *

#READ IMAGE
file_path = 'images/bell_pepper.jpg'
img = io.imread(file_path)
plt.imshow(img)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).to(device).unsqueeze(0)
plt.imshow(gray_img.to('cpu').squeeze(0).squeeze(0).numpy(), cmap='gray')

#CONVOLUTE IMAGE USING PREWITT FILTER
filtered_img = kernel_convolution(gray_img, 'sobel')
plt.imshow(filtered_img, cmap='gray')

#APPLY LOW PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'gausslowpass', 20)
plt.imshow(fourier_img, cmap='gray')

#APPLY BAND PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'gaussbandpass', 10)
plt.imshow(fourier_img, cmap='gray')

#THRESHOLD IMAGE BY INTENSITY RANGE
trsh_img = naive_thresolding(gray_img.clone(), [0.2,0.6,0.8])
plt.imshow(trsh_img, cmap='gray')

#EQUALIZE IMAGE HISTOGRAM
img2 = histogram_equalizer(gray_img.clone(), 255)
plt.imshow(img2.to('cpu').squeeze(0).squeeze(0).numpy(), cmap='gray')