from skimage import io
from torchconv import *

#READ IMAGE
file_path = 'images/building.jpg'
img = io.imread(file_path)
plt.imshow(img)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).to(device).unsqueeze(0)
plt.imshow(gray_img.to('cpu').squeeze(0).squeeze(0).numpy(), cmap='gray')

#EX1
#CONVOLUTE IMAGE USING MEAN FILTER
filtered_img = kernel_convolution(gray_img, 'mean')
plt.imshow(filtered_img, cmap='gray')

#CONVOLUTE IMAGE USING GAUSSIAN FILTER
filtered_img = kernel_convolution(gray_img, 'gauss')
plt.imshow(filtered_img, cmap='gray')

#CONVOLUTE IMAGE USING HIGH PASS FILTER
filtered_img = kernel_convolution(gray_img, 'highpass')
plt.imshow(filtered_img, cmap='gray')

#APPLY LOW PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'gausslowpass', 20)
plt.imshow(fourier_img, cmap='gray')

#APPLY BAND PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'gaussbandpass', 10)
plt.imshow(fourier_img, cmap='gray')

#EX2
#CONVOLUTE IMAGE USING ROBERTS GRADIENT FILTER
filtered_img = kernel_convolution(gray_img, 'roberts')
plt.imshow(filtered_img, cmap='gray')

#CONVOLUTE IMAGE USING PREWITT GRADIENT FILTER
filtered_img = kernel_convolution(gray_img, 'prewitt')
plt.imshow(filtered_img, cmap='gray')

#CONVOLUTE IMAGE USING SOBEL GRADIENT FILTER
filtered_img = kernel_convolution(gray_img, 'sobel')
plt.imshow(filtered_img, cmap='gray')

#CONVOLUTE IMAGE USING LAPLACE GRADIENT FILTER
filtered_img = kernel_convolution(gray_img, 'laplace')
plt.imshow(filtered_img, cmap='gray')

#THRESHOLD IMAGE BY INTENSITY RANGE
trsh_img = naive_thresolding(gray_img.clone(), [0.2,0.6,0.8])
plt.imshow(trsh_img, cmap='gray')

#THRESHOLD IMAGE BY OTSU'S METHOD
otsu = otsu_segmentation(gray_img)
plt.imshow(otsu.to('cpu').squeeze(0).squeeze(0).numpy(), cmap='gray')