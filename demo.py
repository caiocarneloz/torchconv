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

#CONVOLUTE IMAGE USING HIGH PASS FILTER
filtered_img = apply_conv(gray_img, 'sobel', (1,1,3,3))
filtered_img = utils.normalize_image(filtered_img)
plt.imshow(filtered_img, cmap='gray')

#APPLY LOW PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'gausslowpass', 20)
plt.imshow(fourier_img, cmap='gray')

fourier_img = fourier_transform(gray_img.squeeze(0), 'gaussbandpass', 10)
plt.imshow(fourier_img, cmap='gray')