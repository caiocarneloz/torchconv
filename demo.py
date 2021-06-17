from skimage import io
from torchconv import *

#READ IMAGE
file_path = 'images/noisy_statue.jpg'
img = io.imread(file_path)
plt.imshow(img)

#GRAYSCALE AND CONVERT TO TENSOR
gray_img = T.Compose([T.ToPILImage(),T.Grayscale(),
            T.ToTensor()])(img).to(device).unsqueeze(0)

#CONVOLUTE IMAGE USING HIGH PASS FILTER
filtered_img = apply_conv(gray_img, 'sobel', (1,1,3,3))
filtered_img = utils.normalize_image(filtered_img)
plt.imshow(filtered_img, cmap='gray')

#APPLY LOW PASS FILTER IN THE FREQUENCY DOMAIN (FOURIER)
fourier_img = fourier_transform(gray_img.squeeze(0), 'lowpass', 30)
plt.imshow(fourier_img, cmap='gray')