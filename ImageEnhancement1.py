import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.measure import compare_ssim

# Calculates the MSE of two NumPy arrays (which represent images)
def print_mse(image1, image2):
    diff_arr = np.subtract(image1, image2)
    sq_arr = np.square(diff_arr)
    mse = sq_arr.mean()
    print("MSE: " + str(mse))
    # mse = ((image1 - image2)**2).mean(axis=None)
    # print("MSE: " + str(mse))

# Replaces a list of Y positions with a single x position using an unweighted Guassian filter
# nb must use odd kernel size
def kernel_replace_list(image, positionX, positionY, kernelSize):
    value = 0
    diff = int((kernelSize - 1)/2)
    divide = 1/(kernelSize**2)
    for position in positionY:
        for x in range(positionX - diff, positionX + diff):
            for y in range(position - diff, position + diff):
                value = image[x, y]
        new_value = divide * value
        # print(str(image[positionX, position]) + " is being replaced with " + str(new_value))
        image[positionX, position] = new_value
        value = 0


# Replaces a single position using an unweighted Guassian filter
# nb must use odd kernel size
def kernel_replace(image, positionX, positionY, kernelSize):
    value = 0
    diff = int((kernelSize - 1)/2)
    divide = 1/(kernelSize**2)
    for x in range(positionX - diff, positionX + diff):
        for y in range(positionY - diff, positionY + diff):
            value = image[x, y]
    new_value = divide * value
    # print(str(image[positionX, position]) + " is being replaced with " + str(new_value))
    image[positionX, positionY] = new_value

# calculate PSNR (peak signal to noise ratio)
def print_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        print(100)
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print("PSNR: " + str(PSNR))

# Calculates SSIM (structural similarity index measure) using SkImage library
def print_ssim(image1, image2):
    (score, difference) = compare_ssim(image1, image2, full=True)
    print("SSIM: " + str(score))

# Calls all the score functions - this will be used for comparisons
def get_scores(image1, image2):
    print_mse(image1, image2)
#    print_psnr(image1, image2)
    print_ssim(image2, image1)


# this function will change the brightness and contrast of an image based of alpha (contrast) and beta (brightness) values
# alpha is in range 1.0-3.0 and beta is in range 0-100
def change_contrast_brightness(img, alpha, beta):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Changes the value of each cell by multiplier (alpha) and addition (beta) within range 0-255
            img[y, x] = np.clip(alpha * img[y, x] + beta, 0, 255)


# read the noisy image
noise_im = cv2.imread("C:\\Users\\Alienware M15\\PycharmProjects\\ImageAnalysis\\Images\\PandaNoise(1).bmp", 0)
if type(noise_im) is np.ndarray:
    print("Damaged Image loaded successfully")
else:
    print("Failed to load damaged image")

# read the original image
original_im = cv2.imread("C:\\Users\\Alienware M15\\PycharmProjects\\ImageAnalysis\\Images\\PandaOriginal(1).bmp", 0)
if type(noise_im) is np.ndarray:
    print("Good Image loaded successfully")
else:
    print("Failed to load good image")


# create a plot for our transformations to be displayed on
plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

# load our standard image in from file
original_image = cv2.imread(".\\Images\\PandaOriginal(1).bmp", 0)
# load our fuzzy image in from file
noise_im = cv2.imread(".\\Images\\PandaNoise(1).bmp", 0)
# get original MSE between 2 images
print("Images loaded.")
get_scores(noise_im, original_image)

# Frequency domain transformation to remove periodic noise
# Fourier transform and shift in 2D to convert image to frequency domain
pre_ft_img = np.fft.fftshift(np.fft.fft2(noise_im))
# Create a copy of the image so we can successfully show original on plot
ft_img = pre_ft_img.copy()
# Get size and center values of image
rows, cols = ft_img.shape
crow, ccol = rows//2, cols//2

# from observation we can see there is noise along the center of the frequency image
# We loop through the desired cells and replace the values with average from line
# radius is how far from the center the line will stop
radius = 80
for value in range(0, crow-radius):
    # fills the cell with an average of a straight kernel along x axis
    ft_img[value, ccol] = ft_img[value, ccol-5:ccol+5].mean()
    ft_img[value, ccol] = 0
for value in range(crow + radius, rows - 1):
    # fills the area with an average of a straight kernel along x axis
    ft_img[value, ccol] = ft_img[value, ccol-5:ccol+5].mean()
    ft_img[value, ccol] = 0

# For next transforms we need a kernel around the image
# Loop through all columns
for i in range(30, 50):
    position = [65,66,67,197,198,199,461,462,463,593,594,595]
    # Call kernel list replace function to replace values
    kernel_replace_list(ft_img, i, position, 9)

for j in range(325, 345):
    position = [65,66,67,197,198,199,461,462,463,593,594,595]
    # Call kernel list replace function to replace values
    kernel_replace_list(ft_img, j, position, 9)

# Inverse shift then inverse 2D fourier transform
processed_image = abs(np.fft.ifft2(np.fft.ifftshift(ft_img)))

# Increase contrast in image
# We will increase the contrast before blurring the image to try and maintain the quality reduced by blurring
# So we will brighten slightly
# Range: 1.0-3.0
alpha = 1.08 # Contrast control = 1.08 so slight increase in contrast before blur
# Range: 0-100
beta = 0    # Brightness control = 0 so there is no change to brightness

change_contrast_brightness(processed_image, alpha, beta)

get_scores(processed_image, original_image)


# Now we can edit the image in the spatial domain
# we use a median blur to edit the image
spatial_img = scipy.ndimage.median_filter(processed_image, size=5)
print("Calculating MSE after median blur")

# attempt to sharpen
# using digital unsharp masking
# sharpened = original + (original − blurred) × amount.
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(spatial_img,-1,kernel)
# ratio = 0.275
# spatial_img = spatial_img + ((spatial_img - dst) * ratio)


get_scores(spatial_img, original_image)


# this draws the image at each point between the transformations
# log transforming each one - however the log transform isn't good enough
plt.subplot(151), plt.imshow(noise_im, "gray"), plt.title("Noisy Image")
plt.subplot(152), plt.imshow(np.log(1 + np.abs(pre_ft_img)), "gray"), plt.title("FFT2 Image before filter")
# plt.subplot(152), plt.imshow(np.log(1 + np.abs(pre_ft_img)), "gray"), plt.title("FFT2 Image before filter")
plt.subplot(153), plt.imshow(np.log(1 + np.abs(ft_img)), "gray"), plt.title("FFT2 Image after filter")
plt.subplot(154), plt.imshow(np.abs(processed_image), "gray"), plt.title("Processed Image (FFT)")
plt.subplot(155), plt.imshow(np.abs(spatial_img), "gray"), plt.title("Final Image")
# plt.subplot(156), plt.imshow(original_image, "gray"), plt.title("Target Image")

plt.show()