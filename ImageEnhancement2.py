import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim


def print_mse(image1, image2):
    diff_arr = np.subtract(image1, image2)
    sq_arr = np.square(diff_arr)
    mse = sq_arr.mean()


def print_ssim(image1, image2):
    (score, difference) = compare_ssim(image1, image2, full=True)
    print("SSIM: " + str(score))


def process_image(image):
    """
    Post process the image to create a full contrast stretch of the image
    takes as input:
    image: the image obtained from the inverse fourier transform
    return an image with full contrast stretch
    -----------------------------------------------------
    1. Full contrast stretch (fsimage)
    2. take negative (255 - fsimage)
    """
    a = 0
    b = 255
    c = np.min(image)
    d = np.max(image)
    rows, columns = np.shape(image)
    image1 = np.zeros((rows, columns), dtype=int)
    for i in range(rows):
        for j in range(columns):
            if (d - c) == 0:
                image1[i, j] = ((b - a) / 0.000001) * (image[i, j] - c) + a
            else:
                image1[i, j] = ((b - a) / (d - c)) * (image[i, j] - c) + a
    return np.uint8(image1)


def compute_dft(image):
    """
    Computes the Discrete Fourier transformation of the image
    """
    fft = np.fft.fft2(image)
    shift_fft = np.fft.fftshift(fft)
    mag_dft = 20 * np.log(np.abs(shift_fft))
    dft = process_image(mag_dft)
    return shift_fft, dft


def get_butterworth_filter(shape, cutoff, order):
    """
    Computes a butterworth low pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the butterworth filter
    order: the order of the butterworth filter
    returns a butterworth low pass mask
    """
    d0 = cutoff
    n = order
    rows, columns = shape
    mask = np.zeros((rows, columns))
    mid_R, mid_C = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
            mask[i, j] = 1 / (1 + (d / d0) ** (2 * n))

    return mask


def get_ideal_filter(shape, cutoff):
    """
    Computes a Ideal low pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the ideal filter
    returns a ideal low pass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns), dtype=int)
    mid_R, mid_C = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
            if d <= d0:
                mask[i, j] = 1
            else:
                mask[i, j] = 0

    return mask


def get_gaussian_filter(shape, cutoff):
    """
    Computes a gaussian low pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the gaussian filter (sigma)
    returns a gaussian low pass mask
    """
    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns))
    mid_R, mid_C = int(rows / 2), int(columns / 2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_R) ** 2 + (j - mid_C) ** 2)
            mask[i, j] = np.exp(-(d * d) / (2 * d0 * d0))

    return mask


def remove_noise(filter_name, shift_fft, shape, cutoff, order):
    """
    Removes Noise from the image on the basis of the filter name provided
    Returns : filtered discrete fourier transformation, filtered image
    """
    if filter_name == "butterworth":
        mask = get_butterworth_filter(shape, cutoff, order)
    elif filter_name == "ideal":
        mask = get_ideal_filter(shape, cutoff)
    elif filter_name == "gaussian":
        mask = get_gaussian_filter(shape, cutoff)

    filtered_image = np.multiply(mask, shift_fft)
    mag_filtered_dft = 20 * np.log(np.abs(filtered_image) + 1)
    filtered_dft = process_image(mag_filtered_dft)

    return filtered_dft, filtered_image


def compute_ift(filtered_image):
    """
    Computes inverse fourier transformation
    """
    shift_ifft = np.fft.ifftshift(filtered_image)

    ifft = np.fft.ifft2(shift_ifft)
    mag = np.abs(ifft)
    filtered_image = process_image(mag)

    return filtered_image


def remove_periodic_noise(img):
    """
    Removes Periodic noise from the image
    """
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r_out = crow
    r_in = ccol
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_xor(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back, fshift_mask_mag


def plot_images(img_input, img_org, img_dft, img_filtered_i, img_filtered_g, img_filtered_b, img_filter_dft_i,
                img_filter_dft_g, img_filter_dft_b, img_periodic, img_periodic_dft):
    """
    Plot the all images provided on the grid for comparison of each
    step and filter effects.
    """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(6, 1, 1)
    ax1.imshow(img_org, cmap='gray')
    ax1.title.set_text('Original Image')

    plt.axis('off')

    ax9 = fig.add_subplot(6, 2, 3)
    ax9.imshow(img_dft, cmap='gray')
    ax9.title.set_text('DFT of Image')

    plt.axis('off')

    ax2 = fig.add_subplot(6, 2, 4)
    ax2.imshow(img_input, cmap='gray')
    ax2.title.set_text('Noisy Image')

    plt.axis('off')

    ax3 = fig.add_subplot(6, 2, 5)
    ax3.imshow(img_periodic_dft, cmap='gray')
    ax3.title.set_text('DFT after periodic noise removal')

    plt.axis('off')

    ax4 = fig.add_subplot(6, 2, 6)
    ax4.imshow(img_periodic, cmap='gray')
    ax4.title.set_text('After Periodic noise removal')

    plt.axis('off')

    ax3 = fig.add_subplot(6, 2, 7)
    ax3.imshow(img_filter_dft_i, cmap='gray')
    ax3.title.set_text('FFT + Ideal Mask')

    plt.axis('off')

    ax4 = fig.add_subplot(6, 2, 8)
    ax4.imshow(img_filtered_i, cmap='gray')
    ax4.title.set_text('After Ideal Filter')

    plt.axis('off')

    ax5 = fig.add_subplot(6, 2, 9)
    ax5.imshow(img_filter_dft_g, cmap='gray')
    ax5.title.set_text('FFT + Gaussian Mask')

    plt.axis('off')

    ax6 = fig.add_subplot(6, 2, 10)
    ax6.imshow(img_filtered_g, cmap='gray')
    ax6.title.set_text('After Gaussian Filter')

    plt.axis('off')

    ax7 = fig.add_subplot(6, 2, 11)
    ax7.imshow(img_filter_dft_b, cmap='gray')
    ax7.title.set_text('FFT + Butterworth Mask')

    plt.axis('off')

    ax8 = fig.add_subplot(6, 2, 12)
    ax8.imshow(img_filtered_b, cmap='gray')
    ax8.title.set_text('After Butterworth Filter')

    plt.axis('off')

    plt.show()


def main():
    # Initialing the parameters that will be used for filter the noisy image
    input_image = cv2.imread(".\\Images\\PandaNoise(1).bmp", 0)
    rows, cols = input_image.shape
    cutoff_f = 50
    order = 2
    shape = np.shape(input_image)

    # Calculating the Fourier transformation
    shift_fft, dft_org = compute_dft(input_image)

    # Periodic Mask Removal
    img_periodic, img_periodic_dft = remove_periodic_noise(input_image)

    # Initializing all the masks
    mask = ['ideal', 'gaussian', 'butterworth']
    shift_fft, dft = compute_dft(img_periodic)

    ## Ideal Mask
    filtered_dft_i, filtered_image_i = remove_noise(mask[0], shift_fft, shape, cutoff_f, order)
    filtered_image_i = compute_ift(filtered_image_i)

    ## Guassian Mask
    filtered_dft_g, filtered_image_g = remove_noise(mask[1], shift_fft, shape, cutoff_f, order)
    filtered_image_g = compute_ift(filtered_image_g)

    ## Butterworth Mask
    filtered_dft_b, filtered_image_b = remove_noise(mask[2], shift_fft, shape, cutoff_f, order)
    filtered_image_b = compute_ift(filtered_image_b)

    # Declaring all the images
    img_input = input_image
    img_org = cv2.imread(".\\Images\\PandaOriginal(1).bmp", 0)
    print("MSE start: " + str(np.square(np.subtract(img_org, input_image)).mean()))
    img_dft = np.uint8(dft_org)
    img_filtered_i = np.uint8(filtered_image_i)
    img_filter_dft_i = np.uint8(filtered_dft_i)

    img_filtered_g = np.uint8(filtered_image_g)
    img_filter_dft_g = np.uint8(filtered_dft_g)

    img_filtered_b = np.uint8(filtered_image_b)
    img_filter_dft_b = np.uint8(filtered_dft_b)

    plot_images(img_input, img_org, img_dft, img_filtered_i, img_filtered_g, img_filtered_b, img_filter_dft_i,
                      img_filter_dft_g, img_filter_dft_b, img_periodic, img_periodic_dft)

    ## Calculating the MSE between the orginal and filtered images
    Y = np.square(np.subtract(img_org, img_filtered_i)).mean()
    print("MSE of Periodic and Ideal Noise Removal:", Y)

    Y = np.square(np.subtract(img_org, img_filtered_g)).mean()
    print("MSE of Periodic and Gaussian Noise Removal:", Y)

    Y = np.square(np.subtract(img_org, img_filtered_b)).mean()
    print("MSE of Periodic and Butterworth Noise Removal:", Y)


if __name__ == "__main__":
    main()
