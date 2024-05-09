# question 5 - image with blur (RESULT: +/-)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(name_file):
    # reads an image in .jpg format and returns a numpy array representing the pixels.
    # uploading the image using OpenCV
    image = cv2.imread(name_file)

    # checking if uploading the image is done right
    if image is None:
        print(f"Error loading image {name_file}")
        return None

    # convert the image to a NumPy array
    image_numpy = np.array(image)

    return image_numpy

def converts_color_grayscale(image):
    # converts a color (RGB) image to grayscale.
    # use the formula: Y = 0.2989*R + 0.5870*G + 0.1140*B

    # extract the color channels (R, G, B)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # calculate grayscale intensity
    gray_intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray_intensity

def remove_salt_and_pepper_noise(image):
    """
    Remove salt-and-pepper noise from an image using a median filter.

    Args:
        image (numpy.ndarray): Input grayscale image (2D array).

    Returns:
        numpy.ndarray: Image with salt-and-pepper noise removed.
    """
    # create a copy of the input image
    denoised_image = np.copy(image)

    # define the window size for the median filter
    window_size = 3

    # iterate over each pixel in the image
    for i in range(window_size, image.shape[0] - window_size):
        for j in range(window_size, image.shape[1] - window_size):
            # extract the window centered at (i, j)
            window = image[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1]

            # check if the central pixel is corrupted (0 or 255)
            if image[i, j] == 0 or image[i, j] == 255:
                # compute the median of the non-corrupted pixels in the window
                valid_pixels = window[(window != 0) & (window != 255)]
                denoised_image[i, j] = np.median(valid_pixels)

    return denoised_image

def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensidade de Cinza')
    plt.ylabel('Frequência')

def show_results(img1, img2, title_original, title_composition):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plot_histogram(img1, 'Histograma Original')

    plt.subplot(2, 2, 3)
    plt.imshow(img2, cmap='gray')
    plt.title(title_composition)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plot_histogram(img2, 'Histograma - Imagem Sem Blur')

    plt.tight_layout()
    plt.show()

cc_original = 'list1/images/cc-c.jpg'
cc_rgb = load_image(cc_original)
cc_cinza = converts_color_grayscale(cc_rgb)
denoised_image = remove_salt_and_pepper_noise(cc_cinza)
show_results(cc_cinza, denoised_image, "Imagem - Musgo","Imagem Composta")