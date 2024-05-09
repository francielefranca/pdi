# question 1 - resizing images
# FAIL: without interpolation

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_img(name_file):
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

def converts_color_grayscale(image_rgb):
    # converts a color (RGB) image to grayscale.
    # use the formula: Y = 0.2989*R + 0.5870*G + 0.1140*B

    # extract the color channels (R, G, B)
    r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]

    # calculate grayscale intensity
    gray_intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray_intensity

def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensidade de Cinza')
    plt.ylabel('Frequência')

def resizing_img(image, goal):
    height, width = goal
    return np.resize(image, (height, width))

def show_results(image_original, image_resizing, title_original, title_resizing):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(image_original, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plot_histogram(image_original, 'Histograma Original')

    plt.subplot(2, 2, 3)
    plt.imshow(image_resizing, cmap='gray')
    plt.title(title_resizing)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plot_histogram(image_resizing, 'Histograma Redimensionado')

    plt.tight_layout()
    plt.show()

# NAUTICO
nautico_original = 'list1/images/img_1.png'
nautico_rgb = read_img(nautico_original)
nautico_cinza = converts_color_grayscale(nautico_rgb)

nautico_resizing = resizing_img(nautico_cinza, (1280, 720))
show_results(nautico_cinza, nautico_resizing, "image 1", "image 1 - Redimensionada")

# SANTA CRUZ
santa_original = 'list1/images/img_2.png'
santa_rgb = read_img(santa_original)
santa_cinza = converts_color_grayscale(santa_rgb)

santa_resizing = resizing_img(santa_cinza, (640, 480))
show_results(santa_cinza, santa_resizing, "image 2", "image 2 - Redimensionada")

# SPORT
sport_original = 'list1/images/img_3.png'
sport_rgb = read_img(sport_original)
sport_cinza = converts_color_grayscale(sport_rgb)

sport_resizing = resizing_img(sport_cinza, (640, 480))
show_results(sport_cinza, sport_resizing, "image 3", "image 3 - Redimensionada")