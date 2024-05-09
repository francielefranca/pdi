# question 2 - composition
# FAIL: resolution - only final

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

def converts_color_grayscale(image_rgb):
    # converts a color (RGB) image to grayscale.
    # use the formula: Y = 0.2989*R + 0.5870*G + 0.1140*B

    # extract the color channels (R, G, B)
    r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]

    # calculate grayscale intensity
    gray_intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray_intensity

def resizing_img(image, new_height, new_width):
    # resizes an image to the desired resolution.

    original_height, original_width = image.shape

    # calculates scale factors
    height_factor = new_height / original_height
    width_factor = new_width / original_width

    # creates an empty array for the resized image
    img_composition = np.zeros((new_height, new_width), dtype=np.uint8)

    # fills the matrix of the resized image using bilinear interpolation
    for i in range(new_height):
        for j in range(new_width):
            x = int(j / width_factor)
            y = int(i / height_factor)
            dx = j / width_factor - x
            dy = i / height_factor - y

            # bilinear interpolation
            if x + 1 < original_width and y + 1 < original_height:
                img_composition[i, j] = (
                    (1 - dx) * (1 - dy) * image[y, x] +
                    dx * (1 - dy) * image[y, x + 1] +
                    (1 - dx) * dy * image[y + 1, x] +
                    dx * dy * image[y + 1, x + 1]
                )
            else:
                # if the index is out of bounds, copy the original value
                img_composition[i, j] = image[y, x]

    return img_composition

def compose_images(img1, img2, img3):
    # compose imagens side a side
    composite_image = np.hstack((np.array(img1), np.array(img2), np.array(img3)))
    return composite_image

def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensidade de Cinza')
    plt.ylabel('Frequência')

def show_results(img1, img2, img3, img_composition, title_composition):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Imagem 1")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plot_histogram(img1, 'Histograma Original - Imagem 1')

    plt.subplot(2, 4, 3)
    plt.imshow(img2, cmap='gray')
    plt.title("Imagem 2")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plot_histogram(img2, 'Histograma Original - Imagem 2')

    plt.subplot(2, 4, 5)
    plt.imshow(img3, cmap='gray')
    plt.title("Imagem 3")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plot_histogram(img3, 'Histograma Original - Imagem 3')

    plt.subplot(2, 4, 7)
    plt.imshow(img_composition, cmap='gray')
    plt.title(title_composition)
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plot_histogram(img_composition, 'Histograma Composição')

    plt.tight_layout()
    plt.show()

nautico_original = 'list1/images/img_1.png'
nautico_rgb = load_image(nautico_original)
nautico_cinza = converts_color_grayscale(nautico_rgb)
img1 = resizing_img(nautico_cinza, 640, 480)

santa_original = 'list1/images/img_2.png'
santa_rgb = load_image(santa_original)
santa_cinza = converts_color_grayscale(santa_rgb)
img2 = resizing_img(santa_cinza, 640, 480)

sport_original = 'list1/images/img_3.png'
sport_rgb = load_image(sport_original)
sport_cinza = converts_color_grayscale(sport_rgb)
img3 = resizing_img(sport_cinza, 640, 480)

composite_image = compose_images(img1, img2, img3)
composite_image1 = resizing_img(composite_image, 1280, 720)
show_results(img1, img2, img3, composite_image1, "Imagem Composta")