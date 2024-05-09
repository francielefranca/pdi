# question 6 - challenge

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

def enhance_license_plate(image, output_path):
    # Aplicar um filtro de desfoque (motion blur)
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    blurred_image = cv2.filter2D(image, -1, kernel)

    # Subtrair a imagem borrada da imagem original para obter a nitidez
    sharp_image = image - blurred_image

    # Normalizar os valores de pixel para o intervalo [0, 255]
    sharp_image = np.clip(sharp_image, 0, 255).astype(np.uint8)

    # Salvar a imagem resultante
    '''
    cv2.imwrite(output_path, sharp_image)
    '''
    # Somar a imagem original com a imagem nítida
    combined_image = image + sharp_image

    # Normalizar os valores de pixel para o intervalo [0, 255]
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    # Salvar a imagem resultante
    cv2.imwrite(output_path, combined_image)
    show_results(car_cinza, combined_image, "Imagem - Placa de Carro","Imagem Composta")
    print(f"Imagem '{output_path}' salva com sucesso!")

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

car_original = 'list1/images/car.jpg'
car_rgb = load_image(car_original)
car_cinza = converts_color_grayscale(car_rgb)
output_image_path = 'list1/images/sharp_car.jpg'
enhance_license_plate(car_cinza, output_image_path)
