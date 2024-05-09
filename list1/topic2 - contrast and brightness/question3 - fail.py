# question 3 - adjust contrast and brightness
# solucao 1 - convertendo imagens em cinza e depois ajustando brilho/contraste (FAIL - pouco ajuste ou aparente nao modificacao)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(file_path):
    """
    Carrega uma imagem usando OpenCV.
    Args:
        file_path (str): Caminho para o arquivo de imagem.
    Returns:
        numpy.ndarray: Imagem colorida.
    """
    img = cv2.imread(file_path)
    return img

def convert_to_grayscale(img):
    """
    Converte uma imagem colorida para escala de cinza.
    Args:
        img (numpy.ndarray): Imagem colorida.
    Returns:
        numpy.ndarray: Imagem em escala de cinza.
    """
    gray_img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])
    return gray_img.astype(np.uint8)

def adjust_contrast_brightness(img, contrast_factor, brightness_factor):
    """
    Ajusta o contraste e o brilho da imagem.
    Args:
        img (numpy.ndarray): Imagem em escala de cinza.
        contrast_factor (float): Fator de ajuste de contraste (1.0 mantém inalterado).
        brightness_factor (int): Fator de ajuste de brilho (0 mantém inalterado).
    Returns:
        numpy.ndarray: Imagem ajustada.
    """
    adjusted_img = np.clip(contrast_factor * img + brightness_factor, 0, 255).astype(np.uint8)
    return adjusted_img

def show_results(original_img, adjusted_img, title_original, title_adjusted):
    """
    Mostra a imagem original e a imagem ajustada, juntamente com seus histogramas.
    Args:
        original_img (numpy.ndarray): Imagem original.
        adjusted_img (numpy.ndarray): Imagem ajustada.
        title_original (str): Título da imagem original.
        title_adjusted (str): Título da imagem ajustada.
    """
    plt.figure(figsize=(12, 6))

    # Plot da imagem original
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    # Histograma da imagem original
    plt.subplot(2, 2, 2)
    plt.hist(original_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.xlabel('Níveis de intensidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Imagem Original')

    # Plot da imagem ajustada
    plt.subplot(2, 2, 3)
    plt.imshow(adjusted_img, cmap='gray')
    plt.title(title_adjusted)
    plt.axis('off')

    # Histograma da imagem ajustada
    plt.subplot(2, 2, 4)
    plt.hist(adjusted_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.xlabel('Níveis de intensidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Imagem Ajustada')

    plt.tight_layout()
    plt.show()

# Exemplo de uso:
nome_arquivo_imagem_gato = 'list1/images/gato.jpg'
nome_arquivo_imagem_upe = 'list1/images/upe.jpg'
nome_arquivo_imagem_recife = 'list1/images/Recife_antigo.jpg'

gato = load_img(nome_arquivo_imagem_gato)
gato_gray = convert_to_grayscale(gato)
contrast_factor = 1.5  # Ajuste conforme necessário
brightness_factor = 30  # Ajuste conforme necessário
adjusted_gato = adjust_contrast_brightness(gato_gray, contrast_factor, brightness_factor)
show_results(gato_gray, adjusted_gato, 'Imagem Original', 'Imagem Ajustada')

recife = load_img(nome_arquivo_imagem_recife)
recife_gray = convert_to_grayscale(recife)
contrast_factor_rc = 2.5  # Ajuste conforme necessário
brightness_factor_rc = 15.5  # Ajuste conforme necessário
adjusted_recife = adjust_contrast_brightness(recife_gray, contrast_factor_rc, brightness_factor_rc)
show_results(recife_gray, adjusted_recife, 'Imagem Original', 'Imagem Ajustada')

upe = load_img(nome_arquivo_imagem_upe)
upe_gray = convert_to_grayscale(upe)
contrast_factor_upe = 1.5  # Ajuste conforme necessário
brightness_factor_upe = 30  # Ajuste conforme necessário
adjusted_upe = adjust_contrast_brightness(upe_gray, contrast_factor_upe, brightness_factor_upe)
show_results(upe_gray, adjusted_upe, 'Imagem Original', 'Imagem Ajustada')

