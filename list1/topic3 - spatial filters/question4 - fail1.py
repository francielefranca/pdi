# filtro de media - nao funcionou muito bem nas imagens
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

def apply_smoothing(image, kernel_size=5):
    """
    Aplica um filtro de suavização (média) à imagem usando apenas NumPy.
    Args:
        image (numpy.ndarray): Imagem colorida (ou em escala de cinza).
        kernel_size (int): Tamanho do kernel para o filtro (ímpar).
    Returns:
        numpy.ndarray: Imagem suavizada.
    """
    # Cria um kernel de média com valores 1/(kernel_size^2)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Aplica a convolução usando o kernel
    smoothed_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        smoothed_image[:, :, channel] = np.convolve(image[:, :, channel].ravel(), kernel.ravel(), mode='same').reshape(image.shape[:2])

    return smoothed_image.astype(np.uint8)

def enhance_details(original_img, smoothed_img):
    """
    Realça os detalhes da imagem subtraindo a imagem suavizada da imagem original.
    Args:
        original_img (numpy.ndarray): Imagem original.
        smoothed_img (numpy.ndarray): Imagem suavizada.
    Returns:
        numpy.ndarray: Imagem com detalhes nítidos.
    """
    return np.clip(original_img - smoothed_img, 0, 255).astype(np.uint8)

def show_results(original_img, adjusted_img, suavized_img, title_original, title_adjusted, title_suavized):
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
    plt.subplot(2, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(title_original)
    plt.axis('off')

    # Histograma da imagem original
    plt.subplot(2, 3, 4)
    plt.hist(original_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.xlabel('Níveis de intensidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Imagem Original')

    # Plot da imagem ajustada
    plt.subplot(2, 3, 2)
    plt.imshow(adjusted_img, cmap='gray')
    plt.title(title_adjusted)
    plt.axis('off')

    # Histograma da imagem ajustada
    plt.subplot(2, 3, 5)
    plt.hist(adjusted_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.xlabel('Níveis de intensidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Imagem Ajustada Nítida')

    # Plot da imagem suavizada
    plt.subplot(2, 3, 3)
    plt.imshow(adjusted_img, cmap='gray')
    plt.title(title_suavized)
    plt.axis('off')

    # Histograma da imagem suavizada
    plt.subplot(2, 3, 6)
    plt.hist(suavized_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.xlabel('Níveis de intensidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Imagem Ajustada Suavizada')

    plt.tight_layout()
    plt.show()

# Carregar as imagens
cc_w_img = load_img('list1/images/cc-w.jpg')
saturn_img = load_img('list1/images/saturn.png')
toro_img = load_img('list1/images/toro.jpg')

# Aplicar suavização
cc_w_smoothed = apply_smoothing(cc_w_img)
saturn_smoothed = apply_smoothing(saturn_img)
toro_smoothed = apply_smoothing(toro_img)

# Realçar detalhes
cc_w_enhanced = enhance_details(cc_w_img, cc_w_smoothed)
saturn_enhanced = enhance_details(saturn_img, saturn_smoothed)
toro_enhanced = enhance_details(toro_img, toro_smoothed)

# Mostrar os resultados
show_results(cc_w_img, cc_w_smoothed, cc_w_enhanced, "CC original", "CC - detalhes nítidos", "CC - suavizada")
show_results(saturn_img, saturn_smoothed, saturn_enhanced, "saturn original", "saturn - detalhes nítidos", "saturn - suavizada")
show_results(toro_img, toro_smoothed, toro_enhanced, "toro original", "toro- detalhes nítidos", "toro - suavizada")
