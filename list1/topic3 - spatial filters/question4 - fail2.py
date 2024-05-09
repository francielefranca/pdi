# filtro de media ponderada - nao funcionou bem em nenhuma
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

def apply_weighted_average_filter(image, weights=None):
    """
    Aplica um filtro da média ponderada à imagem usando apenas NumPy.
    Args:
        image (numpy.ndarray): Imagem colorida (ou em escala de cinza).
        weights (numpy.ndarray): Matriz de pesos para a média ponderada.
    Returns:
        numpy.ndarray: Imagem filtrada.
    """
    if weights is None:
        # Usando pesos uniformes se nenhum for fornecido
        weights = np.ones((3, 3)) / 9.0  # Filtro 3x3

    pad = weights.shape[0] // 2
    filtered_image = np.zeros_like(image)

    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            window = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
            filtered_image[i, j] = np.sum(window * weights)

    return filtered_image.astype(np.uint8)

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

    plt.tight_layout()
    plt.show()

# Carregar as imagens
cc_w_img = load_img('list1/images/cc-w.jpg')
saturn_img = load_img('list1/images/saturn.png')
toro_img = load_img('list1/images/toro.jpg')

weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # Exemplo de pesos
#filtered_image = (original_image, weights)

# Aplicar filtro
filtered_cc = apply_weighted_average_filter(cc_w_img, weights)
filtered_saturn = apply_weighted_average_filter(saturn_img, weights)
filtered_toro = apply_weighted_average_filter(toro_img, weights)

# Mostrar os resultados
show_results(cc_w_img, filtered_cc, "CC original", "CC - filtrada")
show_results(saturn_img, filtered_saturn, "saturn original", "saturn - filtrada")
show_results(toro_img, filtered_toro, "toro original", "toro - filtrada")