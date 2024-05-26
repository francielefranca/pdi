'''
remover ruido em imagens coloridas

1. Há várias técnicas para remover ruídos em imagens coloridas, como filtros (média,
gaussiano e mediana) e técnicas avançadas, como filtros adaptativos e algoritmos de
filtragem não local. Na questão apresentada, a imagem image_(2a).jpg contém ruído
gaussiano em alguns canais do formato HSI, enquanto a image_(2b).jpg apresenta ruído
de sal e pimenta em alguns canais RGB. É necessário remover o ruído dessas imagens.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função para plotar histogramas comparativos
def plot_histograms(original_image, no_noisy_img, title_original, title_noisy, image_name):
    # Calcule os histogramas
    original_hist, _ = np.histogram(original_image.ravel(), bins=256, range=(0, 256))
    no_noisy_img_array = np.array(no_noisy_img)  # Converta a lista em um array do NumPy
    noisy_hist, _ = np.histogram(no_noisy_img_array.ravel(), bins=256, range=(0, 256))

    # Crie o gráfico
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{title_original} - {image_name}")  
    plt.plot(original_hist, color='blue')
    plt.xlabel("Valor do Pixel")
    plt.ylabel("Frequência")

    plt.subplot(1, 2, 2)
    plt.title(f"{title_noisy} - {image_name}") 
    plt.plot(noisy_hist, color='red')
    plt.xlabel("Valor do Pixel")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.show()

def plot_original_and_noisy(original_image, no_noisy_img):
    """
    Plots the original and compressed images side by side.

    Args:
        original_image (numpy.ndarray): The original grayscale image.
        no_noisy_img (numpy.ndarray): The compressed grayscale image.
    """
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Hide axes

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(no_noisy_img, cv2.COLOR_BGR2RGB))
    plt.title("No Noisy Image")
    plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def filtro_bilateral(imagem):
    # Parâmetros do filtro bilateral
    sigma_color = 75
    sigma_space = 75
    
    # Aplicando filtro bilateral em cada canal
    canal_r = cv2.bilateralFilter(imagem[:, :, 0], 9, sigma_color, sigma_space)
    canal_g = cv2.bilateralFilter(imagem[:, :, 1], 9, sigma_color, sigma_space)
    canal_b = cv2.bilateralFilter(imagem[:, :, 2], 9, sigma_color, sigma_space)
    
    # Criando a imagem filtrada
    imagem_filtrada = np.stack([canal_r, canal_g, canal_b], axis=-1)
    
    return imagem_filtrada

def filtro_media(imagem):
    # Tamanho da janela de média
    janela = 5
    
    # Aplicando filtro de média em cada canal
    canal_r = np.mean(imagem[:, :, 0], axis=(0, 1))
    canal_g = np.mean(imagem[:, :, 1], axis=(0, 1))
    canal_b = np.mean(imagem[:, :, 2], axis=(0, 1))
    
    # Criando a imagem filtrada
    imagem_filtrada = np.stack([canal_r, canal_g, canal_b], axis=-1)
    
    return imagem_filtrada

def remove_salt_and_pepper_noise(image, window_size=5):
    """
    Remove ruído de sal e pimenta de uma imagem colorida.

    Args:
        image (numpy.ndarray): Imagem colorida (formato BGR).
        window_size (int): Tamanho da janela (padrão: 5).

    Returns:
        numpy.ndarray: Imagem sem ruído de sal e pimenta (colorida).
    """
    pad = window_size // 2
    filtered_image = np.zeros_like(image)

    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            window = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Calcular a mediana para cada canal de cor
            for channel in range(3):
                filtered_image[i, j, channel] = np.median(window[:, :, channel])

    return filtered_image.astype(np.uint8)

# Carregando as imagens
escada = cv2.imread("list3/images/image_(2a).jpg")
rachadura = cv2.imread("list3/images/image_(2b).jpg")

escada_ar = np.asarray(escada)
rachadura_ar = np.asarray(rachadura)

escada_gauss = filtro_bilateral(escada_ar)
rachadura_salt = remove_salt_and_pepper_noise(rachadura_ar)

plot_original_and_noisy(escada, escada_gauss)
plot_histograms(escada_ar, escada_gauss, 'Histograma Original', 'Histograma Sem Ruído (Gauss-Bilateral)', 'escada')
plot_original_and_noisy(rachadura, rachadura_salt)
plot_histograms(rachadura_ar, rachadura_salt, 'Histograma Original', 'Histograma Sem Ruído (Salt And Pepper)', 'rachadura')

salvar_imagem(escada_gauss, 'list3/images/escada_gauss.jpeg')
salvar_imagem(rachadura_salt, 'list3/images/rachadura_salt.jpeg')
