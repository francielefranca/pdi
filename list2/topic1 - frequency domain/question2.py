# question 2 - enhancement in the frequency domain

'''
utilizar filtros de dominio de frequencia para aguçar as imagens:
- sharp1

Nesta função, calculamos a função H(u,v) com base no raio r 
e aplicamos o filtro passa-alta multiplicando a transformada 
de Fourier da imagem pela função H(u,v). 
A imagem resultante realça as características de alta frequência. 

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagem_e_histograma(imagem, titulo, imgAgu, tituloAgu):
    plt.figure(figsize=(12, 6))  # Aumentei o tamanho da figura para acomodar os subplots

    # Mostra a imagem original
    plt.subplot(2, 2, 1)  # Duas linhas, duas colunas, primeiro subplot
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem ' + titulo)

    # Calcula e mostra o histograma da imagem original
    plt.subplot(2, 2, 2)  # Duas linhas, duas colunas, segundo subplot
    plt.hist(imagem.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    # Mostra a imagem suavizada
    plt.subplot(2, 2, 3)  # Duas linhas, duas colunas, terceiro subplot
    plt.imshow(imgAgu, cmap='gray')
    plt.title('Imagem ' + tituloAgu)

    # Calcula e mostra o histograma da imagem suavizada
    plt.subplot(2, 2, 4)  # Duas linhas, duas colunas, quarto subplot
    plt.hist(imgAgu.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    plt.tight_layout()  # Ajusta o espaçamento entre os subplots
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def high_pass_filter(img_path, r):
    """
    Aplica um filtro passa-alta em uma imagem.

    Args:
        image (numpy.ndarray): A imagem de entrada (em escala de cinza).
        r (float): O raio do filtro (limiar para a função H(u,v)).

    Returns:
        numpy.ndarray: A imagem filtrada.
    """
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img_data)
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2
    # Cria uma matriz de coordenadas (u, v) para calcular a distância do centro
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((u - center_x)**2 + (v - center_y)**2)

    # Calcula a função H(u,v) com base no raio r
    H = np.where(distance < r, 0, 1)

    # Transformada de Fourier da imagem
    f_transform = np.fft.fft2(image)

    # Aplica o filtro multiplicando pela função H(u,v)
    filtered_transform = f_transform * H

    # Transformada inversa para obter a imagem filtrada
    filtered_image = np.abs(np.fft.ifft2(filtered_transform))

    mostrar_imagem_e_histograma(img_data, 'original', filtered_image.astype(np.uint8), 'aguçada')

    return filtered_image.astype(np.uint8)

# Aguçando as imagens
a1_sharp = high_pass_filter('list2/images/A1.webp', 30)
a2_sharp = high_pass_filter('list2/images/A2.jpg', 50)
a3_sharp = high_pass_filter('list2/images/A3.jpg', 250)
a4_sharp = high_pass_filter('list2/images/A4.jpg', 50)

# Salvando as imagens aguçadas
salvar_imagem(a1_sharp, 'list2/images/a1_sharp.jpg')
salvar_imagem(a2_sharp, 'list2/images/a2_sharp.jpg')
salvar_imagem(a3_sharp, 'list2/images/a3_sharp.jpg')
salvar_imagem(a4_sharp, 'list2/images/a4_sharp.jpg')