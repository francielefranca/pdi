# question 2 - enhancement in the frequency domain

'''
utilizar filtros de dominio de frequencia para aguçar as imagens:
- sharp1
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagem(caminho):
    # Lê a imagem usando o OpenCV
    imagem = cv2.imread(caminho)
    #cv2.IMREAD_UNCHANGED
    return imagem

def converter_para_escala_de_cinza(imagem):
    # Verifica o número de canais da imagem
    num_canais = imagem.shape[2]

    if num_canais > 1:
        # Calcula a média dos canais de cor para obter a escala de cinza
        imagem_escala_de_cinza = np.mean(imagem, axis=2, keepdims=True)
    else:
        # A imagem já está em escala de cinza (ou tem apenas um canal)
        imagem_escala_de_cinza = imagem

    return imagem_escala_de_cinza

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
    plt.title('Imagem Suavizada ' + tituloAgu)

    # Calcula e mostra o histograma da imagem suavizada
    plt.subplot(2, 2, 4)  # Duas linhas, duas colunas, quarto subplot
    plt.hist(imgAgu.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    plt.tight_layout()  # Ajusta o espaçamento entre os subplots
    plt.show()

def salvar_imagem(imagem, caminho):
    # Salva a imagem em escala de cinza
    cv2.imwrite(caminho, imagem)

def apply_sharpening(image, alpha):
    """
    Aplica afiação (sharpening) em uma imagem em escala de cinza.

    Args:
        image (numpy.ndarray): Matriz da imagem em escala de cinza (valores de 0 a 255).
        alpha (float): Fator de afiação (padrão é 1.0).

    Returns:
        numpy.ndarray: Imagem afiada.
    """
    # Crie um filtro de nitidez (sharpening filter)
    sharpen_filter = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    # Aplica o filtro à imagem usando convolução 2D
    sharpened_image = np.zeros_like(image, dtype=np.float64)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sharpened_image[i, j] = image[i, j] + alpha * np.sum(image[i-1:i+2, j-1:j+2] * sharpen_filter)

    # Garante que os valores estejam no intervalo [0, 255]
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

    return sharpened_image

# Carrega as imagens
imagem_a1 = carregar_imagem('list2/images/a1.webp')
imagem_a2 = carregar_imagem('list2/images/a2.jpg')
imagem_a3 = carregar_imagem('list2/images/a3.jpg')
imagem_a4 = carregar_imagem('list2/images/a4.jpg')

# Converte para escala de cinza
a1_cinza = converter_para_escala_de_cinza(imagem_a1)
a2_cinza = converter_para_escala_de_cinza(imagem_a2)
a3_cinza = converter_para_escala_de_cinza(imagem_a3)
a4_cinza = converter_para_escala_de_cinza(imagem_a4)

# Suavizar imagens - Filtro de Sharp
a1_sharp = apply_sharpening(a1_cinza, 1.5)
a2_sharp = apply_sharpening(a2_cinza, 2.0)
a3_sharp = apply_sharpening(a3_cinza, 2.0)
a4_sharp = apply_sharpening(a4_cinza, 2.0)
mostrar_imagem_e_histograma(imagem_a1, 'a1 - original', a1_sharp, 'a1 - sharp')
mostrar_imagem_e_histograma(imagem_a2, 'a2 - original', a2_sharp, 'a2 - sharp')
mostrar_imagem_e_histograma(imagem_a3, 'a3 - original', a3_sharp, 'a3 - sharp')
mostrar_imagem_e_histograma(imagem_a4, 'a4 - original', a4_sharp, 'a4 - sharp')

# Salvando as imagens suavizadas
salvar_imagem(a1_sharp, 'list2/images/a1_sharp.jpg')
salvar_imagem(a2_sharp, 'list2/images/a2_sharp.jpg')
salvar_imagem(a3_sharp, 'list2/images/a3_sharp.jpg')
salvar_imagem(a4_sharp, 'list2/images/a4_sharp.jpg')