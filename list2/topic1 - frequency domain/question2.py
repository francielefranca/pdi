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

'''
Valor Maior de r:
Quando você aumenta o valor de r, o filtro passa-alta se torna mais seletivo em relação às frequências altas.
Isso significa que apenas as bordas e detalhes mais significativos (com frequências mais altas) serão realçados.
A imagem resultante terá menos ruído e menos detalhes finos.
É útil quando você deseja realçar apenas as características mais proeminentes da imagem.
Valor Menor de r:
Com um valor menor de r, o filtro passa-alta se torna menos seletivo.
Ele realçará mais detalhes de alta frequência, incluindo bordas finas e pequenos detalhes.
A imagem resultante pode conter mais ruído e detalhes indesejados.
É útil quando você deseja realçar todos os detalhes, mesmo os menos proeminentes.
'''

def high_pass_filter(img_path, r):
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

def mostrar_imagem_no_dominio_da_frequencia(img_path, segunda_imagem):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imagem = np.asarray(img_data)

    # Aplicar a transformada de Fourier
    img_fft = np.fft.fft2(imagem)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Calcular o espectro de frequência (magnitude)
    magnitude_spectrum = np.abs(img_fft_shifted)

    # Visualizar o espectro de frequência
    plt.figure(figsize=(12, 6))

    # Primeira imagem no domínio da frequência
    plt.subplot(1, 2, 1)
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.title('Espectro de Frequência - Original')
    plt.axis('off')

    segunda_img_fft = np.fft.fft2(segunda_imagem)
    segunda_img_fft_shifted = np.fft.fftshift(segunda_img_fft)
    segunda_magnitude_spectrum = np.abs(segunda_img_fft_shifted)

    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(segunda_magnitude_spectrum), cmap='gray')
    plt.title('Espectro de Frequência - Suavizada')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Aguçando as imagens
sharp1 = high_pass_filter('list2/images/sharp1.png', 60)
mostrar_imagem_no_dominio_da_frequencia('list2/images/sharp1.png', sharp1)
# Salvando as imagens aguçadas
salvar_imagem(sharp1, 'list2/images/sharp_high.jpg')