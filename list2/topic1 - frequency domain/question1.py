# question 1 - enhancement in the frequency domain

'''
utilizar filtros de dominio de frequencia para suavizar as imagens:
- a1
- a2
- a3
- a4

model for colab: "/content/drive/MyDrive/pdi241/lista2/a1_sharp.jpg"
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagem_e_histograma(imagem, titulo, imgSuav, tituloSuav):
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
    plt.imshow(imgSuav, cmap='gray')
    plt.title('Imagem Suavizada ' + tituloSuav)

    # Calcula e mostra o histograma da imagem suavizada
    plt.subplot(2, 2, 4)  # Duas linhas, duas colunas, quarto subplot
    plt.hist(imgSuav.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    plt.tight_layout()  # Ajusta o espaçamento entre os subplots
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def suavizar_img(img_path, cutoff_frequency):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)
    
    # Aplicar a transformada de Fourier
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Criar um filtro passa-baixa
    rows, cols = img.shape
    rows, cols = img.shape

    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    mask[center_row - int(cutoff_frequency * center_row):center_row + int(cutoff_frequency * center_row),
         center_col - int(cutoff_frequency * center_col):center_col + int(cutoff_frequency * center_col)] = 1

    # Aplicar o filtro na transformada de Fourier
    img_fft_filtered = img_fft_shifted * mask

    # Inverter a transformada de Fourier
    img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_filtered)))
    
    mostrar_imagem_e_histograma(img_data, 'original', img_filtered, 'suavizada')

    return img_filtered

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

# Suavizar imagens 
a1_suav = suavizar_img('list2/images/A1.webp', 0.5)
a2_suav = suavizar_img('list2/images/A2.jpg', 0.2)
a3_suav = suavizar_img('list2/images/A3.jpg', 0.3)
a4_suav = suavizar_img('list2/images/A4.jpg', 0.2)

# Mostrar as duas imagens no domínio da frequência
mostrar_imagem_no_dominio_da_frequencia('list2/images/A1.webp', a1_suav)
mostrar_imagem_no_dominio_da_frequencia('list2/images/A2.jpg', a2_suav)
mostrar_imagem_no_dominio_da_frequencia('list2/images/A3.jpg', a3_suav)
mostrar_imagem_no_dominio_da_frequencia('list2/images/A4.jpg', a4_suav)

# Salvando as imagens suavizadas
salvar_imagem(a1_suav, 'list2/images/a1_suav.jpg')
salvar_imagem(a2_suav, 'list2/images/a2_suav.jpg')
salvar_imagem(a3_suav, 'list2/images/a3_suav.jpg')
salvar_imagem(a4_suav, 'list2/images/a4_suav.jpg')
