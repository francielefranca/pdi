'''
topic 2 - question 1
generate images with ruid
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagem_e_histograma(imagem, titulo, imgAgu, tituloAgu):
    plt.figure(figsize=(12, 6))

    # Mostra a imagem original
    plt.subplot(2, 2, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem ' + titulo)

    # Calcula e mostra o histograma da imagem original
    plt.subplot(2, 2, 2)
    plt.hist(imagem.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    # Mostra a imagem suavizada
    plt.subplot(2, 2, 3)
    plt.imshow(imgAgu, cmap='gray')
    plt.title('Imagem ' + tituloAgu)

    # Calcula e mostra o histograma da imagem suavizada
    plt.subplot(2, 2, 4)
    plt.hist(imgAgu.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma')
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

    # Calcula e mostra o histograma da diferença entre as imagens
    plt.figure(figsize=(8, 4))
    hist_diferenca = np.abs(imagem - imgAgu).ravel()
    plt.hist(hist_diferenca, bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histograma da Diferença')
    plt.xlabel('Diferença de Intensidade de Pixel')
    plt.ylabel('Frequência')
    plt.show()

    # Calcula e mostra os valores de PSNR e SSIM
    mse = np.mean((imagem - imgAgu) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    ssim = np.mean((2 * imagem * imgAgu + 1e-10) / (imagem ** 2 + imgAgu ** 2 + 1e-10))
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

def mostrar_imagem_no_dominio_da_frequencia(imagem, imagem_filtrada):
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

    img_filtrada = np.fft.fft2(imagem_filtrada)
    img_filtrada_shifted = np.fft.fftshift(img_filtrada)
    segunda_magnitude_spectrum = np.abs(img_filtrada_shifted)

    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(segunda_magnitude_spectrum), cmap='gray')
    plt.title('Espectro de Frequência - Ruidosa')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def gerar_imagem_ruidosa_unipolar(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)
    mascara_unipolar = np.random.choice([0, 1], size=img.shape, p=[0.2, 0.8])
    imagem_ruidosa_unipolar = img * mascara_unipolar
    mostrar_imagem_e_histograma(img_data, 'original', imagem_ruidosa_unipolar, 'ruidosa')
    mostrar_imagem_no_dominio_da_frequencia(img_data, imagem_ruidosa_unipolar)
    return imagem_ruidosa_unipolar

def gerar_imagem_ruidosa_bipolar(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)
    mascara_bipolar = np.random.choice([0, 1, 255], size=img.shape, p=[0.075, 0.075, 0.85])
    imagem_ruidosa_bipolar = img * mascara_bipolar
    mostrar_imagem_e_histograma(img_data, 'original', imagem_ruidosa_bipolar, 'ruidosa')
    mostrar_imagem_no_dominio_da_frequencia(img_data, imagem_ruidosa_bipolar)
    return imagem_ruidosa_bipolar

def gerar_imagem_ruidosa_gaussiano(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)
    ruido_gaussiano = np.random.normal(loc=20, scale=12, size=img.shape)
    imagem_ruidosa_gaussiano = np.clip(img + ruido_gaussiano, 0, 255).astype(np.uint8)
    mostrar_imagem_e_histograma(img_data, 'original', imagem_ruidosa_gaussiano, 'ruidosa')
    mostrar_imagem_no_dominio_da_frequencia(img_data, imagem_ruidosa_gaussiano)
    return imagem_ruidosa_gaussiano

s1_ruido = gerar_imagem_ruidosa_unipolar('list2/images/S1.jpg .webp')
s2_ruido = gerar_imagem_ruidosa_bipolar('list2/images/S2.jpg .webp')
s3_ruido = gerar_imagem_ruidosa_gaussiano('list2/images/S3.jpg .webp')

salvar_imagem(s1_ruido, 'list2/images/s1_ruido.jpg')
salvar_imagem(s2_ruido, 'list2/images/s2_ruido.jpg')
salvar_imagem(s3_ruido, 'list2/images/s3_ruido.jpg')