# question 1 - enhancement in the frequency domain

'''
explorar o dominio de fourier nas imagens:
- cubo
- wave
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
    plt.title('Espectro de Frequência - Filtrada')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def filter_pass(img_path, frequencia_min, frequencia_max):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    # Calcula a transformada de Fourier 2D da imagem
    F = np.fft.fftshift(np.fft.fft2(img))
    
    # Define a região de frequências permitida (passa-banda)
    mascara = np.zeros_like(F)
    mascara[(frequencia_min <= np.abs(F)) & (np.abs(F) <= frequencia_max)] = 1
    
    # Aplica a máscara no domínio das frequências
    F_filtrado = F * mascara
    
    # Calcula a transformada inversa para obter a imagem filtrada
    img_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtrado)))

    mostrar_imagem_e_histograma(img_data, 'original', img_filtrada.astype(np.uint8), 'passa banda')
    mostrar_imagem_no_dominio_da_frequencia(img_data, img_filtrada)
    return img_filtrada

def filter_retang(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    altura, largura = img.shape

    # Calcula o centro da imagem
    centro_x, centro_y = img.shape[1] // 2, img.shape[0] // 2
    
    # Cria uma máscara retangular
    mascara = np.zeros_like(img)
    mascara[centro_y - altura // 2:centro_y + altura // 2,
            centro_x - largura // 2:centro_x + largura // 2] = 1
    
    # Aplica a máscara no domínio das frequências
    F = np.fft.fftshift(np.fft.fft2(img))
    F_filtrado = F * mascara
    
    # Calcula a transformada inversa para obter a imagem filtrada
    img_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtrado)))

    mostrar_imagem_e_histograma(img_data, 'original', img_filtrada.astype(np.uint8), 'retangular')
    mostrar_imagem_no_dominio_da_frequencia(img_data, img_filtrada)
    return img_filtrada

def filter_vertical(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    # Calcula a transformada de Fourier 2D da imagem
    F = np.fft.fftshift(np.fft.fft2(img))
    
    # Cria uma máscara vertical
    mascara = np.zeros_like(F)
    mascara[:, F.shape[1] // 2] = 1
    
    # Aplica a máscara no domínio das frequências
    F_filtrado = F * mascara
    
    # Calcula a transformada inversa para obter a imagem filtrada
    img_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtrado)))

    mostrar_imagem_e_histograma(img_data, 'original', img_filtrada.astype(np.uint8), 'vertical')
    mostrar_imagem_no_dominio_da_frequencia(img_data, img_filtrada)
    return img_filtrada

def filtro_circular_nao_centrado(img_path, raio, centro_x, centro_y):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    # Calcula a transformada de Fourier 2D da imagem
    F = np.fft.fftshift(np.fft.fft2(img))
    
    # Cria uma máscara circular
    mascara = np.zeros_like(F)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            distancia_centro = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)
            if distancia_centro < raio:
                mascara[y, x] = 1
    
    # Aplica a máscara no domínio das frequências
    F_filtrado = F * mascara
    
    # Calcula a transformada inversa para obter a imagem filtrada
    img_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtrado)))

    mostrar_imagem_e_histograma(img_data, 'original', img_filtrada.astype(np.uint8), 'circular')
    mostrar_imagem_no_dominio_da_frequencia(img_data, img_filtrada)
    return img_filtrada

def filtro_fan(img_path, angulo_inicial, angulo_final):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    # Calcula a transformada de Fourier 2D da imagem
    F = np.fft.fftshift(np.fft.fft2(img))
    
    # Cria uma máscara em forma de leque
    mascara = np.zeros_like(F)
    h, w = img.shape
    centro_x, centro_y = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            angulo = np.arctan2(y - centro_y, x - centro_x)
            if angulo_inicial <= angulo <= angulo_final:
                mascara[y, x] = 1
    
    # Aplica a máscara no domínio das frequências
    F_filtrado = F * mascara
    
    # Calcula a transformada inversa para obter a imagem filtrada
    img_filtrada = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filtrado)))

    mostrar_imagem_e_histograma(img_data, 'original', img_filtrada.astype(np.uint8), 'fan')
    mostrar_imagem_no_dominio_da_frequencia(img_data, img_filtrada)
    return img_filtrada

# aplicando em cubo
cube_pass = filter_pass('list2/images/cubo.jpg', 10, 20)
cube_retang = filter_retang('list2/images/cubo.jpg')
cube_vert = filter_vertical('list2/images/cubo.jpg')

raio_circular = 50
centro_x = 128
centro_y = 128
cube_circular = filtro_circular_nao_centrado('list2/images/cubo.jpg', raio_circular, centro_x, centro_y)

angulo_inicial = np.pi / 4
angulo_final = 3 * np.pi / 4
cube_fan = filtro_fan('list2/images/cubo.jpg', angulo_inicial, angulo_final)

# aplicando em wave
wave_pass = filter_pass('list2/images/wave.jpg', 10, 20)
wave_retang = filter_retang('list2/images/wave.jpg')
wave_vert = filter_vertical('list2/images/wave.jpg')

raio_circular = 50
centro_x = 128
centro_y = 128
wave_circular = filtro_circular_nao_centrado('list2/images/wave.jpg', raio_circular, centro_x, centro_y)

angulo_inicial = np.pi / 4
angulo_final = 3 * np.pi / 4
wave_fan = filtro_fan('list2/images/wave.jpg', angulo_inicial, angulo_final)

# salvando as imagens
salvar_imagem(cube_pass, 'list2/images/cube_pass.jpg')
salvar_imagem(cube_retang, 'list2/images/cube_retang.jpg')
salvar_imagem(cube_vert, 'list2/images/cube_vert.jpg')
salvar_imagem(cube_circular, 'list2/images/cube_circular.jpg')
salvar_imagem(cube_fan, 'list2/images/cube_fan.jpg')

salvar_imagem(wave_pass, 'list2/images/wave_retang.jpg')
salvar_imagem(wave_retang, 'list2/images/wave_retang.jpg')
salvar_imagem(wave_vert, 'list2/images/cube_retang.jpg')
salvar_imagem(wave_circular, 'list2/images/wave_circular.jpg')
salvar_imagem(wave_fan, 'list2/images/wave_fan.jpg')