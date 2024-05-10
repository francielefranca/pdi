# question 1 - enhancement in the frequency domain

'''
utilizar filtros de dominio de frequencia para suavizar as imagens:
- a1
- a2
- a3
- a4
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
    # Salva a imagem em escala de cinza
    cv2.imwrite(caminho, imagem)

'''
Filtro da Média (Box Blur): 
O filtro da média é um filtro simples que substitui 
cada pixel pela média dos valores dos pixels vizinhos. 
Ele ajuda a suavizar a imagem, reduzindo o ruído.
'''
def filtro_media(data, filter_size):
    indexer = filter_size // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            temp = []
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data[i][j] = temp[len(temp) // 2]
            temp = []
    return data
    
def blur(img):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(img)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

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

# Suavizar imagens - Filtro de Media
a1_media = filtro_media(a1_cinza, 3)
a2_media = filtro_media(a2_cinza, 3)
a3_media = filtro_media(a3_cinza, 3)
a4_media = filtro_media(a4_cinza, 3)
mostrar_imagem_e_histograma(imagem_a1, 'a1 - original', a1_media, 'a1 - media')
mostrar_imagem_e_histograma(imagem_a2, 'a2 - original', a2_media, 'a2 - media')
mostrar_imagem_e_histograma(imagem_a3, 'a3 - original', a3_media, 'a3 - media')
mostrar_imagem_e_histograma(imagem_a4, 'a4 - original', a4_media, 'a4 - media')

# Suavizar imagens - Filtro de Blur
a1_blur = blur(a1_cinza)
a2_blur = blur(a2_cinza)
a3_blur = blur(a3_cinza)
a4_blur = blur(a4_cinza)
mostrar_imagem_e_histograma(imagem_a1, 'a1 - original', a1_blur, 'a1 - blur')
mostrar_imagem_e_histograma(imagem_a2, 'a2 - original', a2_blur, 'a2 - blur')
mostrar_imagem_e_histograma(imagem_a3, 'a3 - original', a3_blur, 'a3 - blur')
mostrar_imagem_e_histograma(imagem_a4, 'a4 - original', a4_blur, 'a4 - blur')

# Salvando as imagens suavizadas
salvar_imagem(a1_media, 'list2/images/a1_media.jpg')
salvar_imagem(a2_media, 'list2/images/a2_media.jpg')
salvar_imagem(a3_media, 'list2/images/a3_media.jpg')
salvar_imagem(a4_media, 'list2/images/a4_media.jpg')

salvar_imagem(a1_blur, 'list2/images/a1_blur.jpg')
salvar_imagem(a2_blur, 'list2/images/a2_blur.jpg')
salvar_imagem(a3_blur, 'list2/images/a3_blur.jpg')
salvar_imagem(a4_blur, 'list2/images/a4_blur.jpg')
