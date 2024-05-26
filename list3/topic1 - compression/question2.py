'''
images: 
- capivara-3.webp
- commander.jpg
- passarela.jpg
- wagner_moura.webp

compressao de imagens: eliminar diferentes formas de redundancia existentes
redundancia mais comum: codificacao

objetivo: aplicar dois metodos de compressao com perda visual em todas as images.
as com perda visam reduzir o tamanho da imagem retirando informações não significantes para esse fim. 

model for colab: "/content/drive/MyDrive/pdi241/lista2/a1_sharp.jpg"
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função para plotar histogramas comparativos
def plot_histograms(original_image, compressed_image, title_original, title_compressed, image_name):
    # Calcule os histogramas
    original_hist, _ = np.histogram(original_image.ravel(), bins=256, range=(0, 256))
    compressed_image_array = np.array(compressed_image)  # Converta a lista em um array do NumPy
    compressed_hist, _ = np.histogram(compressed_image_array.ravel(), bins=256, range=(0, 256))

    # Crie o gráfico
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{title_original} - {image_name}")  
    plt.plot(original_hist, color='blue')
    plt.xlabel("Valor do Pixel")
    plt.ylabel("Frequência")

    plt.subplot(1, 2, 2)
    plt.title(f"{title_compressed} - {image_name}") 
    plt.plot(compressed_hist, color='red')
    plt.xlabel("Valor do Pixel")
    plt.ylabel("Frequência")

    plt.tight_layout()
    plt.show()

def plot_original_and_compressed(original_image, compressed_image):
    """
    Plots the original and compressed images side by side.

    Args:
        original_image (numpy.ndarray): The original grayscale image.
        compressed_image (numpy.ndarray): The compressed grayscale image.
    """
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')  # Hide axes

    # Plot the compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image, cmap='gray')
    plt.title("Compressed Image")
    plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def comprimir_com_perdas_interpolacao(imagem, fator):
    """
    Comprime uma imagem com perdas usando interpolação linear.

    Args:
        imagem (numpy.ndarray): A imagem original (em escala de cinza).
        fator (float): O fator de compressão (0.0 a 1.0).

    Returns:
        numpy.ndarray: A imagem comprimida.
    """
    # Calcula as novas dimensões da imagem comprimida
    nova_altura = int(imagem.shape[0] * fator)
    nova_largura = int(imagem.shape[1] * fator)

    # Redimensiona a imagem usando interpolação linear
    imagem_comprimida = np.zeros((nova_altura, nova_largura), dtype=np.uint8)
    for i in range(nova_altura):
        for j in range(nova_largura):
            imagem_comprimida[i, j] = imagem[int(i / fator), int(j / fator)]

    return imagem_comprimida

def comprimir_com_perdas_subamostragem(imagem, fator):
    """
    Comprime uma imagem com perdas usando subamostragem.

    Args:
        imagem (numpy.ndarray): A imagem original (em escala de cinza).
        fator (float): O fator de compressão (0.0 a 1.0).

    Returns:
        numpy.ndarray: A imagem comprimida.
    """
    # Calcula as novas dimensões da imagem comprimida
    nova_altura = int(imagem.shape[0] * fator)
    nova_largura = int(imagem.shape[1] * fator)

    # Seleciona apenas um subconjunto de pixels da imagem original
    imagem_comprimida = imagem[::int(1 / fator), ::int(1 / fator)]

    return imagem_comprimida

# Carregando as imagens
capivara = cv2.imread("list3/images/capivara-3.webp", cv2.IMREAD_GRAYSCALE)
commander = cv2.imread("list3/images/commander.jpg", cv2.IMREAD_GRAYSCALE)
passarela = cv2.imread("list3/images/passarela.jpg", cv2.IMREAD_GRAYSCALE)
wagner = cv2.imread("list3/images/wagner_moura.webp", cv2.IMREAD_GRAYSCALE)

capivara_ar = np.asarray(capivara)
commander_ar = np.asarray(commander)
passarela_ar = np.asarray(passarela)
wagner_ar = np.asarray(wagner)

capivara_inter = comprimir_com_perdas_interpolacao(capivara_ar, 0.5)
commander_inter = comprimir_com_perdas_interpolacao(commander_ar, 0.5)
passarela_inter = comprimir_com_perdas_interpolacao(passarela_ar, 0.5)
wagner_inter = comprimir_com_perdas_interpolacao(wagner_ar, 0.5)

plot_original_and_compressed(capivara, capivara_inter)
plot_histograms(capivara_ar, capivara_inter, 'Histograma Original', 'Histograma Comprimido (INTER)', 'capivara')
plot_original_and_compressed(commander, commander_inter)
plot_histograms(commander_ar, commander_inter, 'Histograma Original', 'Histograma Comprimido (INTER)', 'commander')
plot_original_and_compressed(passarela, passarela_inter)
plot_histograms(passarela_ar, passarela_inter, 'Histograma Original', 'Histograma Comprimido (INTER)', 'passarela')
plot_original_and_compressed(wagner, wagner_inter)
plot_histograms(wagner_ar, wagner_inter, 'Histograma Original', 'Histograma Comprimido (INTER)', 'wagner')

salvar_imagem(capivara_inter, 'list3/images/capivara_inter.jpeg')
salvar_imagem(commander_inter, 'list3/images/commander_inter.jpeg')
salvar_imagem(passarela_inter, 'list3/images/passarela_inter.jpeg')
salvar_imagem(wagner_inter, 'list3/images/wagner_inter.jpeg')

capivara_sub = comprimir_com_perdas_subamostragem(capivara_ar, 0.5)
commander_sub = comprimir_com_perdas_subamostragem(commander_ar, 0.5)
passarela_sub = comprimir_com_perdas_subamostragem(passarela_ar, 0.5)
wagner_sub = comprimir_com_perdas_subamostragem(wagner_ar, 0.5)

plot_original_and_compressed(capivara, capivara_sub)
plot_histograms(capivara_ar, capivara_sub, 'Histograma Original', 'Histograma Comprimido (SUB)', 'capivara')
plot_original_and_compressed(commander, commander_sub)
plot_histograms(commander_ar, commander_sub, 'Histograma Original', 'Histograma Comprimido (SUB)', 'commander')
plot_original_and_compressed(passarela, passarela_sub)
plot_histograms(passarela_ar, passarela_sub, 'Histograma Original', 'Histograma Comprimido (SUB)', 'passarela')
plot_original_and_compressed(wagner, wagner_sub)
plot_histograms(wagner_ar, wagner_sub, 'Histograma Original', 'Histograma Comprimido (SUB)', 'wagner')

salvar_imagem(capivara_sub, 'list3/images/capivara_sub.webp')
salvar_imagem(commander_sub, 'list3/images/commander_sub.webp')
salvar_imagem(passarela_sub, 'list3/images/passarela_sub.webp')
salvar_imagem(wagner_sub, 'list3/images/wagner_sub.webp')