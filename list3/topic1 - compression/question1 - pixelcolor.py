'''
images: 
- capivara-3.webp
- commander.jpg
- passarela.jpg
- wagner_moura.webp

compressao de imagens: eliminar diferentes formas de redundancia existentes
redundancia mais comum: codificacao

objetivo: aplicar dois metodos de compressao sem perda visual em todas as images.
as tecnicas de compressao sem perda, sao rle (do formato pcx e bmp), lz e 
lzw(do utilitario winzip e dos formatos png e gif) utilizam-se da redundancia dos dados da imagem ou de tabelas rlenas

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
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Hide axes

    # Plot the compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB))
    plt.title("Compressed Image")
    plt.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

def salvar_imagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

def comprimir_imagem(image):
    """
    Compresses an image using quantization without loss of data.

    Args:
        image (numpy.ndarray): Input image (grayscale).

    Returns:
        numpy.ndarray: Compressed image.
    """
    # Define the quantization levels (e.g., 16 levels in your example)
    quantization_levels = np.arange(0, 256, 6)

    # Quantize the image
    compressed_image = np.digitize(image, quantization_levels) * 6

    return compressed_image.astype(np.uint8)

def calcular_redundancia_taxa_compressao(imagem_original, imagem_comprimida):
    tamanho_original = imagem_original.size
    tamanho_comprimido = imagem_comprimida.size

    redundancia_relativa = 1 - (tamanho_comprimido / tamanho_original)
    taxa_compressao = tamanho_original / tamanho_comprimido

    return redundancia_relativa, taxa_compressao

# Carregando as imagens
capivara = cv2.imread("list3/images/capivara-3.webp")
commander = cv2.imread("list3/images/commander.jpg")
passarela = cv2.imread("list3/images/passarela.jpg")
wagner = cv2.imread("list3/images/wagner_moura.webp")

capivara_ar = np.asarray(capivara)
commander_ar = np.asarray(commander)
passarela_ar = np.asarray(passarela)
wagner_ar = np.asarray(wagner)

capivara_pixel = comprimir_imagem(capivara_ar)
commander_pixel = comprimir_imagem(commander_ar)
passarela_pixel = comprimir_imagem(passarela_ar)
wagner_pixel = comprimir_imagem(wagner_ar)

plot_original_and_compressed(capivara, capivara_pixel)
plot_histograms(capivara_ar, capivara_pixel, 'Histograma Original', 'Histograma Comprimido (pixel)', 'capivara')
plot_original_and_compressed(commander, commander_pixel)
plot_histograms(commander_ar, commander_pixel, 'Histograma Original', 'Histograma Comprimido (pixel)', 'commander')
plot_original_and_compressed(passarela, passarela_pixel)
plot_histograms(passarela_ar, passarela_pixel, 'Histograma Original', 'Histograma Comprimido (pixel)', 'passarela')
plot_original_and_compressed(wagner, wagner_pixel)
plot_histograms(wagner_ar, wagner_pixel, 'Histograma Original', 'Histograma Comprimido (pixel)', 'wagner')

cap_red_pixel, cap_taxa_pixel = calcular_redundancia_taxa_compressao(capivara_ar, capivara_pixel)
print(f"Redundância Relativa (pixel) - Capivara: {cap_red_pixel:.4f}")
print(f"Taxa de Compressão (pixel) - Capivara: {cap_taxa_pixel:.4f}")

com_red_pixel, com_taxa_pixel = calcular_redundancia_taxa_compressao(commander_ar, commander_pixel)
print(f"Redundância Relativa (pixel) - Commander: {com_red_pixel:.4f}")
print(f"Taxa de Compressão (pixel) - Commander: {com_taxa_pixel:.4f}")

pas_red_pixel, pas_taxa_pixel = calcular_redundancia_taxa_compressao(passarela_ar, passarela_pixel)
print(f"Redundância Relativa (pixel) - Passarela: {pas_red_pixel:.4f}")
print(f"Taxa de Compressão (pixel) - Passarela: {pas_taxa_pixel:.4f}")

wag_red_pixel, wag_taxa_pixel = calcular_redundancia_taxa_compressao(wagner_ar, wagner_pixel)
print(f"Redundância Relativa (pixel) - Wagner: {wag_red_pixel:.4f}")
print(f"Taxa de Compressão (pixel) - Wagner: {wag_taxa_pixel:.4f}")

salvar_imagem(capivara_pixel, 'list3/images/capivara_pixel.png')
salvar_imagem(commander_pixel, 'list3/images/commander_pixel.png')
salvar_imagem(passarela_pixel, 'list3/images/passarela_pixel.png')
salvar_imagem(wagner_pixel, 'list3/images/wagner_pixel.png')

