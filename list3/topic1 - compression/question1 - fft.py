'''
images: 
- capivara-3.webp
- commander.jpg
- passarela.jpg
- wagner_moura.webp

compressao de imagens: eliminar diferentes formas de redundancia existentes
redundancia mais comum: codificacao

objetivo: aplicar dois metodos de compressao sem perda visual em todas as images.
as tecnicas de compressao sem perda, sao fft (do formato pcx e bmp), lz e 
lzw(do utilitario winzip e dos formatos png e gif) utilizam-se da redundancia dos dados da imagem ou de tabelas fftnas

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

def compress_image_fft(image):
    # Separar os canais de cores
    #blue_channel, green_channel, red_channel = cv2.split(image)
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # Calcular a FFT para cada canal
    fft_blue = np.fft.fft2(blue_channel)
    fft_green = np.fft.fft2(green_channel)
    fft_red = np.fft.fft2(red_channel)

    # Combinar os resultados da FFT (pode ser uma média ou outra combinação)
    combined_fft = (fft_blue + fft_green + fft_red) / 3

    # Inverter a FFT para obter a imagem comprimida
    compressed_image = np.abs(np.fft.ifft2(combined_fft))

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

capivara_fft = compress_image_fft(capivara_ar)
commander_fft = compress_image_fft(commander_ar)
passarela_fft = compress_image_fft(passarela_ar)
wagner_fft = compress_image_fft(wagner_ar)

plot_original_and_compressed(capivara, capivara_fft)
plot_histograms(capivara_ar, capivara_fft, 'Histograma Original', 'Histograma Comprimido (fft)', 'capivara')
plot_original_and_compressed(commander, commander_fft)
plot_histograms(commander_ar, commander_fft, 'Histograma Original', 'Histograma Comprimido (fft)', 'commander')
plot_original_and_compressed(passarela, passarela_fft)
plot_histograms(passarela_ar, passarela_fft, 'Histograma Original', 'Histograma Comprimido (fft)', 'passarela')
plot_original_and_compressed(wagner, wagner_fft)
plot_histograms(wagner_ar, wagner_fft, 'Histograma Original', 'Histograma Comprimido (fft)', 'wagner')

cap_red_fft, cap_taxa_fft = calcular_redundancia_taxa_compressao(capivara_ar, capivara_fft)
print(f"Redundância Relativa (fft) - Capivara: {cap_red_fft:.4f}")
print(f"Taxa de Compressão (fft) - Capivara: {cap_taxa_fft:.4f}")

com_red_fft, com_taxa_fft = calcular_redundancia_taxa_compressao(commander_ar, commander_fft)
print(f"Redundância Relativa (fft) - Commander: {com_red_fft:.4f}")
print(f"Taxa de Compressão (fft) - Commander: {com_taxa_fft:.4f}")

pas_red_fft, pas_taxa_fft = calcular_redundancia_taxa_compressao(passarela_ar, passarela_fft)
print(f"Redundância Relativa (fft) - Passarela: {pas_red_fft:.4f}")
print(f"Taxa de Compressão (fft) - Passarela: {pas_taxa_fft:.4f}")

wag_red_fft, wag_taxa_fft = calcular_redundancia_taxa_compressao(wagner_ar, wagner_fft)
print(f"Redundância Relativa (fft) - Wagner: {wag_red_fft:.4f}")
print(f"Taxa de Compressão (fft) - Wagner: {wag_taxa_fft:.4f}")

salvar_imagem(capivara_fft, 'list3/images/capivara_fft.png')
salvar_imagem(commander_fft, 'list3/images/commander_fft.png')
salvar_imagem(passarela_fft, 'list3/images/passarela_fft.png')
salvar_imagem(wagner_fft, 'list3/images/wagner_fft.png')
