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
lzw(do utilitario winzip e dos formatos png e gif) utilizam-se da redundancia dos dados da imagem ou de tabelas internas

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

def run_length_encode(image, image_name):
    flat_image = image.ravel()
    rle_pairs = []
    current_value = flat_image[0]
    count = 1

    for pixel in flat_image[1:]:
        if pixel == current_value:
            count += 1
        else:
            rle_pairs.append((current_value, count))
            current_value = pixel
            count = 1

    rle_pairs.append((current_value, count))

    plot_histograms(image, rle_pairs, title_original="Histograma Original (RLE)", title_compressed="Histograma Comprimido (RLE)", image_name=image_name)
    
    return rle_pairs

def arithmetic_encode(image, image_name):
    # Transforma a imagem em uma sequência unidimensional
    flat_image = image.ravel()

    # Calcula a frequência de ocorrência de cada valor de pixel
    unique_values, counts = np.unique(flat_image, return_counts=True)
    total_pixels = flat_image.size
    probabilities = counts / total_pixels

    # Calcula os intervalos cumulativos
    cumulative_probs = np.cumsum(probabilities)
    cumulative_intervals = np.insert(cumulative_probs, 0, 0)

    # Codifica cada valor de pixel
    encoded_values = []
    lower_bound = 0
    upper_bound = 1

    for pixel in flat_image:
        symbol_index = np.where(unique_values == pixel)[0][0]
        new_lower = lower_bound + (upper_bound - lower_bound) * cumulative_intervals[symbol_index]
        new_upper = lower_bound + (upper_bound - lower_bound) * cumulative_intervals[symbol_index + 1]
        encoded_values.append((new_lower, new_upper))
        lower_bound, upper_bound = new_lower, new_upper

    plot_histograms(image, encoded_values, title_original="Histograma Original (Aritmética)", title_compressed="Histograma Comprimido (Aritmética)", image_name=image_name)

    return encoded_values

# Carregando as imagens
image_paths = ["list3/images/capivara-3.webp", "list3/images/commander.jpg", "list3/images/passarela.jpg", "list3/images/wagner_moura.webp"]

for img_path in image_paths:
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img_data)

    # Compressão RLE
    rle_result = run_length_encode(img, img_path)
    rle_size = len(rle_result) * 16  # Cada par (valor, contagem) ocupa 16 bits
    print(f"Compressão RLE para {img_path}: {rle_result}")

    # Compressão Aritmética
    encoded_result = arithmetic_encode(img, img_path)
    encoded_size = len(encoded_result) * 16
    print(f"Codificação Aritmética para {img_path}: {encoded_result}")

    # Redundância relativa
    original_size = img.size * 8  # Tamanho original em bits
    rle_redundancy = original_size / rle_size
    encoded_redudancy = original_size / encoded_size

    # Taxa de compressão
    rle_compression_ratio = original_size / rle_size
    encoded_compression_ratio = original_size / encoded_size

    print(f"Redundância RLE para {img_path}: {rle_redundancy:.2f}")
    print(f"Taxa de Compressão RLE para {img_path}: {rle_compression_ratio:.2f}")
    print(f"Redundância Encoded (Aritmética) para {img_path}: {encoded_redudancy:.2f}")
    print(f"Taxa de Compressão Encoded (Aritmética) para {img_path}: {encoded_compression_ratio:.2f}")

    

