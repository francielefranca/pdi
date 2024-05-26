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

def run_length_encode(image):
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
    
    flat_image = []
    for value, count in rle_pairs:
        flat_image.extend([value] * count)

    decoded_image = np.array(flat_image).reshape(image.shape)  
    return decoded_image

def comprimir_imagem_svd(imagem, fator_compressao):
    if len(imagem.shape) == 3:
        # Converte a imagem para escala de cinza
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplica a SVD à matriz da imagem
    U, S, Vt = np.linalg.svd(imagem, full_matrices=False)

    # Calcula o número de valores singulares a serem mantidos
    num_valores_singulares = int(fator_compressao * min(imagem.shape))

    # Reduz a matriz S mantendo apenas os primeiros valores singulares
    S_comprimido = np.diag(S[:num_valores_singulares])

    # Reconstrói a imagem comprimida
    imagem_comprimida = np.dot(U[:, :num_valores_singulares], np.dot(S_comprimido, Vt[:num_valores_singulares, :]))

    if len(imagem.shape) == 3:
        imagem_comprimida = cv2.cvtColor(imagem_comprimida, cv2.COLOR_GRAY2BGR)

    return imagem_comprimida.astype(np.uint8)

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

capivara_rle = run_length_encode(capivara_ar)
commander_rle = run_length_encode(commander_ar)
passarela_rle = run_length_encode(passarela_ar)
wagner_rle = run_length_encode(wagner_ar)

plot_original_and_compressed(capivara, capivara_rle)
plot_histograms(capivara_ar, capivara_rle, 'Histograma Original', 'Histograma Comprimido (rle)', 'capivara')
plot_original_and_compressed(commander, commander_rle)
plot_histograms(commander_ar, commander_rle, 'Histograma Original', 'Histograma Comprimido (rle)', 'commander')
plot_original_and_compressed(passarela, passarela_rle)
plot_histograms(passarela_ar, passarela_rle, 'Histograma Original', 'Histograma Comprimido (rle)', 'passarela')
plot_original_and_compressed(wagner, wagner_rle)
plot_histograms(wagner_ar, wagner_rle, 'Histograma Original', 'Histograma Comprimido (rle)', 'wagner')

cap_red_rle, cap_taxa_rle = calcular_redundancia_taxa_compressao(capivara_ar, capivara_rle)
print(f"Redundância Relativa (RLE) - Capivara: {cap_red_rle:.4f}")
print(f"Taxa de Compressão (RLE) - Capivara: {cap_taxa_rle:.4f}")

com_red_rle, com_taxa_rle = calcular_redundancia_taxa_compressao(commander_ar, commander_rle)
print(f"Redundância Relativa (RLE) - Commander: {com_red_rle:.4f}")
print(f"Taxa de Compressão (RLE) - Commander: {com_taxa_rle:.4f}")

pas_red_rle, pas_taxa_rle = calcular_redundancia_taxa_compressao(passarela_ar, passarela_rle)
print(f"Redundância Relativa (RLE) - Passarela: {pas_red_rle:.4f}")
print(f"Taxa de Compressão (RLE) - Passarela: {pas_taxa_rle:.4f}")

wag_red_rle, wag_taxa_rle = calcular_redundancia_taxa_compressao(wagner_ar, wagner_rle)
print(f"Redundância Relativa (RLE) - Wagner: {wag_red_rle:.4f}")
print(f"Taxa de Compressão (RLE) - Wagner: {wag_taxa_rle:.4f}")

salvar_imagem(capivara_rle, 'list3/images/capivara_rle.png')
salvar_imagem(commander_rle, 'list3/images/commander_rle.png')
salvar_imagem(passarela_rle, 'list3/images/passarela_rle.png')
salvar_imagem(wagner_rle, 'list3/images/wagner_rle.png')

capivara_svd = comprimir_imagem_svd(capivara_ar, 0.5)
commander_svd = comprimir_imagem_svd(commander_ar, 0.5)
passarela_svd = comprimir_imagem_svd(passarela_ar, 0.5)
wagner_svd = comprimir_imagem_svd(wagner_ar, 0.5)

plot_original_and_compressed(capivara, capivara_svd)
plot_histograms(capivara_ar, capivara_svd, 'Histograma Original', 'Histograma Comprimido (svd)', 'capivara')
plot_original_and_compressed(commander, commander_svd)
plot_histograms(commander_ar, commander_svd, 'Histograma Original', 'Histograma Comprimido (svd)', 'commander')
plot_original_and_compressed(passarela, passarela_svd)
plot_histograms(passarela_ar, passarela_svd, 'Histograma Original', 'Histograma Comprimido (svd)', 'passarela')
plot_original_and_compressed(wagner, wagner_svd)
plot_histograms(wagner_ar, wagner_svd, 'Histograma Original', 'Histograma Comprimido (svd)', 'wagner')

cap_red_svd, cap_taxa_svd = calcular_redundancia_taxa_compressao(capivara_ar, capivara_svd)
print(f"Redundância Relativa (svdman) - Capivara: {cap_red_svd:.4f}")
print(f"Taxa de Compressão (svdman) - Capivara: {cap_taxa_svd:.4f}")

com_red_svd, com_taxa_svd = calcular_redundancia_taxa_compressao(commander_ar, commander_svd)
print(f"Redundância Relativa (svdman) - Commander: {com_red_svd:.4f}")
print(f"Taxa de Compressão (svdman) - Commander: {com_taxa_svd:.4f}")

pas_red_svd, pas_taxa_svd = calcular_redundancia_taxa_compressao(passarela_ar, passarela_svd)
print(f"Redundância Relativa (svdman) - Passarela: {pas_red_svd:.4f}")
print(f"Taxa de Compressão (svdman) - Passarela: {pas_taxa_svd:.4f}")

wag_red_svd, wag_taxa_svd = calcular_redundancia_taxa_compressao(wagner_ar, wagner_svd)
print(f"Redundância Relativa (svdman) - Wagner: {wag_red_svd:.4f}")
print(f"Taxa de Compressão (svdman) - Wagner: {wag_taxa_svd:.4f}")

salvar_imagem(capivara_svd, 'list3/images/capivara_svd.png')
salvar_imagem(commander_svd, 'list3/images/commander_svd.png')
salvar_imagem(passarela_svd, 'list3/images/passarela_svd.png')
salvar_imagem(wagner_svd, 'list3/images/wagner_svd.png')

