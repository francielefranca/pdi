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

def medianBlur(img_path, kernel_size):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img_data)
    """
    Aplica um filtro de mediana a uma imagem.

    Args:
        image (numpy.ndarray): A imagem de entrada (escala de cinza).
        kernel_size (int): O tamanho da janela do filtro de mediana.

    Returns:
        numpy.ndarray: A imagem filtrada.
    """
    # Verifica se o tamanho do kernel é ímpar
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser ímpar.")

    # Calcula o deslocamento para centralizar o kernel
    offset = kernel_size // 2

    # Cria uma cópia da imagem para armazenar o resultado
    filtered_image = np.copy(image)

    # Percorre os pixels da imagem
    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[1] - offset):
            # Extrai a vizinhança do pixel
            neighborhood = image[i - offset : i + offset + 1, j - offset : j + offset + 1]

            # Calcula a mediana dos valores na vizinhança
            median_value = np.median(neighborhood)

            # Atribui o valor da mediana ao pixel filtrado
            filtered_image[i, j] = median_value


    mostrar_imagem_e_histograma(image, 'original', filtered_image, 'sem efeito jelly')
    mostrar_imagem_no_dominio_da_frequencia(image, filtered_image)

    return filtered_image

no_jello = medianBlur('list2/images/jello.png', 3)
salvar_imagem(no_jello, 'list2/images/no_jello.png')