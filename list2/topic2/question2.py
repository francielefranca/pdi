# question 2
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

def wiener_filter(img_path, psf, noise_var):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img_data)
    """
    Aplica o filtro Wiener para restauração de imagens.

    Args:
        image (numpy.ndarray): A imagem corrompida.
        psf (numpy.ndarray): Função de espalhamento de ponto (PSF).
        noise_var (float): Variância do ruído.

    Returns:
        numpy.ndarray: Imagem restaurada.
    """
    H = np.fft.fft2(psf, s=image.shape)
    G = np.fft.fft2(image)
    F_hat = np.conj(H) / (np.abs(H) ** 2 + noise_var)
    restored_image = np.fft.ifft2(G * F_hat).real
    result = np.clip(restored_image, 0, 255).astype(np.uint8)

    mostrar_imagem_e_histograma(image, 'original', restored_image, 'restaurada - wiener')
    mostrar_imagem_no_dominio_da_frequencia(image, restored_image)
    return restored_image

def bayesian_restoration(img_path, psf, noise_variance):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image_corrupted = np.asarray(img_data)
    """
    Aplica o filtro de restauração de Bayes para restaurar imagens.

    Args:
        image_corrupted (numpy.ndarray): A imagem corrompida.
        psf (numpy.ndarray): Função de espalhamento de ponto (PSF).
        noise_variance (float): Variância do ruído.

    Returns:
        numpy.ndarray: Imagem restaurada.
    """
    H = np.fft.fft2(psf, s=image_corrupted.shape)
    G = np.fft.fft2(image_corrupted)
    F_hat = np.conj(H) / (np.abs(H) ** 2 + noise_variance)
    restored_image = np.fft.ifft2(G * F_hat).real
    np.clip(restored_image, 0, 255).astype(np.uint8)

    mostrar_imagem_e_histograma(image_corrupted, 'original', restored_image, 'restaurada - baye')
    mostrar_imagem_no_dominio_da_frequencia(image_corrupted, restored_image)
    return restored_image

def adaptive_median_filter(img_path, window_size):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img_data)
    """
    Aplica o filtro de mediana adaptativo a uma imagem.

    Args:
        image (numpy.ndarray): A imagem corrompida.
        window_size (int): Tamanho da janela (deve ser ímpar).

    Returns:
        numpy.ndarray: Imagem restaurada.
    """
    half_window = window_size // 2
    rows, cols = image.shape
    result = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            window = image[max(0, i - half_window):min(rows, i + half_window + 1),
                           max(0, j - half_window):min(cols, j + half_window + 1)]
            result[i, j] = np.median(window)

    mostrar_imagem_e_histograma(image, 'original', result, 'restaurada - adaptive_median')
    mostrar_imagem_no_dominio_da_frequencia(image, result)
    return result
    
def create_gaussian_psf(size, sigma):
    """
    Cria uma matriz bidimensional representando uma PSF gaussiana.

    Args:
        size (int): Tamanho da matriz (deve ser ímpar para simetria).
        sigma (float): Parâmetro de espalhamento (desvio padrão) da gaussiana.

    Returns:
        numpy.ndarray: Matriz da PSF gaussiana.
    """
    if size % 2 == 0:
        raise ValueError("O tamanho da matriz deve ser ímpar para simetria.")
    
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2
    psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return psf / np.sum(psf) 

def estimate_noise_variance(img_path):
    img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img_data)
    """
    Estima a variância do ruído em uma imagem.

    Args:
        image (numpy.ndarray): Imagem (em tons de cinza).

    Returns:
        float: Variância do ruído.
    """
    # Calcule a variância dos valores dos pixels
    return np.var(image)

psf = create_gaussian_psf(5, 1.8)
noise_variance1 = estimate_noise_variance('list2/images/s1_ruido.jpg')
noise_variance2 = estimate_noise_variance('list2/images/s2_ruido.jpg')
noise_variance3 = estimate_noise_variance('list2/images/s3_ruido.jpg')

s1_wiener = wiener_filter('list2/images/s1_ruido.jpg', psf, noise_variance1)
s2_wiener = wiener_filter('list2/images/s2_ruido.jpg', psf, noise_variance2)
s3_wiener = wiener_filter('list2/images/s3_ruido.jpg', psf, noise_variance3)

s1_baye = bayesian_restoration('list2/images/s1_ruido.jpg', psf, noise_variance1)
s2_baye = bayesian_restoration('list2/images/s2_ruido.jpg', psf, noise_variance2)
s3_baye = bayesian_restoration('list2/images/s3_ruido.jpg', psf, noise_variance3)

s1_med = adaptive_median_filter('list2/images/s1_ruido.jpg', 3)
s2_med = adaptive_median_filter('list2/images/s2_ruido.jpg', 13)
s3_med = adaptive_median_filter('list2/images/s3_ruido.jpg', 3)

salvar_imagem(s1_wiener, 'list2/images/s1_wiener.jpg')
salvar_imagem(s2_wiener, 'list2/images/s2_wiener.jpg')
salvar_imagem(s3_wiener, 'list2/images/s3_wiener.jpg')

salvar_imagem(s1_baye, 'list2/images/s1_baye.jpg')
salvar_imagem(s2_baye, 'list2/images/s2_baye.jpg')
salvar_imagem(s3_baye, 'list2/images/s3_baye.jpg')

salvar_imagem(s1_med, 'list2/images/s1_med.jpg')
salvar_imagem(s2_med, 'list2/images/s2_med.jpg')
salvar_imagem(s3_med, 'list2/images/s3_med.jpg')