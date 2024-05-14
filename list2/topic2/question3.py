# where is original_1.jpg??
import numpy as np
import cv2

def restaured ():
    # Carregar as imagens
    noisy_image = cv2.imread("list2/images/noisy1.png", cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread("original_1.jpg", cv2.IMREAD_GRAYSCALE)

    # Aplicar a Transformada de Fourier
    noisy_fft = np.fft.fft2(noisy_image)
    original_fft = np.fft.fft2(original_image)

    # Filtragem no domínio da frequência
    kernel = np.array([[0.8, 0.4, 0.8], [0.4, 0.4, 0.4], [0.8, 0.4, 0.8]], dtype=np.float32)
    filtered_fft = noisy_fft * np.conj(np.fft.fft2(kernel))

    # Aplicar a Transformada de Fourier inversa
    filtered_image = np.abs(np.fft.ifft2(filtered_fft))

    # Comparar com a imagem original
    mse = np.mean((filtered_image - original_image) ** 2)
    print(f"Erro médio quadrático (MSE): {mse:.2f}")

    return filtered_image




