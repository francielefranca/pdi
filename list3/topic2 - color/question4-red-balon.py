'''
segmentacao de imagens coloridas

2. A segmentação é uma etapa essencial no processamento de imagens para identificar e
extrair objetos de interesse. Segmente os balões vermelhos na imagem red_balon.webp e
os gatos na imagem gatos.jpeg
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Carregar a imagem original
image = cv2.imread("list3/images/red_balon.webp")

# Converter para o espaço de cores HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir os limites da cor vermelha (em HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Criar a máscara
mask = np.logical_and.reduce((hsv_image[:, :, 0] >= lower_red[0],
                               hsv_image[:, :, 1] >= lower_red[1],
                               hsv_image[:, :, 2] >= lower_red[2],
                               hsv_image[:, :, 0] <= upper_red[0],
                               hsv_image[:, :, 1] <= upper_red[1],
                               hsv_image[:, :, 2] <= upper_red[2]))

# Aplicar a máscara à imagem original
segmented_image = np.zeros_like(image)
segmented_image[mask] = image[mask]

# Salvar a imagem segmentada
cv2.imwrite("list3/images/segmented_red_balon.jpg", segmented_image)

# Exibir as imagens com Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("Imagem Segmentada")
plt.axis("off")

plt.show()

