
####Binarizacion###
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarizacion_global(imagen):
    _, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada

def binarizacion_local_adaptativa(imagen, tamano_bloque, constante):
    binarizada = cv2.adaptiveThreshold(imagen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, tamano_bloque, constante)
    return binarizada

def binarizacion_sauvola(imagen, ventana_tamano, k):
    margen = ventana_tamano // 2
    imagen_con_margen = cv2.copyMakeBorder(imagen, margen, margen, margen, margen, cv2.BORDER_CONSTANT, value=255)
    binarizada = np.zeros_like(imagen)

    for i in range(margen, imagen_con_margen.shape[0] - margen):
        for j in range(margen, imagen_con_margen.shape[1] - margen):
            ventana = imagen_con_margen[i - margen:i + margen + 1, j - margen:j + margen + 1]
            media = np.mean(ventana)
            desviacion_estandar = np.std(ventana)
            umbral = media * (1 + k * ((desviacion_estandar / 128) - 1))

            if imagen[i - margen, j - margen] > umbral:
                binarizada[i - margen, j - margen] = 255

    return binarizada

# Función para procesar la imagen
def procesar_imagen(ruta_imagen):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(ruta_imagen, 0)

    # Aplicar ecualización de histograma
    ecualizada = cv2.equalizeHist(imagen)

    return ecualizada

# Cargar la imagen original
ruta_imagen = "dataset/101_1.tif"
imagen_original = cv2.imread(ruta_imagen, 0)

# Procesar la imagen
imagen_procesada = procesar_imagen(ruta_imagen)

# Aplicar diferentes métodos de binarización
metodos = {
    "Global": binarizacion_global,
    "Local Adaptativa": lambda img: binarizacion_local_adaptativa(img, 31, 10),
    "Sauvola": lambda img: binarizacion_sauvola(img, 15, 0.2)
}

# Crear una figura para mostrar todas las imágenes
fig, axs = plt.subplots(1, len(metodos) + 1)
fig.suptitle("Comparación de Métodos de Binarización")

# Mostrar la imagen original
axs[0].imshow(imagen_original, cmap="gray")
axs[0].set_title("Original")

# Aplicar los diferentes métodos de binarización y mostrar los resultados
for i, (nombre_metodo, metodo) in enumerate(metodos.items()):
    binarizada = metodo(imagen_procesada)
    axs[i+1].imshow(binarizada, cmap="gray")
    axs[i+1].set_title(nombre_metodo)

# Ajustar los márgenes y espacios entre las imágenes
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, wspace=0.2)

# Mostrar todas las imágenes
plt.show()"""
#### Fin Binarizacion###

###Ridge direction
"""
import cv2
import numpy as np

def procesar_imagen(ruta_imagen):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(ruta_imagen, 0)

    # Aplicar ecualización de histograma
    ecualizada = cv2.equalizeHist(imagen)
    binarizada = cv2.adaptiveThreshold(ecualizada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)

    return binarizada

# Ruta de la imagen
ruta_imagen = "dataset/101_1.tif"

# Cargar imagen original y procesada
imagen_original = cv2.imread(ruta_imagen, 0)
imagen_procesada = procesar_imagen(ruta_imagen)

# Aplicar técnicas de estimación de dirección de crestas
filtro_gabor = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
imagen_gabor = cv2.filter2D(imagen_procesada, -1, filtro_gabor)

sobel_x = cv2.Sobel(imagen_procesada, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen_procesada, cv2.CV_64F, 0, 1, ksize=3)
gradiente_orientacion = np.arctan2(sobel_y, sobel_x)

fft = np.fft.fftshift(np.fft.fft2(imagen_procesada))
fft_log = np.log(1 + np.abs(fft))
fft_mag = np.abs(fft)
fft_mag = (fft_mag - np.min(fft_mag)) / (np.max(fft_mag) - np.min(fft_mag))
fft_mag = fft_mag * 255

# Convertir los valores de magnitud en una imagen visual
fft_img = np.uint8(fft_mag)

# Mostrar las imágenes resultantes
cv2.imshow("Imagen original", imagen_original)
cv2.imshow("Imagen procesada", imagen_procesada)
cv2.imshow("Filtro de Gabor", imagen_gabor)
cv2.imshow("Gradiente de orientación", gradiente_orientacion)
cv2.imshow("Transformada de Fourier", fft_log)
cv2.imshow("Transformada de Fourier", fft_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
###Fin Ridge direction