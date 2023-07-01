import cv2
import numpy as np

def procesar_imagen(ruta_imagen):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(ruta_imagen, 0)

    # Aplicar ecualización de histograma
    ecualizada = cv2.equalizeHist(imagen)

    # Aplicar binarización local adaptativa
    ventana = 31
    constante = 5
    binarizada = cv2.adaptiveThreshold(ecualizada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ventana, constante)

    # Aplicar el filtro de Sobel
    sobel_x = cv2.Sobel(binarizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binarizada, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular el gradiente de orientación
    gradiente_orientacion = np.arctan2(sobel_y, sobel_x)

    return gradiente_orientacion

# Ejemplo de uso
ruta_imagen = "dataset/101_1.tif"
imagen_original = cv2.imread(ruta_imagen, 0)
imagen_procesada = procesar_imagen(ruta_imagen)

# Mostrar la comparación entre la imagen original y la imagen procesada
cv2.imshow("Imagen original", imagen_original)
cv2.imshow("Imagen procesada", imagen_procesada)
cv2.waitKey(0)
cv2.destroyAllWindows()