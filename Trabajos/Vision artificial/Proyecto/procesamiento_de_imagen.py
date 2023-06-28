import cv2
import numpy as np

def procesamiento_de_imagen(imagen):
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Parámetros del filtro bilateral
    diametro_bilateral = 9  # Diámetro del filtro bilateral
    sigma_color_bilateral = 75  # Valor sigma para el componente de color en el filtro bilateral
    sigma_space_bilateral = 75  # Valor sigma para el componente espacial en el filtro bilateral

    # Aplicar filtro bilateral a la imagen en escala de grises
    imagen_filtrada = cv2.bilateralFilter(imagen_gris, diametro_bilateral, sigma_color_bilateral, sigma_space_bilateral)

    # Reducción de resolución utilizando decimación
    factor_reduccion = 2
    imagen_reducida = imagen_filtrada[::factor_reduccion, ::factor_reduccion]

    # Binarización utilizando el método de Otsu
    _, imagen_binarizada = cv2.threshold(imagen_reducida, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detección de minutiae
    puntos_bifurcacion, puntos_terminacion = detectar_minutiae(imagen_binarizada)

    return imagen_binarizada, puntos_bifurcacion, puntos_terminacion


def detectar_minutiae(imagen_binaria):
    # Crear un kernel de 3x3 para el operador de cruzamiento de crestas
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Aplicar el operador de cruzamiento de crestas para detectar los puntos de bifurcación y terminación
    puntos_bifurcacion = []
    puntos_terminacion = []

    # Recorrer la imagen binaria
    for i in range(1, imagen_binaria.shape[0] - 1):
        for j in range(1, imagen_binaria.shape[1] - 1):
            if imagen_binaria[i, j] == 0:  # Punto de la cresta
                vecindario = imagen_binaria[i-1:i+2, j-1:j+2]
                cruzamientos = np.sum(vecindario * kernel) / 255  # Contar los cruzamientos

                if cruzamientos == 3:  # Bifurcación
                    puntos_bifurcacion.append((j, i))
                elif cruzamientos == 1:  # Terminación
                    puntos_terminacion.append((j, i))

    return puntos_bifurcacion, puntos_terminacion