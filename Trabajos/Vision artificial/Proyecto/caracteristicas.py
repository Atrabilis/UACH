import cv2
import numpy as np

def detectar_minutiae(imagen_binaria):
    # Obtener los contornos de la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar listas para almacenar los puntos de bifurcación y terminación
    puntos_bifurcacion = []
    puntos_terminacion = []

    # Iterar sobre los contornos
    for contorno in contornos:
        # Calcular el perímetro del contorno
        perimetro = cv2.arcLength(contorno, True)

        # Aproximar el contorno a un polígono
        epsilon = 0.01 * perimetro
        poligono = cv2.approxPolyDP(contorno, epsilon, True)

        # Calcular el número de vértices del polígono
        num_vertices = len(poligono)

        # Si el número de vértices es menor a un umbral, es un punto de terminación
        if num_vertices < 10:
            puntos_terminacion.append(poligono[0][0])

        # Si el número de vértices es mayor a un umbral, es un punto de bifurcación
        elif num_vertices > 10:
            puntos_bifurcacion.append(poligono[0][0])

    return puntos_bifurcacion, puntos_terminacion

# Cargar la imagen de huella dactilar
imagen_huella = cv2.imread("Dataset/101_1.tif", 0)  # Cargar como imagen en escala de grises

# Binarizar la imagen utilizando el método de Otsu
_, imagen_binarizada = cv2.threshold(imagen_huella, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Detectar los puntos de bifurcación y terminación en la imagen binarizada
puntos_bifurcacion, puntos_terminacion = detectar_minutiae(imagen_binarizada)

# Dibujar los puntos de bifurcación en la imagen original
imagen_resultante = cv2.cvtColor(imagen_huella, cv2.COLOR_GRAY2BGR)
for punto in puntos_bifurcacion:
    cv2.circle(imagen_resultante, tuple(punto), 3, (0, 255, 0), -1)

# Dibujar los puntos de terminación en la imagen original
for punto in puntos_terminacion:
    cv2.circle(imagen_resultante, tuple(punto), 3, (0, 0, 255), -1)

# Mostrar la imagen resultante con los puntos de bifurcación y terminación
cv2.imshow("Minutiae", imagen_resultante)
cv2.waitKey(0)
cv2.destroyAllWindows()