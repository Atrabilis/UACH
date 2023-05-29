#Este codigo codifica cada uno de los objetos de la imagen “objetos.jpg” utilizando
#código cadena y luego crea una nueva imagen en donde los objetos son generados
#a partir del código cadena obtenido anteriormente.

import numpy as np
import cv2 as cv

def detectar_bordes(imagen, umbral=240):
    # Convertir la imagen a escala de grises
    imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    # Transformar a imagen binaria utilizando el umbral
    _, imagen_binaria = cv.threshold(imagen_gris, umbral, 255, cv.THRESH_BINARY)

    # Obtener dimensiones de la imagen
    alto, ancho = imagen_binaria.shape[:2]

    # Matriz con coordenadas (fil, col) de los bordes
    bordes = []

    # Barrido de la matriz
    for fil in range(1, alto - 1):
        for col in range(1, ancho - 1):
            # Buscar bordes
            if imagen_binaria[fil, col] == 255:
                if (imagen_binaria[fil - 1, col] == 0 or imagen_binaria[fil + 1, col] == 0 or
                        imagen_binaria[fil, col - 1] == 0 or imagen_binaria[fil, col + 1] == 0):
                    bordes.append((fil, col))

    # Crear una matriz de bordes
    imagen_bordes = np.zeros((alto, ancho), dtype='uint8')
    for punto in bordes:
        imagen_bordes[punto] = 255

    return imagen_bordes


# Ejemplo de uso:
imagen_entrada = cv.imread('./Codigo_de_ayuda/objetos.jpg')
imagen_bordes = detectar_bordes(imagen_entrada)

# Mostrar la imagen con los bordes detectados
cv.imshow("Bordes detectados", imagen_bordes)
cv.imwrite("cadena.jpg", imagen_bordes)
cv.waitKey(0)
cv.destroyAllWindows()
